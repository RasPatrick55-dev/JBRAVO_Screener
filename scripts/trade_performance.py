"""Trade performance metrics, excursions, and cache management."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from utils import atomic_write_bytes
from utils.env import get_alpaca_creds

logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("JBRAVO_HOME", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE_DIR / "data"
TRADES_LOG_PATH = DATA_DIR / "trades_log.csv"
CACHE_PATH = DATA_DIR / "trade_performance_cache.json"
CACHE_MAX_AGE_HOURS = 6
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "7"))
DEFAULT_LOOKBACK_DAYS = 400
DEFAULT_MISSED_PROFIT_THRESHOLD = 3.0
DEFAULT_EXIT_EFFICIENCY_THRESHOLD = 60.0
DEFAULT_REBOUND_DAYS = 5
DEFAULT_REBOUND_THRESHOLD_PCT = 3.0
DEFAULT_TRAILING_STOP_TOLERANCE = 0.005
SUMMARY_WINDOWS = ("7D", "30D", "365D", "ALL")

BarsFetcher = Callable[[str, datetime, datetime, Any], pd.DataFrame]


def _safe_to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _ensure_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def fetch_with_backoff(
    fetcher: Callable[[], pd.DataFrame],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch data with exponential backoff when Alpaca responds with 429.

    Returns an empty DataFrame after exhausting retries to allow best-effort callers
    to continue processing other trades.
    """

    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fetcher()
        except Exception as exc:  # pragma: no cover - exercised via integration
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "error", None), "code", None)
            if status == 429 or "429" in str(exc):
                if attempt == max_attempts:
                    logger.warning("Alpaca 429 rate limit reached after %s attempts.", attempt)
                    break
                time.sleep(delay)
                delay *= 2
                continue
            logger.debug("Backoff fetch failed (attempt %s/%s): %s", attempt, max_attempts, exc, exc_info=True)
            break
    return pd.DataFrame()


def _normalize_pnl(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in ("net_pnl", "pnl", "netPnL", "net_pnl_usd"):
        if candidate in df.columns:
            if candidate != "pnl":
                df = df.rename(columns={candidate: "pnl"})
            break
    return df


def load_trades_log(base_dir: Path | str | None = None) -> pd.DataFrame:
    """Load trades_log.csv with safe defaults and normalized columns."""

    base = Path(base_dir) if base_dir else BASE_DIR
    path = Path(base) / "data" / "trades_log.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.warning("Unable to read trades log at %s", path, exc_info=True)
        return pd.DataFrame()

    df = _normalize_pnl(df)
    if "symbol" in df.columns:
        df["symbol"] = (
            df["symbol"].astype("string").str.upper().str.strip().fillna("")
        )
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for price_col in ("entry_price", "exit_price"):
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    if "pnl" not in df.columns and {"qty", "entry_price", "exit_price"}.issubset(df.columns):
        df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["qty"]
    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    if "return_pct" not in df.columns and {"entry_price", "exit_price"}.issubset(df.columns):
        entry_price = pd.to_numeric(df["entry_price"], errors="coerce")
        exit_price = pd.to_numeric(df["exit_price"], errors="coerce")
        df["return_pct"] = np.where(
            entry_price > 0,
            (exit_price / entry_price - 1) * 100,
            0.0,
        )
    elif "return_pct" in df.columns:
        df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df = df.dropna(subset=["symbol"])
    if "exit_time" in df.columns:
        df = df.dropna(subset=["exit_time"])
    return df


def _normalize_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    if "timestamp" not in frame.columns and frame.index.name == "timestamp":
        frame = frame.reset_index()
    if "timestamp" not in frame.columns and "time" in frame.columns:
        frame = frame.rename(columns={"time": "timestamp"})
    if "timestamp" not in frame.columns:
        frame["timestamp"] = frame.index
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for col in ("high", "low", "close", "open"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.dropna(subset=["timestamp"], inplace=True)
    return frame


def _prepare_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frame = _normalize_pnl(df).copy()

    for col in ("entry_time", "exit_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

    qty = pd.to_numeric(frame.get("qty", pd.Series(np.nan, index=frame.index)), errors="coerce")
    entry_price = pd.to_numeric(frame.get("entry_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exit_price = pd.to_numeric(frame.get("exit_price", pd.Series(np.nan, index=frame.index)), errors="coerce")

    pnl_series = frame.get("pnl")
    computed_pnl = (exit_price - entry_price) * qty
    if pnl_series is None:
        frame["pnl"] = computed_pnl
    else:
        frame["pnl"] = pd.to_numeric(pnl_series, errors="coerce")
        missing_mask = frame["pnl"].isna()
        if missing_mask.any():
            frame.loc[missing_mask, "pnl"] = computed_pnl.loc[missing_mask]

    computed_returns = np.where(
        entry_price > 0,
        (exit_price / entry_price - 1) * 100,
        0.0,
    )
    return_pct = frame.get("return_pct")
    if return_pct is None:
        frame["return_pct"] = computed_returns
    else:
        frame["return_pct"] = pd.to_numeric(return_pct, errors="coerce").fillna(
            pd.Series(computed_returns, index=frame.index)
        )

    frame["exit_efficiency_pct"] = pd.to_numeric(
        frame.get("exit_efficiency_pct", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    if "hold_days" in frame.columns:
        frame["hold_days"] = pd.to_numeric(frame["hold_days"], errors="coerce")
    elif {"entry_time", "exit_time"}.issubset(frame.columns):
        frame["hold_days"] = (
            frame["exit_time"] - frame["entry_time"]
        ).dt.total_seconds() / (24 * 3600)

    return frame


def fetch_bars_with_backoff(
    data_client: StockHistoricalDataClient | None,
    request: StockBarsRequest,
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> pd.DataFrame:
    if data_client is None:
        return pd.DataFrame()

    def _do_fetch() -> pd.DataFrame:
        response = data_client.get_stock_bars(request)
        candidate = getattr(response, "df", pd.DataFrame())
        if candidate is None:
            return pd.DataFrame()
        return _normalize_bars_frame(candidate)

    return fetch_with_backoff(_do_fetch, max_attempts=max_attempts, base_delay=base_delay)


def _fetch_bars_for_trade(
    symbol: str,
    start: datetime,
    end: datetime,
    data_client: StockHistoricalDataClient | None,
    feed: str,
    bar_fetcher: BarsFetcher | None = None,
    *,
    use_intraday: bool = False,
) -> pd.DataFrame:
    if bar_fetcher is not None:
        try:
            fetched = bar_fetcher(symbol, start, end, None)
            if isinstance(fetched, pd.DataFrame):
                normalized = _normalize_bars_frame(fetched)
                if not normalized.empty:
                    return normalized
        except Exception:
            logger.debug("Custom bar_fetcher failed for %s", symbol, exc_info=True)
    if data_client is None:
        return pd.DataFrame()

    timeframe_candidates: list[Any] = [TimeFrame.Day]
    if use_intraday:
        timeframe_candidates.extend([TimeFrame.Hour, TimeFrame(15, TimeFrameUnit.Minute)])
    for timeframe in timeframe_candidates:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
        )
        bars = fetch_bars_with_backoff(data_client, request)
        if bars.empty:
            continue
        bars = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)]
        if not bars.empty:
            return bars
    return pd.DataFrame()


def compute_trade_excursions(
    df: pd.DataFrame,
    data_client: StockHistoricalDataClient | None,
    *,
    bar_fetcher: BarsFetcher | None = None,
    post_exit_days: int = DEFAULT_REBOUND_DAYS,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Compute per-trade excursions (MFE/MAE) using intraday bars."""

    if df.empty:
        return df

    feed = os.getenv("ALPACA_DATA_FEED", "iex")

    frame = df.copy()
    now = datetime.now(timezone.utc)
    if lookback_days is not None and "exit_time" in frame.columns:
        cutoff = now - timedelta(days=lookback_days)
        frame = frame[frame["exit_time"] >= cutoff]

    peak_prices: list[float | None] = []
    trough_prices: list[float | None] = []
    post_exit_peaks: list[float | None] = []

    for _, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").upper()
        entry_ts = row.get("entry_time")
        exit_ts = row.get("exit_time")
        entry_price = _safe_to_float(row.get("entry_price"))
        exit_price = _safe_to_float(row.get("exit_price"))
        exit_reason = _infer_exit_reason(row)
        trailing_pct = _safe_to_float(
            row.get("trailing_pct")
            or row.get("trailing_percent")
            or row.get("trailing_stop_pct")
        )
        estimated_peak = exit_price / 0.97 if exit_reason == "TrailingStop" and trailing_pct == 3.0 and exit_price else None

        if not isinstance(entry_ts, pd.Timestamp) or pd.isna(entry_ts) or not isinstance(exit_ts, pd.Timestamp):
            peak_prices.append(estimated_peak if estimated_peak is not None else np.nan)
            trough_prices.append(np.nan)
            post_exit_peaks.append(estimated_peak if estimated_peak is not None else np.nan)
            continue

        start = entry_ts.to_pydatetime()
        end = exit_ts.to_pydatetime()
        post_end = end + timedelta(days=max(post_exit_days, 0))

        bars = _fetch_bars_for_trade(symbol, start, post_end, data_client, feed, bar_fetcher=bar_fetcher)
        if bars.empty:
            peak_prices.append(estimated_peak if estimated_peak is not None else np.nan)
            trough_prices.append(np.nan)
            post_exit_peaks.append(estimated_peak if estimated_peak is not None else np.nan)
            continue

        window_bars = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)]
        post_bars = bars[(bars["timestamp"] > end)]

        def _series_peak(candidate: pd.Series) -> float | None:
            if candidate.empty:
                return None
            value = candidate.max()
            return float(value) if not pd.isna(value) else None

        def _series_trough(candidate: pd.Series) -> float | None:
            if candidate.empty:
                return None
            value = candidate.min()
            return float(value) if not pd.isna(value) else None

        peak = _series_peak(window_bars.get("high", pd.Series(dtype=float)))
        trough = _series_trough(window_bars.get("low", pd.Series(dtype=float)))
        if peak is None and "close" in window_bars.columns:
            peak = _series_peak(window_bars["close"])
        if trough is None and "close" in window_bars.columns:
            trough = _series_trough(window_bars["close"])

        if peak is None and estimated_peak is not None:
            peak = estimated_peak

        peak_prices.append(peak if peak is not None else np.nan)
        trough_prices.append(trough if trough is not None else (entry_price or exit_price or np.nan))

        post_peak = _series_peak(post_bars.get("high", pd.Series(dtype=float))) or _series_peak(
            post_bars.get("close", pd.Series(dtype=float))
        )
        post_exit_peaks.append(post_peak if post_peak is not None else peak_prices[-1])

    frame["peak_price"] = peak_prices
    frame["trough_price"] = trough_prices
    frame["post_exit_peak"] = post_exit_peaks
    return frame


def _infer_exit_reason(row: Mapping[str, Any]) -> str:
    raw_reason = row.get("exit_reason") or row.get("reason")
    if isinstance(raw_reason, str) and raw_reason.strip():
        return raw_reason.strip()

    haystack_parts: list[str] = []
    for key in ("order_type", "notes", "exit_notes"):
        value = row.get(key)
        if isinstance(value, str) and value:
            haystack_parts.append(value)
    haystack = " ".join(haystack_parts).lower()
    hold_days = _safe_to_float(row.get("hold_days")) or 0.0

    if "trail" in haystack or "trailing" in haystack:
        return "TrailingStop"
    if hold_days >= MAX_HOLD_DAYS:
        return "MaxHold"
    return "Other"


def _normalize_trailing_pct(value: Any) -> float | None:
    pct = _safe_to_float(value)
    if pct is None:
        return None
    if pct > 1:
        return pct / 100.0
    return pct


def _is_trailing_stop_exit(row: Mapping[str, Any], *, tolerance: float = DEFAULT_TRAILING_STOP_TOLERANCE) -> bool:
    reason = (row.get("exit_reason") or row.get("reason") or "").strip()
    if reason == "TrailingStop":
        return True

    trailing_pct = _normalize_trailing_pct(
        row.get("trailing_pct")
        or row.get("trailing_percent")
        or row.get("trailing_stop_pct")
    )
    peak_price = _safe_to_float(row.get("peak_price"))
    exit_price = _safe_to_float(row.get("exit_price"))
    if trailing_pct is None or peak_price is None or exit_price is None:
        return False
    if peak_price <= 0 or exit_price >= peak_price:
        return False
    observed = (peak_price - exit_price) / peak_price
    return abs(observed - trailing_pct) < tolerance


def compute_exit_quality_columns(
    df: pd.DataFrame,
    *,
    missed_profit_threshold: float = DEFAULT_MISSED_PROFIT_THRESHOLD,
    exit_eff_threshold: float = DEFAULT_EXIT_EFFICIENCY_THRESHOLD,
    rebound_threshold_pct: float = DEFAULT_REBOUND_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Add return_pct, hold_days, exit efficiency, and sold-too-soon signals."""

    if df.empty:
        return df
    frame = _prepare_summary_frame(df)

    entry_price = pd.to_numeric(frame.get("entry_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exit_price = pd.to_numeric(frame.get("exit_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    peak_price = pd.to_numeric(frame.get("peak_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    trough_price = pd.to_numeric(frame.get("trough_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    post_exit_peak = pd.to_numeric(
        frame.get("post_exit_peak", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )

    frame["mfe_pct"] = np.where(
        (entry_price > 0) & (peak_price.notna()),
        (peak_price - entry_price) / entry_price * 100,
        np.nan,
    )
    frame["mae_pct"] = np.where(
        (entry_price > 0) & (trough_price.notna()),
        (trough_price - entry_price) / entry_price * 100,
        np.nan,
    )

    valid_peak_mask = (peak_price.notna()) & (entry_price.notna()) & (peak_price > entry_price)
    exit_eff = np.where(
        valid_peak_mask,
        (exit_price / peak_price) * 100,
        0.0,
    )
    frame["exit_efficiency_pct"] = np.clip(exit_eff, 0, 100)
    frame["missed_profit_pct"] = np.where(
        peak_price > 0,
        (peak_price - exit_price) / peak_price * 100,
        np.nan,
    )
    frame["post_exit_rebound"] = np.where(
        (post_exit_peak.notna()) & (exit_price.notna()),
        post_exit_peak >= exit_price * (1 + rebound_threshold_pct / 100),
        False,
    )

    if "exit_reason" not in frame.columns:
        frame["exit_reason"] = [
            _infer_exit_reason(row) for row in frame.to_dict(orient="records")
        ]

    frame["sold_too_soon"] = (
        (frame["missed_profit_pct"] > missed_profit_threshold)
        | (frame["exit_efficiency_pct"] < exit_eff_threshold)
        | frame["post_exit_rebound"].fillna(False)
    )
    frame["is_trailing_stop_exit"] = [
        _is_trailing_stop_exit(row) for row in frame.to_dict(orient="records")
    ]
    return frame


def compute_rebound_metrics(
    df: pd.DataFrame,
    data_client: StockHistoricalDataClient | None,
    *,
    rebound_window_days: int = DEFAULT_REBOUND_DAYS,
    rebound_threshold_pct: float = DEFAULT_REBOUND_THRESHOLD_PCT,
    bar_fetcher: BarsFetcher | None = None,
    trailing_tolerance: float = DEFAULT_TRAILING_STOP_TOLERANCE,
) -> pd.DataFrame:
    """Compute post-exit rebound metrics for trailing-stop exits using daily bars."""

    frame = df.copy()
    if frame.empty:
        frame["rebound_window_days"] = rebound_window_days
        frame["post_exit_high"] = np.nan
        frame["rebound_pct"] = np.nan
        frame["rebounded"] = False
        if "is_trailing_stop_exit" not in frame.columns:
            frame["is_trailing_stop_exit"] = False
        return frame

    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    if "exit_reason" not in frame.columns:
        frame["exit_reason"] = [
            _infer_exit_reason(row) for row in frame.to_dict(orient="records")
        ]
    if "is_trailing_stop_exit" not in frame.columns:
        frame["is_trailing_stop_exit"] = [
            _is_trailing_stop_exit(row, tolerance=trailing_tolerance)
            for row in frame.to_dict(orient="records")
        ]

    rebound_window_days = int(rebound_window_days) if rebound_window_days is not None else DEFAULT_REBOUND_DAYS
    frame["rebound_window_days"] = rebound_window_days

    post_exit_highs: list[float | None] = [np.nan] * len(frame.index)
    rebound_pcts: list[float | None] = [np.nan] * len(frame.index)
    rebound_flags: list[bool] = [False] * len(frame.index)

    for idx, row in frame.iterrows():
        if not bool(row.get("is_trailing_stop_exit")):
            continue
        symbol = str(row.get("symbol") or "").upper()
        exit_ts = row.get("exit_time")
        exit_price = _safe_to_float(row.get("exit_price"))
        if not symbol or exit_price is None or not isinstance(exit_ts, pd.Timestamp):
            continue
        start = (exit_ts + timedelta(days=1)).to_pydatetime()
        end = (exit_ts + timedelta(days=max(rebound_window_days, 0))).to_pydatetime()

        try:
            bars = _fetch_bars_for_trade(
                symbol,
                start,
                end,
                data_client,
                feed,
                bar_fetcher=bar_fetcher,
                use_intraday=False,
            )
        except Exception:
            logger.debug("Post-exit bar fetch failed for %s", symbol, exc_info=True)
            continue
        if bars.empty:
            continue
        highs = bars.get("high", pd.Series(dtype=float))
        if highs.empty:
            continue
        post_high = highs.max()
        post_exit_highs[idx] = post_high
        if exit_price and not pd.isna(post_high):
            pct = (post_high - exit_price) / exit_price * 100
            rebound_pcts[idx] = pct
            rebound_flags[idx] = pct >= rebound_threshold_pct

    frame["post_exit_high"] = post_exit_highs
    frame["rebound_pct"] = rebound_pcts
    frame["rebounded"] = rebound_flags
    return frame


def summarize_by_window(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize trades for predefined time windows."""

    def _safe_mean(series: pd.Series | None) -> float:
        if series is None or len(series.index) == 0:
            return 0.0
        series = series.dropna()
        if series.empty:
            return 0.0
        try:
            value = float(series.mean())
        except Exception:
            return 0.0
        return 0.0 if math.isnan(value) else value

    frame = _prepare_summary_frame(df)
    now = datetime.now(timezone.utc)
    windows: dict[str, datetime | None] = {
        "7D": now - timedelta(days=7),
        "30D": now - timedelta(days=30),
        "365D": now - timedelta(days=365),
        "ALL": None,
    }
    summaries: dict[str, dict[str, float]] = {}

    time_series: pd.Series | None = None
    if "exit_time" in frame.columns:
        time_series = frame["exit_time"]
    elif "entry_time" in frame.columns:
        time_series = frame["entry_time"]

    default_window = {
        "trades": 0,
        "net_pnl": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "win_rate_pct": 0.0,
        "avg_return_pct": 0.0,
        "avg_hold_days": 0.0,
        "avg_exit_efficiency_pct": 0.0,
        "sold_too_soon": 0,
        "stop_exits": 0,
        "rebounds": 0,
        "rebound_rate": 0.0,
        "avg_rebound_pct": 0.0,
    }
    for label, cutoff in windows.items():
        subset = frame
        if cutoff is not None and time_series is not None:
            subset = frame[time_series >= cutoff]
        trades = int(len(subset.index))
        pnl_series = subset.get("pnl", pd.Series(dtype=float))
        net_pnl = float(np.nansum(pnl_series)) if trades else 0.0
        win_rate = float((pnl_series > 0).mean()) if trades else 0.0
        win_rate_pct = win_rate * 100.0
        avg_return = _safe_mean(subset.get("return_pct", pd.Series(dtype=float))) if trades else 0.0
        hold_days = _safe_mean(subset.get("hold_days", pd.Series(dtype=float))) if trades else 0.0
        if "exit_efficiency_pct" in subset.columns:
            exit_eff = _safe_mean(subset["exit_efficiency_pct"]) if trades else 0.0
        else:
            exit_eff = 0.0
        sold_soon = int(subset.get("sold_too_soon", pd.Series(dtype=bool)).sum()) if trades else 0
        trailing_mask = subset.get("is_trailing_stop_exit")
        if trailing_mask is None:
            trailing_mask = pd.Series(False, index=subset.index)
        trailing_subset = subset[trailing_mask.fillna(False)] if trades else pd.DataFrame()
        stop_exits = int(len(trailing_subset.index)) if not trailing_subset.empty else 0
        rebounds = int(trailing_subset.get("rebounded", pd.Series(dtype=bool)).sum()) if stop_exits else 0
        avg_rebound_pct = _safe_mean(trailing_subset.get("rebound_pct", pd.Series(dtype=float))) if stop_exits else 0.0
        rebound_rate = rebounds / stop_exits if stop_exits else 0.0
        summaries[label] = {
            "trades": trades,
            "avg_return_pct": avg_return,
            "win_rate_pct": win_rate_pct,
            "win_rate": win_rate,
            "avg_hold_days": hold_days,
            "avg_exit_efficiency_pct": exit_eff,
            "sold_too_soon": sold_soon,
            "total_pnl": net_pnl,
            "net_pnl": net_pnl,
            "stop_exits": stop_exits,
            "rebounds": rebounds,
            "rebound_rate": rebound_rate,
            "avg_rebound_pct": avg_rebound_pct,
        }
    # ensure all required windows exist even if frame was filtered
    for label in SUMMARY_WINDOWS:
        summaries.setdefault(label, default_window.copy())
    return summaries


def _coerce_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timedelta,)):
        return value.total_seconds()
    return value


def _parse_mtime(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return None
    return None


def write_cache(
    df: pd.DataFrame,
    summary: Mapping[str, Any],
    path: Path | str = CACHE_PATH,
    *,
    trades_log_time: float | None = None,
) -> Path:
    """Persist cache to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        cleaned = {k: _coerce_json(v) for k, v in record.items()}
        records.append(cleaned)
    trades_log_iso: str | None = None
    if trades_log_time is not None:
        try:
            trades_log_iso = datetime.fromtimestamp(float(trades_log_time), tz=timezone.utc).isoformat()
        except Exception:
            trades_log_iso = None
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "trades": records,
        "trades_log_time": trades_log_iso,
    }
    atomic_write_bytes(target, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    return target


def read_cache(path: Path | str = CACHE_PATH) -> dict[str, Any] | None:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to read trade performance cache at %s", path, exc_info=True)
        return None
    df = pd.DataFrame(payload.get("trades", []))
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    payload["df"] = df
    return payload


def is_cache_stale(
    cache_path: Path | str = CACHE_PATH,
    trades_log_path: Path | str = TRADES_LOG_PATH,
    *,
    max_age_hours: int = CACHE_MAX_AGE_HOURS,
) -> bool:
    cache = Path(cache_path)
    trades_path = Path(trades_log_path)
    if not cache.exists():
        return True
    try:
        cache_age_hours = (datetime.now(timezone.utc) - datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600
    except Exception:
        return True
    if cache_age_hours > max_age_hours:
        return True
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        cached_mtime = _parse_mtime(payload.get("trades_log_time"))
    except Exception:
        return True
    try:
        trades_mtime = trades_path.stat().st_mtime
    except FileNotFoundError:
        return False
    except Exception:
        return True
    if cached_mtime is None:
        return True
    return abs(trades_mtime - cached_mtime) > 1e-6


def refresh_trade_performance_cache(
    *,
    base_dir: Path | str | None = None,
    data_client: StockHistoricalDataClient | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force: bool = False,
    cache_path: Path | str = CACHE_PATH,
    bar_fetcher: BarsFetcher | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute trade performance cache (best effort)."""

    base = Path(base_dir) if base_dir else BASE_DIR
    trades_log = Path(base) / "data" / "trades_log.csv"
    cache = Path(cache_path)

    if not force and not is_cache_stale(cache, trades_log):
        cached = read_cache(cache)
        if cached is not None:
            return cached.get("df", pd.DataFrame()), cached.get("summary", {})

    trades_df = load_trades_log(base)
    if trades_df.empty:
        summary = summarize_by_window(trades_df)
        write_cache(trades_df, summary, cache, trades_log_time=None)
        return trades_df, summary

    try:
        excursions = compute_trade_excursions(
            trades_df,
            data_client,
            lookback_days=lookback_days,
            bar_fetcher=bar_fetcher,
        )
    except Exception:
        logger.warning("Excursion enrichment failed; proceeding without peaks.", exc_info=True)
        excursions = trades_df.copy()
        for col in ("peak_price", "trough_price", "post_exit_peak"):
            if col not in excursions.columns:
                excursions[col] = np.nan
    try:
        enriched = compute_exit_quality_columns(excursions)
    except Exception:
        logger.warning("Exit quality enrichment failed; proceeding with raw trades.", exc_info=True)
        enriched = _prepare_summary_frame(excursions)
        for col, default in (("hold_days", 0.0), ("exit_efficiency_pct", 0.0), ("sold_too_soon", 0)):
            if col not in enriched.columns:
                enriched[col] = default
        for col in ("mfe_pct", "mae_pct", "missed_profit_pct"):
            if col not in enriched.columns:
                enriched[col] = np.nan
    try:
        rebound_enriched = compute_rebound_metrics(
            enriched,
            data_client,
            rebound_window_days=DEFAULT_REBOUND_DAYS,
            rebound_threshold_pct=DEFAULT_REBOUND_THRESHOLD_PCT,
            bar_fetcher=bar_fetcher,
        )
    except Exception:
        logger.warning("Rebound enrichment failed; proceeding without rebound metrics.", exc_info=True)
        rebound_enriched = enriched.copy()
        for col, default in (
            ("rebound_window_days", DEFAULT_REBOUND_DAYS),
            ("post_exit_high", np.nan),
            ("rebound_pct", np.nan),
            ("rebounded", False),
            ("is_trailing_stop_exit", False),
        ):
            if col not in rebound_enriched.columns:
                rebound_enriched[col] = default
    summary = summarize_by_window(rebound_enriched)
    try:
        trades_mtime = trades_log.stat().st_mtime
    except Exception:
        trades_mtime = None
    write_cache(rebound_enriched, summary, cache, trades_log_time=trades_mtime)
    return rebound_enriched, summary


def build_data_client() -> StockHistoricalDataClient | None:
    key, secret, _, _ = get_alpaca_creds()
    if not key or not secret:
        logger.warning("Missing Alpaca credentials; proceeding without data client.")
        return None
    try:
        return StockHistoricalDataClient(key, secret)
    except Exception:
        logger.warning("Unable to initialize StockHistoricalDataClient.", exc_info=True)
        return None


def cache_refresh_summary_token(trades: int, summary: Mapping[str, Any], rc: int) -> str:
    windows = ",".join(summary.keys()) if summary else ""
    return f"[INFO] TRADEPERF_REFRESH trades={trades} windows={windows} rc={rc}"


__all__ = [
    "BASE_DIR",
    "CACHE_PATH",
    "CACHE_MAX_AGE_HOURS",
    "DEFAULT_REBOUND_DAYS",
    "DEFAULT_REBOUND_THRESHOLD_PCT",
    "DEFAULT_TRAILING_STOP_TOLERANCE",
    "SUMMARY_WINDOWS",
    "compute_exit_quality_columns",
    "compute_trade_excursions",
    "compute_rebound_metrics",
    "is_cache_stale",
    "load_trades_log",
    "fetch_with_backoff",
    "read_cache",
    "refresh_trade_performance_cache",
    "summarize_by_window",
    "write_cache",
    "build_data_client",
    "cache_refresh_summary_token",
]
