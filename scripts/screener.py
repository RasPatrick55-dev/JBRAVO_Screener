from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - preferred module execution path
    from .indicators import adx, aroon, macd, obv, rsi
    from .utils.models import BarData, classify_exchange
except Exception:  # pragma: no cover - fallback for direct script execution
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from scripts.indicators import adx, aroon, macd, obv, rsi  # type: ignore
    from scripts.utils.models import BarData, classify_exchange  # type: ignore

from utils.env import load_env, get_alpaca_creds
from utils.io_utils import atomic_write_bytes

load_env()
_, _, _, _DEFAULT_FEED = get_alpaca_creds()
DEFAULT_FEED = (_DEFAULT_FEED or "iex").lower()

try:  # pragma: no cover - optional Alpaca dependency import guard
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetStatus, AssetClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    try:  # pragma: no cover - enum availability varies across versions
        from alpaca.data.enums import DataFeed
    except Exception:  # pragma: no cover - ``DataFeed`` introduced later
        DataFeed = None  # type: ignore
except Exception:  # pragma: no cover - allow running without Alpaca SDK
    TradingClient = None  # type: ignore
    GetAssetsRequest = None  # type: ignore
    AssetStatus = None  # type: ignore
    AssetClass = None  # type: ignore
    StockHistoricalDataClient = None  # type: ignore
    StockBarsRequest = None  # type: ignore
    TimeFrame = None  # type: ignore
    DataFeed = None  # type: ignore

try:  # pragma: no cover - compatibility across pydantic versions
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - older pydantic exposes it elsewhere
    from pydantic.error_wrappers import ValidationError  # type: ignore


LOGGER = logging.getLogger(__name__)

INPUT_COLUMNS = [
    "symbol",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

SCORED_COLUMNS = [
    "symbol",
    "exchange",
    "timestamp",
    "score",
    "close",
    "volume",
    "score_breakdown",
    "rsi",
    "macd",
    "macd_hist",
    "adx",
    "aroon_up",
    "aroon_down",
    "sma9",
    "ema20",
    "sma180",
    "atr",
]

TOP_COLUMNS = [
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
    "rsi",
    "macd",
    "macd_hist",
    "adx",
    "aroon_up",
    "aroon_down",
    "sma9",
    "ema20",
    "sma180",
    "atr",
]

SKIP_KEYS = [
    "UNKNOWN_EXCHANGE",
    "NON_EQUITY",
    "VALIDATION_ERROR",
    "NAN_DATA",
    "INSUFFICIENT_HISTORY",
]

DEFAULT_TOP_N = 15
DEFAULT_MIN_HISTORY = 180
ASSET_CACHE_RELATIVE = Path("data") / "reference" / "assets_cache.json"
ASSET_HTTP_TIMEOUT = 15
BARS_BATCH_SIZE = 150
BAR_RETRY_STATUSES = {429, 500, 502, 503, 504}
BAR_MAX_RETRIES = 3


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _resolve_feed(feed: str):
    value = (feed or "").lower()
    if DataFeed is not None:
        try:  # pragma: no cover - depends on Alpaca SDK version
            return DataFeed(value)
        except Exception:
            LOGGER.debug("Falling back to raw feed string for %s", feed)
    return value


def _create_trading_client() -> "TradingClient":
    if TradingClient is None:
        raise RuntimeError("alpaca-py TradingClient is unavailable")
    api_key, api_secret, base_url, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials for trading client")
    env = os.environ.get("APCA_API_ENV", "paper").lower()
    paper = env != "live"
    if base_url:
        lowered = base_url.lower()
        if "paper" in lowered:
            paper = True
        elif "live" in lowered:
            paper = False
    return TradingClient(api_key, api_secret, paper=paper)


def _create_data_client() -> "StockHistoricalDataClient":
    if StockHistoricalDataClient is None:
        raise RuntimeError("alpaca-py StockHistoricalDataClient is unavailable")
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials for data client")
    return StockHistoricalDataClient(api_key, api_secret)


def _normalize_bars_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    data = df.copy()
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    else:
        data = data.reset_index()
    if "symbol" not in data.columns:
        if data.index.name == "symbol":
            data.reset_index(inplace=True)
        else:
            data["symbol"] = ""
    if "timestamp" not in data.columns and "time" in data.columns:
        data.rename(columns={"time": "timestamp"}, inplace=True)
    if "timestamp" not in data.columns and "t" in data.columns:
        data.rename(columns={"t": "timestamp"}, inplace=True)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    required = {"open", "high", "low", "close", "volume"}
    for col in required:
        if col not in data.columns:
            data[col] = np.nan
    if "exchange" not in data.columns:
        data["exchange"] = ""
    result = data[[col for col in INPUT_COLUMNS if col in data.columns]].copy()
    missing = [col for col in INPUT_COLUMNS if col not in result.columns]
    for col in missing:
        result[col] = "" if col in ("symbol", "exchange") else np.nan
    return result[INPUT_COLUMNS]


def _fetch_daily_bars(
    data_client: "StockHistoricalDataClient",
    symbols: List[str],
    *,
    days: int,
    feed: str,
    now: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, int]:
    if not symbols:
        return pd.DataFrame(columns=INPUT_COLUMNS), 0
    if StockBarsRequest is None or TimeFrame is None:
        raise RuntimeError("alpaca-py market data components unavailable")

    window_days = max(days + 30, int(days * 1.5))
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=window_days)
    end = now

    frames: List[pd.DataFrame] = []
    batch_failures = 0
    for batch in _chunked(symbols, BARS_BATCH_SIZE):
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=_resolve_feed(feed),
        )
        success = False
        last_error: Optional[Exception] = None
        for attempt in range(BAR_MAX_RETRIES):
            try:
                response = data_client.get_stock_bars(request)
                bars_df = getattr(response, "df", None)
                if bars_df is None:
                    LOGGER.warning(
                        "Alpaca bars response missing DataFrame for batch of %d", len(batch)
                    )
                    break
                normalized = _normalize_bars_dataframe(bars_df)
                if not normalized.empty:
                    frames.append(normalized)
                success = True
                break
            except Exception as exc:  # pragma: no cover - network errors exercised in integration
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status in BAR_RETRY_STATUSES and attempt < BAR_MAX_RETRIES - 1:
                    sleep_for = 2 ** attempt
                    LOGGER.warning(
                        "Retrying Alpaca bars fetch (batch size %d, attempt %d/%d) after error %s",
                        len(batch),
                        attempt + 1,
                        BAR_MAX_RETRIES,
                        exc,
                    )
                    time.sleep(sleep_for)
                    continue
                LOGGER.warning("Failed to fetch bars for batch starting %s: %s", batch[:3], exc)
                break
        if not success:
            batch_failures += 1
            if last_error is not None:
                LOGGER.debug("Final error for batch %s: %s", batch[:3], last_error)

    if not frames:
        return pd.DataFrame(columns=INPUT_COLUMNS), batch_failures

    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["symbol", "timestamp"], inplace=True)
    combined.sort_values(["symbol", "timestamp"], inplace=True)
    grouped = combined.groupby("symbol", group_keys=False)
    try:
        limited = grouped.apply(
            lambda g: g.sort_values("timestamp").tail(days), include_groups=False
        )
    except TypeError:  # pragma: no cover - for older pandas without include_groups
        limited = grouped.apply(lambda g: g.sort_values("timestamp").tail(days))
    limited = limited.reset_index(drop=True)
    return limited, batch_failures


def _load_alpaca_universe(
    *,
    base_dir: Path,
    days: int,
    feed: str,
    limit: Optional[int],
    now: datetime,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    try:
        trading_client = _create_trading_client()
    except Exception as exc:
        LOGGER.error("Unable to create Alpaca trading client: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}

    try:
        symbols, asset_meta = fetch_active_equity_symbols(trading_client, base_dir=base_dir)
    except Exception as exc:
        LOGGER.error("Failed to fetch Alpaca asset universe: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}

    if limit is not None and limit > 0:
        symbols = symbols[:limit]
    LOGGER.info("Loaded assets: %d", len(symbols))

    if not symbols:
        return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta

    try:
        data_client = _create_data_client()
    except Exception as exc:
        LOGGER.error("Unable to create Alpaca market data client: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta

    bars_df, batch_failures = _fetch_daily_bars(
        data_client,
        symbols,
        days=days,
        feed=feed,
        now=now,
    )
    if batch_failures:
        LOGGER.warning("Failed to fetch %d symbol batches from Alpaca market data.", batch_failures)
    if not bars_df.empty and asset_meta:
        meta_rows = [
            (
                str(symbol).strip().upper(),
                str(meta.get("exchange", "") or "").strip().upper(),
                str(meta.get("asset_class", "") or "").strip().upper(),
                bool(meta.get("tradable", False)),
            )
            for symbol, meta in asset_meta.items()
        ]
        if meta_rows:
            meta_df = pd.DataFrame(
                meta_rows,
                columns=["symbol", "exchange_meta", "asset_class", "tradable"],
            )
            meta_df.drop_duplicates(subset=["symbol"], keep="first", inplace=True)
            bars_df = bars_df.copy()
            bars_df["symbol"] = bars_df["symbol"].astype(str).str.strip().str.upper()
            merged = bars_df.merge(meta_df, on="symbol", how="left")
            if "exchange" in merged.columns:
                merged["exchange"] = (
                    merged["exchange"].fillna("").astype(str).str.strip().str.upper()
                )
            else:
                merged["exchange"] = ""
            merged["exchange_meta"] = (
                merged["exchange_meta"].fillna("").astype(str).str.strip().str.upper()
            )
            merged["exchange"] = merged["exchange"].where(
                merged["exchange"].str.len() > 0,
                merged["exchange_meta"],
            )
            merged.drop(columns=["exchange_meta"], inplace=True)
            merged["asset_class"] = (
                merged["asset_class"].fillna("").astype(str).str.strip().str.upper()
            )
            merged["tradable"] = merged["tradable"].fillna(False).astype(bool)
            bars_df = merged
    return bars_df, asset_meta


def _ensure_logger() -> None:
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _assets_cache_path(base_dir: Optional[Path] = None) -> Path:
    base = base_dir or Path(__file__).resolve().parents[1]
    return base / ASSET_CACHE_RELATIVE


def _read_assets_cache(cache_path: Path) -> List[str]:
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - cache corruption is unexpected
        LOGGER.warning("Failed to read assets cache %s: %s", cache_path, exc)
        return []
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        return []
    cleaned: List[str] = []
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if symbol:
            cleaned.append(symbol)
    return cleaned


def _write_assets_cache(cache_path: Path, symbols: List[str]) -> None:
    if not symbols:
        return
    payload = {
        "cached_utc": _format_timestamp(datetime.now(timezone.utc)),
        "symbols": sorted({str(sym).strip().upper() for sym in symbols if sym}),
    }
    atomic_write_bytes(cache_path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _determine_alpaca_base_url() -> str:
    _, _, base_url, _ = get_alpaca_creds()
    if base_url:
        return base_url.rstrip("/")
    env = os.environ.get("APCA_API_ENV", "").lower()
    if env == "live":
        return "https://api.alpaca.markets"
    return "https://paper-api.alpaca.markets"


def _asset_attr(asset: object, *names: str) -> object:
    if isinstance(asset, dict):
        for name in names:
            if name in asset:
                return asset.get(name)
        return None
    for name in names:
        if hasattr(asset, name):
            return getattr(asset, name)
    return None


def _filter_equity_assets(assets: Iterable[object]) -> Tuple[List[str], Dict[str, dict]]:
    symbols: set[str] = set()
    asset_meta: dict[str, dict[str, object]] = {}
    for asset in assets:
        tradable_raw = _asset_attr(asset, "tradable")
        tradable = bool(tradable_raw) if tradable_raw is not None else True
        if not tradable:
            continue

        status_raw = _asset_attr(asset, "status")
        status = str(status_raw or "").strip().lower()
        if status and status != "active":
            continue

        cls_raw = _asset_attr(asset, "class_", "asset_class", "class")
        cls = str(cls_raw or "").strip().upper()
        if cls.lower() not in {"us_equity", "equity"}:
            continue

        symbol_raw = _asset_attr(asset, "symbol")
        symbol = str(symbol_raw or "").strip().upper()
        if not symbol:
            continue

        exchange_raw = _asset_attr(asset, "exchange", "primary_exchange")
        exchange = str(exchange_raw or "").strip().upper()

        symbols.add(symbol)
        asset_meta[symbol] = {
            "exchange": exchange,
            "asset_class": cls,
            "tradable": tradable,
        }

    LOGGER.info("Asset meta ready: %d symbols", len(asset_meta))
    return sorted(symbols), asset_meta


def _fetch_assets_via_sdk(trading_client) -> Tuple[List[str], Dict[str, dict]]:
    if trading_client is None:
        raise RuntimeError("Trading client is unavailable")
    if GetAssetsRequest is None or AssetStatus is None or AssetClass is None:
        assets = trading_client.get_all_assets()
    else:
        request = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
        )
        assets = trading_client.get_all_assets(request)
    return _filter_equity_assets(assets)


def _fetch_assets_via_http() -> Tuple[List[str], Dict[str, dict]]:
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        LOGGER.warning("Alpaca credentials missing; cannot fetch assets via HTTP fallback.")
        return [], {}

    base_url = _determine_alpaca_base_url()
    url = f"{base_url}/v2/assets"
    try:
        response = requests.get(
            url,
            params={"status": "active", "asset_class": "us_equity"},
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=ASSET_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        LOGGER.warning("Raw Alpaca asset fetch failed: %s", exc)
        return [], {}

    if not isinstance(payload, list):
        LOGGER.warning("Unexpected payload from Alpaca assets endpoint: %s", type(payload))
        return [], {}

    filtered, asset_meta = _filter_equity_assets(payload)
    return filtered, asset_meta


def fetch_active_equity_symbols(
    trading_client, *, base_dir: Optional[Path] = None
) -> Tuple[List[str], Dict[str, dict]]:
    """Fetch active US equity symbols from Alpaca with resilient fallbacks."""

    cache_path = _assets_cache_path(base_dir)
    try:
        symbols, asset_meta = _fetch_assets_via_sdk(trading_client)
    except Exception as exc:  # pragma: no cover - requires Alpaca SDK error
        LOGGER.warning("Alpaca assets validation failed; falling back to raw HTTP: %s", exc)
        symbols, asset_meta = _fetch_assets_via_http()
        if symbols:
            _write_assets_cache(cache_path, symbols)
            return symbols, asset_meta

        cached = _read_assets_cache(cache_path)
        if cached:
            LOGGER.warning(
                "Using cached Alpaca assets (%d symbols) after validation failure.",
                len(cached),
            )
            return cached, {}

        LOGGER.error("Failed to fetch Alpaca assets; continuing with empty universe.")
        return [], {}

    if symbols:
        _write_assets_cache(cache_path, symbols)

    return symbols, asset_meta


def _prepare_input_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    prepared = df.copy()
    for column in INPUT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[INPUT_COLUMNS]


def _safe_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _ensure_indicator_columns(df: pd.DataFrame) -> None:
    if "sma9" not in df.columns:
        df["sma9"] = df["close"].rolling(9, min_periods=9).mean()
    if "ema20" not in df.columns:
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    if "sma180" not in df.columns:
        df["sma180"] = df["close"].rolling(180, min_periods=180).mean()
    if "rsi" not in df.columns:
        df["rsi"] = rsi(df["close"])
    if "macd" not in df.columns or "macd_hist" not in df.columns:
        macd_line, _, macd_hist = macd(df["close"])
        if "macd" not in df.columns:
            df["macd"] = macd_line
        if "macd_hist" not in df.columns:
            df["macd_hist"] = macd_hist
    if "adx" not in df.columns:
        df["adx"] = adx(df)
    if "aroon_up" not in df.columns or "aroon_down" not in df.columns:
        aroon_up, aroon_down = aroon(df)
        if "aroon_up" not in df.columns:
            df["aroon_up"] = aroon_up
        if "aroon_down" not in df.columns:
            df["aroon_down"] = aroon_down
    if "atr" not in df.columns:
        df["atr"] = _compute_atr(df)


def _format_timestamp(ts: datetime) -> str:
    return ts.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    data = df.to_csv(index=False).encode("utf-8")
    atomic_write_bytes(path, data)


def _write_json_atomic(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _score_symbol(df: pd.DataFrame) -> Tuple[float, str]:
    returns = df["close"].pct_change().dropna()
    recent_return = returns.tail(5).mean() if not returns.empty else 0.0
    baseline_window = min(len(df), 30)
    baseline = df["close"].rolling(baseline_window).mean().iloc[-1]
    latest_close = df["close"].iloc[-1]
    if pd.isna(baseline) or baseline == 0:
        trend = 0.0
    else:
        trend = (latest_close / baseline) - 1
    volatility = returns.tail(20).std() if len(returns) >= 2 else 0.0
    score = (np.nan_to_num(recent_return) + np.nan_to_num(trend)) * 100 - np.nan_to_num(volatility) * 10
    breakdown = f"recent_return={recent_return:.4f}; trend={trend:.4f}; vol={volatility:.4f}"
    return round(float(score), 2), breakdown


def run_screener(
    df: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    min_history: int = DEFAULT_MIN_HISTORY,
    now: Optional[datetime] = None,
    asset_meta: Optional[Dict[str, dict]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """Run the screener on ``df`` returning (top, scored, stats, skips)."""

    now = now or datetime.now(timezone.utc)
    prepared = _prepare_input_frame(df)
    stats = {"symbols_in": 0, "candidates_out": 0}
    skip_reasons = {key: 0 for key in SKIP_KEYS}
    scored_records: list[dict[str, object]] = []
    asset_meta = {str(key).upper(): value for key, value in (asset_meta or {}).items()}
    promoted_symbols: set[str] = set()
    reject_samples: list[dict[str, str]] = []

    def record_reject(symbol: object, reason: str) -> None:
        sym = str(symbol or "").strip().upper() or "<UNKNOWN>"
        if len(reject_samples) < 50:
            reject_samples.append({"symbol": sym, "reason": reason})

    if prepared.empty:
        LOGGER.info("No input rows supplied to screener; outputs will be empty.")
    else:
        for symbol, group in prepared.groupby("symbol"):
            stats["symbols_in"] += 1
            if group.empty:
                skip_reasons["NAN_DATA"] += 1
                LOGGER.info("[SKIP] %s has no rows", symbol or "<UNKNOWN>")
                record_reject(symbol, "NAN_DATA")
                continue

            bars: list[BarData] = []
            skip_symbol = False
            for row in group.to_dict("records"):
                row = dict(row)
                sym = str(row.get("symbol") or symbol or "").strip().upper()
                meta = asset_meta.get(sym, {})
                exch = (
                    str(row.get("exchange") or "").strip().upper()
                    or str(meta.get("exchange", "") or "").strip().upper()
                )
                asset_class = (
                    str(row.get("asset_class") or "").strip().upper()
                    or str(meta.get("asset_class", "") or "").strip().upper()
                )
                row["symbol"] = sym
                row["exchange"] = exch
                try:
                    bar = BarData(**row)
                except ValidationError as exc:
                    LOGGER.warning("[SKIP] %s ValidationError: %s", row.get("symbol") or "<UNKNOWN>", exc)
                    skip_reasons["VALIDATION_ERROR"] += 1
                    record_reject(sym, "VALIDATION_ERROR")
                    skip_symbol = True
                    break
                kind = classify_exchange(bar.exchange)
                if kind != "EQUITY":
                    if not bar.exchange:
                        if asset_class in {"US_EQUITY", "EQUITY"}:
                            kind = "EQUITY"
                            promoted_symbols.add(sym)
                        else:
                            LOGGER.info(
                                "[SKIP] %s exchange=%s kind=%s",
                                bar.symbol or "<UNKNOWN>",
                                bar.exchange or "",
                                kind,
                            )
                            skip_reasons["NON_EQUITY"] += 1
                            record_reject(sym, "NON_EQUITY")
                            skip_symbol = True
                            break
                    else:
                        LOGGER.info(
                            "[SKIP] %s exchange=%s kind=%s",
                            bar.symbol or "<UNKNOWN>",
                            bar.exchange or "",
                            kind,
                        )
                        key = "UNKNOWN_EXCHANGE" if kind == "OTHER" else "NON_EQUITY"
                        skip_reasons[key] += 1
                        record_reject(sym, key)
                        skip_symbol = True
                        break
                bars.append(bar)

            if skip_symbol or not bars:
                if not bars and not skip_symbol:
                    record_reject(symbol, "NAN_DATA")
                continue

            bars_df = pd.DataFrame([bar.to_dict() for bar in bars])
            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
            bars_df.sort_values("timestamp", inplace=True)

            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                bars_df[col] = pd.to_numeric(bars_df[col], errors="coerce")

            clean_df = bars_df.dropna(subset=["close"]).copy()
            if clean_df.empty:
                skip_reasons["NAN_DATA"] += 1
                LOGGER.info("[SKIP] %s dropped due to NaN close", symbol or "<UNKNOWN>")
                record_reject(symbol, "NAN_DATA")
                continue

            clean_df.dropna(subset=numeric_cols, inplace=True)
            if clean_df.empty:
                skip_reasons["NAN_DATA"] += 1
                LOGGER.info("[SKIP] %s dropped due to NaN data", symbol or "<UNKNOWN>")
                record_reject(symbol, "NAN_DATA")
                continue

            if len(clean_df) < min_history:
                skip_reasons["INSUFFICIENT_HISTORY"] += 1
                LOGGER.info(
                    "[SKIP] %s insufficient history (%d < %d)",
                    symbol or "<UNKNOWN>",
                    len(clean_df),
                    min_history,
                )
                record_reject(symbol, "INSUFFICIENT_HISTORY")
                continue

            _ensure_indicator_columns(clean_df)

            if len(clean_df) < 2:
                LOGGER.info("[FILTER] %s does not have enough bars for crossover check", symbol)
                record_reject(symbol, "INSUFFICIENT_HISTORY")
                continue

            prev_row = clean_df.iloc[-2]
            curr_row = clean_df.iloc[-1]
            required_keys = [
                prev_row.get("close"),
                prev_row.get("sma9"),
                curr_row.get("close"),
                curr_row.get("sma9"),
                curr_row.get("ema20"),
                curr_row.get("sma180"),
                curr_row.get("rsi"),
            ]
            if any(pd.isna(val) for val in required_keys):
                LOGGER.info(
                    "[FILTER] %s missing indicator data for JBravo criteria",
                    symbol or "<UNKNOWN>",
                )
                record_reject(symbol, "NAN_DATA")
                continue

            crossed_up = prev_row["close"] < prev_row["sma9"] and curr_row["close"] > curr_row["sma9"]
            if not crossed_up:
                LOGGER.info("[FILTER] %s failed crossover condition", symbol or "<UNKNOWN>")
                record_reject(symbol, "FAILED_CROSS")
                continue

            stacked = curr_row["sma9"] > curr_row["ema20"] > curr_row["sma180"]
            if not stacked:
                LOGGER.info("[FILTER] %s failed moving-average stack", symbol or "<UNKNOWN>")
                record_reject(symbol, "FAILED_SMA_STACK")
                continue

            strong_rsi = curr_row["rsi"] > 50
            if not strong_rsi:
                LOGGER.info("[FILTER] %s failed RSI threshold", symbol or "<UNKNOWN>")
                record_reject(symbol, "FAILED_RSI")
                continue

            clean_df.set_index("timestamp", inplace=True)
            score, breakdown = _score_symbol(clean_df)
            latest = clean_df.iloc[-1]
            record = {
                "symbol": symbol,
                "exchange": latest.get("exchange", ""),
                "timestamp": _format_timestamp(now),
                "score": score,
                "close": _safe_float(latest.get("close")),
                "volume": _safe_float(latest.get("volume")),
                "score_breakdown": breakdown,
                "rsi": _safe_float(latest.get("rsi")),
                "macd": _safe_float(latest.get("macd")),
                "macd_hist": _safe_float(latest.get("macd_hist")),
                "adx": _safe_float(latest.get("adx")),
                "aroon_up": _safe_float(latest.get("aroon_up")),
                "aroon_down": _safe_float(latest.get("aroon_down")),
                "sma9": _safe_float(latest.get("sma9")),
                "ema20": _safe_float(latest.get("ema20")),
                "sma180": _safe_float(latest.get("sma180")),
                "atr": _safe_float(latest.get("atr")),
            }
            scored_records.append(record)

    scored_df = pd.DataFrame(scored_records)
    if not scored_df.empty:
        scored_df.sort_values("score", ascending=False, inplace=True)
    scored_df = scored_df.reindex(columns=SCORED_COLUMNS)
    stats["candidates_out"] = int(scored_df.shape[0])

    top_df = scored_df.head(top_n).copy()
    top_df["universe_count"] = stats["symbols_in"]
    top_df = top_df.reindex(columns=TOP_COLUMNS)

    if promoted_symbols:
        LOGGER.warning(
            "Promoting blank exchange to EQUITY for %d symbols based on asset_class",
            len(promoted_symbols),
        )

    if stats["candidates_out"] == 0 and reject_samples:
        sample = reject_samples[:10]
        LOGGER.info(
            "No candidates passed JBravo gates; sample rejections: %s",
            json.dumps(sample, sort_keys=True),
        )

    return top_df, scored_df, stats, skip_reasons, reject_samples


def _load_source_dataframe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - file corruption is rare
            LOGGER.error("Failed to read screener source %s: %s", path, exc)
            return pd.DataFrame(columns=INPUT_COLUMNS)
    LOGGER.warning("Screener source %s missing; proceeding with empty frame.", path)
    return pd.DataFrame(columns=INPUT_COLUMNS)


def write_outputs(
    base_dir: Path,
    top_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    stats: dict[str, int],
    skip_reasons: dict[str, int],
    reject_samples: Optional[list[dict[str, str]]] = None,
    *,
    status: str = "ok",
    now: Optional[datetime] = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    top_path = data_dir / "top_candidates.csv"
    scored_path = data_dir / "scored_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"

    _write_csv_atomic(top_path, top_df)
    _write_csv_atomic(scored_path, scored_df)

    metrics = {
        "last_run_utc": _format_timestamp(now),
        "rows": int(scored_df.shape[0]),
        "symbols_in": int(stats.get("symbols_in", 0)),
        "candidates_out": int(stats.get("candidates_out", 0)),
        "skips": {key: int(skip_reasons.get(key, 0)) for key in SKIP_KEYS},
        "status": status,
    }
    if reject_samples:
        metrics["reject_samples"] = reject_samples[:10]
    else:
        metrics["reject_samples"] = []
    _write_json_atomic(metrics_path, metrics)
    return metrics_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly screener")
    source_default = os.environ.get("SCREENER_SOURCE")
    parser.add_argument(
        "--universe",
        choices=["alpaca-active", "csv"],
        default="alpaca-active",
        help="Universe source to use for screener input",
    )
    parser.add_argument(
        "--source-csv",
        dest="source_csv",
        help="Path to input OHLC CSV (used when --universe csv)",
        default=source_default,
    )
    parser.add_argument(
        "--source",
        help=argparse.SUPPRESS,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write outputs (defaults to repo root)",
    )
    parser.add_argument("--days", type=int, default=750, help="Number of trading days to request")
    parser.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default=DEFAULT_FEED,
        help="Market data feed to request from Alpaca",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional symbol limit for development/testing",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(parsed, "source", None):
        parsed.source_csv = parsed.source
    return parsed


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    input_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> int:
    _ensure_logger()
    args = parse_args(argv)
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        LOGGER.error("Missing Alpaca credentials; set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        return 2
    base_dir = Path(output_dir or args.output_dir or Path(__file__).resolve().parents[1])
    now = datetime.now(timezone.utc)

    frame: pd.DataFrame
    asset_meta: Dict[str, dict] = {}
    if input_df is not None:
        frame = input_df
    else:
        universe_mode = args.universe
        frame = pd.DataFrame(columns=INPUT_COLUMNS)
        if universe_mode == "csv":
            csv_path = (
                Path(args.source_csv)
                if getattr(args, "source_csv", None)
                else base_dir / "data" / "screener_source.csv"
            )
            frame = _load_source_dataframe(csv_path)
            if frame.empty:
                LOGGER.warning(
                    "CSV universe empty or missing; falling back to Alpaca active universe."
                )
                universe_mode = "alpaca-active"
        if universe_mode == "alpaca-active":
            frame, asset_meta = _load_alpaca_universe(
                base_dir=base_dir,
                days=max(1, args.days),
                feed=args.feed,
                limit=args.limit,
                now=now,
            )
    top_df, scored_df, stats, skip_reasons, reject_samples = run_screener(
        frame,
        top_n=args.top_n,
        min_history=args.min_history,
        now=now,
        asset_meta=asset_meta,
    )
    write_outputs(
        base_dir,
        top_df,
        scored_df,
        stats,
        skip_reasons,
        reject_samples,
        status="ok",
        now=now,
    )
    LOGGER.info(
        "Screener complete: %d symbols examined, %d candidates.",
        stats["symbols_in"],
        stats["candidates_out"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
