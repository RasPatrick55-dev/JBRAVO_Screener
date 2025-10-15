from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - preferred module execution path
    from .indicators import adx, aroon, macd, obv, rsi
    from .utils.normalize import to_bars_df, BARS_COLUMNS
    from .utils.calendar import calc_daily_window
    from .utils.http_alpaca import fetch_bars_http
    from .utils.rate import TokenBucket
    from .utils.models import BarData, classify_exchange, KNOWN_EQUITY
    from .utils.env import trading_base_url
    from .utils.frame_guards import ensure_symbol_column
    from .features import ALL_FEATURE_COLUMNS, compute_all_features, REQUIRED_FEATURE_COLUMNS
    from .ranking import (
        apply_gates,
        score_universe,
        DEFAULT_COMPONENT_MAP,
    )
    from .backtest import compute_recent_performance
except Exception:  # pragma: no cover - fallback for direct script execution
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from scripts.indicators import adx, aroon, macd, obv, rsi  # type: ignore
    from scripts.utils.normalize import to_bars_df, BARS_COLUMNS  # type: ignore
    from scripts.utils.calendar import calc_daily_window  # type: ignore
    from scripts.utils.http_alpaca import fetch_bars_http  # type: ignore
    from scripts.utils.rate import TokenBucket  # type: ignore
    from scripts.utils.models import BarData, classify_exchange, KNOWN_EQUITY  # type: ignore
    from scripts.utils.env import trading_base_url  # type: ignore
    from scripts.utils.frame_guards import ensure_symbol_column  # type: ignore
    from scripts.features import (  # type: ignore
        ALL_FEATURE_COLUMNS,
        compute_all_features,
        REQUIRED_FEATURE_COLUMNS,
    )
    from scripts.ranking import (  # type: ignore
        apply_gates,
        score_universe,
        DEFAULT_COMPONENT_MAP,
    )
    from scripts.backtest import compute_recent_performance  # type: ignore

from scripts.health_check import probe_trading_only
from scripts.utils.env import load_env
from utils.env import (
    AlpacaCredentialsError,
    AlpacaUnauthorizedError,
    assert_alpaca_creds,
    get_alpaca_creds,
    write_auth_error_artifacts,
)
from utils.io_utils import atomic_write_bytes

DEFAULT_FEED = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"

REQUIRED_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)

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


def _bootstrap_env() -> list[str]:
    loaded_files, missing = load_env(REQUIRED_ENV_KEYS)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    missing_keys: list[str] = []
    try:
        files_repr = json.dumps(loaded_files) if loaded_files else "[]"
        LOGGER.info("[INFO] ENV_LOADED files=%s", files_repr)
        if missing:
            missing_keys = list(missing)
            LOGGER.error("[ERROR] ENV_MISSING_KEYS=%s", json.dumps(missing_keys))
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    if missing_keys:
        raise SystemExit(2)
    _, _, _, feed_value = get_alpaca_creds()
    resolved_feed = (
        (feed_value or os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
    )
    globals()["DEFAULT_FEED"] = resolved_feed
    return loaded_files


class T:
    def __init__(self) -> None:
        self.t = time.time()
        self.m: dict[str, float] = {}

    def lap(self, key: str) -> float:
        now = time.time()
        elapsed = round(now - self.t, 3)
        self.m[key] = elapsed
        self.t = time.time()
        return elapsed


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

RANKER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ranker.yml"

CORE_FEATURE_COLUMNS = [
    "TS",
    "MS",
    "BP",
    "PT",
    "MH",
    "RSI",
    "ADX",
    "AROON",
    "VCP",
    "VOLexp",
    "GAPpen",
    "LIQpen",
]

COARSE_RANK_COLUMNS = tuple(dict.fromkeys(DEFAULT_COMPONENT_MAP.values()))

TOP_CANDIDATE_COLUMNS = [
    "timestamp",
    "symbol",
    "Score",
    "score",
    "coarse_score",
    "coarse_rank",
    "backtest_expectancy",
    "backtest_win_rate",
    "backtest_adjustment",
    "backtest_samples",
    "exchange",
    "close",
    "entry_price",
    "volume",
    "adv20",
    "atrp",
    "score_breakdown",
    "RSI",
    "ADX",
    "AROON",
    "VOLexp",
    "VCP",
    "SMA9",
    "EMA20",
    "SMA50",
    "SMA100",
    "ATR14",
    "universe_count",
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
BARS_BATCH_SIZE = 200
BAR_RETRY_STATUSES = {429, 500, 502, 503, 504}
BAR_MAX_RETRIES = 3

SCREENER_METRICS_PATH = Path("data") / "screener_metrics.json"
METRICS_SUMMARY_PATH = Path("data") / "metrics_summary.csv"
AUTH_CONTEXT: dict[str, object] = {
    "base_dir": Path(__file__).resolve().parents[1],
    "creds": {},
}


def _auth_paths(base_dir: Optional[Path] = None) -> tuple[Path, Path]:
    root = base_dir or AUTH_CONTEXT.get("base_dir")
    if not isinstance(root, Path):
        root = Path(root) if isinstance(root, str) else Path(__file__).resolve().parents[1]
    return root / "data" / "screener_metrics.json", root / "data" / "metrics_summary.csv"


def _persist_auth_error(
    reason: str,
    missing: Iterable[str] | None = None,
    *,
    sanitized: Mapping[str, object] | None = None,
    base_dir: Optional[Path] = None,
) -> None:
    metrics_path, summary_path = _auth_paths(base_dir)
    snapshot: Mapping[str, object]
    if sanitized is not None:
        snapshot = sanitized
    else:
        stored = AUTH_CONTEXT.get("creds")
        snapshot = stored if isinstance(stored, Mapping) else {}
    write_auth_error_artifacts(
        reason=reason,
        sanitized=snapshot,
        missing=missing or [],
        metrics_path=metrics_path,
        summary_path=summary_path,
    )
DEFAULT_SHORTLIST_SIZE = 800
DEFAULT_BACKTEST_TOP_K = 100
DEFAULT_BACKTEST_LOOKBACK = 90
BACKTEST_EXPECTANCY_WEIGHT = 5.0
BACKTEST_WIN_RATE_WEIGHT = 1.0


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return default


def _write_universe_prefix_metrics(
    universe_df: pd.DataFrame, metrics: MutableMapping[str, Any]
) -> Dict[str, int]:
    """Populate universe prefix counts and guardrail warnings for Alpaca draws."""
    if metrics is None:
        return {}
    if "symbol" not in universe_df.columns or universe_df.empty:
        metrics["universe_prefix_counts"] = {}
        return {}

    prefix_counts = (
        universe_df["symbol"].astype(str).str.upper().str[0].value_counts().to_dict()
    )
    metrics["universe_prefix_counts"] = prefix_counts
    if prefix_counts:
        LOGGER.info("Universe prefix counts: %s", prefix_counts)
        letter, count = max(prefix_counts.items(), key=lambda kv: kv[1])
        share = count / max(len(universe_df), 1)
        if share > 0.70:
            LOGGER.warning(
                "ALERT: Universe draw biased toward '%s' (%.1f%%)",
                letter,
                100 * share,
            )
    return prefix_counts


def make_verify_hook(enabled: bool):
    if not enabled:
        return None

    from pathlib import Path

    Path("debug").mkdir(parents=True, exist_ok=True)

    def hook(url: str, params: dict[str, object]):
        safe: dict[str, object] = {}
        for key, value in (params or {}).items():
            key_str = str(key)
            if "secret" in key_str.lower():
                safe[key_str] = "<redacted>"
            else:
                safe[key_str] = value
        query = "&".join(f"{k}={safe[k]}" for k in safe)
        LOGGER.info("Bars request preview: %s ? %s", url, query)

    return hook


def _fallback_daily_window(days: int, *, now: Optional[datetime] = None) -> tuple[str, str, str]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    days = max(1, int(days))
    current_day = np.datetime64(current.date(), "D")
    last_session = np.busday_offset(current_day, 0, roll="backward")
    start_session = (
        np.busday_offset(last_session, -(days - 1), roll="backward") if days > 1 else last_session
    )
    start_day = pd.Timestamp(start_session).date()
    end_day = pd.Timestamp(last_session).date()
    start_iso = f"{start_day.isoformat()}T00:00:00Z"
    end_iso = f"{end_day.isoformat()}T23:59:59Z"
    return start_iso, end_iso, end_day.isoformat()


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _enum_to_str(value: object) -> str:
    if value is None:
        return ""
    try:  # pragma: no cover - depends on enum type
        return str(value.name).upper()
    except Exception:
        pass
    try:  # pragma: no cover - some enums expose ``value``
        return str(value.value).upper()
    except Exception:
        pass
    return str(value).upper()


def _resolve_feed(feed: str):
    value = (feed or "").lower()
    if DataFeed is not None:
        try:  # pragma: no cover - depends on Alpaca SDK version
            return DataFeed(value)
        except Exception:
            LOGGER.debug("Falling back to raw feed string for %s", feed)
    return value


def _make_stock_bars_request(**kwargs: Any):
    if StockBarsRequest is None:
        raise RuntimeError("alpaca-py market data components unavailable")
    attempts: list[dict[str, Any]] = []
    base = dict(kwargs)
    attempts.append(base)
    attempts.append({k: v for k, v in base.items() if k != "adjustment"})
    attempts.append({k: v for k, v in base.items() if k not in {"adjustment", "feed"}})
    attempts.append({k: v for k, v in base.items() if k not in {"adjustment", "feed", "page_token"}})
    attempts.append({k: v for k, v in base.items() if k not in {"adjustment", "feed", "page_token", "limit"}})
    last_exc: Optional[Exception] = None
    for attempt in attempts:
        try:
            return StockBarsRequest(**attempt)
        except TypeError as exc:  # pragma: no cover - depends on SDK version
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return StockBarsRequest(**base)


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


def _normalize_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame.dropna(subset=["symbol", "timestamp"], inplace=True)
    frame = frame[frame["symbol"] != ""]
    if frame.empty:
        return frame
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["symbol", "timestamp"], inplace=True)
    return frame


def _collect_batch_pages(
    data_client: "StockHistoricalDataClient",
    request_kwargs: dict[str, Any],
) -> Tuple[list[pd.DataFrame], int, bool, str, bool]:
    frames: list[pd.DataFrame] = []
    next_token: Optional[str] = None
    page_count = 0
    paged = False
    columns_desc = ""
    while True:
        kwargs = dict(request_kwargs)
        if next_token:
            kwargs["page_token"] = next_token
        try:
            request = _make_stock_bars_request(**kwargs)
        except Exception as exc:
            LOGGER.warning(
                "Failed to build StockBarsRequest for batch starting %s: %s",
                request_kwargs.get("symbol_or_symbols"),
                exc,
            )
            return [], page_count, paged, columns_desc, False
        try:
            response = data_client.get_stock_bars(request)
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch bars for batch starting %s: %s",
                request_kwargs.get("symbol_or_symbols"),
                exc,
            )
            return [], page_count, paged, columns_desc, False
        page_count += 1
        raw_df = to_bars_df(response)
        if "symbol" not in raw_df.columns:
            LOGGER.error(
                "Bars normalize failed: type=%s has attributes df=%s data=%s; df_cols=%s",
                type(response).__name__,
                hasattr(response, "df"),
                hasattr(response, "data"),
                list(getattr(getattr(response, "df", pd.DataFrame()), "columns", [])),
            )
        df = ensure_symbol_column(raw_df)
        columns_desc = ",".join(df.columns)
        if df.empty:
            return [], page_count, paged, columns_desc, False
        normalized = _normalize_bars_frame(df)
        if not normalized.empty:
            frames.append(normalized)
        next_token = getattr(response, "next_page_token", None)
        if next_token:
            paged = True
        else:
            break
    return frames, page_count, paged, columns_desc, True


def _fetch_daily_bars(
    data_client: Optional["StockHistoricalDataClient"],
    symbols: List[str],
    *,
    days: int,
    start_iso: str,
    end_iso: str,
    feed: str,
    fetch_mode: str,
    batch_size: int,
    max_workers: int,
    min_history: int,
    bars_source: str,
    run_date: Optional[str] = None,
    reuse_cache: bool = True,
    verify_hook: Optional[Callable[[str, dict[str, object]], None]] = None,
) -> Tuple[pd.DataFrame, dict[str, Any], Dict[str, str]]:
    if not symbols:
        return pd.DataFrame(columns=INPUT_COLUMNS), {
            "batches_total": 0,
            "batches_paged": 0,
            "pages_total": 0,
            "bars_rows_total": 0,
            "symbols_with_bars": 0,
            "symbols_no_bars": 0,
            "symbols_no_bars_sample": [],
            "fallback_batches": 0,
            "insufficient_history": 0,
            "rate_limited": 0,
        }, {}

    source = (bars_source or "http").strip().lower()
    if source not in {"http", "sdk"}:
        source = "http"
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if source == "sdk":
        if StockBarsRequest is None or TimeFrame is None:
            raise RuntimeError("alpaca-py market data components unavailable")
        if data_client is None:
            raise RuntimeError("StockHistoricalDataClient required when bars_source=sdk")

    days = max(1, int(days))
    cache_feed = (feed or "iex").strip().lower() or "iex"
    cache_root = Path("data") / "cache" / f"bars_1d_{cache_feed}"
    run_date_str = (
        str(run_date)
        if run_date
        else datetime.now(timezone.utc).date().isoformat()
    )

    def _cache_paths(batch_idx: int) -> tuple[Path, Path]:
        batch_dir = cache_root / run_date_str
        batch_path = batch_dir / f"{batch_idx:04d}.parquet"
        meta_path = batch_path.with_suffix(".json")
        return batch_path, meta_path

    def _load_batch_cache(batch_idx: int) -> tuple[Optional[pd.DataFrame], dict]:
        if not reuse_cache:
            return None, {}
        cache_path, meta_path = _cache_paths(batch_idx)
        if not cache_path.exists():
            return None, {}
        try:
            cached = pd.read_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - cache corruption is rare
            LOGGER.warning("Failed to read cached batch %s: %s", cache_path, exc)
            return None, {}
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # pragma: no cover - JSON corruption is rare
                LOGGER.warning("Failed to read cache metadata %s: %s", meta_path, exc)
                meta = {}
        return cached, meta

    def _write_batch_cache(batch_idx: int, frame: pd.DataFrame, meta: dict) -> None:
        if frame is None or frame.empty:
            return
        cache_path, meta_path = _cache_paths(batch_idx)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
            if meta:
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - disk issues are unexpected
            LOGGER.warning("Failed to persist batch cache %s: %s", cache_path, exc)

    def _log_batch_progress(
        batch_idx: int,
        batches_total: int,
        symbols_in_batch: int,
        pages: int,
        rows: int,
        elapsed: float,
        *,
        from_cache: bool = False,
    ) -> None:
        total_width = max(len(str(batches_total)), 2)
        batch_label = f"{batch_idx:0{total_width}d}"
        total_label = f"{batches_total:0{total_width}d}"
        cache_state = "hit" if from_cache else "miss"
        LOGGER.info(
            "[INFO] bars: batch %s/%s syms=%d pages=%d rows=~%d elapsed=%.1fs cache=%s",
            batch_label,
            total_label,
            symbols_in_batch,
            pages,
            rows,
            elapsed,
            cache_state,
        )

    def _parse_iso(value: str) -> datetime:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc)

    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)

    unique_symbols = [str(sym or "").strip().upper() for sym in symbols if sym]
    unique_symbols = list(dict.fromkeys(unique_symbols))
    metrics: dict[str, Any] = {
        "batches_total": 0,
        "batches_paged": 0,
        "pages_total": 0,
        "bars_rows_total": 0,
        "symbols_with_bars": 0,
        "symbols_no_bars": 0,
        "symbols_no_bars_sample": [],
        "fallback_batches": 0,
        "insufficient_history": 0,
        "rate_limited": 0,
        "http_404_batches": 0,
        "http_empty_batches": 0,
        "symbols_in": len(unique_symbols),
        "raw_bars_count": 0,
        "parsed_rows_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "http": {
            "requests": 0,
            "rows": 0,
            "rate_limit_hits": 0,
            "retries": 0,
        },
    }
    prescreened: dict[str, str] = {}
    symbols_with_history: set[str] = set()

    frames: list[pd.DataFrame] = []

    verify_consumed = False
    preview_enabled = bool(verify_hook)
    preview_written = False

    def _acquire_verify_hook():
        nonlocal verify_consumed
        if verify_hook and not verify_consumed:
            verify_consumed = True
            return verify_hook
        return None

    def _write_preview(bars_raw: object, bars_df: pd.DataFrame) -> None:
        nonlocal preview_written
        if not preview_enabled or preview_written:
            return
        preview_written = True
        try:
            if isinstance(bars_raw, dict) and "bars" in bars_raw:
                sample = list(bars_raw.get("bars", []))[:5]
            elif isinstance(bars_raw, list):
                sample = list(bars_raw)[:5]
            else:
                sample = bars_df.head(5).to_dict("records")
            preview_dir = Path("debug")
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = Path(PurePath("debug", "bars_preview.json"))
            preview_path.write_text(json.dumps({"bars": sample}, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("Could not write bars_preview.json: %s", exc)

    def _dump_parse_failure(bars_raw: object, bars_df: pd.DataFrame) -> None:
        try:
            debug_dir = Path("debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(bars_raw, dict) and "bars" in bars_raw:
                sample = list(bars_raw.get("bars", []))[:10]
            elif isinstance(bars_raw, (list, tuple)):
                sample = list(bars_raw)[:10]
            else:
                sample = []
            Path(debug_dir / "raw_bars_sample.json").write_text(
                json.dumps(sample, indent=2), encoding="utf-8"
            )
            Path(debug_dir / "parsed_empty_schema.json").write_text(
                json.dumps({"columns": list(bars_df.columns)}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("Could not write parse debug artifacts: %s", exc)

    def fetch_single_collection(
        target_symbols: List[str], *, use_http: bool
    ) -> Tuple[pd.DataFrame, int, Dict[str, str], dict[str, int]]:
        local_frames: list[pd.DataFrame] = []
        local_prescreened: dict[str, str] = {}
        token_bucket = TokenBucket(200) if not use_http else None
        local_metrics: dict[str, int] = {
            "symbols_in": 0,
            "rate_limited": 0,
            "http_404_batches": 0,
            "http_empty_batches": 0,
            "raw_bars_count": 0,
            "parsed_rows_count": 0,
            "http_requests": 0,
            "http_rows": 0,
            "http_retries": 0,
        }
        def _merge_metrics(payload: dict[str, int]) -> None:
            for key, value in (payload or {}).items():
                if not isinstance(value, int):
                    continue
                local_metrics[key] = local_metrics.get(key, 0) + int(value)

        def _http_worker(symbol: str) -> Tuple[str, pd.DataFrame, dict[str, int]]:
            try:
                hook = _acquire_verify_hook()
                raw, http_stats = fetch_bars_http(
                    [symbol],
                    start_iso,
                    end_iso,
                    timeframe="1Day",
                    feed=feed,
                    verify_hook=hook,
                )
                raw_bars_count = len(raw)
                bars_df = to_bars_df(raw)
                bars_df = ensure_symbol_column(bars_df).reset_index(drop=True)

                parsed_rows_count = int(bars_df.shape[0])
                if raw_bars_count > 0 and parsed_rows_count == 0:
                    LOGGER.error(
                        "Raw bars > 0 but parsed rows == 0; dumping debug artifacts."
                    )
                    _dump_parse_failure(raw, bars_df)
                if not bars_df.empty and "symbol" in bars_df.columns:
                    bars_df["symbol"] = bars_df["symbol"].fillna(symbol)
                    mask = bars_df["symbol"].str.strip() == ""
                    if mask.any():
                        bars_df.loc[mask, "symbol"] = symbol
                missing = [c for c in required_cols if c not in bars_df.columns]
                if missing:
                    LOGGER.error("Bars normalized but missing columns: %s", missing)
                    bars_df = bars_df.iloc[0:0]
                _write_preview(raw, bars_df)
                stats_block = {
                    "symbols_in": 1,
                    "rate_limited": int(
                        http_stats.get("rate_limit_hits", http_stats.get("rate_limited", 0))
                    ),
                    "http_404_batches": int(http_stats.get("http_404_batches", 0)),
                    "http_empty_batches": int(http_stats.get("http_empty_batches", 0)),
                    "raw_bars_count": raw_bars_count,
                    "parsed_rows_count": parsed_rows_count,
                    "http_requests": int(http_stats.get("requests", 0)),
                    "http_rows": int(http_stats.get("rows", parsed_rows_count)),
                    "http_retries": int(http_stats.get("retries", 0)),
                }
                return symbol, bars_df, stats_block
            except AlpacaUnauthorizedError:
                raise
            except Exception as exc:  # pragma: no cover - network errors hit in integration
                LOGGER.warning("Failed HTTP bars fetch for %s: %s", symbol, exc)
                df = pd.DataFrame(columns=BARS_COLUMNS)
                return (
                    symbol,
                    df,
                    {
                        "symbols_in": 1,
                        "rate_limited": 0,
                        "http_404_batches": 0,
                        "http_empty_batches": 0,
                        "raw_bars_count": 0,
                        "parsed_rows_count": 0,
                        "http_requests": 0,
                        "http_rows": 0,
                        "http_retries": 0,
                    },
                )

        def _sdk_worker(symbol: str) -> Tuple[str, pd.DataFrame, dict[str, int]]:
            assert data_client is not None  # for type checker
            request_kwargs = {
                "symbol_or_symbols": symbol,
                "timeframe": TimeFrame.Day,
                "start": start_dt,
                "end": end_dt,
                "feed": _resolve_feed(feed),
                "adjustment": "raw",
                "limit": days,
            }
            for attempt in range(BAR_MAX_RETRIES):
                try:
                    request = _make_stock_bars_request(**request_kwargs)
                    assert token_bucket is not None
                    token_bucket.acquire()
                    response = data_client.get_stock_bars(request)
                    df = to_bars_df(response)
                    df = ensure_symbol_column(df)
                    if not df.empty and "symbol" in df.columns and symbol:
                        df["symbol"] = df["symbol"].fillna(symbol)
                        mask = df["symbol"].str.strip() == ""
                        if mask.any():
                            df.loc[mask, "symbol"] = symbol
                    parsed_count = int(df.shape[0])
                    if parsed_count > 0:
                        df = df.reset_index(drop=True)
                    _write_preview(response, df)
                    return (
                        symbol,
                        df,
                        {
                            "symbols_in": 1,
                            "rate_limited": 0,
                            "pages": 1,
                            "requests": 1,
                            "chunks": 1,
                            "raw_bars_count": parsed_count,
                            "parsed_rows_count": parsed_count,
                            "http_requests": 1,
                            "http_rows": parsed_count,
                            "http_retries": 0,
                        },
                    )
                except Exception as exc:  # pragma: no cover - network errors hit in integration
                    status = getattr(exc, "status_code", None)
                    if status in BAR_RETRY_STATUSES and attempt < BAR_MAX_RETRIES - 1:
                        sleep_for = 2 ** attempt
                        LOGGER.warning(
                            "Retrying single-symbol SDK fetch for %s (attempt %d/%d) after %s",
                            symbol,
                            attempt + 1,
                            BAR_MAX_RETRIES,
                            exc,
                        )
                        time.sleep(sleep_for)
                        continue
                    LOGGER.warning("Failed to fetch single-symbol bars for %s: %s", symbol, exc)
                    break
            return symbol, pd.DataFrame(columns=BARS_COLUMNS), {
                "symbols_in": 1,
                "rate_limited": 0,
                "http_404_batches": 0,
                "http_empty_batches": 0,
                "raw_bars_count": 0,
                "parsed_rows_count": 0,
                "http_requests": 0,
                "http_rows": 0,
                "http_retries": 0,
            }

        worker = _http_worker if use_http else _sdk_worker
        pages = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(worker, symbol): symbol for symbol in target_symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, df, http_stats = future.result()
                except AlpacaUnauthorizedError:
                    raise
                except Exception as exc:  # pragma: no cover - unexpected failure path
                    LOGGER.error("Single-symbol worker failed for %s: %s", symbol, exc)
                    sym = symbol
                    df = pd.DataFrame(columns=BARS_COLUMNS)
                    http_stats = {
                        "rate_limited": 0,
                        "pages": 0,
                        "requests": 0,
                        "chunks": 0,
                        "http_404_batches": 0,
                        "http_empty_batches": 0,
                        "raw_bars_count": 0,
                        "parsed_rows_count": 0,
                    }
                _merge_metrics(http_stats)
                pages += int(http_stats.get("pages", 0)) or 0
                if df.empty:
                    local_prescreened[sym] = "NAN_DATA"
                    continue
                if "symbol" not in df.columns:
                    LOGGER.error(
                        "Normalize returned unexpected shape; got columns=%s", list(df.columns)
                    )
                    local_prescreened[sym] = "NAN_DATA"
                    continue
                normalized = _normalize_bars_frame(df)
                if normalized.empty:
                    local_prescreened[sym] = "NAN_DATA"
                    continue
                local_frames.append(normalized)

        merged = (
            pd.concat(local_frames, ignore_index=True)
            if local_frames
            else pd.DataFrame(columns=BARS_COLUMNS)
        )
        return merged, pages, local_prescreened, local_metrics

    if fetch_mode == "single":
        (
            single_frame,
            single_pages,
            single_prescreened,
            single_metrics,
        ) = fetch_single_collection(unique_symbols, use_http=(source == "http"))
        frames.append(single_frame)
        metrics["pages_total"] += single_pages
        for key in ["rate_limited", "http_404_batches", "http_empty_batches"]:
            metrics[key] += int(single_metrics.get(key, 0))
        metrics["raw_bars_count"] += int(single_metrics.get("raw_bars_count", 0))
        metrics["parsed_rows_count"] += int(single_metrics.get("parsed_rows_count", 0))
        metrics["http"]["requests"] += int(single_metrics.get("http_requests", 0))
        metrics["http"]["rows"] += int(single_metrics.get("http_rows", 0))
        metrics["http"]["rate_limit_hits"] += int(
            single_metrics.get("rate_limit_hits", single_metrics.get("rate_limited", 0))
        )
        metrics["http"]["retries"] += int(single_metrics.get("http_retries", 0))
        prescreened.update(single_prescreened)
    else:
        batches = list(_chunked(unique_symbols, max(1, batch_size)))
        metrics["batches_total"] = len(batches)
        for index, batch in enumerate(batches, start=1):
            if source == "http":
                batch_start = time.time()
                cached_frame, cached_meta = _load_batch_cache(index)
                if cached_frame is not None:
                    normalized_cached = _normalize_bars_frame(cached_frame)
                    rows_cached = int(normalized_cached.shape[0])
                    metrics["cache_hits"] += 1
                    metrics.setdefault("cache", {})
                    metrics["cache"]["batches_hit"] = int(
                        metrics["cache"].get("batches_hit", 0)
                    ) + 1
                    metrics["raw_bars_count"] += rows_cached
                    metrics["parsed_rows_count"] += rows_cached
                    metrics["bars_rows_total"] = metrics.get("bars_rows_total", 0) + rows_cached
                    keep_cached = {
                        str(sym).strip().upper()
                        for sym in (cached_meta.get("keep") or [])
                        if sym
                    }
                    if not keep_cached and not normalized_cached.empty:
                        keep_cached = set(
                            normalized_cached["symbol"].astype(str).str.upper().unique()
                        )
                    insufficient_cached = {
                        str(sym).strip().upper()
                        for sym in (cached_meta.get("insufficient") or [])
                        if sym
                    }
                    missing_cached = {
                        str(sym).strip().upper()
                        for sym in (cached_meta.get("missing") or [])
                        if sym
                    }
                    if not missing_cached:
                        missing_cached = {str(sym).upper() for sym in batch} - keep_cached
                    symbols_with_history.update(keep_cached)
                    for sym in insufficient_cached:
                        prescreened.setdefault(sym, "INSUFFICIENT_HISTORY")
                    metrics["insufficient_history"] += len(insufficient_cached)
                    for sym in missing_cached:
                        prescreened.setdefault(sym, "NAN_DATA")
                    if not normalized_cached.empty:
                        frames.append(normalized_cached)
                    elapsed_cached = time.time() - batch_start
                    pages_logged = cached_meta.get("pages")
                    pages_logged = int(pages_logged) if isinstance(pages_logged, int) else 0
                    metrics["pages_total"] += pages_logged
                    if pages_logged > 1:
                        metrics["batches_paged"] += 1
                    _log_batch_progress(
                        index,
                        len(batches),
                        len(batch),
                        pages_logged,
                        rows_cached,
                        elapsed_cached,
                        from_cache=True,
                    )
                    continue
                metrics["cache_misses"] += 1
                if reuse_cache:
                    metrics.setdefault("cache", {})
                    metrics["cache"]["batches_miss"] = int(
                        metrics["cache"].get("batches_miss", 0)
                    ) + 1
                try:
                    hook = _acquire_verify_hook()
                    raw, http_stats = fetch_bars_http(
                        batch,
                        start_iso,
                        end_iso,
                        timeframe="1Day",
                        feed=feed,
                        verify_hook=hook,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "HTTP batch fetch failed for batch %d/%d (size=%d): %s",
                        index,
                        len(batches),
                        len(batch),
                        exc,
                    )
                    metrics["fallback_batches"] += 1
                    fallback_frame, fallback_pages, fallback_prescreened, fallback_metrics = (
                        fetch_single_collection(batch, use_http=True)
                    )
                    if not fallback_frame.empty:
                        frames.append(fallback_frame)
                        rows_logged = int(fallback_frame.shape[0])
                    else:
                        rows_logged = 0
                    metrics["pages_total"] += fallback_pages
                    for key in ["rate_limited", "http_404_batches", "http_empty_batches"]:
                        metrics[key] += int(fallback_metrics.get(key, 0))
                    metrics["raw_bars_count"] += int(
                        fallback_metrics.get("raw_bars_count", 0)
                    )
                    metrics["parsed_rows_count"] += int(
                        fallback_metrics.get("parsed_rows_count", 0)
                    )
                    metrics["http"]["requests"] += int(
                        fallback_metrics.get("http_requests", 0)
                    )
                    metrics["http"]["rows"] += int(fallback_metrics.get("http_rows", 0))
                    metrics["http"]["rate_limit_hits"] += int(
                        fallback_metrics.get("rate_limit_hits", fallback_metrics.get("rate_limited", 0))
                    )
                    metrics["http"]["retries"] += int(
                        fallback_metrics.get("http_retries", 0)
                    )
                    prescreened.update(fallback_prescreened)
                    keep_fallback = (
                        set(fallback_frame["symbol"].astype(str).str.upper().unique())
                        if not fallback_frame.empty
                        else set()
                    )
                    insufficient_fallback = {
                        str(sym).strip().upper()
                        for sym, reason in (fallback_prescreened or {}).items()
                        if str(reason).upper() == "INSUFFICIENT_HISTORY"
                    }
                    missing_fallback = {str(sym).upper() for sym in batch} - keep_fallback
                    batch_meta = {
                        "keep": sorted(keep_fallback),
                        "insufficient": sorted(insufficient_fallback),
                        "missing": sorted(missing_fallback),
                        "rows": rows_logged,
                        "pages": int(fallback_pages),
                    }
                    _write_batch_cache(index, fallback_frame, batch_meta)
                    elapsed_fallback = time.time() - batch_start
                    _log_batch_progress(
                        index,
                        len(batches),
                        len(batch),
                        int(fallback_pages),
                        rows_logged,
                        elapsed_fallback,
                    )
                    continue
                raw_bars_count = len(raw)
                bars_df = to_bars_df(raw)
                bars_df = ensure_symbol_column(bars_df).reset_index(drop=True)

                parsed_rows_count = int(bars_df.shape[0])
                if raw_bars_count > 0 and parsed_rows_count == 0:
                    LOGGER.error(
                        "Raw bars > 0 but parsed rows == 0; dumping debug artifacts."
                    )
                    _dump_parse_failure(raw, bars_df)
                missing = [c for c in required_cols if c not in bars_df.columns]
                if missing:
                    LOGGER.error("Bars normalized but missing columns: %s", missing)
                    bars_df = bars_df.iloc[0:0]
                _write_preview(raw, bars_df)
                metrics["http"]["requests"] += int(http_stats.get("requests", 0))
                metrics["http"]["rows"] += int(http_stats.get("rows", parsed_rows_count))
                metrics["http"]["rate_limit_hits"] += int(
                    http_stats.get("rate_limit_hits", http_stats.get("rate_limited", 0))
                )
                metrics["http"]["retries"] += int(http_stats.get("retries", 0))
                pages = int(http_stats.get("pages", 0)) if http_stats else 0
                metrics["pages_total"] += pages
                if pages > 1:
                    metrics["batches_paged"] += 1
                metrics["rate_limited"] += int(http_stats.get("rate_limited", 0))
                metrics["http_404_batches"] += int(http_stats.get("http_404_batches", 0))
                metrics["http_empty_batches"] += int(http_stats.get("http_empty_batches", 0))
                metrics["raw_bars_count"] += raw_bars_count
                metrics["parsed_rows_count"] += parsed_rows_count

                hist = (
                    bars_df.dropna(subset=["timestamp"])
                    .groupby("symbol", as_index=False)["timestamp"]
                    .size()
                    .rename(columns={"size": "n"})
                    if not bars_df.empty
                    else pd.DataFrame(columns=["symbol", "n"])
                )
                if hist.empty:
                    keep: set[str] = set()
                    insufficient: set[str] = set()
                else:
                    keep = set(hist.loc[hist["n"] >= min_history, "symbol"].tolist())
                    insufficient = set(hist.loc[hist["n"] < min_history, "symbol"].tolist())
                symbols_with_history.update(keep)
                for sym in insufficient:
                    prescreened.setdefault(sym, "INSUFFICIENT_HISTORY")
                metrics["insufficient_history"] += len(insufficient)
                bars_df = bars_df[bars_df["symbol"].isin(keep)] if keep else bars_df.iloc[0:0]
                metrics["bars_rows_total"] = metrics.get("bars_rows_total", 0) + int(
                    len(bars_df)
                )
                normalized = _normalize_bars_frame(bars_df)
                if normalized.empty:
                    for sym in keep:
                        prescreened.setdefault(sym, "NAN_DATA")
                    continue
                frames.append(normalized)
                rows_logged = int(normalized.shape[0])
                batch_meta = {
                    "keep": sorted(keep),
                    "insufficient": sorted(insufficient),
                    "missing": sorted({str(sym).upper() for sym in batch} - set(keep)),
                    "rows": rows_logged,
                    "pages": pages,
                }
                _write_batch_cache(index, normalized, batch_meta)
                elapsed_batch = time.time() - batch_start
                _log_batch_progress(
                    index,
                    len(batches),
                    len(batch),
                    pages,
                    rows_logged,
                    elapsed_batch,
                )
                continue

            request_kwargs = {
                "symbol_or_symbols": batch if len(batch) > 1 else batch[0],
                "timeframe": TimeFrame.Day,
                "start": start_dt,
                "end": end_dt,
                "feed": _resolve_feed(feed),
                "adjustment": "raw",
            }
            batch_frames, page_count, paged, columns_desc, batch_success = _collect_batch_pages(
                data_client, request_kwargs
            )
            metrics["pages_total"] += page_count
            if paged:
                metrics["batches_paged"] += 1
            if batch_success and batch_frames:
                parsed_rows = sum(frame.shape[0] for frame in batch_frames)
                metrics["raw_bars_count"] += int(parsed_rows)
                metrics["parsed_rows_count"] += int(parsed_rows)
                frames.append(pd.concat(batch_frames, ignore_index=True))
                continue

            metrics["fallback_batches"] += 1
            LOGGER.warning(
                "[WARN] Batch %d/%d: bars normalize failed (cols=%s); falling back to single-symbol HTTP for %d symbols",
                index,
                len(batches),
                columns_desc or "<unknown>",
                len(batch),
            )
            fallback_frame, fallback_pages, fallback_prescreened, fallback_metrics = (
                fetch_single_collection(batch, use_http=True)
            )
            if not fallback_frame.empty:
                frames.append(fallback_frame)
            metrics["pages_total"] += fallback_pages
            for key in ["rate_limited", "http_404_batches", "http_empty_batches"]:
                metrics[key] += int(fallback_metrics.get(key, 0))
            metrics["raw_bars_count"] += int(fallback_metrics.get("raw_bars_count", 0))
            metrics["parsed_rows_count"] += int(
                fallback_metrics.get("parsed_rows_count", 0)
            )
            metrics["http"]["requests"] += int(fallback_metrics.get("http_requests", 0))
            metrics["http"]["rows"] += int(fallback_metrics.get("http_rows", 0))
            metrics["http"]["rate_limit_hits"] += int(
                fallback_metrics.get("rate_limit_hits", fallback_metrics.get("rate_limited", 0))
            )
            metrics["http"]["retries"] += int(fallback_metrics.get("http_retries", 0))
            prescreened.update(fallback_prescreened)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BARS_COLUMNS)
    combined = _normalize_bars_frame(combined)
    combined = ensure_symbol_column(combined)
    if not combined.empty:
        combined = combined.dropna(subset=["timestamp"])
        if not combined.empty:
            combined = (
                combined.sort_values("timestamp")
                .groupby("symbol", as_index=False, group_keys=False)
                .tail(days)
            )
        combined = ensure_symbol_column(combined).reset_index(drop=True)

    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in combined.columns]
    if missing_cols:
        LOGGER.error(
            "Pre-gating frame missing columns: %s; gating skipped.",
            missing_cols,
        )
        combined = pd.DataFrame(columns=required_cols)

    combined = ensure_symbol_column(combined)

    clean = combined.dropna(subset=["timestamp"]) if not combined.empty else combined
    if not clean.empty:
        hist = (
            clean.groupby("symbol", as_index=False)["timestamp"]
            .size()
            .rename(columns={"size": "n"})
        )
    else:
        hist = pd.DataFrame(columns=["symbol", "n"])

    if min_history > 0 and not hist.empty:
        keep = set(hist.loc[hist["n"] >= min_history, "symbol"].tolist())
        insufficient = set(hist.loc[hist["n"] < min_history, "symbol"].tolist())
    else:
        keep = set(hist["symbol"].tolist()) if not hist.empty else set()
        insufficient = set()
    symbols_with_history = keep
    existing_insufficient = {
        sym for sym, reason in prescreened.items() if reason == "INSUFFICIENT_HISTORY"
    }
    insufficient = (insufficient | existing_insufficient)
    for sym in insufficient:
        prescreened.setdefault(sym, "INSUFFICIENT_HISTORY")
    if keep:
        combined = combined[combined["symbol"].isin(keep)].copy()
    else:
        combined = combined.iloc[0:0]
    if insufficient:
        metrics["insufficient_history"] = len(insufficient)

    symbols_with_bars = int(combined["symbol"].nunique()) if not combined.empty else 0
    symbols_no_bars = max(len(unique_symbols) - symbols_with_bars, 0)
    bars_rows_total = int(len(combined))
    metrics.update(
        {
            "symbols_in": len(unique_symbols),
            "symbols_with_bars": symbols_with_bars,
            "symbols_no_bars": symbols_no_bars,
            "bars_rows_total": bars_rows_total,
        }
    )
    missing = [sym for sym in unique_symbols if sym not in keep]
    metrics["symbols_no_bars_sample"] = missing[:10]
    for sym in missing:
        if sym in insufficient:
            continue
        prescreened.setdefault(sym, "NAN_DATA")

    return combined, metrics, prescreened


def merge_asset_metadata(bars_df: pd.DataFrame, asset_meta: Dict[str, dict]) -> pd.DataFrame:
    result = ensure_symbol_column(bars_df.copy())

    if asset_meta:
        meta_df = pd.DataFrame.from_dict(asset_meta, orient="index")
        meta_df = meta_df.reindex(columns=["exchange", "asset_class", "tradable"])
        meta_df = meta_df.rename_axis("symbol").reset_index()
        if not meta_df.empty:
            meta_df["symbol"] = meta_df["symbol"].astype(str).str.upper()
            m_exchange = dict(zip(meta_df["symbol"], meta_df["exchange"]))
            m_class = dict(zip(meta_df["symbol"], meta_df["asset_class"]))
            tradable_map = (
                dict(zip(meta_df["symbol"], meta_df["tradable"]))
                if "tradable" in meta_df.columns
                else {}
            )
            result["exchange"] = result["symbol"].map(m_exchange)
            result["asset_class"] = result["symbol"].map(m_class)
            if tradable_map:
                result["tradable"] = result["symbol"].map(tradable_map)

    result = ensure_symbol_column(result)

    for column in ["exchange", "asset_class"]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip().str.upper()
    if "tradable" not in result.columns:
        result["tradable"] = True
    else:
        result["tradable"] = result["tradable"].fillna(True).astype(bool)

    def _kind_from_row(row: pd.Series) -> str:
        exchange = str(row.get("exchange", "") or "").strip().upper()
        asset_class = str(row.get("asset_class", "") or "").strip().upper()
        if exchange in KNOWN_EQUITY:
            return "EQUITY"
        if not exchange and asset_class in {"US_EQUITY", "EQUITY"}:
            return "EQUITY"
        if exchange:
            classified = classify_exchange(exchange)
            if classified == "EQUITY":
                return classified
            if classified == "OTHER" and asset_class in {"US_EQUITY", "EQUITY"}:
                return "EQUITY"
            return classified
        if asset_class in {"US_EQUITY", "EQUITY"}:
            return "EQUITY"
        return "OTHER"

    result["kind"] = result.apply(_kind_from_row, axis=1)
    return result


def _load_alpaca_universe(
    *,
    base_dir: Path,
    days: int,
    feed: str,
    limit: Optional[int],
    fetch_mode: str,
    batch_size: int,
    max_workers: int,
    min_history: int,
    bars_source: str,
    exclude_otc: bool,
    iex_only: bool,
    liquidity_top: int,
    symbols_override: Optional[List[str]] = None,
    verify_request: bool = False,
    min_days_fallback: int = 365,
    min_days_final: int = 90,
    reuse_cache: bool = True,
    metrics: Optional[MutableMapping[str, Any]] = None,
) -> Tuple[
    pd.DataFrame,
    Dict[str, dict],
    dict[str, Any],
    Dict[str, str],
    dict[str, Any],
]:
    """
    Load eligible Alpaca symbols, apply filters, and return sampled bars metadata.

    The caller provides a ``metrics`` mapping which is updated in place with
    guardrail statistics such as ``universe_prefix_counts``.
    """
    empty_metrics = {
        "batches_total": 0,
        "batches_paged": 0,
        "pages_total": 0,
        "bars_rows_total": 0,
        "symbols_with_bars": 0,
        "symbols_no_bars": 0,
        "symbols_no_bars_sample": [],
        "fallback_batches": 0,
        "insufficient_history": 0,
        "rate_limited": 0,
        "http_404_batches": 0,
        "http_empty_batches": 0,
        "cache_hits": 0,
        "window_attempts": [],
        "window_used": 0,
        "symbols_override": 0,
        "symbols_in": 0,
        "universe_prefix_counts": {},
        "cache": {"batches_hit": 0, "batches_miss": 0},
    }
    empty_asset_metrics = {
        "assets_total": 0,
        "assets_tradable_equities": 0,
        "assets_after_filters": 0,
        "symbols_after_iex_filter": 0,
    }
    agg_metrics: MutableMapping[str, Any] = metrics if metrics is not None else {}
    for key, value in empty_metrics.items():
        if isinstance(value, list):
            agg_metrics[key] = list(value)
        elif isinstance(value, dict):
            agg_metrics[key] = dict(value)
        else:
            agg_metrics[key] = int(value)
    try:
        trading_client = _create_trading_client()
    except Exception as exc:
        LOGGER.error("Unable to create Alpaca trading client: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}, agg_metrics, {}, empty_asset_metrics

    try:
        symbols, asset_meta, asset_metrics = fetch_active_equity_symbols(
            trading_client,
            base_dir=base_dir,
            exclude_otc=exclude_otc,
        )
    except Exception as exc:
        LOGGER.error("Failed to fetch Alpaca asset universe: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}, agg_metrics, {}, empty_asset_metrics

    LOGGER.info(
        "Asset metrics: total=%d tradable_equities=%d after_filters=%d",
        int(asset_metrics.get("assets_total", 0)),
        int(asset_metrics.get("assets_tradable_equities", 0)),
        int(asset_metrics.get("assets_after_filters", 0)),
    )
    if asset_meta:
        sample = list(asset_meta.items())[:5]
        sample_str = ", ".join(f"{sym}:{meta.get('exchange', '')}" for sym, meta in sample)
        LOGGER.info("Asset sample: %s", sample_str)

    asset_meta = {
        str(symbol).strip().upper(): dict(meta or {})
        for symbol, meta in (asset_meta or {}).items()
    }
    override_cleaned: list[str] = []
    if symbols_override:
        seen_override: set[str] = set()
        for raw in symbols_override:
            sym = str(raw or "").strip().upper()
            if sym and sym not in seen_override:
                seen_override.add(sym)
                override_cleaned.append(sym)
        if override_cleaned:
            LOGGER.info(
                "Symbols override enabled (%d): %s",
                len(override_cleaned),
                ", ".join(override_cleaned[:10]),
            )

    raw_symbols = [str(sym).strip().upper() for sym in symbols]
    iex_exchanges = {"NASDAQ", "NYSE", "ARCA", "AMEX"}
    filtered_symbols: list[str] = []
    filtered_skips: Dict[str, str] = {}

    if override_cleaned:
        seen_override: set[str] = set()
        for sym in override_cleaned:
            if sym in seen_override:
                continue
            seen_override.add(sym)
            meta = asset_meta.get(sym, {})
            exchange = str(meta.get("exchange", "") or "").strip().upper()
            if iex_only and exchange and exchange not in iex_exchanges:
                filtered_skips.setdefault(sym, "UNKNOWN_EXCHANGE")
                continue
            filtered_symbols.append(sym)
        LOGGER.info("Universe sample size (override): %d", len(filtered_symbols))
    else:
        ordered_symbols = raw_symbols or list(asset_meta.keys())
        seen: set[str] = set()
        for sym in ordered_symbols:
            if sym in seen:
                continue
            meta = asset_meta.get(sym, {})
            tradable = bool(meta.get("tradable", True))
            asset_class = str(meta.get("asset_class", "") or "").strip().upper()
            exchange = str(meta.get("exchange", "") or "").strip().upper()
            if not tradable:
                filtered_skips.setdefault(sym, "NON_EQUITY")
                continue
            if asset_class not in {"US_EQUITY", "EQUITY"}:
                reason = "NON_EQUITY"
                if asset_class in {"OTC"} and exchange:
                    reason = "UNKNOWN_EXCHANGE"
                filtered_skips.setdefault(sym, reason)
                continue
            if iex_only and exchange not in iex_exchanges:
                filtered_skips.setdefault(sym, "UNKNOWN_EXCHANGE")
                continue
            filtered_symbols.append(sym)
            seen.add(sym)
        total_tradable = len(raw_symbols)
        LOGGER.info(
            "Universe sample size: %d (of %d tradable equities)",
            len(filtered_symbols),
            total_tradable or asset_metrics.get("assets_tradable_equities", 0),
        )

    asset_metrics["symbols_after_iex_filter"] = len(filtered_symbols)
    asset_metrics["assets_total"] = int(asset_metrics.get("assets_total", len(asset_meta)))
    asset_metrics["assets_after_filters"] = len(filtered_symbols)

    assets_df = pd.DataFrame({"symbol": filtered_symbols})
    seed = int(pd.Timestamp.utcnow().strftime("%Y%m%d"))
    if limit and len(assets_df) > limit:
        universe_df = assets_df.sample(n=limit, random_state=seed)
    else:
        universe_df = assets_df.copy()

    universe_df = universe_df.reset_index(drop=True)
    _write_universe_prefix_metrics(universe_df, agg_metrics)

    symbols = universe_df["symbol"].astype(str).str.upper().tolist()

    if asset_meta:
        asset_meta = {sym: asset_meta.get(sym, {}) for sym in symbols if sym in asset_meta}

    if not symbols:
        return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta, agg_metrics, {}, asset_metrics

    agg_metrics["symbols_in"] = len(symbols)

    data_client: Optional["StockHistoricalDataClient"] = None
    if (bars_source or "http").strip().lower() == "sdk":
        try:
            data_client = _create_data_client()
        except Exception as exc:
            LOGGER.error("Unable to create Alpaca market data client: %s", exc)
            return (
                pd.DataFrame(columns=INPUT_COLUMNS),
                asset_meta,
                agg_metrics,
                {},
                asset_metrics,
            )

    verify_hook_fn = make_verify_hook(bool(verify_request))
    attempt_candidates = [days, min_days_fallback, min_days_final]
    attempt_days: list[int] = []
    for candidate in attempt_candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        value = max(1, value)
        if value not in attempt_days:
            attempt_days.append(value)
    if not attempt_days:
        attempt_days.append(max(1, int(days)))

    bars_df = pd.DataFrame(columns=BARS_COLUMNS)
    prescreened: Dict[str, str] = {}
    fetch_metrics: dict[str, Any] = {}
    window_attempts: list[int] = []
    final_window = 0
    symbols_override_count = len(override_cleaned)

    for idx, window_days in enumerate(attempt_days):
        window_attempts.append(window_days)
        try:
            start_iso, end_iso, last_day = calc_daily_window(trading_client, window_days)
        except Exception as exc:
            LOGGER.error(
                "Failed to determine trading window for %d days: %s",
                window_days,
                exc,
            )
            start_iso, end_iso, last_day = _fallback_daily_window(window_days)
            LOGGER.info(
                "Using fallback trading window for %d days (%s → %s)",
                window_days,
                start_iso,
                end_iso,
            )
        LOGGER.info(
            "Requesting %d trading days ending %s (%s → %s)",
            window_days,
            last_day,
            start_iso,
            end_iso,
        )
        use_verify = bool(verify_request) and idx == 0
        bars_df_candidate, metrics_candidate, prescreened_candidate = _fetch_daily_bars(
            data_client,
            symbols,
            days=window_days,
            start_iso=start_iso,
            end_iso=end_iso,
            feed=feed,
            fetch_mode=fetch_mode,
            batch_size=max(1, batch_size),
            max_workers=max(1, max_workers),
            min_history=min_history,
            bars_source=bars_source,
            run_date=last_day,
            reuse_cache=reuse_cache,
            verify_hook=verify_hook_fn if use_verify else None,
        )

        metrics_candidate = metrics_candidate or {}
        for key, value in metrics_candidate.items():
            if isinstance(value, list):
                continue
            if key in {
                "bars_rows_total",
                "symbols_with_bars",
                "symbols_no_bars",
                "raw_bars_count",
                "parsed_rows_count",
            }:
                continue
            if isinstance(value, dict):
                agg_metrics[key] = dict(value)
                continue
            agg_metrics[key] = agg_metrics.get(key, 0) + int(value)

        bars_df = bars_df_candidate
        fetch_metrics = metrics_candidate
        prescreened = prescreened_candidate
        final_window = window_days

        if int(metrics_candidate.get("bars_rows_total", 0)) > 0:
            break
        if idx < len(attempt_days) - 1:
            LOGGER.info(
                "No bars returned for %d-day window; retrying with %d-day fallback",
                window_days,
                attempt_days[idx + 1],
            )

    agg_metrics["bars_rows_total"] = _coerce_int(fetch_metrics.get("bars_rows_total", 0))
    agg_metrics["symbols_with_bars"] = _coerce_int(fetch_metrics.get("symbols_with_bars", 0))
    agg_metrics["symbols_no_bars"] = _coerce_int(fetch_metrics.get("symbols_no_bars", 0))
    agg_metrics["symbols_in"] = _coerce_int(fetch_metrics.get("symbols_in", agg_metrics.get("symbols_in", 0)))
    agg_metrics["symbols_no_bars_sample"] = list(
        fetch_metrics.get("symbols_no_bars_sample", []) or []
    )
    agg_metrics["raw_bars_count"] = _coerce_int(fetch_metrics.get("raw_bars_count", 0))
    agg_metrics["parsed_rows_count"] = _coerce_int(fetch_metrics.get("parsed_rows_count", 0))
    agg_metrics["window_attempts"] = window_attempts
    agg_metrics["window_used"] = final_window
    agg_metrics["symbols_override"] = symbols_override_count

    LOGGER.info(
        "Bars fetch metrics: batches=%d paged=%d pages=%d rows=%d symbols_with_bars=%d",
        int(agg_metrics.get("batches_total", 0)),
        int(agg_metrics.get("batches_paged", 0)),
        int(agg_metrics.get("pages_total", 0)),
        int(agg_metrics.get("bars_rows_total", 0)),
        int(agg_metrics.get("symbols_with_bars", 0)),
    )
    LOGGER.info(
        "Bars window attempts: %s → used=%d",
        window_attempts or ["<none>"],
        final_window,
    )
    if agg_metrics.get("http_404_batches"):
        LOGGER.info(
            "Bars HTTP 404 batches: %d",
            int(agg_metrics.get("http_404_batches", 0)),
        )
    if agg_metrics.get("http_empty_batches"):
        LOGGER.info(
            "Bars HTTP empty batches: %d",
            int(agg_metrics.get("http_empty_batches", 0)),
        )
    missing_sample = agg_metrics.get("symbols_no_bars_sample", [])
    if missing_sample:
        LOGGER.info("Symbols without bars (sample): %s", list(missing_sample)[:10])
    if agg_metrics.get("fallback_batches"):
        LOGGER.info(
            "Fallback batches invoked: %d",
            int(agg_metrics.get("fallback_batches", 0)),
        )
    if agg_metrics.get("insufficient_history"):
        LOGGER.info(
            "Symbols dropped for insufficient history: %d",
            int(agg_metrics.get("insufficient_history", 0)),
        )

    if filtered_skips:
        for sym, reason in filtered_skips.items():
            prescreened.setdefault(sym, reason)

    if bars_df.empty:
        return bars_df, asset_meta, agg_metrics, prescreened, asset_metrics

    bars_df = bars_df.copy()
    if "symbol" not in bars_df.columns:
        LOGGER.error("Normalized bars missing 'symbol' unexpectedly; skipping merge")
        return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta, agg_metrics, prescreened, asset_metrics
    if liquidity_top and liquidity_top > 0:
        try:
            bars_df.sort_values(["symbol", "timestamp"], inplace=True)
            recent = (
                bars_df.dropna(subset=["timestamp"])
                .sort_values("timestamp")
                .groupby("symbol", as_index=False, group_keys=False)
                .tail(20)
            )
            recent = ensure_symbol_column(recent).copy()
            recent["volume"] = pd.to_numeric(recent["volume"], errors="coerce")
            adv = recent.groupby("symbol")["volume"].mean().fillna(0)
            top_symbols = set(adv.sort_values(ascending=False).head(liquidity_top).index)
            if top_symbols:
                bars_df = bars_df[bars_df["symbol"].isin(top_symbols)]
                agg_metrics["symbols_with_bars"] = min(
                    int(agg_metrics.get("symbols_with_bars", 0)), len(top_symbols)
                )
        except Exception as exc:
            LOGGER.warning("Failed liquidity filter computation: %s", exc)
    bars_df = merge_asset_metadata(bars_df, asset_meta)
    return bars_df, asset_meta, agg_metrics, prescreened, asset_metrics


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
    return trading_base_url()


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


def _filter_equity_assets(
    assets: Iterable[object], *, exclude_otc: bool
) -> Tuple[List[str], Dict[str, dict], dict[str, int]]:
    allowed_exchanges = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "NYSEARCA"}
    symbols: set[str] = set()
    asset_meta: dict[str, dict[str, object]] = {}
    metrics = {
        "assets_total": 0,
        "assets_tradable_equities": 0,
        "assets_after_filters": 0,
    }
    for asset in assets:
        metrics["assets_total"] += 1
        tradable_raw = _asset_attr(asset, "tradable")
        tradable = bool(tradable_raw) if tradable_raw is not None else True
        if not tradable:
            continue

        status = _enum_to_str(_asset_attr(asset, "status")).strip().upper()
        if status and status not in {"ACTIVE", ""}:
            continue

        cls = _enum_to_str(_asset_attr(asset, "asset_class", "class_", "class")).strip().upper()
        if cls not in {"US_EQUITY", "EQUITY"}:
            continue

        metrics["assets_tradable_equities"] += 1

        symbol = _enum_to_str(_asset_attr(asset, "symbol")).strip().upper()
        if not symbol:
            continue

        exchange = _enum_to_str(_asset_attr(asset, "exchange", "primary_exchange")).strip().upper()
        if exclude_otc and exchange not in allowed_exchanges:
            continue

        symbols.add(symbol)
        asset_meta[symbol] = {
            "exchange": exchange,
            "asset_class": cls,
            "tradable": tradable,
        }
        metrics["assets_after_filters"] += 1

    LOGGER.info("Asset meta ready: %d symbols (tradable US equities)", len(asset_meta))
    return sorted(symbols), asset_meta, metrics


def _fetch_assets_via_sdk(trading_client, *, exclude_otc: bool) -> Tuple[List[str], Dict[str, dict], dict[str, int]]:
    if trading_client is None:
        raise RuntimeError("Trading client is unavailable")
    try:
        if GetAssetsRequest is None or AssetStatus is None or AssetClass is None:
            assets = trading_client.get_all_assets()
        else:
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = trading_client.get_all_assets(request)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    return _filter_equity_assets(assets, exclude_otc=exclude_otc)


def _fetch_assets_via_http(exclude_otc: bool) -> Tuple[List[str], Dict[str, dict], dict[str, int]]:
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        LOGGER.warning("Alpaca credentials missing; cannot fetch assets via HTTP fallback.")
        return [], {}, {"assets_total": 0, "assets_tradable_equities": 0, "assets_after_filters": 0}

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
        if response.status_code in (401, 403):
            raise AlpacaUnauthorizedError(endpoint="/v2/assets")
        response.raise_for_status()
        payload = response.json()
    except AlpacaUnauthorizedError:
        raise
    except Exception as exc:
        LOGGER.warning("Raw Alpaca asset fetch failed: %s", exc)
        return [], {}, {"assets_total": 0, "assets_tradable_equities": 0, "assets_after_filters": 0}

    if not isinstance(payload, list):
        LOGGER.warning("Unexpected payload from Alpaca assets endpoint: %s", type(payload))
        return [], {}, {"assets_total": 0, "assets_tradable_equities": 0, "assets_after_filters": 0}

    filtered, asset_meta, metrics = _filter_equity_assets(payload, exclude_otc=exclude_otc)
    return filtered, asset_meta, metrics


def fetch_active_equity_symbols(
    trading_client,
    *,
    base_dir: Optional[Path] = None,
    exclude_otc: bool = True,
) -> Tuple[List[str], Dict[str, dict], dict[str, int]]:
    """Fetch active US equity symbols from Alpaca with resilient fallbacks."""

    cache_path = _assets_cache_path(base_dir)
    metrics = {"assets_total": 0, "assets_tradable_equities": 0, "assets_after_filters": 0}
    try:
        symbols, asset_meta, metrics = _fetch_assets_via_sdk(
            trading_client, exclude_otc=exclude_otc
        )
    except Exception as exc:  # pragma: no cover - requires Alpaca SDK error
        LOGGER.warning("SDK assets fetch failed (%s); falling back to HTTP", exc)
        symbols, asset_meta, metrics = _fetch_assets_via_http(exclude_otc)
        if symbols:
            _write_assets_cache(cache_path, symbols)
            return symbols, asset_meta, metrics

        cached = _read_assets_cache(cache_path)
        if cached:
            LOGGER.warning(
                "Using cached Alpaca assets (%d symbols) after validation failure.",
                len(cached),
            )
            return cached, {}, metrics

        LOGGER.error("Failed to fetch Alpaca assets; continuing with empty universe.")
        return [], {}, metrics

    if symbols:
        _write_assets_cache(cache_path, symbols)

    return symbols, asset_meta, metrics


def _prepare_input_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    prepared = df.copy()
    for column in INPUT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[INPUT_COLUMNS]


def _load_ranker_config(path: Optional[Path] = None) -> dict:
    target = path or RANKER_CONFIG_PATH
    if not target.exists():
        LOGGER.warning("Ranker config %s missing; falling back to defaults.", target)
        return {}
    try:
        import yaml
    except ImportError:
        LOGGER.warning("PyYAML unavailable; using built-in ranker defaults.")
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - configuration issues are unexpected
        LOGGER.error("Failed to load ranker config %s: %s", target, exc)
        return {}
    if not isinstance(loaded, dict):
        LOGGER.error("Ranker config %s must be a mapping; ignoring.", target)
        return {}
    return loaded


def _apply_preset_config(cfg: dict, preset_key: str) -> dict:
    if not preset_key:
        return cfg
    presets = cfg.get("presets")
    if not isinstance(presets, Mapping):
        return cfg
    preset = presets.get(str(preset_key).strip().lower())
    if not isinstance(preset, Mapping):
        return cfg
    gates = dict(cfg.get("gates") or {})
    for key, value in preset.items():
        gates[key] = value
    cfg["gates"] = gates
    return cfg


def _apply_relaxations(cfg: dict, mode: str) -> dict:
    gates = dict(cfg.get("gates") or {})
    mode = (mode or "none").strip().lower()
    if mode == "sma_only":
        gates["require_sma_stack"] = True
        for key in [
            "min_rsi",
            "max_rsi",
            "min_adx",
            "min_aroon",
            "min_volexp",
            "max_gap",
            "max_liq_penalty",
            "min_score",
        ]:
            if key in gates:
                gates[key] = None
    elif mode == "cross_or_rsi":
        gates["require_sma_stack"] = False
    cfg["gates"] = gates
    return cfg


def _prepare_ranker_config(
    base_cfg: Optional[Mapping[str, object]],
    *,
    preset_key: str,
    relax_mode: str,
    min_history: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg or {})
    if not isinstance(cfg.get("weights"), Mapping):
        cfg["weights"] = dict(cfg.get("weights", {}))
    if not isinstance(cfg.get("components"), Mapping):
        cfg["components"] = dict(cfg.get("components", {}))
    gates = dict(cfg.get("gates") or {})
    cfg["gates"] = gates
    cfg = _apply_preset_config(cfg, preset_key)
    cfg = _apply_relaxations(cfg, relax_mode)
    gates = dict(cfg.get("gates") or {})
    try:
        gates["min_history"] = max(int(min_history), 0)
    except (TypeError, ValueError):
        pass
    gates.setdefault("history_column", "history")
    cfg["gates"] = gates
    cfg["_gate_preset"] = (preset_key or "standard").strip().lower() or "standard"
    cfg["_gate_relax_mode"] = (relax_mode or "none").strip().lower() or "none"
    return cfg


def build_enriched_bars(
    bars_df: pd.DataFrame,
    cfg: Mapping[str, object],
    *,
    timings: Optional[dict[str, float]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    include_intermediate: Optional[bool] = None,
) -> pd.DataFrame:
    stage_timer = T()
    df = bars_df.copy() if bars_df is not None else pd.DataFrame(columns=INPUT_COLUMNS)
    if df.empty:
        normalize_elapsed = stage_timer.lap("normalize_secs")
        if timings is not None:
            timings["normalize_secs"] = timings.get("normalize_secs", 0.0) + normalize_elapsed
            timings.setdefault("feature_secs", timings.get("feature_secs", 0.0))
        columns = [
            "symbol",
            "timestamp",
            *ALL_FEATURE_COLUMNS,
            "close",
            "volume",
            "exchange",
            "history",
        ]
        return pd.DataFrame(columns=columns)

    df = df.copy()
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "exchange" not in df.columns:
        df["exchange"] = ""
    df["exchange"] = df["exchange"].astype("string").fillna("").str.strip().str.upper()
    df = df.dropna(subset=["symbol", "timestamp"])
    df = df[df["symbol"] != ""]

    normalize_elapsed = stage_timer.lap("normalize_secs")
    if timings is not None:
        timings["normalize_secs"] = timings.get("normalize_secs", 0.0) + normalize_elapsed

    history_counts = (
        df.groupby("symbol", as_index=False)["timestamp"].size().rename(columns={"size": "history"})
    )

    add_intermediate = True if include_intermediate is None else bool(include_intermediate)
    features_df = compute_all_features(df, cfg, add_intermediate=add_intermediate)
    feature_elapsed = stage_timer.lap("feature_secs")
    if timings is not None:
        timings["feature_secs"] = timings.get("feature_secs", 0.0) + feature_elapsed
    if features_df.empty:
        columns = [
            "symbol",
            "timestamp",
            *(feature_columns or ALL_FEATURE_COLUMNS),
            "close",
            "volume",
            "exchange",
            "history",
        ]
        return pd.DataFrame(columns=columns)

    base_cols = ["symbol", "timestamp", "close", "volume", "exchange"]
    base_info = df[base_cols].drop_duplicates(subset=["symbol", "timestamp"], keep="last")

    if feature_columns is not None:
        ordered = [str(col) for col in feature_columns]
        missing = [col for col in ordered if col not in features_df.columns]
        for col in missing:
            features_df[col] = pd.NA
        feature_subset = ["symbol", "timestamp", *ordered]
        features_df = features_df.reindex(columns=feature_subset)
    else:
        feature_subset = ["symbol", "timestamp", *ALL_FEATURE_COLUMNS]
        features_df = features_df.reindex(columns=feature_subset, fill_value=pd.NA)

    enriched = features_df.merge(base_info, on=["symbol", "timestamp"], how="left")
    enriched = enriched.merge(history_counts, on="symbol", how="left")
    enriched["history"] = pd.to_numeric(enriched["history"], errors="coerce").fillna(0)
    enriched.sort_values(["symbol", "timestamp"], inplace=True)
    enriched.reset_index(drop=True, inplace=True)

    for column in ["close", "volume"]:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    if "exchange" in enriched.columns:
        enriched["exchange"] = enriched["exchange"].astype("string").fillna("").str.upper()

    return enriched


def _prepare_top_frame(candidates_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty or top_n <= 0:
        return pd.DataFrame(columns=TOP_CANDIDATE_COLUMNS)
    subset = candidates_df.head(int(top_n)).copy()
    for column in TOP_CANDIDATE_COLUMNS:
        if column not in subset.columns:
            subset[column] = pd.NA
    subset = subset[TOP_CANDIDATE_COLUMNS]
    return subset


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fallback = series.astype("string") if isinstance(series, pd.Series) else pd.Series(dtype="string")
    return formatted.where(~parsed.isna(), fallback).fillna("")


def _ensure_score_breakdown(raw: object, score_value: Optional[float]) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, (dict, list)):
        try:
            return json.dumps(raw, sort_keys=True)
        except TypeError:
            pass
    if score_value is not None and not pd.isna(score_value):
        return json.dumps({"score": float(score_value)})
    return json.dumps({})


def _normalise_top_candidates(
    top_df: pd.DataFrame,
    scored_df: Optional[pd.DataFrame],
    *,
    universe_count: Optional[int] = None,
) -> pd.DataFrame:
    if top_df is None:
        return pd.DataFrame(columns=TOP_CANDIDATE_COLUMNS)
    frame = top_df.copy()
    if frame.empty:
        for column in TOP_CANDIDATE_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame[TOP_CANDIDATE_COLUMNS]

    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.NA
    frame["timestamp"] = _normalise_timestamp(frame["timestamp"])
    frame["_ts"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)

    scored_meta: Optional[pd.DataFrame] = None
    if isinstance(scored_df, pd.DataFrame) and not scored_df.empty:
        scored_meta = scored_df.copy()
        scored_meta["symbol"] = (
            scored_meta.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()
        )
        scored_meta["_ts"] = pd.to_datetime(
            scored_meta.get("timestamp"), errors="coerce", utc=True
        )

    meta_columns = [col for col in ("exchange", "close", "volume", "score_breakdown") if col in frame.columns]
    lookup_columns = {"exchange", "close", "volume", "score_breakdown"}

    if scored_meta is not None:
        available_meta = [col for col in lookup_columns if col in scored_meta.columns]
        if available_meta:
            exact_meta = (
                scored_meta.dropna(subset=["symbol"])
                .drop_duplicates(subset=["symbol", "_ts"], keep="last")
                .loc[:, ["symbol", "_ts", *available_meta]]
                .rename(columns={col: f"__exact_{col}" for col in available_meta})
            )
            frame = frame.merge(exact_meta, on=["symbol", "_ts"], how="left")

            symbol_meta = (
                scored_meta.dropna(subset=["symbol"])
                .sort_values(["symbol", "_ts"], na_position="last")
                .drop_duplicates(subset=["symbol"], keep="last")
                .loc[:, ["symbol", *available_meta]]
                .rename(columns={col: f"__symbol_{col}" for col in available_meta})
            )
            frame = frame.merge(symbol_meta, on="symbol", how="left")
            meta_columns.extend(available_meta)

    meta_columns = sorted({*meta_columns, *lookup_columns})

    if "score" not in frame.columns or frame["score"].isna().all():
        frame["score"] = frame.get("Score", pd.Series(dtype="float64"))
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if "Score" not in frame.columns:
        frame["Score"] = frame["score"]

    if "entry_price" not in frame.columns:
        frame["entry_price"] = pd.NA

    optional_defaults = {
        "adv20": frame.get("adv20", pd.Series(pd.NA, index=frame.index)),
        "atrp": frame.get("atrp", pd.Series(pd.NA, index=frame.index)),
    }
    for column, series in optional_defaults.items():
        if column not in frame.columns:
            frame[column] = series

    for column in meta_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        exact_col = f"__exact_{column}"
        symbol_col = f"__symbol_{column}"
        if exact_col in frame.columns:
            mask = frame[column].isna()
            frame.loc[mask, column] = frame.loc[mask, exact_col]
            frame.drop(columns=[exact_col], inplace=True)
        if symbol_col in frame.columns:
            mask = frame[column].isna()
            frame.loc[mask, column] = frame.loc[mask, symbol_col]
            frame.drop(columns=[symbol_col], inplace=True)

    frame["exchange"] = (
        frame.get("exchange", pd.Series(dtype="string")).astype("string").fillna("").str.upper()
    )
    for column in ("close", "entry_price", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    missing_close = frame["close"].isna()
    if missing_close.any():
        frame.loc[missing_close, "close"] = frame.loc[missing_close, "entry_price"]

    if "ADV20" in frame.columns:
        frame["adv20"] = pd.to_numeric(frame.get("ADV20"), errors="coerce")
    if "ATR_pct" in frame.columns and frame["ATR_pct"].notna().any():
        frame["atrp"] = pd.to_numeric(frame.get("ATR_pct"), errors="coerce")

    frame["entry_price"] = pd.to_numeric(frame.get("entry_price"), errors="coerce")
    entry_missing = frame["entry_price"].isna()
    if entry_missing.any():
        frame.loc[entry_missing, "entry_price"] = frame.loc[entry_missing, "close"]

    frame["score_breakdown"] = [
        _ensure_score_breakdown(raw, score)
        for raw, score in zip(frame.get("score_breakdown"), frame["score"])
    ]

    count = universe_count if universe_count is not None else None
    if count is None:
        count = int(scored_df.shape[0]) if isinstance(scored_df, pd.DataFrame) else None
    if count is None:
        count = int(frame.shape[0])
    frame["universe_count"] = int(count)

    required = [
        "timestamp",
        "symbol",
        "score",
        "exchange",
        "close",
        "volume",
        "universe_count",
        "score_breakdown",
    ]
    ordered = required + [col for col in TOP_CANDIDATE_COLUMNS if col not in required]
    ordered += [col for col in frame.columns if col not in ordered]
    frame = frame.reindex(columns=ordered, fill_value=pd.NA)

    frame.drop(columns=["_ts"], inplace=True, errors="ignore")
    return frame


def _summarise_features(
    scored_df: pd.DataFrame, cfg: Optional[Mapping[str, object]] = None
) -> dict[str, dict[str, Optional[float]]]:
    if scored_df is None or scored_df.empty:
        return {}
    cfg = cfg or {}
    components = dict(DEFAULT_COMPONENT_MAP)
    for key, value in (cfg.get("components") or {}).items():
        if value:
            components[str(key)] = str(value)
    feature_cols = set(components.values()) | set(CORE_FEATURE_COLUMNS)
    summary: dict[str, dict[str, Optional[float]]] = {}
    for column in sorted(feature_cols):
        if column not in scored_df.columns:
            continue
        series = pd.to_numeric(scored_df[column], errors="coerce")
        mask = series.notna()
        if not mask.any():
            summary[column] = {"mean": None, "std": None}
            continue
        mean = float(series[mask].mean())
        std = float(series[mask].std(ddof=0))
        summary[column] = {"mean": round(mean, 4), "std": round(std, 4)}
    return summary


def _format_timestamp(ts: datetime) -> str:
    return ts.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    data = df.to_csv(index=False).encode("utf-8")
    atomic_write_bytes(path, data)


def _write_shortlist_csv(path: Path, shortlist: Optional[pd.DataFrame]) -> Path:
    columns = ["symbol", "coarse_score", "coarse_rank"]
    if shortlist is None or shortlist.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = shortlist.copy()
        frame = frame.reindex(columns=columns, fill_value=pd.NA)
        frame["symbol"] = frame["symbol"].astype("string").str.upper().fillna("")
        if "coarse_score" in frame.columns:
            frame["coarse_score"] = pd.to_numeric(frame["coarse_score"], errors="coerce")
        if "coarse_rank" in frame.columns:
            frame["coarse_rank"] = pd.to_numeric(frame["coarse_rank"], errors="coerce").astype("Int64")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(target, frame)
    return target


def _write_json_atomic(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _load_coverage_table(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Failed to read coverage table %s: %s", path, exc)
        return {}
    coverage: dict[str, dict[str, object]] = {}
    for row in df.to_dict("records"):
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        coverage[symbol] = {
            "last_ok_utc": row.get("last_ok_utc"),
            "last_miss_utc": row.get("last_miss_utc"),
            "ok_count": int(row.get("ok_count", 0) or 0),
            "miss_count": int(row.get("miss_count", 0) or 0),
        }
    return coverage


def _write_coverage_table(path: Path, coverage: Mapping[str, Mapping[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for symbol in sorted(coverage.keys()):
        entry = coverage[symbol]
        rows.append(
            {
                "symbol": symbol,
                "last_ok_utc": entry.get("last_ok_utc"),
                "last_miss_utc": entry.get("last_miss_utc"),
                "ok_count": int(entry.get("ok_count", 0) or 0),
                "miss_count": int(entry.get("miss_count", 0) or 0),
            }
        )
    frame = pd.DataFrame(rows, columns=["symbol", "last_ok_utc", "last_miss_utc", "ok_count", "miss_count"])
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(path, frame)


def _normalize_timings(timings: Optional[Mapping[str, object]] = None) -> dict[str, float]:
    base_keys = {
        "fetch_secs": 0.0,
        "feature_secs": 0.0,
        "rank_secs": 0.0,
        "gates_secs": 0.0,
    }
    normalized: dict[str, float] = {key: float(value) for key, value in base_keys.items()}
    extras: dict[str, float] = {}
    for key, value in (timings or {}).items():
        try:
            val = round(float(value), 3)
        except Exception:
            continue
        if key in normalized:
            normalized[key] = val
        else:
            extras[key] = val
    normalized.update(extras)
    return normalized


def _metrics_defaults() -> dict[str, Any]:
    return {
        "symbols_in": 0,
        "symbols_with_bars": 0,
        "bars_rows_total": 0,
        "rows": 0,
        "universe_prefix_counts": {},
        "gate_fail_counts": {},
        "timings": _normalize_timings({}),
        "http": {
            "requests": 0,
            "rows": 0,
            "rate_limit_hits": 0,
            "retries": 0,
        },
        "cache": {"batches_hit": 0, "batches_miss": 0},
    }


def _merge_dict(target: MutableMapping[str, Any], payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in payload.items():
        if isinstance(value, Mapping):
            node = target.setdefault(key, {})
            if isinstance(node, MutableMapping):
                _merge_dict(node, value)
            else:
                target[key] = dict(value)
        else:
            target[key] = value
    return target


def _update_metrics_file(base_dir: Path, payload: Mapping[str, Any]) -> Path:
    metrics_dir = base_dir / "data"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "screener_metrics.json"
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            LOGGER.warning("Failed to read metrics from %s: %s", metrics_path, exc)
            existing = {}
    merged: dict[str, Any] = dict(existing)
    _merge_dict(merged, _metrics_defaults())
    _merge_dict(merged, payload)
    _write_json_atomic(metrics_path, merged)
    return metrics_path


def _latest_child_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    children = [child for child in root.iterdir() if child.is_dir()]
    if not children:
        return None
    return sorted(children)[-1]


def _is_batch_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    return name.endswith(".parquet") or name.endswith(".parquet.csv.gz")


def _iter_batch_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    shards = sorted(path for path in directory.iterdir() if _is_batch_file(path))
    seen: set[str] = set()
    for shard in shards:
        name = shard.name
        base = name[: -len(".csv.gz")] if name.endswith(".parquet.csv.gz") else name
        if base in seen:
            continue
        seen.add(base)
        yield shard


def _write_parquet_or_csv(df: pd.DataFrame, path: Union[str, Path]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target, index=False)
        return str(target)
    except Exception as exc:
        LOGGER.warning(
            "[DELTA] parquet engine missing/unavailable (%s). Falling back to CSV gzip for %s",
            exc,
            target.name,
        )
        alt = target.with_suffix(target.suffix + ".csv.gz")
        with gzip.open(alt, "wt", encoding="utf-8") as handle:
            df.to_csv(handle, index=False)
        return str(alt)


def _read_batch_safely(path: Path) -> pd.DataFrame:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".gz" and path.name.endswith(".csv.gz"):
            return pd.read_csv(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        LOGGER.warning("Unsupported shard format at %s", path)
    except Exception as exc:
        LOGGER.warning("Failed to read shard %s: %s", path, exc)
    return pd.DataFrame(columns=INPUT_COLUMNS)


def _load_local_bars(
    base_dir: Path,
    feed: str,
    days: int,
    *,
    symbols: Optional[Sequence[str]] = None,
    reuse_cache: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    feed = (feed or DEFAULT_FEED).strip().lower() or DEFAULT_FEED
    cache_root = base_dir / "data" / "cache" / f"bars_1d_{feed}"
    delta_root = base_dir / "data" / "bars" / feed / "1d"

    frames: list[pd.DataFrame] = []
    latest_delta = _latest_child_dir(delta_root)
    if latest_delta is not None:
        for shard in _iter_batch_files(latest_delta):
            frames.append(_read_batch_safely(shard))

    if reuse_cache and cache_root.exists():
        day_limit = max(int(days), 0) if days else 0
        collected = 0
        for day_dir in sorted(cache_root.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            collected += 1
            for shard in _iter_batch_files(day_dir):
                frames.append(_read_batch_safely(shard))
            if day_limit and collected >= day_limit:
                break

    if not frames:
        return pd.DataFrame(columns=INPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalize_bars_frame(combined)
    if "exchange" not in combined.columns:
        combined["exchange"] = ""

    if symbols:
        symbol_set = {str(sym or "").strip().upper() for sym in symbols if sym}
        if symbol_set:
            combined = combined[combined["symbol"].isin(symbol_set)]

    combined.dropna(subset=["timestamp"], inplace=True)
    if days:
        combined["session"] = combined["timestamp"].dt.normalize()
        sessions = combined["session"].dropna().unique()
        if len(sessions) > days:
            keep = set(sorted(sessions)[-days:])
            combined = combined[combined["session"].isin(keep)]
        combined.drop(columns=["session"], inplace=True, errors="ignore")

    if limit and not combined.empty:
        try:
            limit_val = max(int(limit), 0)
        except (TypeError, ValueError):
            limit_val = 0
        if limit_val:
            ordered = (
                combined[["symbol", "timestamp"]]
                .sort_values(["symbol", "timestamp"])
                .drop_duplicates(subset=["symbol"], keep="last")
            )
            keep_symbols = ordered["symbol"].head(limit_val).tolist()
            combined = combined[combined["symbol"].isin(keep_symbols)]

    combined.sort_values(["symbol", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _compute_symbol_stats_frame(bars_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "close", "ADV20", "ATR_pct", "last_bar_date"]
    if bars_df is None or bars_df.empty:
        return pd.DataFrame(columns=columns)

    df = bars_df.copy()
    df["symbol"] = df["symbol"].astype("string").str.upper()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df.dropna(subset=["symbol", "timestamp", "close", "volume", "high", "low"], inplace=True)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df.sort_values(["symbol", "timestamp"], inplace=True)
    df["close_volume"] = df["close"] * df["volume"]
    df["ADV20"] = (
        df.groupby("symbol", group_keys=False)["close_volume"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
    )
    prev_close = df.groupby("symbol")[["close"]].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close["close"]).abs(),
            (df["low"] - prev_close["close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["TR"] = true_range
    df["ATR14"] = df.groupby("symbol", group_keys=False)["TR"].transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    )
    latest = df.groupby("symbol", as_index=False).tail(1).copy()
    latest["ATR_pct"] = (latest.get("ATR14", pd.Series(index=latest.index)) / latest["close"]).replace(
        [np.inf, -np.inf], np.nan
    )
    latest["last_bar_date"] = latest["timestamp"].dt.date.astype("string")
    result = latest.reindex(columns=columns, fill_value=pd.NA)
    result.sort_values("symbol", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _resolve_latest_session(bars_df: pd.DataFrame) -> str:
    if bars_df is None or bars_df.empty or "timestamp" not in bars_df.columns:
        return datetime.now(timezone.utc).date().isoformat()
    sessions = pd.to_datetime(bars_df["timestamp"], utc=True, errors="coerce").dt.date.dropna()
    if sessions.empty:
        return datetime.now(timezone.utc).date().isoformat()
    return max(sessions).isoformat()


def _run_delta_update(args: argparse.Namespace, base_dir: Path) -> int:
    LOGGER.info("[DELTA] Starting delta update run")
    run_ts = datetime.now(timezone.utc)
    run_utc = _format_timestamp(run_ts)

    feed = str(getattr(args, "feed", DEFAULT_FEED) or DEFAULT_FEED).strip().lower()
    batch_size = max(1, int(getattr(args, "batch_size", 50)))

    try:
        start_iso, end_iso, last_day = _fallback_daily_window(1)
    except Exception as exc:  # pragma: no cover - fallback already guards
        LOGGER.warning("Delta update fallback window failed: %s", exc)
        start_iso, end_iso, last_day = _fallback_daily_window(1)

    LOGGER.info("[DELTA] Requesting %s (%s → %s)", last_day, start_iso, end_iso)

    symbols, _, asset_metrics = _fetch_assets_via_http(exclude_otc=False)
    unique_symbols = list(dict.fromkeys(symbols))
    if not unique_symbols:
        LOGGER.error("[DELTA] No tradable symbols available from Alpaca assets endpoint")
        return 1

    coverage_path = base_dir / "data" / "coverage" / f"{feed}_coverage.csv"
    coverage = _load_coverage_table(coverage_path)

    output_dir = base_dir / "data" / "bars" / feed / "1d" / last_day
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = 0
    if output_dir.exists():
        for existing_file in _iter_batch_files(output_dir):
            try:
                if existing_file.stat().st_size == 0:
                    continue
                existing_df = _read_batch_safely(existing_file)
                if existing_df.empty:
                    continue
                existing_rows += int(existing_df.shape[0])
            except Exception as exc:  # pragma: no cover - best effort only
                LOGGER.debug("[DELTA] Could not inspect %s: %s", existing_file, exc)
                continue
            if existing_rows > 0:
                break
    if existing_rows > 0:
        LOGGER.info(
            "[DELTA] No new bars for %s; shards already present. Proceeding.",
            last_day,
        )
        return 0

    metrics = {
        "mode": "delta-update",
        "run_utc": run_utc,
        "feed": feed,
        "trading_day": last_day,
        "batches": 0,
        "symbols_total": len(unique_symbols),
        "symbols_with_rows": 0,
        "symbols_without_rows": 0,
        "http": {
            "requests": 0,
            "rows": 0,
            "rate_limit_hits": 0,
            "retries": 0,
            "miss_symbols": 0,
        },
        "assets": {
            "total": int(asset_metrics.get("assets_total", 0)),
            "tradable_equity": int(asset_metrics.get("assets_tradable_equities", 0)),
            "after_filters": int(asset_metrics.get("assets_after_filters", 0)),
        },
    }

    miss_symbols: set[str] = set()

    for batch_idx, chunk in enumerate(_chunked(unique_symbols, batch_size)):
        batch_syms = [s.upper() for s in chunk if s]
        if not batch_syms:
            continue
        metrics["batches"] += 1
        bars_raw, http_stats = fetch_bars_http(
            batch_syms,
            start_iso,
            end_iso,
            feed=feed,
            batch=len(batch_syms),
        )

        df = to_bars_df(bars_raw)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype("string").str.upper()
            df = df[df["symbol"].isin(batch_syms)]
        df = df.reset_index(drop=True)

        batch_path = output_dir / f"{batch_idx:04d}.parquet"
        if df.empty:
            empty = pd.DataFrame(columns=BARS_COLUMNS)
            written = _write_parquet_or_csv(empty, batch_path)
            LOGGER.info("[DELTA] wrote shard: %s (empty batch)", written)
        else:
            _write_parquet_or_csv(df, batch_path)

        present = set(df["symbol"].unique()) if not df.empty else set()
        missing = set(batch_syms) - present

        LOGGER.info(
            "[DELTA] batch=%04d symbols=%d rows=%d missing=%d",
            batch_idx,
            len(batch_syms),
            int(df.shape[0]),
            len(missing),
        )

        metrics["symbols_with_rows"] += len(present)
        metrics["symbols_without_rows"] += len(missing)

        metrics["http"]["requests"] += int(http_stats.get("requests", 0))
        metrics["http"]["rows"] += int(http_stats.get("rows", df.shape[0]))
        metrics["http"]["rate_limit_hits"] += int(http_stats.get("rate_limit_hits", 0))
        metrics["http"]["retries"] += int(http_stats.get("retries", 0))
        miss_symbols.update(http_stats.get("miss_list", []))
        metrics["http"]["miss_symbols"] += int(http_stats.get("miss_symbols", len(missing)))

        for symbol in batch_syms:
            record = coverage.setdefault(
                symbol,
                {"last_ok_utc": None, "last_miss_utc": None, "ok_count": 0, "miss_count": 0},
            )
            if symbol in present:
                record["last_ok_utc"] = run_utc
                record["ok_count"] = int(record.get("ok_count", 0)) + 1
            else:
                record["last_miss_utc"] = run_utc
                record["miss_count"] = int(record.get("miss_count", 0)) + 1

    metrics["coverage"] = {
        "ok": sum(1 for entry in coverage.values() if int(entry.get("ok_count", 0)) > 0),
        "miss": sum(1 for entry in coverage.values() if int(entry.get("miss_count", 0)) > 0),
    }
    metrics["http"]["miss_list"] = sorted(miss_symbols)

    _write_coverage_table(coverage_path, coverage)

    metrics_dir = base_dir / "data" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"delta_update_{feed}.json"
    _write_json_atomic(metrics_path, metrics)

    LOGGER.info(
        "[DELTA] Completed batches=%d http.requests=%d http.rows=%d",
        metrics["batches"],
        metrics["http"]["requests"],
        metrics["http"]["rows"],
    )

    return 0


def run_build_symbol_stats(args: argparse.Namespace, base_dir: Path) -> int:
    LOGGER.info("[MODE] build-symbol-stats start")
    start_time = time.time()
    feed = getattr(args, "feed", DEFAULT_FEED)
    days = max(int(getattr(args, "prefilter_days", 120) or 120), 20)
    reuse_cache = _as_bool(getattr(args, "reuse_cache", True), True)
    limit = getattr(args, "limit", None)

    bars_df = _load_local_bars(base_dir, feed, days, reuse_cache=reuse_cache, limit=limit)
    stats_df = _compute_symbol_stats_frame(bars_df)

    registry_dir = base_dir / "data" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    stats_path = registry_dir / "symbol_stats.csv"
    _write_csv_atomic(stats_path, stats_df)

    elapsed = round(time.time() - start_time, 3)
    now = datetime.now(timezone.utc)
    symbols_total = int(stats_df["symbol"].nunique()) if not stats_df.empty else 0
    prefix_counts = (
        stats_df["symbol"].astype("string").str.upper().str[0].value_counts().to_dict()
        if not stats_df.empty
        else {}
    )
    metrics_payload = {
        "mode": "build-symbol-stats",
        "last_run_utc": _format_timestamp(now),
        "symbol_stats": {"count": int(stats_df.shape[0])},
        "symbols_in": symbols_total,
        "symbols_with_bars": int(bars_df["symbol"].nunique()) if not bars_df.empty else 0,
        "bars_rows_total": int(bars_df.shape[0]),
        "rows": 0,
        "universe_prefix_counts": prefix_counts,
        "gate_fail_counts": {},
        "http": {
            "requests": 0,
            "rows": 0,
            "rate_limit_hits": 0,
            "retries": 0,
        },
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "timings": _normalize_timings({"feature_secs": elapsed}),
    }
    _update_metrics_file(base_dir, metrics_payload)

    LOGGER.info(
        "Symbol stats written to %s (rows=%d, elapsed=%.3fs)",
        stats_path,
        int(stats_df.shape[0]),
        elapsed,
    )
    if stats_df.empty:
        LOGGER.warning("Symbol stats output is empty; verify delta-update inputs are available.")
    return 0


def _prepare_coarse_rank_export(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "Score_coarse", "coarse_rank"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    df = frame.copy()
    df["symbol"] = df["symbol"].astype("string").str.upper()
    if "Score_coarse" not in df.columns and "Score" in df.columns:
        df.rename(columns={"Score": "Score_coarse"}, inplace=True)
    if "coarse_rank" not in df.columns:
        df["coarse_rank"] = np.arange(1, df.shape[0] + 1, dtype=int)
    z_cols = sorted(col for col in df.columns if col.endswith("_z"))
    payload_cols = [*columns, *z_cols]
    payload_cols = [col for col in payload_cols if col in df.columns]
    ordered = df.reindex(columns=payload_cols, fill_value=pd.NA)
    ordered.sort_values("coarse_rank", inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def run_coarse_features(args: argparse.Namespace, base_dir: Path) -> int:
    LOGGER.info("[MODE] coarse-features start")
    stats_path = base_dir / "data" / "registry" / "symbol_stats.csv"
    if not stats_path.exists():
        LOGGER.error("Missing symbol stats at %s; run build-symbol-stats first.", stats_path)
        return 1
    try:
        stats_df = pd.read_csv(stats_path)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", stats_path, exc)
        return 1

    stats_df["symbol"] = stats_df.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()
    stats_df["ADV20"] = pd.to_numeric(stats_df.get("ADV20"), errors="coerce")
    threshold = float(getattr(args, "dollar_vol_min", 0) or 0)
    eligible = stats_df[stats_df["ADV20"] >= threshold] if threshold else stats_df
    liquidity_fail = int(stats_df.shape[0] - eligible.shape[0])

    limit = getattr(args, "limit", None)
    if limit:
        try:
            limit_val = max(int(limit), 0)
        except (TypeError, ValueError):
            limit_val = 0
        if limit_val:
            eligible = eligible.head(limit_val)

    symbols = (
        eligible["symbol"].dropna().astype("string").str.upper().unique().tolist()
        if not eligible.empty
        else []
    )

    if not symbols:
        LOGGER.warning("No eligible symbols after ADV20 filter; coarse features will be empty.")

    days = max(int(getattr(args, "prefilter_days", 120) or 120), 1)
    reuse_cache = _as_bool(getattr(args, "reuse_cache", True), True)
    feed = getattr(args, "feed", DEFAULT_FEED)
    bars_df = _load_local_bars(
        base_dir,
        feed,
        days,
        symbols=symbols if symbols else None,
        reuse_cache=reuse_cache,
        limit=limit,
    )

    ranker_cfg = _prepare_ranker_config(
        _load_ranker_config(),
        preset_key=str(getattr(args, "gate_preset", "standard") or "standard"),
        relax_mode=str(getattr(args, "relax_gates", "none") or "none"),
        min_history=int(getattr(args, "min_history", DEFAULT_MIN_HISTORY) or DEFAULT_MIN_HISTORY),
    )
    dollar_override = getattr(args, "dollar_vol_min", None)
    if dollar_override is not None:
        try:
            gates = dict(ranker_cfg.get("gates") or {})
            gates["dollar_vol_min"] = float(dollar_override)
            ranker_cfg["gates"] = gates
        except (TypeError, ValueError):
            LOGGER.warning("Invalid dollar volume override: %s", dollar_override)

    timing_info: dict[str, float] = {}
    coarse_enriched = build_enriched_bars(
        bars_df,
        ranker_cfg,
        timings=timing_info,
        feature_columns=None,
        include_intermediate=False,
    )

    run_date = _resolve_latest_session(bars_df if not bars_df.empty else coarse_enriched)
    features_dir = base_dir / "data" / "features" / "1d"
    features_dir.mkdir(parents=True, exist_ok=True)
    features_path = features_dir / f"coarse_{run_date}.parquet"
    coarse_enriched.to_parquet(features_path, index=False)

    rank_timer = T()
    coarse_scored = score_universe(coarse_enriched, ranker_cfg)
    coarse_rank_elapsed = rank_timer.lap("coarse_rank_secs")
    if not coarse_scored.empty and "coarse_rank" not in coarse_scored.columns:
        coarse_scored["coarse_rank"] = np.arange(1, coarse_scored.shape[0] + 1, dtype=int)
    coarse_scored.rename(columns={"Score": "Score_coarse"}, inplace=True)

    coarse_output = _prepare_coarse_rank_export(coarse_scored)
    coarse_path = base_dir / "data" / "tmp" / "coarse_rank.csv"
    coarse_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(coarse_path, coarse_output)

    symbols_total = int(eligible["symbol"].nunique()) if not eligible.empty else 0
    prefix_counts = (
        eligible["symbol"].astype("string").str.upper().str[0].value_counts().to_dict()
        if not eligible.empty
        else {}
    )
    timing_base = _normalize_timings(
        {
            "feature_secs": timing_info.get("feature_secs", 0.0),
            "rank_secs": coarse_rank_elapsed,
        }
    )
    timing_base["coarse_rank_secs"] = round(float(coarse_rank_elapsed), 3)
    metrics_payload = {
        "mode": "coarse-features",
        "last_run_utc": _format_timestamp(datetime.now(timezone.utc)),
        "symbols_in": symbols_total,
        "symbols_with_bars": int(bars_df["symbol"].nunique()) if not bars_df.empty else 0,
        "bars_rows_total": int(bars_df.shape[0]),
        "rows": int(coarse_output.shape[0]) if coarse_output is not None else 0,
        "gate_fail_counts": {"liquidity": max(liquidity_fail, 0)},
        "coarse_ranked": int(coarse_scored.shape[0]) if coarse_scored is not None else 0,
        "universe_prefix_counts": prefix_counts,
        "http": {
            "requests": 0,
            "rows": 0,
            "rate_limit_hits": 0,
            "retries": 0,
        },
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "timings": timing_base,
        "shortlist_path": str(coarse_path),
    }
    _update_metrics_file(base_dir, metrics_payload)

    LOGGER.info(
        "Coarse features complete: eligible=%d symbols=%d rows=%d features=%s",
        int(eligible.shape[0]),
        len(symbols),
        int(bars_df.shape[0]),
        features_path,
    )
    if bars_df.empty:
        LOGGER.warning("No bars available for coarse features; downstream stages may be empty.")
    return 0


def run_full_nightly(args: argparse.Namespace, base_dir: Path) -> int:
    LOGGER.info("[MODE] full-nightly start")
    coarse_path = base_dir / "data" / "tmp" / "coarse_rank.csv"
    if not coarse_path.exists():
        LOGGER.error("Missing coarse rank file at %s; run coarse-features first.", coarse_path)
        return 1
    try:
        coarse_df = pd.read_csv(coarse_path)
    except Exception as exc:
        LOGGER.error("Failed to read coarse rank file %s: %s", coarse_path, exc)
        return 1

    coarse_df["symbol"] = coarse_df.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()
    if "coarse_rank" not in coarse_df.columns:
        coarse_df = coarse_df.sort_values("Score_coarse", ascending=False).reset_index(drop=True)
        coarse_df["coarse_rank"] = np.arange(1, coarse_df.shape[0] + 1, dtype=int)
    shortlist_size = max(int(getattr(args, "prefilter_top", DEFAULT_SHORTLIST_SIZE) or DEFAULT_SHORTLIST_SIZE), 0)
    shortlist_df = (
        coarse_df.sort_values(["coarse_rank", "Score_coarse"], ascending=[True, False])
        .head(shortlist_size)
        .copy()
    )

    limit = getattr(args, "limit", None)
    if limit:
        try:
            limit_val = max(int(limit), 0)
        except (TypeError, ValueError):
            limit_val = 0
        if limit_val:
            shortlist_df = shortlist_df.head(limit_val)

    symbols = shortlist_df["symbol"].dropna().astype("string").str.upper().tolist()
    if not symbols:
        LOGGER.warning("Shortlist is empty; final ranking will have no candidates.")

    days = max(int(getattr(args, "full_days", 750) or 750), 1)
    try:
        start_iso, end_iso, last_day = _fallback_daily_window(days)
    except Exception as exc:  # pragma: no cover - fallback already guards
        LOGGER.warning("Failed to determine fetch window: %s", exc)
        start_iso, end_iso, last_day = _fallback_daily_window(days)

    reuse_cache = _as_bool(getattr(args, "reuse_cache", True), True)
    feed = getattr(args, "feed", DEFAULT_FEED)
    fetch_mode = getattr(args, "fetch_mode", "auto")
    batch_size = int(getattr(args, "batch_size", 50) or 50)
    max_workers = int(getattr(args, "max_workers", 4) or 4)
    min_history = int(getattr(args, "min_history", DEFAULT_MIN_HISTORY) or DEFAULT_MIN_HISTORY)
    bars_source = getattr(args, "bars_source", "http")
    verify_hook = make_verify_hook(bool(getattr(args, "verify_request", False)))

    data_client = None
    if bars_source == "sdk":
        try:
            data_client = _create_data_client()
        except Exception as exc:
            LOGGER.error("Could not create Alpaca data client: %s", exc)
            return 1

    fetch_timer = T()
    bars_df, fetch_metrics, prefiltered = _fetch_daily_bars(
        data_client,
        symbols,
        days=days,
        start_iso=start_iso,
        end_iso=end_iso,
        feed=feed,
        fetch_mode=fetch_mode,
        batch_size=batch_size,
        max_workers=max_workers,
        min_history=min_history,
        bars_source=bars_source,
        run_date=last_day,
        reuse_cache=reuse_cache,
        verify_hook=verify_hook,
    )
    fetch_elapsed = fetch_timer.lap("fetch_secs")

    if not bars_df.empty and symbols:
        bars_df = bars_df[bars_df["symbol"].astype("string").str.upper().isin(set(symbols))]

    fetch_metrics = fetch_metrics or {}
    fetch_metrics["symbols_in"] = len(symbols)
    _write_universe_prefix_metrics(pd.DataFrame({"symbol": shortlist_df["symbol"]}), fetch_metrics)

    now = datetime.now(timezone.utc)
    top_df, scored_df, stats, skip_reasons, reject_samples, gate_counters, ranker_cfg, timing_info = run_screener(
        bars_df,
        top_n=int(getattr(args, "top_n", DEFAULT_TOP_N) or DEFAULT_TOP_N),
        min_history=min_history,
        now=now,
        asset_meta={},
        prefiltered_skips=prefiltered,
        gate_preset=str(getattr(args, "gate_preset", "standard") or "standard"),
        relax_gates=str(getattr(args, "relax_gates", "none") or "none"),
        dollar_vol_min=getattr(args, "dollar_vol_min", None),
        shortlist_size=shortlist_size,
        shortlist_path=coarse_path,
    )
    timing_info["fetch_secs"] = timing_info.get("fetch_secs", 0.0) + round(float(fetch_elapsed), 3)

    output_path = write_outputs(
        base_dir,
        top_df,
        scored_df,
        stats,
        skip_reasons,
        reject_samples,
        status="ok",
        now=now,
        gate_counters=gate_counters,
        fetch_metrics=fetch_metrics,
        asset_metrics={},
        ranker_cfg=ranker_cfg,
        timings=timing_info,
    )
    LOGGER.info("Full nightly outputs written to %s", output_path)

    refresh_latest = _as_bool(getattr(args, "refresh_latest", True), True)
    if refresh_latest:
        try:
            try:
                from .run_pipeline import refresh_latest_candidates  # type: ignore
            except Exception:  # pragma: no cover - fallback for script execution
                from scripts.run_pipeline import refresh_latest_candidates  # type: ignore
            refresh_latest_candidates()
        except Exception as exc:  # pragma: no cover - copy failures unexpected
            LOGGER.warning("Failed to refresh latest candidates: %s", exc)

    _update_metrics_file(
        base_dir,
        {
            "mode": "full-nightly",
            "last_run_utc": _format_timestamp(datetime.now(timezone.utc)),
        },
    )

    return 0


def write_predictions(
    ranked_df: Optional[pd.DataFrame],
    run_meta: Optional[Mapping[str, object]],
    top_n: int = 200,
) -> None:
    run_date = datetime.now(tz=timezone.utc).date().isoformat()
    pred_dir = Path("data") / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    keep = [
        "symbol",
        "Score",
        "rank",
        "timestamp",
        "close",
        "ATR14",
        "ADV20",
        "TS",
        "MS",
        "BP",
        "PT",
        "RSI",
        "MH",
        "ADX",
        "AROON",
        "VCP",
        "VOLexp",
        "GAPpen",
        "LIQpen",
        "score_breakdown",
    ]
    cols = ["run_date"] + keep + ["ranker_version", "gate_preset", "relax_gates"]

    if isinstance(ranked_df, pd.DataFrame):
        out = ranked_df.copy()
    else:
        out = pd.DataFrame(columns=keep)

    if out.empty:
        out = pd.DataFrame(columns=keep)
    else:
        out = out.reset_index(drop=True)

    out["rank"] = range(1, len(out) + 1)
    for column in keep:
        if column not in out.columns:
            out[column] = pd.NA

    meta = run_meta or {}
    out["run_date"] = run_date
    out["ranker_version"] = str(meta.get("ranker_version", "1.0.0"))
    out["gate_preset"] = str(meta.get("gate_preset", "standard"))
    out["relax_gates"] = str(meta.get("relax_gates", "none"))

    out = out[[col for col in cols if col in out.columns]]

    head = out.head(top_n)
    daily_path = pred_dir / f"{run_date}.csv"
    head.to_csv(daily_path, index=False)
    head.to_csv(pred_dir / "latest.csv", index=False)
    LOGGER.info("[STAGE] predictions written: %s (top_n=%d)", daily_path, top_n)


def _prepare_predictions_frame(
    scored_df: pd.DataFrame,
    *,
    run_date: datetime,
    gate_counters: Optional[Mapping[str, object]] = None,
    ranker_cfg: Optional[Mapping[str, object]] = None,
    limit: int = 200,
) -> pd.DataFrame:
    columns = [
        "run_date",
        "symbol",
        "rank",
        "score",
        "passed_gates",
        "ranker_version",
        "gate_preset",
        "adv20",
        "price_close",
        "atr14",
        "ts",
        "ms",
        "bp",
        "pt",
        "rsi",
        "mh",
        "adx",
        "aroon",
        "vcp",
        "volexp",
        "gap_pen",
        "liq_pen",
        "score_breakdown_json",
    ]
    if scored_df is None or scored_df.empty:
        return pd.DataFrame(columns=columns)

    frame = scored_df.copy()
    if "rank" not in frame.columns:
        frame["rank"] = np.arange(1, frame.shape[0] + 1, dtype=int)
    if "gates_passed" not in frame.columns:
        frame["gates_passed"] = False
    frame["gates_passed"] = frame["gates_passed"].fillna(False).astype(bool)

    limit = max(1, int(limit)) if limit else frame.shape[0]
    frame = frame.head(limit)

    run_date_str = run_date.date().isoformat()
    gate_preset = "custom"
    if gate_counters and isinstance(gate_counters, Mapping):
        gate_preset = str(gate_counters.get("gate_preset") or gate_preset)
    ranker_version = None
    if ranker_cfg and isinstance(ranker_cfg, Mapping):
        ranker_version = ranker_cfg.get("version")

    def _coerce_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    score_breakdowns = frame.get("score_breakdown", pd.Series(dtype="string")).fillna("").astype(str)

    payload = pd.DataFrame({
        "run_date": run_date_str,
        "symbol": frame.get("symbol", pd.Series(dtype="string")).astype(str).str.upper(),
        "rank": pd.to_numeric(frame.get("rank"), errors="coerce").astype("Int64"),
        "score": _coerce_float(frame.get("Score")),
        "passed_gates": frame.get("gates_passed", pd.Series(dtype=bool)).astype(bool),
        "ranker_version": str(ranker_version or ""),
        "gate_preset": gate_preset,
        "adv20": _coerce_float(frame.get("ADV20")),
        "price_close": _coerce_float(frame.get("close")),
        "atr14": _coerce_float(frame.get("ATR14")),
        "ts": _coerce_float(frame.get("TS")),
        "ms": _coerce_float(frame.get("MS")),
        "bp": _coerce_float(frame.get("BP")),
        "pt": _coerce_float(frame.get("PT")),
        "rsi": _coerce_float(frame.get("RSI")),
        "mh": _coerce_float(frame.get("MH")),
        "adx": _coerce_float(frame.get("ADX")),
        "aroon": _coerce_float(frame.get("AROON")),
        "vcp": _coerce_float(frame.get("VCP")),
        "volexp": _coerce_float(frame.get("VOLexp")),
        "gap_pen": _coerce_float(frame.get("GAPpen")),
        "liq_pen": _coerce_float(frame.get("LIQpen")),
        "score_breakdown_json": score_breakdowns,
    })

    payload["score_breakdown_json"] = payload["score_breakdown_json"].apply(
        lambda s: s[:1024] if isinstance(s, str) else ""
    )
    payload = payload.reindex(columns=columns)
    payload["rank"] = payload["rank"].astype("Int64")
    return payload


def _coerce_int(value: object) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def run_screener(
    df: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    min_history: int = DEFAULT_MIN_HISTORY,
    now: Optional[datetime] = None,
    asset_meta: Optional[Dict[str, dict]] = None,
    prefiltered_skips: Optional[Dict[str, str]] = None,
    gate_preset: str = "standard",
    relax_gates: str = "none",
    dollar_vol_min: Optional[float] = None,
    ranker_config: Optional[Mapping[str, object]] = None,
    shortlist_size: Optional[int] = None,
    shortlist_path: Optional[Union[str, Path]] = None,
    backtest_top_k: Optional[int] = None,
    backtest_lookback: Optional[int] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[str, int],
    list[dict[str, str]],
    dict[str, Union[int, str]],
    dict[str, object],
    dict[str, float],
]:
    """Run the screener pipeline returning scored outputs and diagnostics."""

    now = now or datetime.now(timezone.utc)
    prepared = _prepare_input_frame(df)
    stats: dict[str, int] = {"symbols_in": 0, "candidates_out": 0}
    skip_reasons = {key: 0 for key in SKIP_KEYS}

    try:
        shortlist_limit = (
            DEFAULT_SHORTLIST_SIZE
            if shortlist_size is None
            else max(int(shortlist_size), 0)
        )
    except (TypeError, ValueError):
        shortlist_limit = DEFAULT_SHORTLIST_SIZE

    try:
        backtest_limit = (
            DEFAULT_BACKTEST_TOP_K
            if backtest_top_k is None
            else max(int(backtest_top_k), 0)
        )
    except (TypeError, ValueError):
        backtest_limit = DEFAULT_BACKTEST_TOP_K

    try:
        lookback_window = (
            DEFAULT_BACKTEST_LOOKBACK
            if backtest_lookback is None
            else max(int(backtest_lookback), 0)
        )
    except (TypeError, ValueError):
        lookback_window = DEFAULT_BACKTEST_LOOKBACK

    shortlist_target = (
        Path(shortlist_path)
        if shortlist_path is not None
        else Path("data") / "tmp" / "shortlist.csv"
    )

    stats["shortlist_requested"] = shortlist_limit
    stats["backtest_target"] = backtest_limit

    prefiltered_map = {
        str(sym or "").strip().upper(): str(reason or "").strip().upper()
        for sym, reason in (prefiltered_skips or {}).items()
    }

    if not prefiltered_map and not prepared.empty:
        allowed_exchanges = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "IEX"}
        auto_prefilter: dict[str, str] = {}
        for sym_raw, exch_raw in prepared[["symbol", "exchange"]].itertuples(index=False):
            sym = str(sym_raw or "").strip().upper()
            if not sym:
                continue
            exchange = str(exch_raw or "").strip().upper()
            reason: str | None
            if not exchange:
                reason = "UNKNOWN_EXCHANGE"
            elif exchange in allowed_exchanges:
                reason = None
            elif exchange.startswith("OTC"):
                reason = "NON_EQUITY"
            elif exchange in {"CRYPTO", "CRYPTO1", "CRYPTOX"}:
                reason = "NON_EQUITY"
            elif exchange in {"PINK", "GREY", "OTCQB", "OTCQX"}:
                reason = "NON_EQUITY"
            else:
                reason = "UNKNOWN_EXCHANGE"
            if reason:
                auto_prefilter.setdefault(sym, reason)
        if auto_prefilter:
            prefiltered_map.update(auto_prefilter)

    initial_rejects: list[dict[str, str]] = []
    for sym, reason in prefiltered_map.items():
        if reason in skip_reasons:
            skip_reasons[reason] += 1
        if len(initial_rejects) < 10:
            initial_rejects.append({"symbol": sym, "reason": reason or "PREFILTERED"})

    if not prepared.empty:
        prepared_symbols = prepared["symbol"].astype("string").str.strip().str.upper()
    else:
        prepared_symbols = pd.Series(dtype="string")

    input_symbols = {sym for sym in prepared_symbols.tolist() if sym}
    input_symbols.update(prefiltered_map.keys())
    stats["symbols_in"] = len(input_symbols)

    if not prepared.empty:
        prepared = prepared.copy()
        prepared["symbol"] = prepared_symbols
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared = prepared.dropna(subset=["timestamp"])
        prepared = prepared[prepared["symbol"] != ""]
        if prefiltered_map:
            prepared = prepared[~prepared["symbol"].isin(prefiltered_map.keys())]
    else:
        prepared = pd.DataFrame(columns=INPUT_COLUMNS)

    base_cfg = ranker_config or _load_ranker_config()
    ranker_cfg = _prepare_ranker_config(
        base_cfg,
        preset_key=str(gate_preset or "standard"),
        relax_mode=str(relax_gates or "none"),
        min_history=min_history,
    )
    if dollar_vol_min is not None:
        try:
            gates = dict(ranker_cfg.get("gates") or {})
            gates["dollar_vol_min"] = float(dollar_vol_min)
            ranker_cfg["gates"] = gates
        except (TypeError, ValueError):
            LOGGER.warning("Invalid dollar_vol_min override: %s", dollar_vol_min)

    timing_info: dict[str, float] = {}

    LOGGER.info("[STAGE] coarse features start")
    coarse_enriched = build_enriched_bars(
        prepared,
        ranker_cfg,
        timings=timing_info,
        feature_columns=COARSE_RANK_COLUMNS,
        include_intermediate=False,
    )
    LOGGER.info(
        "[STAGE] coarse features end (rows=%d)", int(coarse_enriched.shape[0])
    )

    coarse_rank_timer = T()
    LOGGER.info("[STAGE] coarse rank start")
    coarse_scored = score_universe(coarse_enriched, ranker_cfg)
    LOGGER.info(
        "[STAGE] coarse rank end (rows=%d)", int(coarse_scored.shape[0])
    )
    timing_info["coarse_rank_secs"] = timing_info.get(
        "coarse_rank_secs", 0.0
    ) + coarse_rank_timer.lap("coarse_rank_secs")

    if not coarse_scored.empty:
        coarse_scored = coarse_scored.copy()
        coarse_scored["coarse_rank"] = np.arange(
            1, coarse_scored.shape[0] + 1, dtype=int
        )
    else:
        coarse_scored["coarse_rank"] = pd.Series(dtype="int64")

    stats["coarse_ranked"] = int(coarse_scored.shape[0])

    shortlist_view = (
        coarse_scored.head(shortlist_limit).loc[:, ["symbol", "Score", "coarse_rank"]]
        if not coarse_scored.empty
        else pd.DataFrame(columns=["symbol", "Score", "coarse_rank"])
    )
    shortlist_payload = shortlist_view.rename(columns={"Score": "coarse_score"})
    shortlist_payload["symbol"] = (
        shortlist_payload.get("symbol", pd.Series(dtype="object"))
        .astype(str)
        .str.upper()
    )
    shortlist_payload = shortlist_payload.drop_duplicates(subset=["symbol"])
    stats["shortlist_candidates"] = int(shortlist_payload.shape[0])

    try:
        shortlist_written = _write_shortlist_csv(shortlist_target, shortlist_payload)
        LOGGER.info(
            "[STAGE] shortlist written: %s (rows=%d)",
            shortlist_written,
            stats["shortlist_candidates"],
        )
    except Exception as exc:  # pragma: no cover - disk issues unexpected
        LOGGER.warning("Failed to write shortlist %s: %s", shortlist_target, exc)
        shortlist_written = shortlist_target

    stats["shortlist_path"] = str(shortlist_written)

    shortlist_symbols = (
        shortlist_payload["symbol"].astype(str).str.upper().tolist()
        if not shortlist_payload.empty
        else []
    )

    shortlist_prepared = (
        prepared[prepared["symbol"].isin(shortlist_symbols)].copy()
        if shortlist_symbols
        else prepared.iloc[0:0].copy()
    )

    LOGGER.info(
        "[STAGE] full features start (shortlist=%d)", len(shortlist_symbols)
    )
    enriched = build_enriched_bars(
        shortlist_prepared,
        ranker_cfg,
        timings=timing_info,
    )
    LOGGER.info("[STAGE] full features end (rows=%d)", int(enriched.shape[0]))

    rank_timer = T()
    LOGGER.info("[STAGE] full rank start")
    scored_df = score_universe(enriched, ranker_cfg)
    LOGGER.info("[STAGE] full rank end (rows=%d)", int(scored_df.shape[0]))
    timing_info["rank_secs"] = timing_info.get("rank_secs", 0.0) + rank_timer.lap(
        "rank_secs"
    )

    if not scored_df.empty:
        scored_df = scored_df.copy()
        scored_df["rank"] = np.arange(1, scored_df.shape[0] + 1, dtype=int)
    else:
        scored_df["rank"] = pd.Series(dtype="int64")

    scored_df["gates_passed"] = False

    if "symbol" in scored_df.columns and not shortlist_payload.empty:
        scored_df = scored_df.merge(shortlist_payload, on="symbol", how="left")
    else:
        if "coarse_score" not in scored_df.columns:
            scored_df["coarse_score"] = pd.Series(pd.NA, index=scored_df.index, dtype="Float64")
        if "coarse_rank" not in scored_df.columns:
            scored_df["coarse_rank"] = pd.Series(pd.NA, index=scored_df.index, dtype="Int64")

    if "coarse_score" in scored_df.columns:
        scored_df["coarse_score"] = pd.to_numeric(
            scored_df["coarse_score"], errors="coerce"
        )
    if "coarse_rank" in scored_df.columns:
        scored_df["coarse_rank"] = pd.to_numeric(
            scored_df["coarse_rank"], errors="coerce"
        ).astype("Int64")

    gates_timer = T()
    LOGGER.info("[STAGE] gates start")
    candidates_df, gate_fail_counts, gate_rejects = apply_gates(scored_df, ranker_cfg)
    LOGGER.info("[STAGE] gates end (candidates=%d)", int(candidates_df.shape[0]))
    timing_info["gates_secs"] = timing_info.get("gates_secs", 0.0) + gates_timer.lap(
        "gates_secs"
    )

    if not candidates_df.empty:
        candidates_df["gates_passed"] = True
        if "coarse_score" in candidates_df.columns:
            candidates_df["coarse_score"] = pd.to_numeric(
                candidates_df["coarse_score"], errors="coerce"
            )
        if "coarse_rank" in candidates_df.columns:
            candidates_df["coarse_rank"] = pd.to_numeric(
                candidates_df["coarse_rank"], errors="coerce"
            ).astype("Int64")

    if not candidates_df.empty:
        def _gate_key(frame: pd.DataFrame) -> pd.Index:
            symbols = frame.get("symbol", pd.Series(index=frame.index, dtype="object"))
            symbols = symbols.astype(str).str.upper()
            if "timestamp" in frame.columns:
                timestamps = pd.to_datetime(
                    frame["timestamp"], utc=True, errors="coerce"
                )
                timestamp_str = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                timestamp_str = timestamp_str.fillna("")
            else:
                timestamp_str = pd.Series("", index=frame.index, dtype="object")
            return symbols.str.cat(timestamp_str, sep="|")

        passed_index = set(_gate_key(candidates_df))
        scored_df_keys = _gate_key(scored_df)
        scored_df["gates_passed"] = scored_df_keys.isin(passed_index)

    stats["candidates_out"] = int(candidates_df.shape[0])

    backtest_timer = T()
    backtest_rows: list[dict[str, object]] = []
    if backtest_limit and not scored_df.empty and not shortlist_prepared.empty:
        ranked_symbols = (
            scored_df.sort_values("Score", ascending=False)["symbol"]
            .astype(str)
            .str.upper()
            .tolist()
        )
        unique_symbols: list[str] = []
        for sym in ranked_symbols:
            if sym not in unique_symbols:
                unique_symbols.append(sym)
            if len(unique_symbols) >= backtest_limit:
                break

        if unique_symbols:
            bars_sorted = shortlist_prepared.sort_values("timestamp")
            for sym in unique_symbols:
                sym_mask = bars_sorted["symbol"].astype(str).str.upper() == sym
                sym_bars = bars_sorted.loc[sym_mask]
                if sym_bars.empty:
                    continue
                metrics = compute_recent_performance(sym_bars, lookback=lookback_window)
                backtest_rows.append(
                    {
                        "symbol": sym,
                        "backtest_expectancy": float(metrics.get("expectancy", 0.0) or 0.0),
                        "backtest_win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
                        "backtest_samples": int(metrics.get("samples", 0) or 0),
                    }
                )

    backtest_elapsed = backtest_timer.lap("backtest_secs")
    if backtest_elapsed:
        timing_info["backtest_secs"] = timing_info.get("backtest_secs", 0.0) + backtest_elapsed

    backtest_df = pd.DataFrame(
        backtest_rows,
        columns=[
            "symbol",
            "backtest_expectancy",
            "backtest_win_rate",
            "backtest_samples",
        ],
    )
    stats["backtest_evaluated"] = int(backtest_df.shape[0])
    stats["backtest_lookback"] = int(lookback_window)
    if not backtest_df.empty:
        stats["backtest_expectancy_mean"] = float(
            pd.to_numeric(backtest_df["backtest_expectancy"], errors="coerce")
            .dropna()
            .mean()
        )
        stats["backtest_win_rate_mean"] = float(
            pd.to_numeric(backtest_df["backtest_win_rate"], errors="coerce")
            .dropna()
            .mean()
        )
    else:
        stats["backtest_expectancy_mean"] = 0.0
        stats["backtest_win_rate_mean"] = 0.0

    merge_cols = [
        "symbol",
        "backtest_expectancy",
        "backtest_win_rate",
        "backtest_samples",
    ]
    if not backtest_df.empty:
        scored_df = scored_df.merge(backtest_df, on="symbol", how="left")
        if not candidates_df.empty:
            candidates_df = candidates_df.merge(backtest_df, on="symbol", how="left")
    else:
        for frame in (scored_df, candidates_df):
            if frame is None or frame.empty:
                continue
            for col in merge_cols[1:]:
                if col not in frame.columns:
                    frame[col] = pd.NA

    def _apply_backtest_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        expectancy_raw = pd.to_numeric(
            frame.get("backtest_expectancy", pd.Series(dtype="float64")),
            errors="coerce",
        )
        win_rate_raw = pd.to_numeric(
            frame.get("backtest_win_rate", pd.Series(dtype="float64")),
            errors="coerce",
        )
        samples_raw = pd.to_numeric(
            frame.get("backtest_samples", pd.Series(dtype="float64")),
            errors="coerce",
        )

        samples_filled = samples_raw.fillna(0)
        valid_mask = samples_filled > 0

        expectancy_display = expectancy_raw.where(valid_mask, np.nan)
        win_rate_display = win_rate_raw.where(valid_mask, np.nan)
        samples_display = samples_filled.astype("Int64")

        exp_comp = expectancy_display.fillna(0.0)
        win_comp = win_rate_display.fillna(0.5)
        adjustment = exp_comp * BACKTEST_EXPECTANCY_WEIGHT
        adjustment += (win_comp - 0.5) * BACKTEST_WIN_RATE_WEIGHT

        frame["backtest_expectancy"] = expectancy_display
        frame["backtest_win_rate"] = win_rate_display
        frame["backtest_samples"] = samples_display
        frame["backtest_adjustment"] = adjustment
        score_series = pd.to_numeric(frame.get("Score"), errors="coerce").fillna(0.0)
        frame["Score"] = score_series + adjustment
        return frame

    scored_df = _apply_backtest_adjustment(scored_df)
    candidates_df = _apply_backtest_adjustment(candidates_df)

    def _augment_breakdown(frame: pd.DataFrame) -> None:
        if frame is None or frame.empty or "score_breakdown" not in frame.columns:
            return

        def _update(row: pd.Series) -> str:
            raw = row.get("score_breakdown", "")
            try:
                payload = json.loads(raw) if isinstance(raw, str) and raw else {}
            except Exception:
                payload = {}
            samples_value = row.get("backtest_samples")
            expectancy_value = row.get("backtest_expectancy")
            win_rate_value = row.get("backtest_win_rate")
            if pd.isna(samples_value) or float(samples_value) <= 0:
                return json.dumps({k: payload[k] for k in sorted(payload)})
            if pd.notna(expectancy_value):
                payload["BTexp"] = round(float(expectancy_value), 4)
            if pd.notna(win_rate_value):
                payload["BTwin"] = round(float(win_rate_value), 4)
            return json.dumps({k: payload[k] for k in sorted(payload)})

        frame["score_breakdown"] = frame.apply(_update, axis=1)

    _augment_breakdown(scored_df)
    _augment_breakdown(candidates_df)

    if not scored_df.empty:
        scored_df.sort_values("Score", ascending=False, inplace=True)
        scored_df.reset_index(drop=True, inplace=True)
        scored_df["rank"] = np.arange(1, scored_df.shape[0] + 1, dtype=int)

    if not candidates_df.empty:
        candidates_df.sort_values("Score", ascending=False, inplace=True)
        candidates_df.reset_index(drop=True, inplace=True)
        candidates_df["rank"] = np.arange(1, candidates_df.shape[0] + 1, dtype=int)

    skip_reasons["NAN_DATA"] += _coerce_int(gate_fail_counts.get("nan_data"))
    skip_reasons["INSUFFICIENT_HISTORY"] += _coerce_int(
        gate_fail_counts.get("insufficient_history")
    )

    combined_rejects = list(initial_rejects)
    for entry in gate_rejects:
        if len(combined_rejects) >= 10:
            break
        combined_rejects.append(entry)

    top_df = _prepare_top_frame(candidates_df, top_n)
    scored_count = int(scored_df.shape[0]) if isinstance(scored_df, pd.DataFrame) else 0
    top_df = _normalise_top_candidates(
        top_df,
        scored_df,
        universe_count=scored_count if scored_count else stats.get("symbols_in"),
    )

    if stats["candidates_out"] == 0:
        if combined_rejects:
            LOGGER.info(
                "No candidates passed ranking gates; sample rejections: %s",
                json.dumps(combined_rejects, sort_keys=True),
            )
        else:
            LOGGER.info("No candidates passed ranking gates.")

    return (
        top_df,
        scored_df,
        stats,
        skip_reasons,
        combined_rejects,
        gate_fail_counts,
        ranker_cfg,
        timing_info,
    )


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
    gate_counters: Optional[dict[str, Union[int, str]]] = None,
    fetch_metrics: Optional[dict[str, Any]] = None,
    asset_metrics: Optional[dict[str, Any]] = None,
    ranker_cfg: Optional[Mapping[str, object]] = None,
    timings: Optional[Mapping[str, float]] = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if top_df is None:
        top_df = pd.DataFrame(columns=TOP_CANDIDATE_COLUMNS)
    if scored_df is None:
        scored_df = pd.DataFrame()

    top_path = data_dir / "top_candidates.csv"
    scored_path = data_dir / "scored_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"

    _write_csv_atomic(top_path, top_df)
    _write_csv_atomic(scored_path, scored_df)

    cfg_for_summary = ranker_cfg or _load_ranker_config()
    gate_meta = gate_counters or {}
    gate_preset_meta: Optional[str] = None
    gate_relax_meta: Optional[str] = None
    if isinstance(gate_meta, Mapping):
        raw_preset = gate_meta.get("gate_preset") if gate_meta else None
        raw_relax = gate_meta.get("gate_relax_mode") if gate_meta else None
        gate_preset_meta = str(raw_preset) if raw_preset not in (None, "") else None
        gate_relax_meta = str(raw_relax) if raw_relax not in (None, "") else None
    if isinstance(ranker_cfg, Mapping):
        gate_preset_meta = gate_preset_meta or str(ranker_cfg.get("_gate_preset") or "") or None
        gate_relax_meta = gate_relax_meta or str(ranker_cfg.get("_gate_relax_mode") or "") or None
    run_meta = {
        "ranker_version": str(cfg_for_summary.get("version", "unknown")),
        "gate_preset": gate_preset_meta or "standard",
        "relax_gates": gate_relax_meta or "none",
    }
    write_predictions(scored_df, run_meta)

    diagnostics_dir = data_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    run_date = now.date().isoformat()
    diag_columns = [
        "symbol",
        "Score",
        "TS",
        "MS",
        "BP",
        "PT",
        "RSI",
        "MH",
        "ADX",
        "AROON",
        "VCP",
        "VOLexp",
        "GAPpen",
        "LIQpen",
    ]
    diag_frame = (
        scored_df.reindex(columns=diag_columns, fill_value=pd.NA).head(10)
        if not scored_df.empty
        else pd.DataFrame(columns=diag_columns)
    )
    diag_csv_path = diagnostics_dir / f"top10_{run_date}.csv"
    diag_json_path = diagnostics_dir / f"top10_{run_date}.json"
    _write_csv_atomic(diag_csv_path, diag_frame)
    diag_json = diag_frame.to_json(orient="records", indent=2)
    atomic_write_bytes(diag_json_path, diag_json.encode("utf-8"))

    metrics = {
        "last_run_utc": _format_timestamp(now),
        "status": status,
        "rows": int(top_df.shape[0]),
        "ranked_rows": int(scored_df.shape[0]),
        "symbols_in": int(stats.get("symbols_in", 0)),
        "candidates_out": int(stats.get("candidates_out", 0)),
        "shortlist_requested": int(stats.get("shortlist_requested", 0)),
        "shortlist_size": int(stats.get("shortlist_candidates", 0)),
        "coarse_ranked": int(stats.get("coarse_ranked", 0)),
        "shortlist_path": str(stats.get("shortlist_path", "")),
        "backtest_target": int(stats.get("backtest_target", 0)),
        "backtest_evaluated": int(stats.get("backtest_evaluated", 0)),
        "backtest_lookback": int(stats.get("backtest_lookback", 0)),
        "backtest_expectancy_mean": float(stats.get("backtest_expectancy_mean", 0.0)),
        "backtest_win_rate_mean": float(stats.get("backtest_win_rate_mean", 0.0)),
        "skips": {key: int(skip_reasons.get(key, 0)) for key in SKIP_KEYS},
    }
    metrics["gate_preset"] = run_meta["gate_preset"]
    metrics["relax_gates"] = run_meta["relax_gates"]
    gate_counts: dict[str, Union[int, str]] = {}
    for key, value in (gate_counters or {}).items():
        str_key = str(key)
        try:
            gate_counts[str_key] = int(value)
        except (TypeError, ValueError):
            gate_counts[str_key] = value
    for key, value in gate_counts.items():
        metrics[key] = value
    metrics["gate_fail_counts"] = gate_counts
    metrics["ranker_version"] = str(cfg_for_summary.get("version", "unknown"))
    metrics["feature_summary"] = _summarise_features(scored_df, cfg_for_summary)
    fetch_payload = fetch_metrics or {}
    metrics.update(
        {
            "bars_rows_total": _coerce_int(fetch_payload.get("bars_rows_total", 0)),
            "symbols_with_bars": _coerce_int(fetch_payload.get("symbols_with_bars", 0)),
            "symbols_no_bars": _coerce_int(fetch_payload.get("symbols_no_bars", 0)),
            "rate_limited": _coerce_int(fetch_payload.get("rate_limited", 0)),
            "http_404_batches": _coerce_int(fetch_payload.get("http_404_batches", 0)),
            "http_empty_batches": _coerce_int(fetch_payload.get("http_empty_batches", 0)),
            "window_used": _coerce_int(fetch_payload.get("window_used", 0)),
            "symbols_override": _coerce_int(fetch_payload.get("symbols_override", 0)),
            "raw_bars_count": _coerce_int(fetch_payload.get("raw_bars_count", 0)),
            "parsed_rows_count": _coerce_int(fetch_payload.get("parsed_rows_count", 0)),
            "cache_hits": _coerce_int(fetch_payload.get("cache_hits", 0)),
            "cache_misses": _coerce_int(fetch_payload.get("cache_misses", 0)),
            "batches_total": _coerce_int(fetch_payload.get("batches_total", 0)),
        }
    )
    if isinstance(scored_df, pd.DataFrame) and "symbol" in scored_df.columns:
        try:
            symbol_count = int(scored_df["symbol"].astype("string").dropna().nunique())
        except Exception:
            symbol_count = int(scored_df["symbol"].nunique())
    else:
        symbol_count = 0
    rows_total = int(scored_df.shape[0]) if isinstance(scored_df, pd.DataFrame) else 0
    metrics["symbols_with_bars"] = int(symbol_count)
    metrics["bars_rows_total"] = int(rows_total)
    metrics["symbols_no_bars"] = max(int(metrics.get("symbols_in", 0)) - int(symbol_count), 0)
    prefix_counts_payload = fetch_payload.get("universe_prefix_counts")
    if isinstance(prefix_counts_payload, dict):
        metrics["universe_prefix_counts"] = {
            str(k): int(v) for k, v in prefix_counts_payload.items()
        }
    else:
        metrics["universe_prefix_counts"] = {}
    cache_payload = fetch_payload.get("cache")
    cache_hits = metrics.get("cache_hits", 0)
    cache_misses = metrics.get("cache_misses", 0)
    if not cache_misses:
        batches_total = metrics.get("batches_total", 0)
        try:
            cache_misses = max(int(batches_total) - int(cache_hits), 0)
        except Exception:
            cache_misses = 0
        metrics["cache_misses"] = cache_misses
    batches_hit = int(cache_hits or 0)
    batches_miss = int(cache_misses or 0)
    if isinstance(cache_payload, dict):
        try:
            batches_hit = max(batches_hit, int(cache_payload.get("batches_hit", 0)))
        except Exception:
            pass
        try:
            batches_miss = max(batches_miss, int(cache_payload.get("batches_miss", 0)))
        except Exception:
            pass
    metrics["cache"] = {
        "batches_hit": batches_hit,
        "batches_miss": batches_miss,
    }
    http_payload = fetch_payload.get("http")
    if isinstance(http_payload, Mapping):
        http_requests = int(http_payload.get("requests", 0) or 0)
        http_rows = int(http_payload.get("rows", 0) or 0)
        http_rl = int(
            http_payload.get("rate_limit_hits", http_payload.get("rate_limited", 0) or 0)
        )
        http_retries = int(http_payload.get("retries", http_payload.get("http_retries", 0) or 0))
    else:
        http_requests = int(fetch_payload.get("http_requests", 0) or 0)
        http_rows = int(fetch_payload.get("http_rows", 0) or 0)
        http_rl = int(fetch_payload.get("rate_limit_hits", fetch_payload.get("rate_limited", 0) or 0))
        http_retries = int(fetch_payload.get("http_retries", fetch_payload.get("retries", 0) or 0))
    if not http_rows:
        try:
            http_rows = int(metrics.get("bars_rows_total", 0))
        except Exception:
            http_rows = 0
    metrics["http"] = {
        "requests": http_requests,
        "rows": http_rows,
        "rate_limit_hits": http_rl,
        "retries": http_retries,
    }
    window_attempts = fetch_payload.get("window_attempts", [])
    if isinstance(window_attempts, list):
        metrics["window_attempts"] = window_attempts
    else:
        metrics["window_attempts"] = []
    asset_payload = asset_metrics or {}
    metrics.update(
        {
            "assets_total": _coerce_int(asset_payload.get("assets_total", 0)),
            "symbols_after_iex_filter": _coerce_int(
                asset_payload.get("symbols_after_iex_filter", 0)
            ),
        }
    )
    metrics["reject_samples"] = (reject_samples or [])[:10]
    timing_payload = _normalize_timings(timings)
    metrics["timings"] = timing_payload
    _write_json_atomic(metrics_path, metrics)

    hist_path = data_dir / "screener_metrics_history.csv"
    row = {
        "run_utc": metrics.get("last_run_utc"),
        "symbols_in": metrics.get("symbols_in", 0),
        "symbols_with_bars": metrics.get("symbols_with_bars", 0),
        "bars_rows_total": metrics.get("bars_rows_total", 0),
        "rows": metrics.get("rows", 0),
        "fetch_secs": timing_payload.get("fetch_secs", 0),
        "feature_secs": timing_payload.get("feature_secs", 0),
        "rank_secs": timing_payload.get("rank_secs", 0),
        "gates_secs": timing_payload.get("gates_secs", 0),
        "coarse_rank_secs": timing_payload.get("coarse_rank_secs", 0),
        "backtest_secs": timing_payload.get("backtest_secs", 0),
    }
    try:
        if hist_path.exists():
            pd.DataFrame([row]).to_csv(hist_path, mode="a", header=False, index=False)
        else:
            pd.DataFrame([row]).to_csv(hist_path, mode="w", header=True, index=False)
    except Exception as exc:
        LOGGER.warning("Could not append metrics history: %s", exc)

    return metrics_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly screener")
    source_default = os.environ.get("SCREENER_SOURCE")
    parser.add_argument(
        "--mode",
        choices=[
            "screener",
            "delta-update",
            "build-symbol-stats",
            "coarse-features",
            "full-nightly",
        ],
        required=True,
        help="Pipeline mode to run",
    )
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
        "--prefilter-days",
        type=int,
        default=120,
        help="Lookback window (trading days) for coarse prefilter inputs",
    )
    parser.add_argument(
        "--prefilter-top",
        type=int,
        default=DEFAULT_SHORTLIST_SIZE,
        help="Shortlist size to carry into the nightly stage",
    )
    parser.add_argument(
        "--full-days",
        type=int,
        default=750,
        help="Lookback window (trading days) for full feature computation",
    )
    parser.add_argument(
        "--min-days-fallback",
        type=int,
        default=365,
        help="Fallback trading-day window when the primary request returns no bars",
    )
    parser.add_argument(
        "--min-days-final",
        type=int,
        default=90,
        help="Final trading-day window if the fallback still returns no bars",
    )
    parser.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default=DEFAULT_FEED,
        help="Market data feed to request from Alpaca",
    )
    parser.add_argument(
        "--bars-source",
        choices=["http", "sdk"],
        default="http",
        help="Source for fetching bars data",
    )
    parser.add_argument(
        "--fetch-mode",
        choices=["auto", "batch", "single"],
        default="auto",
        help="Strategy for fetching bars: auto (default), batch, or single",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum symbols per batch for bar requests",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum workers for single-symbol fallback",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=450,
        help="Optional symbol limit for development/testing",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to override the Alpaca universe",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument(
        "--iex-only",
        choices=["true", "false"],
        default="true",
        help="Restrict universe to exchanges covered by IEX (default: true)",
    )
    parser.add_argument(
        "--liquidity-top",
        type=int,
        default=500,
        help="Keep only the top N symbols by recent volume (0 disables)",
    )
    parser.add_argument(
        "--exclude-otc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude OTC symbols from the universe (default: true)",
    )
    parser.add_argument(
        "--verify-request",
        action="store_true",
        help="Log and capture the first bars request for debugging",
    )
    parser.add_argument(
        "--reuse-cache",
        choices=["true", "false"],
        default="true",
        help="Reuse cached bars batches when available (default: true)",
    )
    parser.add_argument(
        "--refresh-latest",
        choices=["true", "false"],
        default="true",
        help="Update latest_candidates.csv after full-nightly (default: true)",
    )
    parser.add_argument(
        "--gate-preset",
        choices=["strict", "standard", "mild"],
        default="standard",
        help="Gate thresholds preset (default: standard)",
    )
    parser.add_argument(
        "--relax-gates",
        choices=["none", "sma_only", "cross_or_rsi"],
        default="none",
        help="Optional relaxation override for diagnostics (default: none)",
    )
    parser.add_argument(
        "--dollar-vol-min",
        type=float,
        default=2000000,
        help="Minimum ADV20 dollar volume required for gate pass (default: 2000000)",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(parsed, "source", None):
        parsed.source_csv = parsed.source
    if isinstance(parsed.iex_only, str):
        parsed.iex_only = _as_bool(parsed.iex_only, True)
    parsed.reuse_cache = _as_bool(getattr(parsed, "reuse_cache", True), True)
    parsed.refresh_latest = _as_bool(getattr(parsed, "refresh_latest", True), True)
    return parsed


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    input_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> int:
    _bootstrap_env()
    _ensure_logger()
    arg_list: Optional[List[str]]
    if argv is None:
        arg_list = None
    else:
        arg_list = list(argv)
        if "--mode" not in {item for item in arg_list if isinstance(item, str)}:
            arg_list = ["--mode", "screener", *arg_list]
    args = parse_args(arg_list)
    base_dir = Path(output_dir or args.output_dir or Path(__file__).resolve().parents[1])

    try:
        creds_snapshot = assert_alpaca_creds()
    except AlpacaCredentialsError as exc:
        key_value = os.getenv("APCA_API_KEY_ID", "")
        if exc.reason == "invalid_prefix" and key_value.lower().startswith("test"):
            LOGGER.warning(
                "[WARN] Detected test credential prefix; continuing without strict validation"
            )
            os.environ.pop("APCA_API_KEY_ID", None)
            os.environ.pop("APCA_API_SECRET_KEY", None)
            creds_snapshot = exc.sanitized
        else:
            missing = list(dict.fromkeys(list(exc.missing) + list(exc.whitespace)))
            LOGGER.error(
                "[ERROR] ALPACA_CREDENTIALS_INVALID reason=%s missing=%s whitespace=%s sanitized=%s",
                exc.reason,
                ",".join(exc.missing) or "",
                ",".join(exc.whitespace) or "",
                json.dumps(exc.sanitized, sort_keys=True),
            )
            AUTH_CONTEXT["creds"] = exc.sanitized
            AUTH_CONTEXT["base_dir"] = base_dir
            _persist_auth_error(exc.reason, missing, sanitized=exc.sanitized, base_dir=base_dir)
            return 2

    AUTH_CONTEXT["creds"] = creds_snapshot
    AUTH_CONTEXT["base_dir"] = base_dir
    LOGGER.info(
        "[INFO] ALPACA_CREDENTIALS_OK sanitized=%s",
        json.dumps(creds_snapshot, sort_keys=True),
    )

    def _run() -> int:
        mode = getattr(args, "mode", "screener")

        if mode == "build-symbol-stats":
            return run_build_symbol_stats(args, base_dir)
        if mode == "coarse-features":
            return run_coarse_features(args, base_dir)

        trading_probe = probe_trading_only()
        status_code = (
            trading_probe.get("status")
            if isinstance(trading_probe, Mapping)
            else trading_probe.get("status") if isinstance(trading_probe, dict) else None
        )
        if status_code in (401, 403):
            raise AlpacaUnauthorizedError(endpoint="/v2/account")

        if mode == "delta-update":
            return _run_delta_update(args, base_dir)
        if mode == "full-nightly":
            return run_full_nightly(args, base_dir)

        now = datetime.now(timezone.utc)
        pipeline_timer = T()
        fetch_elapsed = 0.0

        symbols_override: Optional[List[str]] = None
        if args.symbols:
            symbols_override = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

        LOGGER.info("[STAGE] fetch start")
        frame: pd.DataFrame
        asset_meta: Dict[str, dict] = {}
        fetch_metrics: dict[str, Any] = {}
        asset_metrics: dict[str, Any] = {}
        prescreened: Dict[str, str] = {}
        if input_df is not None:
            frame = input_df
            fetch_elapsed = pipeline_timer.lap("fetch_secs")
            LOGGER.info(
                "[STAGE] fetch end (rows=%d, elapsed=%.2fs)",
                int(frame.shape[0]) if hasattr(frame, "shape") else 0,
                fetch_elapsed,
            )
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
                (
                    frame,
                    asset_meta,
                    fetch_metrics,
                    prescreened,
                    asset_metrics,
                ) = _load_alpaca_universe(
                    base_dir=base_dir,
                    days=max(1, args.days),
                    feed=args.feed,
                    limit=args.limit,
                    fetch_mode=args.fetch_mode,
                    batch_size=args.batch_size,
                    max_workers=args.max_workers,
                    min_history=args.min_history,
                    bars_source=args.bars_source,
                    exclude_otc=args.exclude_otc,
                    iex_only=args.iex_only,
                    liquidity_top=args.liquidity_top,
                    symbols_override=symbols_override,
                    verify_request=args.verify_request,
                    min_days_fallback=args.min_days_fallback,
                    min_days_final=args.min_days_final,
                    reuse_cache=bool(args.reuse_cache),
                    metrics=fetch_metrics,
                )
                fetch_elapsed = pipeline_timer.lap("fetch_secs")
            LOGGER.info(
                "[STAGE] fetch end (rows=%d, elapsed=%.2fs)",
                int(frame.shape[0]) if hasattr(frame, "shape") else 0,
                fetch_elapsed,
            )
        (
            top_df,
            scored_df,
            stats,
            skip_reasons,
            reject_samples,
            gate_counters,
            ranker_cfg,
            timing_info,
        ) = run_screener(
            frame,
            top_n=args.top_n,
            min_history=args.min_history,
            now=now,
            asset_meta=asset_meta,
            prefiltered_skips=prescreened,
            gate_preset=args.gate_preset,
            relax_gates=args.relax_gates,
            dollar_vol_min=args.dollar_vol_min,
        )
        timing_info["fetch_secs"] = timing_info.get("fetch_secs", 0.0) + round(fetch_elapsed, 3)
        write_outputs(
            base_dir,
            top_df,
            scored_df,
            stats,
            skip_reasons,
            reject_samples,
            status="ok",
            now=now,
            gate_counters=gate_counters,
            fetch_metrics=fetch_metrics,
            asset_metrics=asset_metrics,
            ranker_cfg=ranker_cfg,
            timings=timing_info,
        )
        LOGGER.info(
            "Screener complete: %d symbols examined, %d candidates.",
            stats["symbols_in"],
            stats["candidates_out"],
        )
        return 0

    try:
        return _run()
    except AlpacaUnauthorizedError as exc:
        LOGGER.error(
            '[ERROR] ALPACA_UNAUTHORIZED endpoint=%s feed=%s hint="check keys/base urls"',
            exc.endpoint or "",
            exc.feed or "",
        )
        _persist_auth_error("unauthorized", base_dir=base_dir)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
