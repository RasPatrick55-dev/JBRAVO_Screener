from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any, Callable

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - preferred module execution path
    from .indicators import adx, aroon, macd, obv, rsi
    from .utils.normalize import to_bars_df, BARS_COLUMNS
    from .utils.http_alpaca import fetch_bars_http
    from .utils.rate import TokenBucket
    from .utils.models import BarData, classify_exchange, KNOWN_EQUITY
    from .utils.env import trading_base_url
except Exception:  # pragma: no cover - fallback for direct script execution
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from scripts.indicators import adx, aroon, macd, obv, rsi  # type: ignore
    from scripts.utils.normalize import to_bars_df, BARS_COLUMNS  # type: ignore
    from scripts.utils.http_alpaca import fetch_bars_http  # type: ignore
    from scripts.utils.rate import TokenBucket  # type: ignore
    from scripts.utils.models import BarData, classify_exchange, KNOWN_EQUITY  # type: ignore
    from scripts.utils.env import trading_base_url  # type: ignore

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
BARS_BATCH_SIZE = 200
BAR_RETRY_STATUSES = {429, 500, 502, 503, 504}
BAR_MAX_RETRIES = 3


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
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame.dropna(subset=["symbol", "timestamp"], inplace=True)
    if frame.empty:
        return frame
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
        df = to_bars_df(response)
        if "symbol" not in df.columns:
            LOGGER.error(
                "Bars normalize failed: type=%s has attributes df=%s data=%s; df_cols=%s",
                type(response).__name__,
                hasattr(response, "df"),
                hasattr(response, "data"),
                list(getattr(getattr(response, "df", pd.DataFrame()), "columns", [])),
            )
        columns_desc = ",".join(df.columns)
        if df.empty or "symbol" not in df.columns:
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
    feed: str,
    fetch_mode: str,
    batch_size: int,
    max_workers: int,
    min_history: int,
    bars_source: str,
    now: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, dict[str, int | list[str]], Dict[str, str]]:
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
    if source == "sdk":
        if StockBarsRequest is None or TimeFrame is None:
            raise RuntimeError("alpaca-py market data components unavailable")
        if data_client is None:
            raise RuntimeError("StockHistoricalDataClient required when bars_source=sdk")

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    days = max(1, int(days))

    def _as_midnight(date_like: np.datetime64) -> datetime:
        ts = pd.Timestamp(date_like).to_pydatetime()
        return datetime.combine(ts.date(), datetime.min.time(), tzinfo=timezone.utc)

    current_day = np.datetime64(now.date(), "D")
    last_session = np.busday_offset(current_day, 0, roll="backward")
    start_session = (
        np.busday_offset(last_session, -(days - 1), roll="backward") if days > 1 else last_session
    )
    start = _as_midnight(start_session)
    end = _as_midnight(last_session) + timedelta(days=1)

    def _to_iso(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    start_iso = _to_iso(start)
    end_iso = _to_iso(end)

    unique_symbols = [str(sym or "").strip().upper() for sym in symbols if sym]
    unique_symbols = list(dict.fromkeys(unique_symbols))
    metrics: dict[str, int | list[str]] = {
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
    }
    prescreened: dict[str, str] = {}

    frames: list[pd.DataFrame] = []

    def fetch_single_collection(
        target_symbols: List[str], *, use_http: bool
    ) -> Tuple[pd.DataFrame, int, Dict[str, str], dict[str, int]]:
        local_frames: list[pd.DataFrame] = []
        local_prescreened: dict[str, str] = {}
        token_bucket = TokenBucket(200) if not use_http else None
        local_metrics: dict[str, int] = {
            "rate_limited": 0,
            "pages": 0,
            "requests": 0,
            "chunks": 0,
        }

        def _merge_metrics(payload: dict[str, int]) -> None:
            for key, value in (payload or {}).items():
                if not isinstance(value, int):
                    continue
                local_metrics[key] = local_metrics.get(key, 0) + int(value)

        def _http_worker(symbol: str) -> Tuple[str, pd.DataFrame, dict[str, int]]:
            try:
                raw, http_stats = fetch_bars_http(
                    [symbol], start_iso, end_iso, timeframe="1Day", feed=feed
                )
                df = to_bars_df(raw, symbol_hint=symbol)
                return symbol, df, http_stats
            except Exception as exc:  # pragma: no cover - network errors hit in integration
                LOGGER.warning("Failed HTTP bars fetch for %s: %s", symbol, exc)
                df = pd.DataFrame(columns=BARS_COLUMNS)
                return symbol, df, {"rate_limited": 0, "pages": 0, "requests": 0, "chunks": 0}

        def _sdk_worker(symbol: str) -> Tuple[str, pd.DataFrame, dict[str, int]]:
            assert data_client is not None  # for type checker
            request_kwargs = {
                "symbol_or_symbols": symbol,
                "timeframe": TimeFrame.Day,
                "start": start,
                "end": end,
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
                    df = to_bars_df(response, symbol_hint=symbol)
                    return symbol, df, {"rate_limited": 0, "pages": 1, "requests": 1, "chunks": 1}
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
                "rate_limited": 0,
                "pages": 0,
                "requests": 0,
                "chunks": 0,
            }

        worker = _http_worker if use_http else _sdk_worker
        pages = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(worker, symbol): symbol for symbol in target_symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, df, http_stats = future.result()
                except Exception as exc:  # pragma: no cover - unexpected failure path
                    LOGGER.error("Single-symbol worker failed for %s: %s", symbol, exc)
                    sym = symbol
                    df = pd.DataFrame(columns=BARS_COLUMNS)
                    http_stats = {"rate_limited": 0, "pages": 0, "requests": 0, "chunks": 0}
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
        metrics["rate_limited"] += int(single_metrics.get("rate_limited", 0))
        prescreened.update(single_prescreened)
    else:
        batches = list(_chunked(unique_symbols, max(1, batch_size)))
        metrics["batches_total"] = len(batches)
        for index, batch in enumerate(batches, start=1):
            if source == "http":
                try:
                    raw, http_stats = fetch_bars_http(
                        batch, start_iso, end_iso, timeframe="1Day", feed=feed
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
                    metrics["pages_total"] += fallback_pages
                    metrics["rate_limited"] += int(fallback_metrics.get("rate_limited", 0))
                    prescreened.update(fallback_prescreened)
                    continue
                metrics["rate_limited"] += int(http_stats.get("rate_limited", 0))
                pages_seen = int(http_stats.get("pages", 0))
                metrics["pages_total"] += pages_seen
                if pages_seen > 1:
                    metrics["batches_paged"] += 1
                bars_df = to_bars_df(raw)
                if "symbol" not in bars_df.columns:
                    LOGGER.error(
                        "Normalize returned unexpected shape; got columns=%s",
                        list(bars_df.columns),
                    )
                    metrics["fallback_batches"] += 1
                    fallback_frame, fallback_pages, fallback_prescreened, fallback_metrics = (
                        fetch_single_collection(batch, use_http=True)
                    )
                    if not fallback_frame.empty:
                        frames.append(fallback_frame)
                    metrics["pages_total"] += fallback_pages
                    metrics["rate_limited"] += int(fallback_metrics.get("rate_limited", 0))
                    prescreened.update(fallback_prescreened)
                    continue
                normalized = _normalize_bars_frame(bars_df)
                if normalized.empty:
                    for sym in batch:
                        prescreened.setdefault(sym, "NAN_DATA")
                    continue
                frames.append(normalized)
                continue

            request_kwargs = {
                "symbol_or_symbols": batch if len(batch) > 1 else batch[0],
                "timeframe": TimeFrame.Day,
                "start": start,
                "end": end,
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
            metrics["rate_limited"] += int(fallback_metrics.get("rate_limited", 0))
            prescreened.update(fallback_prescreened)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BARS_COLUMNS)
    if not combined.empty and "symbol" not in combined.columns:
        LOGGER.error("Normalized bars missing 'symbol'; dropping combined frame")
        combined = pd.DataFrame(columns=BARS_COLUMNS)
    combined = _normalize_bars_frame(combined)
    if not combined.empty and "symbol" not in combined.columns:
        LOGGER.error("Normalized bars missing 'symbol' after normalization; clearing frame")
        combined = pd.DataFrame(columns=BARS_COLUMNS)
    if not combined.empty:
        try:
            combined = (
                combined.groupby("symbol", group_keys=False)
                .apply(lambda g: g.sort_values("timestamp").tail(days), include_groups=False)
            )
        except TypeError:  # pragma: no cover - for older pandas without include_groups
            combined = combined.groupby("symbol", group_keys=False).apply(
                lambda g: g.sort_values("timestamp").tail(days)
            )
        combined = combined.reset_index(drop=True)

    if not combined.empty and min_history > 0:
        counts = combined.groupby("symbol")["timestamp"].count()
        insufficient = [sym for sym, count in counts.items() if count < min_history]
        if insufficient:
            metrics["insufficient_history"] = len(insufficient)
            for sym in insufficient:
                prescreened.setdefault(sym, "INSUFFICIENT_HISTORY")
            combined = combined[~combined["symbol"].isin(insufficient)]

    symbols_with_bars = set(combined["symbol"].unique()) if not combined.empty else set()
    metrics["bars_rows_total"] = int(combined.shape[0])
    metrics["symbols_with_bars"] = len(symbols_with_bars)
    missing = [sym for sym in unique_symbols if sym not in symbols_with_bars]
    for sym in missing:
        prescreened.setdefault(sym, "NAN_DATA")
    metrics["symbols_no_bars"] = len(missing)
    metrics["symbols_no_bars_sample"] = missing[:10]

    return combined, metrics, prescreened


def merge_asset_metadata(bars_df: pd.DataFrame, asset_meta: Dict[str, dict]) -> pd.DataFrame:
    if bars_df.empty:
        if not asset_meta:
            return bars_df
        result = bars_df.copy()
    else:
        result = bars_df.copy()
        result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()

    if asset_meta:
        meta_rows: list[dict[str, object]] = []
        for symbol, meta in asset_meta.items():
            if not symbol:
                continue
            cleaned_symbol = str(symbol).strip().upper()
            details = meta or {}
            meta_rows.append(
                {
                    "symbol": cleaned_symbol,
                    "exchange": str(details.get("exchange", "") or "").strip().upper(),
                    "asset_class": str(details.get("asset_class", "") or "").strip().upper(),
                    "tradable": bool(details.get("tradable", True)),
                }
            )
        meta_df = pd.DataFrame(meta_rows)
        if not meta_df.empty:
            meta_df = meta_df.drop_duplicates(subset=["symbol"], keep="first")
            result = result.merge(meta_df, on="symbol", how="left")

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
    now: datetime,
) -> Tuple[
    pd.DataFrame,
    Dict[str, dict],
    dict[str, int | list[str]],
    Dict[str, str],
    dict[str, int | list[str]],
]:
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
    }
    empty_asset_metrics = {
        "assets_total": 0,
        "assets_tradable_equities": 0,
        "assets_after_filters": 0,
        "symbols_after_iex_filter": 0,
    }
    try:
        trading_client = _create_trading_client()
    except Exception as exc:
        LOGGER.error("Unable to create Alpaca trading client: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}, empty_metrics, {}, empty_asset_metrics

    try:
        symbols, asset_meta, asset_metrics = fetch_active_equity_symbols(
            trading_client,
            base_dir=base_dir,
            exclude_otc=exclude_otc,
        )
    except Exception as exc:
        LOGGER.error("Failed to fetch Alpaca asset universe: %s", exc)
        return pd.DataFrame(columns=INPUT_COLUMNS), {}, empty_metrics, {}, empty_asset_metrics

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
    raw_symbols = [str(sym).strip().upper() for sym in symbols]
    ordered_symbols = raw_symbols or list(asset_meta.keys())
    seen: set[str] = set()
    iex_exchanges = {"NASDAQ", "NYSE", "ARCA", "AMEX"}
    filtered_symbols: list[str] = []
    filtered_skips: Dict[str, str] = {}
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

    if limit is not None and limit > 0:
        filtered_symbols = filtered_symbols[:limit]

    asset_metrics["symbols_after_iex_filter"] = len(filtered_symbols)
    asset_metrics["assets_total"] = int(asset_metrics.get("assets_total", len(asset_meta)))
    asset_metrics["assets_after_filters"] = len(filtered_symbols)
    total_tradable = len(raw_symbols)
    LOGGER.info(
        "Universe sample size: %d (of %d tradable equities)",
        len(filtered_symbols),
        total_tradable or asset_metrics.get("assets_tradable_equities", 0),
    )
    if asset_meta:
        asset_meta = {
            sym: asset_meta.get(sym, {})
            for sym in filtered_symbols
            if sym in asset_meta
        }
    symbols = filtered_symbols

    if not symbols:
        return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta, empty_metrics, {}, asset_metrics

    data_client: Optional["StockHistoricalDataClient"] = None
    if (bars_source or "http").strip().lower() == "sdk":
        try:
            data_client = _create_data_client()
        except Exception as exc:
            LOGGER.error("Unable to create Alpaca market data client: %s", exc)
            return (
                pd.DataFrame(columns=INPUT_COLUMNS),
                asset_meta,
                empty_metrics,
                {},
                asset_metrics,
            )

    bars_df, fetch_metrics, prescreened = _fetch_daily_bars(
        data_client,
        symbols,
        days=days,
        feed=feed,
        fetch_mode=fetch_mode,
        batch_size=max(1, batch_size),
        max_workers=max(1, max_workers),
        min_history=min_history,
        bars_source=bars_source,
        now=now,
    )
    LOGGER.info(
        "Bars fetch metrics: batches=%d paged=%d pages=%d rows=%d symbols_with_bars=%d",
        int(fetch_metrics.get("batches_total", 0)),
        int(fetch_metrics.get("batches_paged", 0)),
        int(fetch_metrics.get("pages_total", 0)),
        int(fetch_metrics.get("bars_rows_total", 0)),
        int(fetch_metrics.get("symbols_with_bars", 0)),
    )
    missing_sample = fetch_metrics.get("symbols_no_bars_sample", [])
    if missing_sample:
        LOGGER.info("Symbols without bars (sample): %s", list(missing_sample)[:10])
    if fetch_metrics.get("fallback_batches"):
        LOGGER.info(
            "Fallback batches invoked: %d",
            int(fetch_metrics.get("fallback_batches", 0)),
        )
    if fetch_metrics.get("insufficient_history"):
        LOGGER.info(
            "Symbols dropped for insufficient history: %d",
            int(fetch_metrics.get("insufficient_history", 0)),
        )

    if filtered_skips:
        for sym, reason in filtered_skips.items():
            prescreened.setdefault(sym, reason)

    if not bars_df.empty:
        bars_df = bars_df.copy()
        if "symbol" not in bars_df.columns:
            LOGGER.error(
                "Normalized bars missing 'symbol' unexpectedly; skipping merge"
            )
            return pd.DataFrame(columns=INPUT_COLUMNS), asset_meta, fetch_metrics, prescreened, asset_metrics
        if liquidity_top and liquidity_top > 0:
            try:
                bars_df.sort_values(["symbol", "timestamp"], inplace=True)
                try:
                    recent = bars_df.groupby("symbol", group_keys=False).apply(
                        lambda g: g.tail(20), include_groups=False
                    )
                except TypeError:
                    recent = bars_df.groupby("symbol", group_keys=False).apply(lambda g: g.tail(20))
                recent = recent.copy()
                recent["volume"] = pd.to_numeric(recent["volume"], errors="coerce")
                adv = recent.groupby("symbol")["volume"].mean().fillna(0)
                top_symbols = set(adv.sort_values(ascending=False).head(liquidity_top).index)
                if top_symbols:
                    bars_df = bars_df[bars_df["symbol"].isin(top_symbols)]
                    metrics["symbols_with_bars"] = min(
                        int(metrics.get("symbols_with_bars", 0)), len(top_symbols)
                    )
            except Exception as exc:
                LOGGER.warning("Failed liquidity filter computation: %s", exc)
        bars_df = merge_asset_metadata(bars_df, asset_meta)
    return bars_df, asset_meta, fetch_metrics, prescreened, asset_metrics


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
        response.raise_for_status()
        payload = response.json()
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


def _coerce_int(value: object) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    prefiltered_skips: Optional[Dict[str, str]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[str, int],
    list[dict[str, str]],
    dict[str, int],
]:
    """Run the screener on ``df`` returning (top, scored, stats, skips)."""

    now = now or datetime.now(timezone.utc)
    prepared = _prepare_input_frame(df)
    stats = {"symbols_in": 0, "candidates_out": 0}
    skip_reasons = {key: 0 for key in SKIP_KEYS}
    gate_counters = {
        "failed_sma_stack": 0,
        "failed_rsi": 0,
        "failed_cross": 0,
        "nan_data": 0,
        "insufficient_history": 0,
    }
    scored_records: list[dict[str, object]] = []
    asset_meta = {str(key).upper(): value for key, value in (asset_meta or {}).items()}
    promoted_symbols: set[str] = set()
    reject_samples: list[dict[str, str]] = []

    def record_reject(symbol: object, reason: str) -> None:
        sym = str(symbol or "").strip().upper() or "<UNKNOWN>"
        if len(reject_samples) < 10:
            reject_samples.append({"symbol": sym, "reason": reason})

    prefiltered_map = {
        str(sym or "").strip().upper(): str(reason or "").strip().upper()
        for sym, reason in (prefiltered_skips or {}).items()
    }
    if prefiltered_map:
        stats["symbols_in"] += len(prefiltered_map)
        for sym, reason in prefiltered_map.items():
            if reason in skip_reasons:
                skip_reasons[reason] += 1
            if reason == "NAN_DATA":
                gate_counters["nan_data"] += 1
            elif reason == "INSUFFICIENT_HISTORY":
                gate_counters["insufficient_history"] += 1
            record_reject(sym, reason or "PREFILTERED")
        if not prepared.empty:
            prepared = prepared[~prepared["symbol"].isin(prefiltered_map.keys())]

    if prepared.empty:
        LOGGER.info("No input rows supplied to screener; outputs will be empty.")
    else:
        for symbol, group in prepared.groupby("symbol"):
            stats["symbols_in"] += 1
            if group.empty:
                skip_reasons["NAN_DATA"] += 1
                gate_counters["nan_data"] += 1
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
                    gate_counters["nan_data"] += 1
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
                gate_counters["nan_data"] += 1
                LOGGER.info("[SKIP] %s dropped due to NaN close", symbol or "<UNKNOWN>")
                record_reject(symbol, "NAN_DATA")
                continue

            clean_df.dropna(subset=numeric_cols, inplace=True)
            if clean_df.empty:
                skip_reasons["NAN_DATA"] += 1
                gate_counters["nan_data"] += 1
                LOGGER.info("[SKIP] %s dropped due to NaN data", symbol or "<UNKNOWN>")
                record_reject(symbol, "NAN_DATA")
                continue

            if len(clean_df) < min_history:
                skip_reasons["INSUFFICIENT_HISTORY"] += 1
                gate_counters["insufficient_history"] += 1
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
                gate_counters["failed_cross"] += 1
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
                gate_counters["nan_data"] += 1
                record_reject(symbol, "NAN_DATA")
                continue

            crossed_up = prev_row["close"] < prev_row["sma9"] and curr_row["close"] > curr_row["sma9"]
            if not crossed_up:
                LOGGER.info("[FILTER] %s failed crossover condition", symbol or "<UNKNOWN>")
                gate_counters["failed_cross"] += 1
                record_reject(symbol, "FAILED_CROSS")
                continue

            stacked = curr_row["sma9"] > curr_row["ema20"] > curr_row["sma180"]
            if not stacked:
                LOGGER.info("[FILTER] %s failed moving-average stack", symbol or "<UNKNOWN>")
                gate_counters["failed_sma_stack"] += 1
                record_reject(symbol, "FAILED_SMA_STACK")
                continue

            strong_rsi = curr_row["rsi"] > 50
            if not strong_rsi:
                LOGGER.info("[FILTER] %s failed RSI threshold", symbol or "<UNKNOWN>")
                gate_counters["failed_rsi"] += 1
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

    if stats["candidates_out"] == 0:
        sample = reject_samples[:10]
        LOGGER.info(
            "No candidates passed JBravo gates; sample rejections: %s",
            json.dumps(sample, sort_keys=True) if sample else "[]",
        )

    return top_df, scored_df, stats, skip_reasons, reject_samples, gate_counters


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
    gate_counters: Optional[dict[str, int]] = None,
    fetch_metrics: Optional[dict[str, int | list[str]]] = None,
    asset_metrics: Optional[dict[str, int | list[str]]] = None,
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
        "status": status,
        "rows": int(scored_df.shape[0]),
        "symbols_in": int(stats.get("symbols_in", 0)),
        "candidates_out": int(stats.get("candidates_out", 0)),
        "skips": {key: int(skip_reasons.get(key, 0)) for key in SKIP_KEYS},
    }
    gate_counts = gate_counters or {}
    metrics.update(
        {
            "failed_sma_stack": int(gate_counts.get("failed_sma_stack", 0)),
            "failed_rsi": int(gate_counts.get("failed_rsi", 0)),
            "failed_cross": int(gate_counts.get("failed_cross", 0)),
            "nan_data": int(gate_counts.get("nan_data", 0)),
            "insufficient_history": int(gate_counts.get("insufficient_history", 0)),
        }
    )
    fetch_payload = fetch_metrics or {}
    metrics.update(
        {
            "bars_rows_total": _coerce_int(fetch_payload.get("bars_rows_total", 0)),
            "symbols_with_bars": _coerce_int(fetch_payload.get("symbols_with_bars", 0)),
            "symbols_no_bars": _coerce_int(fetch_payload.get("symbols_no_bars", 0)),
            "rate_limited": _coerce_int(fetch_payload.get("rate_limited", 0)),
        }
    )
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
        help="Optional symbol limit for development/testing",
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
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(parsed, "source", None):
        parsed.source_csv = parsed.source
    if isinstance(parsed.iex_only, str):
        parsed.iex_only = parsed.iex_only.lower() != "false"
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
    fetch_metrics: dict[str, int | list[str]] = {}
    asset_metrics: dict[str, int | list[str]] = {}
    prescreened: Dict[str, str] = {}
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
                now=now,
            )
    top_df, scored_df, stats, skip_reasons, reject_samples, gate_counters = run_screener(
        frame,
        top_n=args.top_n,
        min_history=args.min_history,
        now=now,
        asset_meta=asset_meta,
        prefiltered_skips=prescreened,
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
        gate_counters=gate_counters,
        fetch_metrics=fetch_metrics,
        asset_metrics=asset_metrics,
    )
    LOGGER.info(
        "Screener complete: %d symbols examined, %d candidates.",
        stats["symbols_in"],
        stats["candidates_out"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
