"""Robust trade execution orchestrator for JBRAVO."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, time as dtime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

import pandas as pd
import requests

from scripts.fallback_candidates import CANON, normalize_candidate_df
from scripts.utils.env import load_env
from utils.env import (
    AlpacaCredentialsError,
    AlpacaUnauthorizedError,
    assert_alpaca_creds,
    get_alpaca_creds,
    write_auth_error_artifacts,
)
import utils.telemetry as telemetry

try:  # pragma: no cover - import guard for optional dependency
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        TrailingStopOrderRequest,
    )
except Exception:  # pragma: no cover - lightweight fallback for tests
    TradingClient = None  # type: ignore

    class _Enum(str):
        def __new__(cls, value: str):
            return str.__new__(cls, value)

    class OrderSide:  # type: ignore
        BUY = _Enum("buy")
        SELL = _Enum("sell")

    class TimeInForce:  # type: ignore
        DAY = _Enum("day")
        GTC = _Enum("gtc")

    class QueryOrderStatus:  # type: ignore
        OPEN = _Enum("open")

    class LimitOrderRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class TrailingStopOrderRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class GetOrdersRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


logger = logging.getLogger("execute_trades")
LOGGER = logger


def log_info(tag: str, **kv: Any) -> None:
    payload = " ".join(f"{key}={kv[key]}" for key in sorted(kv)) if kv else ""
    message = f"{tag} {payload}".strip()
    LOGGER.info(message)
 
 
def _log_call(label, fn, *args, **kwargs):
    t0 = _time.perf_counter()
    try:
        res = fn(*args, **kwargs)
        logger.debug("[CALL] %s ok=1 dt=%.3fs", label, _time.perf_counter() - t0)
        return res, None
    except Exception as e:
        logger.warning("[CALL] %s ok=0 dt=%.3fs err=%r", label, _time.perf_counter() - t0, e)
        return None, e
LOG_PATH = Path("logs") / "execute_trades.log"
METRICS_PATH = Path("data") / "execute_metrics.json"
DEFAULT_BAR_DIRECTORIES: Sequence[Path] = (
    Path("data") / "daily",
    Path("data") / "bars" / "daily",
    Path("data") / "bars",
    Path("data") / "cache" / "bars",
)

REQUIRED_COLUMNS = [
    "symbol",
    "close",
    "score",
    "universe_count",
    "score_breakdown",
]

OPTIONAL_COLUMNS = {"atrp", "exchange", "adv20", "entry_price"}
SKIP_REASON_ORDER: tuple[str, ...] = (
    "CASH",
    "EXISTING_POSITION",
    "MAX_POSITIONS",
    "NO_CANDIDATES",
    "OPEN_ORDER",
    "PRICE_BOUNDS",
    "TIME_WINDOW",
    "ZERO_QTY",
)
SKIP_REASON_KEYS = set(SKIP_REASON_ORDER)
IMPORT_SENTINEL_ENV = "JBRAVO_IMPORT_SENTINEL"
DATA_URL_ENV_VARS = ("APCA_API_DATA_URL", "ALPACA_API_DATA_URL")
DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
REQUIRED_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)

_ENV_FILES_LOADED: list[str] = []


def _mask_key_hint(raw: Optional[str]) -> str:
    if not raw:
        return "missing"
    try:
        value = str(raw).strip()
    except Exception:
        return "invalid"
    if not value:
        return "missing"
    if len(value) <= 4:
        return value[0] + "***"
    prefix = value[:4]
    suffix = value[-2:]
    return f"{prefix}…{suffix}"


def _log_account_probe(creds_snapshot: Mapping[str, Any] | None = None) -> None:
    env_files = [
        Path(path).name
        for path in _ENV_FILES_LOADED
        if isinstance(path, str) and path.strip()
    ]
    has_env_file = any(name.lower().endswith(".env") for name in env_files)

    key, secret, base_url, _ = get_alpaca_creds()
    key_hint = _mask_key_hint(key)
    if (not key or key_hint == "missing") and isinstance(creds_snapshot, Mapping):
        prefix = str(creds_snapshot.get("key_prefix") or "").strip()
        if prefix:
            key_hint = prefix

    auth_ok = False
    bp_display = "n/a"
    trading_base = base_url or ""
    if not trading_base and isinstance(creds_snapshot, Mapping):
        base_urls = creds_snapshot.get("base_urls")
        if isinstance(base_urls, Mapping):
            trading_base = str(base_urls.get("trading") or "")
    trading_base = (trading_base or "https://paper-api.alpaca.markets").rstrip("/")

    if key and secret:
        url = f"{trading_base}/v2/account"
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if isinstance(payload, Mapping):
                    buying_power = payload.get("buying_power")
                    if buying_power not in (None, ""):
                        bp_display = str(buying_power)
                auth_ok = True
            else:
                bp_display = f"status={response.status_code}"
        except Exception as exc:  # pragma: no cover - best-effort logging
            bp_display = f"error={exc}"
    else:
        bp_display = "missing-creds"

    LOGGER.info(
        "[INFO] AUTH_CHECK auth_ok=%s buying_power=%s base_url=%s env_loaded=%s key_hint=%s",
        bool(auth_ok),
        bp_display,
        base_url,
        "yes" if has_env_file else "no",
        key_hint,
    )


def _paper_only_guard(trading_client: Any | None, base_url: str | None = "") -> None:
    if trading_client is None:
        return

    base = (os.getenv("APCA_API_BASE_URL") or base_url or "").lower()
    if "paper-api.alpaca.markets" not in base:
        logger.error(
            "PAPER_ONLY guard: APCA_API_BASE_URL is not a paper endpoint: %s",
            base,
        )
        sys.exit(2)


def _load_execute_metrics() -> Optional[Dict[str, Any]]:
    if not METRICS_PATH.exists():
        return None
    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive metrics parsing
        LOGGER.warning("Failed to load execute metrics: %s", exc)
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    LOGGER.warning(
        "Execute metrics file contained unexpected payload type: %s",
        type(payload).__name__,
    )
    return None


def _ny_now() -> datetime:
    return datetime.now(NY)


def _clock_is_in_window(client: Any, window: str) -> tuple[bool, str]:
    if client is None or not hasattr(client, "get_clock"):
        return False, "00:00"
    try:
        clk = client.get_clock()
        ny = ZoneInfo("America/New_York")
        now_ny = (
            clk.timestamp.astimezone(ny)
            if getattr(clk, "timestamp", None) is not None
            else None
        )
        if now_ny is None:
            now_ny = datetime.now(ny)
        hhmm = now_ny.strftime("%H:%M")
        target = (window or "auto").lower()
        if target == "premarket":
            in_window = "07:00" <= hhmm < "09:30"
        elif target == "regular":
            in_window = "09:30" <= hhmm < "16:00"
        elif target == "any":
            in_window = True
        else:  # auto and unknown default to premarket guard
            in_window = "07:00" <= hhmm < "09:30"
        log_info("MARKET_TIME", ny_now=str(now_ny), window=target, in_window=in_window)
        return in_window, hhmm
    except Exception as exc:  # pragma: no cover - defensive logging
        log_info("CLOCK_FETCH_FAILED", error=str(exc)[:200])
        return False, "00:00"


def _resolve_time_window(requested: str):
    req = (requested or "auto").strip().lower()
    now = _ny_now()
    tnow = now.timetz()
    prem_open = dtime(7, 0, tzinfo=NY)
    prem_close = dtime(9, 30, tzinfo=NY)
    reg_open = dtime(9, 30, tzinfo=NY)
    reg_close = dtime(16, 0, tzinfo=NY)

    in_pre = prem_open <= tnow < prem_close
    in_reg = reg_open <= tnow < reg_close

    if req == "premarket":
        return "premarket", in_pre, now
    if req == "regular":
        return "regular", in_reg, now
    if req == "any":
        return "any", True, now

    if in_pre:
        return "premarket", True, now
    if in_reg:
        return "regular", True, now
    return "closed", False, now


def _record_auth_error(
    reason: str,
    sanitized: Mapping[str, object],
    missing: Iterable[str] | None = None,
) -> None:
    write_auth_error_artifacts(
        reason=reason,
        sanitized=sanitized,
        missing=missing or [],
        metrics_path=METRICS_PATH,
        summary_path=Path("data") / "metrics_summary.csv",
    )


def _fetch_latest_close_from_alpaca(symbol: str) -> Optional[float]:
    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return None
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/bars/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.latest_bar", symbol=symbol)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.latest_bar", f"{symbol}: {exc}")
        return None
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.latest_bar",
            f"{symbol} unauthorized status={response.status_code}",
        )
        return None
    if response.status_code != 200:
        _warn_context(
            "alpaca.latest_bar",
            f"{symbol} status={response.status_code}",
        )
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.latest_bar", f"{symbol} decode_failed: {exc}")
        return None
    if not isinstance(payload, Mapping):
        return None
    bar = payload.get("bar")
    if isinstance(bar, Mapping):
        close_value = bar.get("c") if "c" in bar else bar.get("close")
        if close_value is None:
            return None
        try:
            return float(close_value)
        except (TypeError, ValueError):
            return None
    return None


def _fetch_latest_daily_bars(symbols: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    unique = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    if not unique:
        return {}

    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return {}

    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/bars/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {"symbols": ",".join(unique), "timeframe": "1Day"}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.bars_latest", symbols=len(unique))
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.bars_latest", f"request_failed: {exc}")
        return {}
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.bars_latest",
            f"unauthorized status={response.status_code}",
        )
        return {}
    if response.status_code != 200:
        _warn_context(
            "alpaca.bars_latest",
            f"status={response.status_code}",
        )
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.bars_latest", f"decode_failed: {exc}")
        return {}
    bars = payload.get("bars") if isinstance(payload, Mapping) else None
    results: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(bars, Mapping):
        for symbol in unique:
            entry = bars.get(symbol) or bars.get(symbol.lower()) or bars.get(symbol.upper())
            if not isinstance(entry, Mapping):
                continue
            close_value = entry.get("c") if "c" in entry else entry.get("close")
            volume_value = entry.get("v") if "v" in entry else entry.get("volume")
            close: Optional[float]
            volume: Optional[float]
            try:
                close = float(close_value) if close_value is not None else None
            except (TypeError, ValueError):
                close = None
            try:
                volume = float(volume_value) if volume_value is not None else None
            except (TypeError, ValueError):
                volume = None
            results[symbol] = {"close": close, "volume": volume, "source": "alpaca"}
    return results


class CandidateLoadError(RuntimeError):
    """Raised when candidate data cannot be loaded or validated."""


class AlpacaAuthFailure(RuntimeError):
    """Raised when Alpaca auth probes fail."""


def emit_import_sentinel() -> None:
    """Emit an import sentinel event when requested via environment flag."""

    if os.environ.get(IMPORT_SENTINEL_ENV) != "1":
        return
    version = telemetry.get_version()
    cmd = [
        sys.executable,
        "-m",
        "bin.emit_event",
        "IMPORT_SENTINEL",
        "component=execute_trades",
        f"version={version}",
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:  # pragma: no cover - telemetry best effort
        LOGGER.debug("Import sentinel emission failed: %s", exc)


emit_import_sentinel()


def _format_breakdown(value: Any, score: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:  # pragma: no cover - fallback for unserialisable payload
            pass
    if score is not None and not pd.isna(score):
        try:
            return json.dumps({"score": float(score)})
        except (TypeError, ValueError):
            return json.dumps({"score": score})
    return json.dumps({})


def _apply_candidate_defaults(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    frame = df.copy()
    messages: List[str] = []

    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    for column in missing_columns:
        frame[column] = pd.NA
        messages.append(f"[WARN] MISSING_{column.upper()} column (defaulted)")

    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    if "entry_price" in frame.columns:
        entry_series = pd.to_numeric(frame["entry_price"], errors="coerce")
        close_series = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame.columns else pd.Series(dtype="float64")
        missing_entry = entry_series.isna()
        if "close" in frame.columns and not close_series.empty:
            filled = close_series.reindex_like(entry_series)
            entry_series = entry_series.fillna(filled)
        frame["entry_price"] = entry_series
        if missing_entry.any():
            messages.append("[WARN] DEFAULTED_ENTRY_PRICE_FROM_CLOSE")

    if "score" not in frame.columns or frame["score"].isna().all():
        if "Score" in frame.columns:
            frame["score"] = frame.get("Score")
            messages.append("[WARN] derived score column from Score header (defaulted)")
        else:
            frame["score"] = pd.NA
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")

    if "close" not in frame.columns:
        frame["close"] = pd.NA
    close_series = pd.to_numeric(frame.get("close"), errors="coerce")

    if "entry_price" not in frame.columns:
        frame["entry_price"] = pd.NA
    entry_series = pd.to_numeric(frame.get("entry_price"), errors="coerce")

    close_fallback_mask = close_series.isna() & entry_series.notna()
    if close_fallback_mask.any():
        close_series.loc[close_fallback_mask] = entry_series.loc[close_fallback_mask]
        messages.append(
            f"[WARN] close fallback from entry_price applied to {int(close_fallback_mask.sum())} rows"
        )
    frame["close"] = close_series
    frame["entry_price"] = entry_series

    if "universe_count" not in frame.columns:
        frame["universe_count"] = pd.NA
    universe_series = pd.to_numeric(frame.get("universe_count"), errors="coerce")
    default_universe = int(frame.shape[0])
    universe_missing = universe_series.isna()
    if universe_missing.any():
        universe_series.loc[universe_missing] = default_universe
        messages.append("[WARN] universe_count defaulted to candidate row count")
    frame["universe_count"] = universe_series.fillna(default_universe).astype(int)

    if "score_breakdown" not in frame.columns:
        frame["score_breakdown"] = [None] * frame.shape[0]
        messages.append("[WARN] score_breakdown populated from score (defaulted)")
    frame["score_breakdown"] = [
        _format_breakdown(value, score)
        for value, score in zip(frame.get("score_breakdown"), frame["score"])
    ]

    frame["exchange"] = (
        frame.get("exchange", pd.Series(dtype="string")).astype("string").fillna("").str.upper()
    )

    return frame, messages


@dataclass
class ExecutorConfig:
    source: Path = Path("data/latest_candidates.csv")
    allocation_pct: float = 0.05
    max_positions: int = 4
    entry_buffer_bps: int = 75
    limit_buffer_pct: float = 1.0
    trailing_percent: float = 3.0
    cancel_after_min: int = 35
    extended_hours: bool = True
    time_window: str = "auto"
    market_timezone: str = "America/New_York"
    dry_run: bool = False
    min_adv20: int = 2_000_000
    min_price: float = 1.0
    max_price: float = 1_000.0
    log_json: bool = False
    bar_directories: Sequence[Path] = DEFAULT_BAR_DIRECTORIES
    min_order_usd: float = 200.0
    allow_bump_to_one: bool = True
    allow_fractional: bool = False


@dataclass
class ExecutionMetrics:
    symbols_in: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_canceled: int = 0
    trailing_attached: int = 0
    api_retries: int = 0
    api_failures: int = 0
    latency_samples: List[float] = field(default_factory=list)
    skipped_reasons: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        # Maintain backward compatibility with legacy attribute name
        self.skipped_by_reason = self.skipped_reasons

    def record_skip(self, reason: str, count: int = 1) -> None:
        count = int(count)
        if count <= 0:
            return
        key = reason.upper()
        self.skipped_reasons[key] = self.skipped_reasons.get(key, 0) + count

    def record_latency(self, seconds: float) -> None:
        if seconds > 0:
            self.latency_samples.append(seconds)

    def percentile(self, p: float) -> float:
        if not self.latency_samples:
            return 0.0
        ordered = sorted(self.latency_samples)
        if len(ordered) == 1:
            return round(ordered[0], 3)
        rank = (len(ordered) - 1) * p
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return round(ordered[int(rank)], 3)
        fraction = rank - lower
        value = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        return round(value, 3)

    def as_dict(self) -> Dict[str, Any]:
        skip_payload = {
            key: int(self.skipped_reasons.get(key, 0)) for key in SKIP_REASON_ORDER
        }
        for key, value in sorted(self.skipped_reasons.items()):
            if key not in skip_payload:
                skip_payload[key] = int(value)
        return {
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
            "symbols_in": self.symbols_in,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_canceled": self.orders_canceled,
            "trailing_attached": self.trailing_attached,
            "api_retries": self.api_retries,
            "api_failures": self.api_failures,
            "latency_secs": {
                "p50": self.percentile(0.5),
                "p95": self.percentile(0.95),
            },
            "skips": skip_payload,
        }

    def flush(self) -> None:
        return None

    @property
    def skips(self) -> Counter:
        return self.skipped_reasons


def round_to_tick(value: float, tick: float = 0.01) -> float:
    decimal_value = Decimal(str(value))
    quant = Decimal(str(tick))
    return float(decimal_value.quantize(quant, rounding=ROUND_HALF_UP))


def compute_limit_price(row: Dict[str, Any], buffer_bps: int = 75) -> float:
    base_price = row.get("entry_price") if row.get("entry_price") not in (None, "") else row.get("close")
    if base_price is None or pd.isna(base_price):
        raise ValueError("Row must contain either entry_price or close")
    base_price_f = float(base_price)
    limit = base_price_f * (1 + buffer_bps / 10_000)
    return round_to_tick(limit, 0.01)


def _compute_qty(
    buying_power: float,
    limit_price: float,
    allocation_pct: float,
    min_order_usd: float,
) -> int:
    limit_floor = max(0.01, float(limit_price or 0))
    alloc_qty = math.floor(max(0.0, buying_power) * max(0.0, allocation_pct) / limit_floor)
    min_qty = (
        math.floor(max(0.0, min_order_usd) / limit_floor)
        if min_order_usd
        else 0
    )
    qty = max(alloc_qty, min_qty)
    return max(0, qty)


def compute_quantity(buying_power: float, allocation_pct: float, limit_price: float) -> int:
    if buying_power <= 0 or allocation_pct <= 0:
        return 0
    notional_cap = buying_power * allocation_pct
    if notional_cap <= 0:
        return 0
    qty = int(notional_cap // limit_price)
    if qty <= 0 and notional_cap >= limit_price:
        qty = 1
    if qty <= 0:
        return 0
    max_affordable = int(buying_power // limit_price)
    if max_affordable <= 0:
        return 0
    qty = min(qty, max_affordable)
    if qty < 1:
        return 0
    return qty


class DailyBarCache:
    def __init__(self, search_paths: Sequence[Path]) -> None:
        self.search_paths = search_paths
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _candidate_paths(self, symbol: str) -> Iterator[Path]:
        for directory in self.search_paths:
            candidate = directory / f"{symbol.upper()}.csv"
            yield candidate

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        key = symbol.upper()
        if key in self._cache:
            return self._cache[key]
        for path in self._candidate_paths(key):
            if path.exists():
                try:
                    frame = pd.read_csv(path)
                    self._cache[key] = frame
                    return frame
                except Exception:  # pragma: no cover - defensive parsing
                    break
        self._cache[key] = None
        return None

    def atr_percent(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        required = {"high", "low", "close"}
        if not required.issubset(lower_map.keys()):
            return None
        try:
            highs = pd.to_numeric(frame[lower_map["high"]], errors="coerce")
            lows = pd.to_numeric(frame[lower_map["low"]], errors="coerce")
            closes = pd.to_numeric(frame[lower_map["close"]], errors="coerce")
        except KeyError:
            return None
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}).dropna()
        if df.empty or len(df) < 15:
            return None
        df.sort_index(inplace=True)
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=14).mean()
        latest_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else None
        latest_close = df["close"].iloc[-1]
        if latest_atr is None or latest_close <= 0:
            return None
        return round(float(latest_atr) / float(latest_close) * 100, 3)

    def adv20(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        volume_column = lower_map.get("volume") or lower_map.get("vol")
        if not close_column or not volume_column:
            return None
        try:
            closes = pd.to_numeric(frame[close_column], errors="coerce")
            volumes = pd.to_numeric(frame[volume_column], errors="coerce")
        except Exception:
            return None
        df = pd.DataFrame({"close": closes, "volume": volumes}).dropna()
        if df.empty:
            return None
        window = df.tail(20)
        if window.empty:
            return None
        adv = (window["close"] * window["volume"]).mean()
        if pd.isna(adv):
            return None
        return float(adv)

    def latest_close(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        if not close_column or close_column not in frame.columns:
            return None
        try:
            closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
        except Exception:  # pragma: no cover - defensive conversion
            return None
        if closes.empty:
            return None
        return float(closes.iloc[-1])

    def latest_bar(self, symbol: str) -> Optional[Dict[str, Optional[float]]]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        volume_column = lower_map.get("volume") or lower_map.get("vol")
        close_value: Optional[float] = None
        volume_value: Optional[float] = None
        if close_column and close_column in frame.columns:
            try:
                closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
                if not closes.empty:
                    close_value = float(closes.iloc[-1])
            except Exception:  # pragma: no cover - defensive conversion
                close_value = None
        if volume_column and volume_column in frame.columns:
            try:
                volumes = pd.to_numeric(frame[volume_column], errors="coerce").dropna()
                if not volumes.empty:
                    volume_value = float(volumes.iloc[-1])
            except Exception:  # pragma: no cover - defensive conversion
                volume_value = None
        if close_value is None and volume_value is None:
            return None
        return {"close": close_value, "volume": volume_value, "source": "cache"}


class OptionalFieldHydrator:
    def __init__(self, client: Any, cache: DailyBarCache) -> None:
        self.client = client
        self.cache = cache
        self.exchange_cache: Dict[str, Optional[str]] = {}
        self.latest_bar_cache: Dict[str, Dict[str, Optional[float]]] = {}
        self._missing_logged: set[Tuple[str, str]] = set()
        self.missing_counts: Counter[str] = Counter()
        self._summary_logged = False
        self._entry_summary_logged = False
        self._entry_defaults = 0

    def _log_missing(
        self, field: str, symbol: str, detail: str = "", *, qualifier: str = "defaulted"
    ) -> None:
        key = (field.lower(), symbol.upper())
        if key in self._missing_logged:
            return
        qualifier_text = qualifier.strip()
        if qualifier_text:
            parts = [f"[WARN] MISSING_{field.upper()} ({qualifier_text})"]
        else:
            parts = [f"[WARN] MISSING_{field.upper()}"]
        if symbol:
            parts.append(f"symbol={symbol.upper()}")
        if detail:
            parts.append(detail)
        LOGGER.warning(" ".join(parts))
        self._missing_logged.add(key)
        self.missing_counts[field.upper()] += 1
        if field.lower() == "entry_price" and "defaulted from close" in qualifier_text.lower():
            self._entry_defaults += 1

    def get_exchange(self, symbol: str) -> Optional[str]:
        key = symbol.upper()
        if key in self.exchange_cache:
            return self.exchange_cache[key]
        exchange: Optional[str] = None
        if self.client is not None:
            try:
                asset = self.client.get_asset(key)
                exchange = getattr(asset, "exchange", None)
            except Exception:  # pragma: no cover - API errors surfaced elsewhere
                exchange = None
        self.exchange_cache[key] = exchange
        return exchange

    @staticmethod
    def _value_missing(value: Any) -> bool:
        if value in (None, ""):
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def prime_latest_bars(self, records: Iterable[Mapping[str, Any]]) -> None:
        symbols_to_fetch: List[str] = []
        for row in records:
            raw_symbol = row.get("symbol")
            symbol = str(raw_symbol).upper() if raw_symbol not in (None, "") else ""
            if not symbol:
                continue
            needs_close = self._value_missing(row.get("close"))
            needs_volume = self._value_missing(row.get("volume"))
            cache_entry = self.latest_bar_cache.get(symbol)
            if cache_entry:
                if not needs_close or cache_entry.get("close") is not None:
                    needs_close = False
                if not needs_volume or cache_entry.get("volume") is not None:
                    needs_volume = False
            if not needs_close and not needs_volume:
                continue
            local = self.cache.latest_bar(symbol)
            if local:
                cached = self.latest_bar_cache.get(symbol, {})
                if local.get("close") is not None:
                    cached["close"] = local.get("close")
                if local.get("volume") is not None:
                    cached["volume"] = local.get("volume")
                cached["source"] = "cache"
                self.latest_bar_cache[symbol] = cached
                if (
                    (not needs_close or cached.get("close") is not None)
                    and (not needs_volume or cached.get("volume") is not None)
                ):
                    continue
            symbols_to_fetch.append(symbol)

        unique_symbols = sorted(set(symbols_to_fetch))
        remote_payload = _fetch_latest_daily_bars(unique_symbols) if unique_symbols else {}
        for symbol, payload in remote_payload.items():
            cached = self.latest_bar_cache.get(symbol, {})
            if "close" in payload and payload.get("close") is not None:
                cached["close"] = payload.get("close")
            if "volume" in payload and payload.get("volume") is not None:
                cached["volume"] = payload.get("volume")
            if payload:
                cached["source"] = payload.get("source") or "alpaca"
            self.latest_bar_cache[symbol] = cached

    def latest_bar(self, symbol: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
        key = symbol.upper()
        if key in self.latest_bar_cache:
            cached = self.latest_bar_cache[key]
            return (
                cached.get("close"),
                cached.get("volume"),
                cached.get("source"),
            )
        local = self.cache.latest_bar(symbol)
        if local:
            self.latest_bar_cache[key] = dict(local)
            cached = self.latest_bar_cache[key]
            return (
                cached.get("close"),
                cached.get("volume"),
                cached.get("source"),
            )
        price = _fetch_latest_close_from_alpaca(symbol)
        if price is not None:
            cached = {"close": price, "volume": None, "source": "alpaca"}
            self.latest_bar_cache[key] = cached
            return price, None, "alpaca"
        return None, None, None

    def log_missing_summary(self) -> None:
        if not self.missing_counts or self._summary_logged:
            return
        summary = " ".join(
            f"{field.lower()}={count}" for field, count in sorted(self.missing_counts.items())
        )
        if summary:
            LOGGER.info("[INFO] DEFAULTED_OPTIONALS %s", summary)
        if self._entry_defaults and not self._entry_summary_logged:
            LOGGER.warning(
                "[WARN] entry_price defaulted from close on %d rows", self._entry_defaults
            )
            self._entry_summary_logged = True
        self._summary_logged = True

    def hydrate(self, row: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(row)
        symbol = enriched.get("symbol", "").upper()
        if symbol:
            exchange_value = enriched.get("exchange")
            if exchange_value in (None, "") or pd.isna(exchange_value):
                exchange = self.get_exchange(symbol)
                enriched["exchange"] = exchange
                self._log_missing(
                    "exchange",
                    symbol,
                    detail=f"source={'asset' if exchange else 'default'}",
                )
            close_value = enriched.get("close")
            if self._value_missing(close_value):
                fallback = enriched.get("entry_price")
                fallback_source: Optional[str] = None
                source_label = "entry_price"
                if self._value_missing(fallback):
                    latest_close, _, fallback_source = self.latest_bar(symbol)
                    fallback = latest_close
                    source_label = fallback_source or "default"
                if fallback is not None and not pd.isna(fallback):
                    enriched["close"] = fallback
                    self._log_missing(
                        "close",
                        symbol,
                        detail=f"source={source_label}",
                        qualifier="defaulted",
                    )
                else:
                    self._log_missing(
                        "close",
                        symbol,
                        detail="source=unavailable",
                        qualifier="missing",
                    )
            volume_value = enriched.get("volume")
            if self._value_missing(volume_value):
                _, latest_volume, source_label = self.latest_bar(symbol)
                if latest_volume is not None and not pd.isna(latest_volume):
                    enriched["volume"] = latest_volume
                    self._log_missing(
                        "volume",
                        symbol,
                        detail=f"source={source_label or 'default'}",
                        qualifier="defaulted",
                    )
                else:
                    self._log_missing(
                        "volume",
                        symbol,
                        detail="source=unavailable",
                        qualifier="missing",
                    )
            atr_value = enriched.get("atrp")
            if self._value_missing(atr_value):
                atr = self.cache.atr_percent(symbol)
                enriched["atrp"] = atr
                self._log_missing(
                    "atrp",
                    symbol,
                    detail=f"source={'bars' if atr is not None else 'default'}",
                )
            adv_value = enriched.get("adv20")
            if self._value_missing(adv_value):
                adv = self.cache.adv20(symbol)
                enriched["adv20"] = adv
                self._log_missing(
                    "adv20",
                    symbol,
                    detail=f"source={'bars' if adv is not None else 'default'}",
                )
        entry_value = enriched.get("entry_price")
        if self._value_missing(entry_value):
            close_value = enriched.get("close")
            if not self._value_missing(close_value):
                enriched["entry_price"] = close_value
                self._log_missing(
                    "entry_price",
                    symbol,
                    qualifier="defaulted from close",
                )
            else:
                self._log_missing(
                    "entry_price",
                    symbol,
                    detail="source=unavailable",
                    qualifier="missing",
                )
        if "adv20" not in enriched or pd.isna(enriched.get("adv20")):
            enriched["adv20"] = None
        return enriched


class TradeExecutor:
    def __init__(
        self,
        config: ExecutorConfig,
        client: Any,
        metrics: ExecutionMetrics,
        *,
        sleep_fn: Optional[Any] = None,
        base_url: str = "",
        account_snapshot: Optional[Mapping[str, Any]] = None,
        clock_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.metrics = metrics
        self.sleep = sleep_fn or time.sleep
        self.bar_cache = DailyBarCache(config.bar_directories)
        self.hydrator = OptionalFieldHydrator(client, self.bar_cache)
        self.log_json = config.log_json
        self._tz_fallback_logged = False
        self._clock_warning_logged = False
        self._base_url = base_url
        self.account_snapshot: Optional[Mapping[str, Any]] = account_snapshot
        self.clock_snapshot: Optional[Mapping[str, Any]] = clock_snapshot
        self._last_buying_power_raw: Any = None

    def log_info(self, event: str, **payload: Any) -> None:
        log_info(event, **payload)
        if self.log_json:
            try:
                LOGGER.info(json.dumps({"event": event, **payload}))
            except TypeError:  # pragma: no cover - serialization guard
                LOGGER.info(json.dumps({"event": event, "message": str(payload)}))

    def _market_label(self) -> str:
        tz_name = str(self.config.market_timezone or "America/New_York")
        mapping = {
            "America/New_York": "NY",
            "America/Chicago": "Chicago",
        }
        if tz_name in mapping:
            return mapping[tz_name]
        suffix = tz_name.split("/")[-1].strip()
        return suffix or tz_name

    def record_skip_reason(
        self,
        reason: str,
        *,
        symbol: str = "",
        detail: str = "",
        count: int = 1,
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        reason_key = reason.upper()
        parts = [f"[INFO] {reason_key}"]
        if symbol:
            parts.append(f"symbol={symbol}")
        if detail:
            parts.append(str(detail))
        if count > 1:
            parts.append(f"count={count}")
        LOGGER.info(" ".join(parts))

        self.metrics.record_skip(reason_key, count=count)
        if aliases:
            for alias in aliases:
                if alias:
                    self.metrics.record_skip(alias.upper(), count=count)
        payload: Dict[str, Any] = {"reason": reason_key}
        if symbol:
            payload["symbol"] = symbol
        if detail:
            payload["detail"] = detail
        if count > 1:
            payload["count"] = count
        self.log_info("SKIP", **payload)

    def _resolve_market_timezone(self) -> ZoneInfo:
        tz_name = self.config.market_timezone or "America/New_York"
        try:
            return ZoneInfo(tz_name)
        except Exception:
            if not self._tz_fallback_logged:
                LOGGER.warning(
                    "[WARN] invalid market timezone '%s'; falling back to America/New_York",
                    tz_name,
                )
                self._tz_fallback_logged = True
            try:
                return ZoneInfo("America/New_York")
            except Exception:  # pragma: no cover - ZoneInfo fallback safety
                return ZoneInfo("UTC")

    def _log_clock_warning(self, status: str) -> None:
        if self._clock_warning_logged:
            return
        LOGGER.warning(
            "[WARN] clock_fetch_failed status=%s -> using tz_fallback=America/New_York",
            status,
        )
        self._clock_warning_logged = True

    def _probe_trading_clock(self) -> Optional[Any]:
        api_key, api_secret, base_url, _ = get_alpaca_creds()
        base = (base_url or "").strip()
        if not api_key or not api_secret or not base:
            return None
        url = f"{base.rstrip('/')}/v2/clock"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        log_info("alpaca.clock_probe")
        try:
            response = requests.get(url, headers=headers, timeout=5)
        except Exception as exc:
            self._log_clock_warning("ERR")
            _warn_context("alpaca.clock_probe", str(exc))
            LOGGER.debug("Trading clock probe failed: %s", exc, exc_info=False)
            return None
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                return None
            if isinstance(payload, Mapping):
                return SimpleNamespace(**payload)
            return None
        self._log_clock_warning(str(response.status_code))
        _warn_context(
            "alpaca.clock_probe",
            f"status={response.status_code}",
        )
        return None

    def _get_trading_clock(self) -> Optional[Any]:
        if self.clock_snapshot is not None:
            if isinstance(self.clock_snapshot, Mapping):
                return SimpleNamespace(**self.clock_snapshot)
            return None
        if self.client is None or not hasattr(self.client, "get_clock"):
            return self._probe_trading_clock()
        try:
            log_info("alpaca.get_clock")
            return self.client.get_clock()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            status_label = str(status) if status else "ERR"
            self._log_clock_warning(status_label)
            if status not in (401, 403):
                _warn_context("alpaca.get_clock", str(exc))
            return self._probe_trading_clock()

    def evaluate_time_window(self, *, log: bool = True) -> Tuple[bool, str, str]:
        tz = self._resolve_market_timezone()
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz)
        ny_now_dt = now_utc.astimezone(ZoneInfo("America/New_York"))
        ny_now = ny_now_dt.isoformat()
        current_time = dtime(
            now_local.hour, now_local.minute, now_local.second, now_local.microsecond
        )
        ny_time = dtime(
            ny_now_dt.hour,
            ny_now_dt.minute,
            ny_now_dt.second,
            ny_now_dt.microsecond,
        )

        premarket_start = dtime(4, 0)
        regular_start = dtime(9, 30)
        regular_end = dtime(16, 0)
        postmarket_end = dtime(20, 0)

        within_premarket = premarket_start <= current_time < regular_start
        within_regular = regular_start <= current_time < regular_end
        within_post = regular_end <= current_time < postmarket_end
        ny_within_premarket = premarket_start <= ny_time < regular_start
        ny_within_regular = regular_start <= ny_time < regular_end

        mode = (self.config.time_window or "auto").lower()
        resolved_window = mode
        auto_branch = None
        if mode == "auto":
            if dtime(7, 0) <= ny_time < regular_start:
                resolved_window = "premarket"
                auto_branch = "premarket"
            elif ny_within_regular:
                resolved_window = "regular"
                auto_branch = "regular"
            else:
                resolved_window = "regular"
                auto_branch = "regular"
            log_info("TIME_WINDOW_AUTO", branch=auto_branch, ny_time=ny_now)
        elif mode not in {"premarket", "regular", "any"}:
            resolved_window = "any"

        clock = self._get_trading_clock()
        clock_is_open = bool(getattr(clock, "is_open", False))

        message: str
        allowed = False
        market_label = self._market_label()

        if resolved_window == "premarket":
            if not self.config.extended_hours:
                allowed = False
                message = f"outside premarket ({market_label})"
            else:
                allowed = ny_within_premarket if mode == "auto" else within_premarket
                message = (
                    f"premarket window open ({market_label})"
                    if allowed
                    else f"outside premarket ({market_label})"
                )
        elif resolved_window == "regular":
            allowed = ny_within_regular if mode == "auto" else within_regular
            message = (
                f"regular session ({market_label})"
                if allowed
                else f"outside regular session ({market_label})"
            )
        else:  # any
            allowed = True
            message = f"any window ({market_label})"

        if not allowed and clock is not None and resolved_window in {"premarket", "regular"}:
            # When Alpaca reports the venue open, treat it as authoritative for overrides.
            if (
                resolved_window == "premarket"
                and self.config.extended_hours
                and clock_is_open
                and not within_regular
            ):
                allowed = True
                message = f"premarket window open ({market_label})"
            elif resolved_window == "regular" and clock_is_open and within_regular:
                allowed = True
                message = f"regular session ({market_label})"

        if log:
            log_info(
                "MARKET_TIME",
                ny_now=ny_now,
                mode=mode,
                resolved=resolved_window,
                in_window=bool(allowed),
            )
            log_info("TIME_WINDOW", message=message)
        return allowed, message, resolved_window

    def load_candidates(self) -> pd.DataFrame:
        path = self.config.source
        if not path.exists():
            raise CandidateLoadError(f"Candidate file not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            LOGGER.info("[INFO] NO_CANDIDATES_IN_SOURCE")
            return df
        canonical_available = set()
        for column in df.columns:
            key = str(column).strip()
            canonical = CANON.get(key, CANON.get(key.lower(), key))
            canonical_available.add(str(canonical))
        normalized = normalize_candidate_df(df, now_ts=None)
        missing_required = [column for column in REQUIRED_COLUMNS if column not in canonical_available]
        if missing_required:
            joined = ", ".join(sorted(missing_required))
            raise CandidateLoadError(f"Missing required columns: {joined}")
        df, warnings = _apply_candidate_defaults(normalized)
        for message in warnings:
            LOGGER.warning(message)
        return df

    def hydrate_candidates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records = df.to_dict(orient="records")
        self.hydrator.prime_latest_bars(records)
        enriched = [self.hydrator.hydrate(record) for record in records]
        self.hydrator.log_missing_summary()
        return enriched

    def guard_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        buffer_pct = 0.0
        if self.config.extended_hours:
            try:
                buffer_pct = max(0.0, float(self.config.limit_buffer_pct)) / 100.0
            except (TypeError, ValueError):
                buffer_pct = 0.0
        min_threshold = max(0.0, float(self.config.min_price) * (1 - buffer_pct))
        max_threshold = float(self.config.max_price) * (1 + buffer_pct)
        for record in records:
            raw_symbol = record.get("symbol")
            symbol = ""
            if raw_symbol not in (None, "") and not pd.isna(raw_symbol):
                symbol = str(raw_symbol).upper()
            price = record.get("entry_price")
            if price in (None, "") or pd.isna(price):
                price = record.get("close")
            if price in (None, "") or pd.isna(price):
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail="missing_price",
                    aliases=("MISSING_PRICE",),
                )
                continue
            price_f = float(price)
            if price_f < min_threshold:
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail=f"lt_min({price_f:.2f}) thr={min_threshold:.2f}",
                    aliases=("PRICE_LT_MIN",),
                )
                continue
            if price_f > max_threshold:
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail=f"gt_max({price_f:.2f}) thr={max_threshold:.2f}",
                    aliases=("PRICE_GT_MAX",),
                )
                continue
            adv = record.get("adv20")
            if adv not in (None, "") and not pd.isna(adv):
                adv_value = pd.to_numeric(pd.Series([adv]), errors="coerce").iloc[0]
                if (
                    not pd.isna(adv_value)
                    and float(adv_value) > 0
                    and float(adv_value) < self.config.min_adv20
                ):
                    detail = f"adv20_lt_min({float(adv_value):.0f})"
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=detail,
                        aliases=("ADV20_LT_MIN",),
                    )
                    continue
            filtered.append(record)
        return filtered

    def execute(
        self,
        df: pd.DataFrame,
        *,
        prefiltered: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        self.metrics.symbols_in = len(df)
        if prefiltered is not None:
            candidates = prefiltered
        else:
            records = self.hydrate_candidates(df)
            candidates = self.guard_candidates(records)
        if not candidates:
            self.metrics.record_skip("NO_CANDIDATES", count=max(1, len(df)))
            LOGGER.info("No candidates passed guardrails; nothing to do.")
            self.persist_metrics()
            self.log_summary()
            return 0

        allowed, status, _ = self.evaluate_time_window()
        if not allowed:
            self.record_skip_reason("TIME_WINDOW", detail=status, count=len(candidates))
            self.persist_metrics()
            self.log_summary()
            return 0

        if self.config.dry_run:
            LOGGER.info(
                "[INFO] DRY_RUN=True — no orders will be submitted, but still perform sizing and emit skip reasons"
            )

        existing_positions = self.fetch_existing_positions()
        open_order_symbols = self.fetch_open_order_symbols()
        account_buying_power = self.fetch_buying_power()
        bp_raw = self._last_buying_power_raw
        if bp_raw is None:
            bp_raw = account_buying_power
        bp_display: str
        try:
            bp_display = f"{float(bp_raw):.2f}"
        except Exception:
            bp_display = str(bp_raw)
        if account_buying_power <= 0 or bp_raw in (None, "", 0, "0"):
            LOGGER.error("[ERROR] NO_BUYING_POWER buying_power=%s", bp_display)
            self.metrics.api_failures += 1
            self.persist_metrics()
            self.log_summary()
            return 2

        slots = max(1, min(self.config.max_positions, len(candidates)))
        try:
            allocation_pct = float(self.config.allocation_pct)
        except (TypeError, ValueError):
            allocation_pct = 0.0
        limit_buffer_ratio = max(0.0, float(self.config.limit_buffer_pct)) / 100.0

        for record in candidates:
            symbol = record.get("symbol", "").upper()
            if not symbol:
                continue
            if symbol in existing_positions:
                self.record_skip_reason("EXISTING_POSITION", symbol=symbol)
                continue
            if symbol in open_order_symbols:
                self.record_skip_reason("OPEN_ORDER", symbol=symbol)
                continue
            if len(existing_positions) >= self.config.max_positions:
                self.record_skip_reason("MAX_POSITIONS", symbol=symbol)
                break

            price_value = record.get("entry_price")
            if price_value in (None, "") or (isinstance(price_value, float) and math.isnan(price_value)):
                price_value = record.get("close")
            price_series = pd.to_numeric(pd.Series([price_value]), errors="coerce").fillna(0.0)
            price_f = float(price_series.iloc[0])
            if price_f <= 0:
                self.record_skip_reason("ZERO_QTY", symbol=symbol, detail="invalid_price")
                continue

            base_notional = max(0.0, allocation_pct * max(account_buying_power, 0.0))
            min_order = max(0.0, float(self.config.min_order_usd))
            target_notional = max(base_notional, min_order)
            limit_px = price_f * (1 + limit_buffer_ratio)
            limit_px = max(limit_px, 0.01)
            qty = _compute_qty(account_buying_power, limit_px, allocation_pct, min_order)
            if qty < 1 and self.config.allow_bump_to_one and not self.config.allow_fractional:
                qty = 1 if limit_px > 0 and target_notional >= limit_px else qty
            if qty < 1:
                if not self.config.allow_fractional and target_notional >= min_order > 0:
                    _warn_context(
                        "sizing",
                        f"ZERO_QTY post min-notional symbol={symbol} limit={limit_px:.2f}",
                    )
                    if limit_px > 0 and min_order < limit_px:
                        LOGGER.info(
                            "[INFO] SIZING_HINT symbol=%s required_min_order_usd=%.2f current_min=%.2f",
                            symbol,
                            limit_px,
                            min_order,
                        )
                LOGGER.info(
                    "[INFO] ZERO_QTY_AFTER_BUMP symbol=%s limit=%.2f min_usd=%.2f",
                    symbol,
                    limit_px,
                    min_order,
                )
                self.record_skip_reason("ZERO_QTY", symbol=symbol)
                continue
            LOGGER.debug(
                "CALC symbol=%s price=%.2f bp=%.2f alloc_pct=%.4f slots=%d notional=%.2f limit_px=%.2f qty=%d",
                symbol,
                price_f,
                account_buying_power,
                allocation_pct,
                slots,
                target_notional,
                limit_px,
                qty,
            )

            limit_price = compute_limit_price(record, self.config.entry_buffer_bps)
            limit_price = max(limit_price, limit_px)
            notional = qty * limit_price
            if notional > account_buying_power:
                detail = f"required={notional:.2f} available={account_buying_power:.2f}"
                self.record_skip_reason("CASH", symbol=symbol, detail=detail)
                continue

            if self.config.dry_run:
                self.log_info("DRY_RUN_ORDER", symbol=symbol, qty=qty, limit_price=f"{limit_price:.2f}")
                continue

            outcome = self.execute_order(symbol, qty, limit_price)
            if outcome.get("filled_qty", 0) > 0:
                existing_positions.add(symbol)
                account_buying_power = max(0.0, account_buying_power - notional)
            if outcome.get("submitted"):
                open_order_symbols.add(symbol)

        self.persist_metrics()
        self.log_summary()
        return 0

    def fetch_existing_positions(self) -> set[str]:
        symbols: set[str] = set()
        if self.client is None:
            return symbols
        try:
            log_info("alpaca.get_positions")
            positions = self.client.get_all_positions()
            for pos in positions:
                symbol = getattr(pos, "symbol", "")
                if symbol:
                    symbols.add(symbol.upper())
        except Exception as exc:
            _warn_context("alpaca.get_positions", str(exc))
        return symbols

    @staticmethod
    def _coerce_buying_power(value: Any) -> Optional[float]:
        if value in (None, "", False):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            try:
                numeric = float(str(value))
            except Exception:
                return None
        if math.isnan(numeric):
            return None
        return numeric

    def fetch_open_order_symbols(self) -> set[str]:
        symbols: set[str] = set()
        if self.client is None:
            return symbols
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            log_info("alpaca.get_orders", status=request.status)
            orders = self.client.get_orders(request)
            for order in orders or []:
                symbol = getattr(order, "symbol", "")
                if symbol:
                    symbols.add(symbol.upper())
        except Exception as exc:
            _warn_context("alpaca.get_orders", str(exc))
        return symbols

    def fetch_buying_power(self) -> float:
        def _extract(snapshot: Mapping[str, Any], source: str) -> Optional[float]:
            snapshot_map = dict(snapshot)
            for field in ("cash", "buying_power", "non_marginable_buying_power"):
                raw_value = snapshot_map.get(field)
                parsed = self._coerce_buying_power(raw_value)
                if parsed is None:
                    continue
                self._last_buying_power_raw = raw_value
                if field != "buying_power" and "buying_power" not in snapshot_map:
                    snapshot_map["buying_power"] = raw_value
                if field != "cash":
                    LOGGER.info(
                        "[INFO] BUYING_POWER_FALLBACK source=%s field=%s", source, field
                    )
                self.account_snapshot = snapshot_map
                return parsed
            return None

        if isinstance(self.account_snapshot, Mapping):
            parsed_snapshot = _extract(self.account_snapshot, "snapshot")
            if parsed_snapshot is not None:
                return parsed_snapshot
        if self.client is None:
            _warn_context("alpaca.buying_power", "client_unavailable")
            return 0.0
        try:
            log_info("alpaca.get_account")
            account = self.client.get_account()
        except Exception as exc:
            _warn_context("alpaca.get_account", str(exc))
            return 0.0
        snapshot: Dict[str, Any] = {}
        for field in ("cash", "buying_power", "non_marginable_buying_power"):
            buying_power = getattr(account, field, None)
            snapshot[field] = buying_power
        parsed = _extract(snapshot, "account")
        if parsed is not None:
            return parsed
        _warn_context("alpaca.buying_power", "Unable to fetch buying power")
        self.account_snapshot = snapshot
        return 0.0

    def execute_order(self, symbol: str, qty: int, limit_price: float) -> Dict[str, Any]:
        outcome: Dict[str, Any] = {"submitted": False, "filled_qty": 0.0}
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type="limit",
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=self.config.extended_hours,
        )
        submitted_order = self.submit_with_retries(order_request)
        if submitted_order is None:
            return outcome
        outcome["submitted"] = True
        order_id = str(getattr(submitted_order, "id", ""))
        self.metrics.orders_submitted += 1
        extended_hours_flag = "true" if self.config.extended_hours else "false"
        self.log_info(
            "BUY_SUBMIT",
            symbol=symbol,
            qty=str(qty),
            limit=f"{limit_price:.2f}",
            extended_hours=extended_hours_flag,
            order_id=order_id or "",
        )

        fill_deadline = datetime.now(timezone.utc) + timedelta(minutes=self.config.cancel_after_min)
        submit_ts = time.time()

        filled_qty = 0.0
        filled_avg_price = None
        status = "open"

        while datetime.now(timezone.utc) < fill_deadline:
            try:
                log_info("alpaca.get_order", order_id=order_id)
                order = self.client.get_order_by_id(order_id)
            except Exception as exc:
                _warn_context("alpaca.get_order", f"{order_id}: {exc}")
                break
            status = str(getattr(order, "status", "")).lower()
            filled_qty = float(getattr(order, "filled_qty", filled_qty) or 0)
            filled_avg_price = getattr(order, "filled_avg_price", filled_avg_price)
            if status == "filled":
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_info(
                    "BUY_FILL",
                    symbol=symbol,
                    filled_qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
                outcome["filled_qty"] = filled_qty
                return outcome
            if status in {"canceled", "expired", "rejected"}:
                break
            if filled_qty >= qty:
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_info(
                    "BUY_FILL",
                    symbol=symbol,
                    filled_qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
                outcome["filled_qty"] = filled_qty
                return outcome
            self.sleep(5)

        # Deadline reached or not fully filled
        remaining = max(qty - filled_qty, 0)
        if remaining > 0:
            self.cancel_order(order_id, symbol)
            self.metrics.orders_canceled += 1
            self.log_info(
                "BUY_CANCELLED",
                symbol=symbol,
                remaining_qty=f"{remaining:.0f}",
                status=status,
                order_id=order_id,
            )
        if filled_qty > 0:
            latency = time.time() - submit_ts
            self.metrics.orders_filled += 1
            self.metrics.record_latency(latency)
            self.log_info(
                "BUY_FILL",
                symbol=symbol,
                filled_qty=f"{filled_qty:.0f}",
                avg_price=f"{float(filled_avg_price or 0):.2f}",
                partial="true",
                order_id=order_id,
            )
            self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
            outcome["filled_qty"] = filled_qty
        return outcome

    def cancel_order(self, order_id: str, symbol: str) -> None:
        if self.client is None or not order_id:
            return
        try:
            if hasattr(self.client, "cancel_order_by_id"):
                log_info("alpaca.cancel_order", order_id=order_id)
                self.client.cancel_order_by_id(order_id)
            elif hasattr(self.client, "cancel_order"):
                log_info("alpaca.cancel_order", order_id=order_id)
                self.client.cancel_order(order_id)
            else:  # pragma: no cover - defensive fallback
                LOGGER.warning("Client has no cancel method; unable to cancel %s", order_id)
        except Exception as exc:
            _warn_context("alpaca.cancel_order", f"{order_id}: {exc}")

    def attach_trailing_stop(self, symbol: str, qty: float, avg_price: Optional[Any]) -> None:
        if qty <= 0 or self.client is None:
            return
        qty_int = int(qty) if int(qty) == qty else math.floor(qty)
        if qty_int <= 0:
            return
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty_int,
            side=OrderSide.SELL,
            trail_percent=self.config.trailing_percent,
            time_in_force=TimeInForce.GTC,
        )
        trail_value = self.config.trailing_percent
        if float(trail_value).is_integer():
            trail_display = str(int(trail_value))
        else:
            trail_display = f"{trail_value:g}"
        LOGGER.info(
            "TRAIL_SUBMIT symbol=%s trail_pct=%s route=trailing_stop",
            symbol,
            trail_display,
        )
        self.log_info(
            "TRAIL_SUBMIT",
            symbol=symbol,
            trail_pct=trail_display,
            route="trailing_stop",
        )
        trailing_order = self.submit_with_retries(request)
        if trailing_order is None:
            LOGGER.error("[ERROR] TRAIL_FAILED symbol=%s reason=submit_failed", symbol)
            self.log_info("TRAIL_FAILED", symbol=symbol, reason="submit_failed")
            return
        self.metrics.trailing_attached += 1
        order_id = str(getattr(trailing_order, "id", ""))
        LOGGER.info(
            "TRAIL_CONFIRMED symbol=%s qty=%s order_id=%s",
            symbol,
            qty_int,
            order_id,
        )
        self.log_info(
            "TRAIL_CONFIRMED",
            symbol=symbol,
            qty=str(qty_int),
            order_id=order_id,
        )

    def submit_with_retries(self, request: Any) -> Optional[Any]:
        if self.client is None:
            return None
        attempts = 3
        backoff = 1.5
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                log_info("alpaca.submit_order", attempt=attempt)
                if isinstance(request, LimitOrderRequest):
                    result = self.client.submit_order(request)
                else:
                    result = self.client.submit_order(request)
                if attempt > 1:
                    self.log_info("API_RETRY_SUCCESS", attempt=attempt)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    self.metrics.api_retries += 1
                    self.log_info("API_RETRY", attempt=attempt, error=str(exc))
                    self.sleep(delay)
                    delay *= backoff
                    continue
                self.metrics.api_failures += 1
                _warn_context("alpaca.submit_order", f"failed: {exc}")
        if last_error is not None:
            self.log_info("API_FAILURE", error=str(last_error))
        return None

    def log_summary(self) -> None:
        skips = {key.upper(): int(value) for key, value in self.metrics.skipped_reasons.items()}
        payload: Dict[str, Any] = {
            "orders_submitted": self.metrics.orders_submitted,
            "trailing_attached": self.metrics.trailing_attached,
        }
        for key in SKIP_REASON_ORDER:
            payload[f"skips.{key}"] = int(skips.get(key, 0))
        extra_keys = sorted(key for key in skips.keys() if key not in SKIP_REASON_KEYS)
        for key in extra_keys:
            payload[f"skips.{key}"] = int(skips.get(key, 0))
        log_info("EXECUTE_SUMMARY", **payload)

    def persist_metrics(self) -> None:
        payload = self.metrics.as_dict()
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if METRICS_PATH.exists():
            try:
                existing_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive merge
                _warn_context("metrics.persist", f"decode_failed: {exc}")
                existing_payload = {}
            if isinstance(existing_payload, Mapping):
                existing = dict(existing_payload)
        merged = dict(existing)
        merged.update(payload)
        existing_skips = existing.get("skips") if isinstance(existing.get("skips"), Mapping) else {}
        payload_skips = payload.get("skips") if isinstance(payload.get("skips"), Mapping) else {}
        skips_merged = dict(existing_skips)
        skips_merged.update(payload_skips)
        merged["skips"] = skips_merged
        try:
            METRICS_PATH.write_text(
                json.dumps(merged, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            _warn_context("metrics.persist", f"write_failed: {exc}")
            return
        log_info("METRICS_UPDATED", path=str(METRICS_PATH))


def configure_logging(log_json: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stream_handler)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handlers.append(file_handler)

    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
    for handler in handlers:
        if log_json:
            handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _create_trading_client() -> tuple[Any, str, bool, bool | None]:
    if TradingClient is None:
        raise RuntimeError("alpaca-py TradingClient is unavailable")
    api_key, api_secret, base_url, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials for trading client")
    forced_raw = os.getenv("JBR_EXEC_PAPER")
    forced_mode: bool | None = None
    paper_mode: bool
    if forced_raw is not None:
        forced_mode = forced_raw.strip().lower() in {"1", "true", "yes", "on"}
        paper_mode = forced_mode
    else:
        env = (base_url or "paper").lower()
        paper_mode = "live" not in env
    resolved_base = (base_url or "https://paper-api.alpaca.markets").rstrip("/")
    if not resolved_base:
        resolved_base = "https://paper-api.alpaca.markets"
    client = TradingClient(api_key, api_secret, paper=paper_mode)
    return client, resolved_base, paper_mode, forced_mode


def _ensure_trading_auth(
    base_url: str, creds_snapshot: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        body = "missing credentials"
        LOGGER.error("[ERROR] ALPACA_AUTH_FAILED status=MISSING body=%s", body)
        _record_auth_error("unauthorized", creds_snapshot)
        raise AlpacaAuthFailure("missing credentials")

    resolved_base = (base_url or "https://paper-api.alpaca.markets").rstrip("/")
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    def fetch(path: str) -> Mapping[str, Any]:
        url = f"{resolved_base}{path}"
        context = f"alpaca.auth{path}"
        log_info(context)
        try:
            response = requests.get(url, headers=headers, timeout=10)
        except Exception as exc:
            body_text = str(exc)
            _warn_context(context, body_text)
            LOGGER.error(
                "[ERROR] ALPACA_AUTH_FAILED status=ERR body=%s endpoint=%s",
                body_text,
                path,
            )
            _record_auth_error("unauthorized", creds_snapshot)
            raise AlpacaAuthFailure(body_text) from exc
        if response.status_code != 200:
            raw_text = getattr(response, "text", "")
            body_text = (raw_text or "")[:500].replace("\n", " ")
            _warn_context(context, f"status={response.status_code}")
            LOGGER.error(
                "[ERROR] ALPACA_AUTH_FAILED status=%s body=%s endpoint=%s",
                response.status_code,
                body_text,
                path,
            )
            _record_auth_error("unauthorized", creds_snapshot)
            raise AlpacaAuthFailure(f"status={response.status_code} body={body_text}")
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return payload if isinstance(payload, Mapping) else {}

    try:
        account_payload = fetch("/v2/account")
        clock_payload = fetch("/v2/clock")
        return account_payload, clock_payload
    except AlpacaAuthFailure as exc:
        status_hint = str(exc) or "unknown"
        LOGGER.error(
            "[ERROR] TRADING_AUTH_FAILED base_url=%s status=%s", resolved_base, status_hint
        )
        LOGGER.error(
            "Reload ~/.config/jbravo/.env or export APCA_API_KEY_ID/APCA_API_SECRET_KEY"
        )
        raise SystemExit(2) from exc


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades for pipeline candidates")
    parser.add_argument("--source", type=Path, default=ExecutorConfig.source, help="Path to the candidate CSV file")
    parser.add_argument(
        "--allocation-pct",
        type=float,
        default=ExecutorConfig.allocation_pct,
        help="Fraction of buying power allocated per position (0-1)",
    )
    parser.add_argument(
        "--min-order-usd",
        type=float,
        default=ExecutorConfig.min_order_usd,
        help="Minimum USD notional target per slot before rounding",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=ExecutorConfig.max_positions,
        help="Maximum concurrent positions the executor will open",
    )
    parser.add_argument(
        "--entry-buffer-bps",
        type=int,
        default=ExecutorConfig.entry_buffer_bps,
        help="Entry buffer in basis points added to the reference price",
    )
    parser.add_argument(
        "--limit-buffer-pct",
        type=float,
        default=ExecutorConfig.limit_buffer_pct,
        help="Percent tolerance applied to price guards (default 1.0)",
    )
    parser.add_argument(
        "--trailing-percent",
        type=float,
        default=ExecutorConfig.trailing_percent,
        help="Percent trail for the protective stop order",
    )
    parser.add_argument(
        "--cancel-after-min",
        type=int,
        default=ExecutorConfig.cancel_after_min,
        help="Minutes after regular market open to cancel unfilled orders",
    )
    parser.add_argument(
        "--extended-hours",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.extended_hours,
        help="Whether to submit orders eligible for extended hours",
    )
    parser.add_argument(
        "--dry-run",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.dry_run,
        help="If true, only log intended actions without submitting orders",
    )
    parser.add_argument(
        "--allow-bump-to-one",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.allow_bump_to_one,
        help="Allow bumping position size to one share when sizing rounds to zero",
    )
    parser.add_argument(
        "--allow-fractional",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.allow_fractional,
        help="Allow fractional share orders when full shares are not affordable",
    )
    parser.add_argument(
        "--min-adv20",
        type=int,
        default=ExecutorConfig.min_adv20,
        help="Minimum 20-day average dollar volume required",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=ExecutorConfig.min_price,
        help="Minimum allowed price for candidates",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=ExecutorConfig.max_price,
        help="Maximum allowed price for candidates",
    )
    parser.add_argument(
        "--log-json",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.log_json,
        help="Emit structured JSON logs in addition to human readable ones",
    )
    parser.add_argument(
        "--time-window",
        choices=("premarket", "regular", "any", "auto"),
        default=ExecutorConfig.time_window,
        help="Trading time window gate controlling when orders may be submitted",
    )
    parser.add_argument(
        "--market-tz",
        "--market-timezone",
        dest="market_timezone",
        default=ExecutorConfig.market_timezone,
        help="IANA timezone name used for market window evaluation",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    market_timezone = args.market_timezone or ExecutorConfig.market_timezone
    return ExecutorConfig(
        source=args.source,
        allocation_pct=args.allocation_pct,
        max_positions=args.max_positions,
        entry_buffer_bps=args.entry_buffer_bps,
        trailing_percent=args.trailing_percent,
        limit_buffer_pct=args.limit_buffer_pct,
        cancel_after_min=args.cancel_after_min,
        extended_hours=args.extended_hours,
        dry_run=args.dry_run,
        min_order_usd=args.min_order_usd,
        allow_bump_to_one=args.allow_bump_to_one,
        allow_fractional=args.allow_fractional,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        log_json=args.log_json,
        time_window=args.time_window,
        market_timezone=market_timezone,
    )


def run_executor(
    config: ExecutorConfig,
    *,
    client: Optional[Any] = None,
    creds_snapshot: Mapping[str, Any] | None = None,
) -> int:
    configure_logging(config.log_json)
    metrics = ExecutionMetrics()
    loader = TradeExecutor(config, client, metrics)
    if config.dry_run:
        banner = "=" * 72
        LOGGER.info(banner)
        LOGGER.info(
            "[INFO] DRY_RUN=True — no orders will be submitted, but still perform sizing and emit skip reasons"
        )
        LOGGER.info(banner)
    try:
        frame = loader.load_candidates()
    except CandidateLoadError as exc:
        LOGGER.error("%s", exc)
        metrics.api_failures += 1
        loader.persist_metrics()
        return 1

    candidates_df = frame
    configured_window = config.time_window or "auto"
    win, in_window, now_ny = _resolve_time_window(configured_window)
    trading_probe = client if client is not None else None
    clock_hhmm = now_ny.strftime("%H:%M")
    clock_window = win if win in {"premarket", "regular", "any"} else configured_window
    if trading_probe is not None and hasattr(trading_probe, "get_clock"):
        clock_allowed, hhmm = _clock_is_in_window(trading_probe, clock_window)
        if hhmm != "00:00":
            clock_hhmm = hhmm
        if clock_window == "any":
            in_window = clock_allowed
        elif clock_window in {"premarket", "regular"}:
            in_window = bool(in_window) and clock_allowed
        else:
            in_window = bool(in_window) and clock_allowed
        if hhmm == "00:00":
            log_info(
                "MARKET_TIME",
                ny_now=now_ny.isoformat(),
                window=clock_window,
                in_window=bool(in_window),
            )
    else:
        log_info(
            "MARKET_TIME",
            ny_now=now_ny.isoformat(),
            window=clock_window,
            in_window=bool(in_window),
        )

    log_info(
        "EXEC_START",
        dry_run=bool(config.dry_run),
        time_window=configured_window,
        resolved=win,
        ny_now=now_ny.isoformat(),
        hhmm=clock_hhmm,
        in_window=bool(in_window),
        candidates=len(candidates_df),
    )
    acc = None
    acc_err = None
    if trading_probe is not None and hasattr(trading_probe, "get_account"):
        acc, acc_err = _log_call("get_account", trading_probe.get_account)
    buying_power = 0.0
    if acc and hasattr(acc, "buying_power"):
        try:
            buying_power = float(acc.buying_power or 0)
        except Exception:
            buying_power = 0.0
    log_info(
        "AUTH_STATUS",
        ok=acc is not None and acc_err is None,
        buying_power=f"{buying_power:.2f}",
    )

    if not in_window:
        log_info(
            "TIME_WINDOW",
            window=win,
            status="outside",
            count=len(candidates_df),
        )
        metrics.skips["TIME_WINDOW"] += len(candidates_df)
        metrics.flush()
        loader.persist_metrics()
        loader.log_summary()
        return 0

    _log_account_probe(creds_snapshot)

    if candidates_df.empty:
        loader.metrics.record_skip("NO_CANDIDATES", count=1)
        loader.persist_metrics()
        return 0

    records = loader.hydrate_candidates(candidates_df)
    filtered = loader.guard_candidates(records)

    account_payload: Optional[Mapping[str, Any]] = None
    clock_payload: Optional[Mapping[str, Any]] = None
    if client is not None:
        trading_client = client
        base_url = os.getenv("APCA_API_BASE_URL", "")
        paper_mode = None
    elif config.dry_run or not filtered:
        trading_client = None
        base_url = ""
        paper_mode = None
    else:
        trading_client, base_url, paper_mode, forced_mode = _create_trading_client()
        forced_label = forced_mode if forced_mode is not None else "auto"
        LOGGER.info(
            "[INFO] TRADING_MODE base=%s paper_mode=%s forced=%s",
            base_url or "",
            bool(paper_mode),
            forced_label,
        )
    _paper_only_guard(trading_client, base_url)
    if trading_client is not None and client is None:
        try:
            account_payload, clock_payload = _ensure_trading_auth(base_url or "", creds_snapshot or {})
        except AlpacaAuthFailure:
            metrics.api_failures += 1
            loader.persist_metrics()
            return 2

    executor = TradeExecutor(
        config,
        trading_client,
        metrics,
        base_url=base_url or "",
        account_snapshot=account_payload,
        clock_snapshot=clock_payload,
    )
    return executor.execute(candidates_df, prefiltered=filtered)


def load_candidates(path: Path) -> pd.DataFrame:
    config = ExecutorConfig(source=path)
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    return executor.load_candidates()


def apply_guards(df: pd.DataFrame, config: ExecutorConfig, metrics: ExecutionMetrics) -> pd.DataFrame:
    executor = TradeExecutor(config, None, metrics)
    records = executor.hydrate_candidates(df)
    filtered = executor.guard_candidates(records)
    if not filtered:
        return df.iloc[0:0].copy()
    return pd.DataFrame(filtered)


def _bootstrap_env() -> list[str]:
    global _ENV_FILES_LOADED
    loaded_files, missing = load_env(REQUIRED_ENV_KEYS)
    _ENV_FILES_LOADED = list(loaded_files)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    missing_keys: list[str] = []
    try:
        files_repr = f"[{', '.join(loaded_files)}]" if loaded_files else "[]"
        LOGGER.info("[INFO] ENV_LOADED files=%s", files_repr)
        if missing:
            missing_keys = list(missing)
            LOGGER.error("[ERROR] ENV_MISSING_KEYS=%s", f"[{', '.join(missing_keys)}]")
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    if missing_keys:
        raise SystemExit(2)
    return loaded_files


def main(argv: Optional[Iterable[str]] = None) -> int:
    _bootstrap_env()
    try:
        creds_snapshot = assert_alpaca_creds()
    except AlpacaCredentialsError as exc:
        missing = list(dict.fromkeys(list(exc.missing) + list(exc.whitespace)))
        LOGGER.error(
            "[ERROR] ALPACA_CREDENTIALS_INVALID reason=%s missing=%s whitespace=%s sanitized=%s",
            exc.reason,
            ",".join(exc.missing) or "",
            ",".join(exc.whitespace) or "",
            json.dumps(exc.sanitized, sort_keys=True),
        )
        _record_auth_error(exc.reason, exc.sanitized, missing)
        return 2

    LOGGER.info(
        "[INFO] ALPACA_CREDENTIALS_OK sanitized=%s",
        json.dumps(creds_snapshot, sort_keys=True),
    )

    args = parse_args(argv)
    LOGGER.info(
        "[INFO] EXEC_CONFIG ext_hours=%s alloc=%.2f max_pos=%d trail_pct=%.1f",
        args.extended_hours,
        args.allocation_pct,
        args.max_positions,
        args.trailing_percent,
    )
    config = build_config(args)
    try:
        rc = run_executor(config, creds_snapshot=creds_snapshot)
        metrics_payload = _load_execute_metrics()
        if metrics_payload is not None:
            def _as_int(value: Any) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            LOGGER.info(
                "EXECUTE END submitted=%d trailing=%d skips=%s",
                _as_int(metrics_payload.get("orders_submitted")),
                _as_int(metrics_payload.get("trailing_attached")),
                metrics_payload.get("skips", {}),
            )
        return rc
    except AlpacaUnauthorizedError as exc:
        LOGGER.error(
            '[ERROR] ALPACA_UNAUTHORIZED endpoint=%s feed=%s hint="check keys/base urls"',
            exc.endpoint or "",
            exc.feed or "",
        )
        _record_auth_error("unauthorized", creds_snapshot)
        return 2
    except Exception as exc:  # pragma: no cover - top-level guard
        LOGGER.exception("Executor failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
def _warn_context(context: str, message: str) -> None:
    LOGGER.warning("[WARN] %s: %s", context, message)


