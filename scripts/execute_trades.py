# --- Minimal, import-proof telemetry header (NO project imports) ---
import os, sys, subprocess, json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def emit(evt: str, **kvs):
    """Write an event via bootstrap emitter; never throw."""
    cmd = [sys.executable, "-m", "bin.emit_event", evt]
    for k, v in kvs.items():
        cmd.append(f"{k}={v}")
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        pass


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# Import sentinel to prove this module loaded
emit("IMPORT_SENTINEL", component="execute_trades")

import argparse
import logging
import pandas as pd
import secrets
from datetime import timedelta, time
import pytz
from time import sleep
from typing import Optional
from decimal import Decimal, ROUND_HALF_UP
from statistics import median, quantiles
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    GetOrdersRequest,
    TrailingStopOrderRequest,
    CancelOrderResponse,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
# Alerting
import requests
from .utils import cache_bars

from .exit_signals import should_exit_early
from .monitor_positions import log_trade_exit
from utils.env import load_env, get_alpaca_creds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logfile = os.path.join(LOG_DIR, "execute_trades.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

load_env()
start_time = datetime.utcnow()
logger.info("Trade execution script started.")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(message: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)

API_KEY, API_SECRET, BASE_URL, _ = get_alpaca_creds()


def detect_trading_env() -> str:
    """Detect the Alpaca trading environment (paper or live)."""

    env = os.environ.get("TRADING_ENV")
    if env:
        env = env.lower()
        if env in ("paper", "live"):
            return env
    base_url = BASE_URL or os.environ.get("ALPACA_BASE_URL", "")
    return "paper" if base_url and "paper" in base_url.lower() else "live"


TRADING_ENV = detect_trading_env()

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca credentials")

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=TRADING_ENV == "paper")
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Constants
# Maximum number of concurrent open trades allowed
MAX_OPEN_TRADES = 10
ALLOC_PERCENT = 0.03  # Changed allocation to 3%
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7

# Order handling constants
ORDER_POLL_INTERVAL = 10  # seconds between status checks
ORDER_TIMEOUT_SECONDS = 120  # total time to wait before cancelling and retrying
MAX_ORDER_RETRIES = 3  # maximum number of retries per symbol

# Structured logging helpers -------------------------------------------------


def log_trailing_stop_event(symbol: str, trail_percent: float, order_id: Optional[str], status: str) -> None:
    """Emit a structured log entry for trailing-stop attachments."""

    payload = {
        "symbol": symbol,
        "trail_percent": trail_percent,
        "order_id": order_id,
        "status": status,
    }
    logger.info("TRAILING_STOP_ATTACH %s", payload)


def log_exit_submit(symbol: str, qty: int, order_type: str, reason_code: str, side: str = "sell") -> None:
    """Emit the EXIT_SUBMIT log required for downstream monitoring."""

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "reason_code": reason_code,
    }
    logger.info("EXIT_SUBMIT %s", payload)


def log_exit_final(status: str, latency_ms: int) -> None:
    """Emit the EXIT_FINAL log when an exit order reaches a terminal state."""

    payload = {
        "status": status,
        "latency_ms": latency_ms,
    }
    logger.info("EXIT_FINAL %s", payload)
    order_latencies_ms.append(latency_ms)


# Directory used for caching market history
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "history_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Runtime metrics recorded for dashboard display
metrics = {
    "symbols_processed": 0,
    "orders_submitted": 0,
    "symbols_skipped": 0,  # backward compatible total skips
    "orders_skipped_existing_positions": 0,
    "orders_skipped_pending_orders": 0,
    "orders_skipped_existing_position": 0,
    "orders_skipped_pending_order": 0,
    "orders_skipped_risk_limit": 0,
    "orders_skipped_market_data": 0,
    "orders_skipped_session_window": 0,
    "orders_skipped_duplicate_candidate": 0,
    "orders_skipped_other": 0,
    "api_retries": 0,
    "api_failures": 0,
    "skips_time_window": 0,
}
skip_reason_counts: Counter[str] = Counter()

CANDIDATE_SOURCE_DEFAULT = os.path.join(BASE_DIR, "data", "latest_candidates.csv")
CANDIDATE_SOURCE_PATH = CANDIDATE_SOURCE_DEFAULT
DRY_RUN = False


def resolve_candidates_path(source: Optional[str] = None) -> Path:
    """Return an absolute path for the candidates CSV."""

    raw = source or CANDIDATE_SOURCE_PATH
    candidate_path = Path(raw)
    if not candidate_path.is_absolute():
        candidate_path = Path(BASE_DIR) / candidate_path
    return candidate_path

VALID_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}


def sanitize_candidate_row(row: dict) -> dict:
    """Normalize symbol/exchange data sourced from screener outputs."""

    symbol = (row.get("symbol") or "").strip().upper()
    exchange = (row.get("exchange") or "").strip().upper()
    if not exchange or exchange not in VALID_EXCHANGES:
        exchange = "UNKNOWN"
    row["symbol"] = symbol
    row["exchange"] = exchange
    return row


LEGACY_SKIP_METRIC_ALIASES = {
    "orders_skipped_existing_positions": (
        "orders_skipped_existing_positions",
        "orders_skipped_existing_position",
    ),
    "orders_skipped_pending_orders": (
        "orders_skipped_pending_orders",
        "orders_skipped_pending_order",
    ),
    "orders_skipped_risk_limits": ("orders_skipped_risk_limits", "orders_skipped_risk_limit"),
    "orders_skipped_market_data": ("orders_skipped_market_data",),
    "orders_skipped_session": ("orders_skipped_session", "orders_skipped_session_window"),
    "orders_skipped_duplicate": (
        "orders_skipped_duplicate",
        "orders_skipped_duplicate_candidate",
    ),
    "orders_skipped_other": ("orders_skipped_other",),
}
metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")
order_latencies_ms: list[int] = []
retry_backoff_ms_total = 0

COMPONENT_NAME = "execute_trades"


def log_event(event: dict) -> None:
    payload = {"component": COMPONENT_NAME}
    payload.update(event)
    evt = payload.pop("event", None)
    if not evt:
        return
    emit(evt, **{k: str(v) for k, v in payload.items()})


def inc(metric: str, by: int = 1) -> None:
    """Increment ``metric`` in the global metrics dictionary."""

    current = metrics.get(metric, 0)
    metrics[metric] = current + by


def _resolve_metric_alias(*names: str) -> int:
    """Return the first non-zero metric value among ``names``."""

    for name in names:
        value = metrics.get(name, 0)
        if value:
            return value
    return metrics.get(names[0], 0) if names else 0


def build_execute_metrics_snapshot() -> dict[str, int]:
    """Return the metrics payload compatible with the legacy dashboard."""

    snapshot = {
        "symbols_processed": metrics.get("symbols_processed", 0),
        "orders_submitted": metrics.get("orders_submitted", 0),
        "symbols_skipped": metrics.get("symbols_skipped", 0),
        "api_retries": metrics.get("api_retries", 0),
        "api_failures": metrics.get("api_failures", 0),
    }

    for legacy_key, aliases in LEGACY_SKIP_METRIC_ALIASES.items():
        snapshot[legacy_key] = _resolve_metric_alias(*aliases)

    latency_samples = order_latencies_ms
    if latency_samples:
        order_latency_ms_p50 = int(median(latency_samples))
        if len(latency_samples) == 1:
            order_latency_ms_p95 = order_latency_ms_p50
        else:
            order_latency_ms_p95 = int(
                round(quantiles(latency_samples, n=100, method="inclusive")[94])
            )
    else:
        order_latency_ms_p50 = 0
        order_latency_ms_p95 = 0

    snapshot.update(
        {
            "order_latency_ms_p50": order_latency_ms_p50,
            "order_latency_ms_p95": order_latency_ms_p95,
            "retry_backoff_ms_sum": retry_backoff_ms_total,
        }
    )

    if metrics.get("run_aborted_reason"):
        snapshot["run_aborted_reason"] = metrics["run_aborted_reason"]
    snapshot["skips_time_window"] = metrics.get("skips_time_window", 0)

    return snapshot


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def persist_execute_metrics() -> dict[str, object]:
    """Persist the current execution metrics snapshot to disk."""

    snapshot = build_execute_metrics_snapshot()
    payload: dict[str, object] = {
        "generated_at_utc": _current_utc_iso(),
        "submitted": {
            "orders": snapshot.get("orders_submitted", 0),
            "symbols": snapshot.get("symbols_processed", 0),
        },
        "skipped": {
            "total": snapshot.get("symbols_skipped", 0),
            "by_reason": dict(sorted(skip_reason_counts.items())),
        },
        "errors": {
            "api_failures": snapshot.get("api_failures", 0),
        },
        "legacy": snapshot,
    }

    if metrics.get("run_aborted_reason"):
        payload["run_aborted_reason"] = metrics["run_aborted_reason"]

    summary = {
        "orders_submitted": int(metrics.get("orders_submitted", 0)),
        "skips_existing_position": int(skip_reason_counts.get("EXISTING_POSITION", 0)),
        "skips_pending_order": int(skip_reason_counts.get("PENDING_ORDER", 0)),
        "skips_time_window": int(
            metrics.get("skips_time_window", 0)
            or skip_reason_counts.get("SESSION_WINDOW", 0)
        ),
        "api_failures": int(metrics.get("api_failures", 0)),
        "api_retries": int(metrics.get("api_retries", 0)),
    }
    payload.update(summary)

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("[METRICS] Execution results: %s", payload)
    return payload


def log_order_submit_event(
    symbol: str,
    side: str,
    qty: int,
    limit_price: float,
    attempt: int,
    session: str,
) -> None:
    """Log an ``ORDER_SUBMIT`` event with the provided context."""

    log_event(
        {
            "event": "ORDER_SUBMIT",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "limit_price": limit_price,
            "attempt": attempt,
            "session": session,
        }
    )


def log_order_final_event(
    symbol: str,
    attempt: int,
    status: str,
    latency_ms: int,
    filled_avg_price: Optional[float] = None,
) -> None:
    """Log an ``ORDER_FINAL`` event with the final status of the order."""

    event = {
        "event": "ORDER_FINAL",
        "symbol": symbol,
        "attempt": attempt,
        "status": status,
        "latency_ms": latency_ms,
    }
    if filled_avg_price is not None:
        event["filled_avg_price"] = float(filled_avg_price)
    log_event(event)
    order_latencies_ms.append(latency_ms)


def log_retry_event(
    phase: str,
    symbol: str,
    attempt: int,
    reason_code: str,
    backoff_ms: int,
) -> None:
    """Increment retry metrics and log a ``RETRY`` event."""

    inc("api_retries")
    global retry_backoff_ms_total
    retry_backoff_ms_total += backoff_ms
    log_event(
        {
            "event": "RETRY",
            "phase": phase,
            "symbol": symbol,
            "attempt": attempt,
            "reason_code": reason_code,
            "backoff_ms": backoff_ms,
        }
    )


def log_api_error_event(phase: str, symbol: str, message: str) -> None:
    """Increment failure metrics and log an ``API_ERROR`` event."""

    inc("api_failures")
    log_event(
        {
            "event": "API_ERROR",
            "phase": phase,
            "symbol": symbol,
            "message": message,
        }
    )


def skip(symbol: str, code: str, reason: str, **kvs) -> None:
    """Record a skipped candidate and log the associated event."""

    inc("symbols_skipped")
    inc(f"orders_skipped_{code.lower()}")
    normalized_code = code.strip().upper() or "UNKNOWN"
    skip_reason_counts[normalized_code] += 1
    if normalized_code == "SESSION_WINDOW":
        inc("skips_time_window")
    log_event(
        {
            "event": "CANDIDATE_SKIPPED",
            "symbol": symbol,
            "reason_code": code,
            "reason_text": reason,
            "kvs": kvs,
        }
    )


def load_cached_prices(symbols: list[str], cache_dir: str = os.path.join(BASE_DIR, "data", "history_cache")) -> dict[str, float]:
    """Return a mapping of symbol to the last cached close price."""
    prices: dict[str, float] = {}
    for sym in symbols:
        path = Path(cache_dir) / f"{sym}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "close" in df.columns and not df.empty:
                    prices[sym] = float(df["close"].iloc[-1])
                else:
                    logger.warning("No 'close' data found in cache for %s.", sym)
            except Exception as exc:  # pragma: no cover - unexpected read errors
                logger.error("Failed reading cache for %s: %s", sym, exc)
        else:
            logger.warning("No cache file found for %s.", sym)
    return prices


def round_price(value: float) -> float:
    """Return ``value`` rounded to the nearest cent."""
    return round(value + 1e-6, 2)


def get_latest_price(symbol: str) -> Optional[float]:
    """Return the most recent trade price for ``symbol``."""
    if hasattr(trading_client, "get_stock_latest_trade"):
        try:
            trade = trading_client.get_stock_latest_trade(symbol)
            return float(trade.price)
        except Exception as exc:  # pragma: no cover - API errors
            logger.warning("get_stock_latest_trade failed for %s: %s", symbol, exc)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now(timezone.utc) - timedelta(minutes=5),
            end=datetime.now(timezone.utc),
            feed="iex",
        )
        bars = data_client.get_stock_bars(req).df
        if bars.empty:
            logger.warning("No bars available for %s using IEX feed", symbol)
            return None
    except Exception as exc:  # pragma: no cover - API errors
        logger.error("Failed to fetch latest price for %s via IEX: %s", symbol, exc)
        return None
    if not bars.empty:
        return float(bars["close"].iloc[-1])
    return None


def get_symbol_price(symbol: str, client, fallback_prices: dict[str, float]) -> Optional[float]:
    """Return a recent price for ``symbol`` using multiple fallbacks."""
    now_utc = datetime.utcnow()
    start_time = now_utc - timedelta(minutes=75)
    end_time = now_utc - timedelta(minutes=15)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time,
            feed="iex",
        )
        bars = client.get_stock_bars(request).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.droplevel(0)
        if not bars.empty:
            price = float(bars["close"].iloc[-1])
            logger.info("IEX intraday price for %s: %s", symbol, price)
            return price
        raise ValueError("No intraday bars available from IEX.")
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("IEX intraday data failed for %s: %s", symbol, exc)

    try:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
        trade_resp = client.get_stock_latest_trade(req)
        trade_price = (
            trade_resp[symbol].price if isinstance(trade_resp, dict) else getattr(trade_resp, "price", None)
        )
        if trade_price:
            logger.info("Latest IEX trade price for %s: %s", symbol, trade_price)
            return float(trade_price)
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Latest trade fetch failed for %s: %s", symbol, exc)

    if symbol in fallback_prices:
        cached = fallback_prices[symbol]
        logger.info("Using cached previous close for %s: %s", symbol, cached)
        return cached

    logger.error("No price available for %s. Skipping trade.", symbol)
    return None


def is_extended_hours(now_et: time) -> bool:
    """Return True if ``now_et`` falls in extended trading hours."""
    return time(4, 0) <= now_et < time(9, 30) or now_et >= time(16, 0)


def is_market_open_via_alpaca(trading_env: str) -> tuple[bool, str, Optional[str]]:
    """Return ``(is_open, clock_source, error_message)`` for the market guard."""

    clock_source = "unknown"

    try:
        clock_source = "TradingClient"
        market_clock = trading_client.get_clock()
        is_open = bool(getattr(market_clock, "is_open", False))
        return is_open, clock_source, None
    except Exception as exc:  # pragma: no cover - API errors
        logger.warning("Trading client clock lookup failed: %s", exc)

    try:
        clock_source = "RESTv2"
        import alpaca_trade_api as trade_api  # type: ignore import-not-found

        base_url = BASE_URL or os.environ.get("ALPACA_BASE_URL", "")
        api = trade_api.REST(API_KEY, API_SECRET, base_url, api_version="v2")
        market_clock = api.get_clock()
        is_open = bool(getattr(market_clock, "is_open", False))
        return is_open, clock_source, None
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to determine market status via Alpaca REST API: %s", exc)
        return True, clock_source, str(exc)


def get_clock_open(trading_env: str) -> tuple[bool, str, Optional[str]]:
    """Wrapper to retrieve Alpaca market clock state for telemetry guards."""

    return is_market_open_via_alpaca(trading_env)


def is_market_open(trading_client) -> bool:
    """Return True if the market is currently open."""

    is_open, _, _ = is_market_open_via_alpaca(TRADING_ENV)
    return is_open

# Candidate selection happens dynamically from ``top_candidates.csv``.


# Ensure executed trades file exists
exec_trades_path = os.path.join(BASE_DIR, 'data', 'executed_trades.csv')
if not os.path.exists(exec_trades_path):
    pd.DataFrame(
        columns=[
            "order_id",
            "symbol",
            "qty",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "order_status",
            "net_pnl",
            "order_type",
            "side",
        ]
    ).to_csv(exec_trades_path, index=False)

# Ensure open positions file exists
open_pos_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
if not os.path.exists(open_pos_path):
    pd.DataFrame(
        columns=[
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "entry_price",
            "entry_time",
            "days_in_trade",
            "side",
            "order_status",
            "net_pnl",
            "pnl",
            "order_type",
        ]
    ).to_csv(open_pos_path, index=False)

# Warn if open_positions.csv hasn't been updated recently
max_stale_minutes = 15
if os.path.exists(open_pos_path):
    last_update_time = os.path.getmtime(open_pos_path)
    file_age_minutes = (datetime.now().timestamp() - last_update_time) / 60
    if file_age_minutes > max_stale_minutes:
        logger.warning(
            "open_positions.csv is stale (%.1f minutes old)", file_age_minutes
        )

def get_buying_power():
    acc = trading_client.get_account()
    return float(acc.buying_power)

def get_open_positions():
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}


def get_available_qty(symbol: str) -> int:
    """Return the available quantity for ``symbol`` or 0 on error."""
    try:
        positions = trading_client.get_all_positions()
        for p in positions:
            if p.symbol == symbol:
                return int(getattr(p, "qty_available", p.qty))
    except Exception as exc:
        logger.error("Failed retrieving available qty for %s: %s", symbol, exc)
    return 0



def load_top_candidates(source_path: Optional[str] = None, limit: int = 3) -> pd.DataFrame:
    """Load ranked candidates from the configured source CSV."""

    path = resolve_candidates_path(source_path)
    if not path.exists():
        logger.error("Candidates source missing: %s", path)
        return pd.DataFrame(columns=["symbol", "score", "win_rate", "avg_return", "net_pnl"])

    try:
        candidates_df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read candidates from %s: %s", path, exc)
        return pd.DataFrame(columns=["symbol", "score", "win_rate", "avg_return", "net_pnl"])

    if "symbol" not in candidates_df.columns:
        logger.error("Candidates file missing 'symbol' column: %s", path)
        return pd.DataFrame(columns=["symbol", "score", "win_rate", "avg_return", "net_pnl"])

    candidates_df["symbol"] = (
        candidates_df["symbol"].fillna("").astype(str).str.strip().str.upper()
    )

    # Normalise score-related columns for downstream compatibility
    if "score" not in candidates_df.columns and "Score" in candidates_df.columns:
        candidates_df["score"] = candidates_df["Score"]
    if "win_rate" not in candidates_df.columns:
        if "backtest_win_rate" in candidates_df.columns:
            candidates_df["win_rate"] = candidates_df["backtest_win_rate"]
        else:
            candidates_df["win_rate"] = 0.0
    if "avg_return" not in candidates_df.columns and "backtest_expectancy" in candidates_df.columns:
        candidates_df["avg_return"] = candidates_df["backtest_expectancy"]
    if "net_pnl" not in candidates_df.columns:
        candidates_df["net_pnl"] = candidates_df.get("pnl", 0.0)

    if "exchange" in candidates_df.columns:
        sanitized_records: list[dict] = []
        for record in candidates_df.to_dict("records"):
            record = sanitize_candidate_row(record)
            if record.get("exchange") == "UNKNOWN":
                symbol = record.get("symbol") or "UNKNOWN"
                skip(symbol, "MARKET_DATA", "Unknown exchange")
                continue
            sanitized_records.append(record)
        candidates_df = pd.DataFrame(sanitized_records) if sanitized_records else pd.DataFrame(columns=candidates_df.columns)

    sort_col = "score" if "score" in candidates_df.columns else None
    if sort_col:
        candidates_df = candidates_df.sort_values(sort_col, ascending=False)
    if limit:
        candidates_df = candidates_df.head(limit)

    logger.info("Loaded candidates from %s (rows=%d)", path, len(candidates_df))
    if not candidates_df.empty:
        logger.info("Selected symbols: %s", candidates_df["symbol"].tolist())
    return candidates_df

def save_open_positions_csv():
    """Fetch current open positions from Alpaca and save to CSV."""
    try:
        positions = trading_client.get_all_positions()
        csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
        existing_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

        def get_entry_time(sym: str, default: str) -> str:
            if not existing_df.empty and sym in existing_df.get('symbol', []).values:
                try:
                    return existing_df.loc[existing_df['symbol'] == sym, 'entry_time'].iloc[0]
                except Exception:
                    return default
            return default

        data = []
        for p in positions:
            ts = getattr(p, 'created_at', None)
            entry_iso = ts.isoformat() if ts is not None else 'N/A'
            entry_dt = pd.to_datetime(get_entry_time(p.symbol, entry_iso), utc=True, errors='coerce')
            days_in_trade = (pd.Timestamp.utcnow() - entry_dt).days if not pd.isna(entry_dt) else 0
            data.append({
                'symbol': p.symbol,
                'qty': p.qty,
                'avg_entry_price': p.avg_entry_price,
                'current_price': p.current_price,
                'unrealized_pl': p.unrealized_pl,
                'entry_price': p.avg_entry_price,
                'entry_time': get_entry_time(p.symbol, entry_iso),
                'days_in_trade': days_in_trade,
                'side': getattr(p, 'side', 'long'),
                'order_status': 'open',
                'net_pnl': p.unrealized_pl,
                'pnl': p.unrealized_pl,
                'order_type': getattr(p, 'order_type', 'market'),
            })

        columns = [
            'symbol',
            'qty',
            'avg_entry_price',
            'current_price',
            'unrealized_pl',
            'entry_price',
            'entry_time',
            'days_in_trade',
            'side',
            'order_status',
            'net_pnl',
            'pnl',
            'order_type',
        ]

        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(columns=columns)

        df['side'] = df.get('side', 'long')
        df['order_status'] = df.get('order_status', 'open')
        df['net_pnl'] = df.get('unrealized_pl', 0.0)
        df['pnl'] = df['net_pnl']
        df['order_type'] = df.get('order_type', 'limit')
        df = df[columns]

        csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
        df.to_csv(csv_path, index=False)
        logger.info("Saved open positions to %s", csv_path)
    except Exception as e:
        logger.error("Failed to save open positions: %s", e)

def update_trades_log() -> None:
    """Fetch recent closed orders and append to trades_log.csv."""
    try:
        request = GetOrdersRequest(limit=100)
        orders = trading_client.get_orders(request)
        closed_orders = [
            o
            for o in orders
            if getattr(o, "status", "").lower() in ("filled", "canceled", "rejected")
        ]
        records = []
        for order in closed_orders:
            entry_price = order.filled_avg_price if order.side.value == "buy" else ""
            exit_price = order.filled_avg_price if order.side.value == "sell" else ""

            if order.side.value == 'buy':
                ts = order.filled_at
                entry_time = ts.isoformat() if ts is not None else 'N/A'
            else:
                entry_time = ''

            if order.side.value == 'sell':
                ts = order.filled_at
                exit_time = ts.isoformat() if ts is not None else 'N/A'
            else:
                exit_time = ''

            qty = float(order.filled_qty or 0)
            pnl = (float(exit_price) - float(entry_price)) * qty if exit_price and entry_price else 0.0

            status_val = (
                order.status.value if hasattr(order.status, "value") else order.status
            )
            records.append(
                {
                    "symbol": order.symbol,
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "order_status": status_val,
                    "net_pnl": pnl,
                    "pnl": pnl,
                    "order_type": getattr(order, "order_type", ""),
                    "side": getattr(order.side, "value", order.side),
                }
            )

        df_new = pd.DataFrame(
            records,
            columns=[
                "symbol",
                "qty",
                "entry_price",
                "exit_price",
                "entry_time",
                "exit_time",
                "order_status",
                "net_pnl",
                "pnl",
                "order_type",
                "side",
            ],
        )
        csv_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if existing_df.empty:
                    df = df_new.copy()
                elif df_new.empty:
                    df = existing_df.copy()
                else:
                    df = pd.concat([existing_df, df_new], ignore_index=True)
            except Exception as exc:
                logger.error("Failed reading existing trades log: %s", exc)
                df = df_new
        else:
            df = df_new
        df.to_csv(csv_path, index=False)
        logger.info("Trades log updated successfully.")
    except Exception as e:
        logger.error("Failed to update trades log: %s", e)

def record_executed_trade(
    symbol,
    qty,
    entry_price,
    order_type,
    side,
    order_id="",
    order_status="submitted",
):
    """Append executed trade details to CSV using the unified schema."""

    csv_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
    row = {
        "order_id": order_id,
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": "",
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "exit_time": "",
        "order_status": order_status,
        "net_pnl": 0.0,
        "order_type": order_type,
        "side": side,
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)


def submit_order_with_retry(
    trading_client,
    symbol: str,
    qty: int,
    limit_price: float,
    side: str = "buy",
    retries: int = 0,
):
    """Submit a limit order and retry with timeout if stuck."""

    if retries > MAX_ORDER_RETRIES:
        logger.error("Max retries exceeded for %s. Skipping.", symbol)
        return None, "max_retries"

    attempt = retries + 1
    session_label = "regular"
    log_order_submit_event(symbol, side, qty, limit_price, attempt, session_label)
    submit_ts = datetime.utcnow()
    try:
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_data)
        submit_ts = datetime.utcnow()
        order_id = order.id
        logger.info(
            "[SUBMIT] Order for %s: qty=%s, price=%s, session=regular, order_id=%s",
            symbol,
            qty,
            limit_price,
            order_id,
        )
    except Exception as exc:
        log_api_error_event("submit", symbol, str(exc))
        logger.error("Failed to submit order for %s: %s", symbol, exc)
        return None, "submit_failed"

    start_time = submit_ts
    status = "new"
    final_logged = False
    filled_avg_price = None
    while True:
        try:
            current_order = trading_client.get_order_by_id(order_id)
            status = getattr(current_order, "status", "unknown")
            filled_avg_price = getattr(current_order, "filled_avg_price", None)
            logger.info("Polling order %s for %s, status: %s", order_id, symbol, status)

            if status in ["filled", "partially_filled"]:
                logger.info(
                    "Order %s for %s executed successfully with status %s.",
                    order_id,
                    symbol,
                    status,
                )
                final_ts = datetime.utcnow()
                latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
                log_order_final_event(
                    symbol,
                    attempt,
                    status,
                    latency_ms,
                    filled_avg_price,
                )
                final_logged = True
                break
            if status in ["canceled", "rejected"]:
                logger.warning("Order %s for %s was %s.", order_id, symbol, status)
                final_ts = datetime.utcnow()
                latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
                log_order_final_event(
                    symbol,
                    attempt,
                    status,
                    latency_ms,
                    filled_avg_price,
                )
                final_logged = True
                break
            if (datetime.utcnow() - start_time).total_seconds() > ORDER_TIMEOUT_SECONDS:
                logger.warning(
                    "Order %s for %s stuck in '%s' status for over %ss. Cancelling and retrying.",
                    order_id,
                    symbol,
                    status,
                    ORDER_TIMEOUT_SECONDS,
                )
                try:
                    trading_client.cancel_order_by_id(order_id)
                except Exception as cancel_exc:
                    log_api_error_event("cancel", symbol, str(cancel_exc))
                    logger.error(
                        "[ORDER ERROR] Failed to cancel order %s for %s: %s",
                        order_id,
                        symbol,
                        cancel_exc,
                    )
                    final_ts = datetime.utcnow()
                    latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
                    log_order_final_event(
                        symbol,
                        attempt,
                        "cancel_failed",
                        latency_ms,
                        filled_avg_price,
                    )
                    final_logged = True
                    break
                final_ts = datetime.utcnow()
                latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
                log_order_final_event(
                    symbol,
                    attempt,
                    "timeout",
                    latency_ms,
                    filled_avg_price,
                )
                final_logged = True
                backoff_ms = 5000
                sleep(backoff_ms / 1000)
                logger.info("[RETRY] Attempt #%s for order %s", retries + 1, symbol)
                log_retry_event("poll", symbol, attempt, "timeout", backoff_ms)
                return submit_order_with_retry(
                    trading_client,
                    symbol,
                    qty,
                    limit_price,
                    side,
                    retries + 1,
                )
        except Exception as exc:
            log_api_error_event("poll", symbol, str(exc))
            logger.error("Error polling order %s for %s: %s", order_id, symbol, exc)
            status = "error"
            break

        sleep(ORDER_POLL_INTERVAL)

    if not final_logged:
        final_ts = datetime.utcnow()
        latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
        log_order_final_event(
            symbol,
            attempt,
            status,
            latency_ms,
            filled_avg_price,
        )

    return order_id, status


def submit_new_trailing_stop(symbol: str, qty: int, trail_percent: float) -> None:
    """Submit a trailing stop sell order for ``symbol``."""
    try:
        position = trading_client.get_open_position(symbol)
        entry_price = getattr(position, "avg_entry_price", 0)
    except Exception:
        entry_price = 0
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            trail_percent=trail_percent,
            time_in_force=TimeInForce.GTC,
        )
        order = trading_client.submit_order(request)
        record_executed_trade(
            symbol,
            qty,
            entry_price,
            order_type="trailing_stop",
            side="sell",
            order_id=str(order.id),
        )
        logger.info(
            "Placed trailing stop for %s: qty=%s, trail_pct=%s",
            symbol,
            qty,
            trail_percent,
        )
        log_trailing_stop_event(symbol, trail_percent, str(getattr(order, "id", None)), "submitted")
    except Exception as exc:
        logger.error("Failed to submit trailing stop for %s: %s", symbol, exc)
        log_trailing_stop_event(symbol, trail_percent, None, "error")


def poll_order_until_complete(order_id: str) -> str:
    """Poll Alpaca for ``order_id`` until it is filled or cancelled."""
    while True:
        try:
            order = trading_client.get_order_by_id(order_id)
            status = order.status.value if hasattr(order.status, "value") else order.status
        except Exception as exc:
            logger.error("Failed fetching status for %s: %s", order_id, exc)
            status = "unknown"
            break
        if status in {"filled", "canceled", "expired"}:
            break
        logger.info("Polling order %s, status: %s", order_id, status)
        sleep(10)
    logger.info("Order %s completed with status %s", order_id, status)
    return status


def poll_order_status(
    trading_client: TradingClient,
    order_id: str,
    symbol: str,
    submit_ts: datetime,
    attempt: int = 1,
    timeout_seconds: int = ORDER_TIMEOUT_SECONDS,
    poll_interval: int = ORDER_POLL_INTERVAL,
) -> Optional["Order"]:
    """Poll order ``order_id`` until completion or timeout.

    Returns the final order object on success or ``None`` on cancel/timeout.
    """

    start_time = datetime.utcnow()
    while True:
        try:
            order = trading_client.get_order_by_id(order_id)
            status = getattr(order, "status", "unknown")
            filled_avg_price = getattr(order, "filled_avg_price", None)
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed fetching status for %s: %s", order_id, exc)
            log_api_error_event("poll", symbol, str(exc))
            final_ts = datetime.utcnow()
            latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
            log_order_final_event(
                symbol,
                attempt,
                "poll_error",
                latency_ms,
            )
            return None

        if status in [
            "filled",
            "partially_filled",
            "canceled",
            "rejected",
            "expired",
        ]:
            logger.info(
                "Order %s for %s completed with status %s.", order_id, order.symbol, status
            )
            final_ts = datetime.utcnow()
            latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
            log_order_final_event(
                symbol,
                attempt,
                status,
                latency_ms,
                filled_avg_price,
            )
            return order

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if elapsed > timeout_seconds:
            logger.warning(
                "Order %s for %s timed out after %s seconds. Attempting cancellation.",
                order_id,
                order.symbol,
                timeout_seconds,
            )
            try:
                trading_client.cancel_order_by_id(order_id)
            except Exception as cancel_exc:  # pragma: no cover - API errors
                logger.error(
                    "Failed to cancel order %s for %s: %s",
                    order_id,
                    order.symbol,
                    cancel_exc,
                )
                log_api_error_event("cancel", symbol, str(cancel_exc))
            final_ts = datetime.utcnow()
            latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
            log_order_final_event(
                symbol,
                attempt,
                "timeout",
                latency_ms,
                filled_avg_price,
            )
            return None

        if not is_market_open(trading_client):
            logger.warning(
                "Market closed while polling order %s for %s. Cancelling and exiting polling.",
                order_id,
                order.symbol,
            )
            try:
                trading_client.cancel_order_by_id(order_id)
            except Exception as cancel_exc:  # pragma: no cover - API errors
                logger.error(
                    "Failed to cancel order %s for %s: %s",
                    order_id,
                    order.symbol,
                    cancel_exc,
                )
                log_api_error_event("cancel", symbol, str(cancel_exc))
            final_ts = datetime.utcnow()
            latency_ms = int((final_ts - submit_ts).total_seconds() * 1000)
            log_order_final_event(
                symbol,
                attempt,
                "market_closed",
                latency_ms,
                filled_avg_price,
            )
            return None

        logger.info(
            "Polling order %s for %s, status: %s",
            order_id,
            order.symbol,
            status,
        )
        sleep(poll_interval)


def submit_order_with_polling(
    trading_client: TradingClient,
    symbol: str,
    qty: int,
    limit_price: float,
    side: str = "buy",
) -> Optional["Order"]:
    """Submit a limit order and poll until it completes or times out."""

    attempt = 1
    session_label = "regular"
    log_order_submit_event(symbol, side, qty, limit_price, attempt, session_label)
    submit_ts = datetime.utcnow()
    try:
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=False,
        )
        order = trading_client.submit_order(order_request)
        submit_ts = datetime.utcnow()
        logger.info(
            "[SUBMIT] Order submitted for %s qty=%s at price=%s, id=%s",
            symbol,
            qty,
            limit_price,
            order.id,
        )
    except Exception as exc:
        log_api_error_event("submit", symbol, str(exc))
        logger.error("Exception submitting order for %s: %s", symbol, exc)
        return None

    final_order = poll_order_status(
        trading_client,
        order.id,
        symbol,
        submit_ts,
        attempt=attempt,
    )
    if final_order is None:
        logger.warning(
            "[CANCELLED] Order for %s was cancelled or expired after timeout.", symbol
        )
    else:
        logger.info(
            "[FILLED] Order for %s executed successfully at price %s",
            symbol,
            getattr(final_order, "filled_avg_price", "N/A"),
        )
    return final_order

def allocate_position(symbol: str, fallback_prices: dict[str, float]):
    open_positions = get_open_positions()
    if symbol in open_positions:
        reason = "already in open positions"
        logger.debug("Skipping %s: %s", symbol, reason)
        return None, reason
    if len(open_positions) >= MAX_OPEN_TRADES:
        reason = "max open trades reached"
        logger.info(
            "Skipped %s due to reaching max open trades (%s).",
            symbol,
            MAX_OPEN_TRADES,
        )
        return None, reason

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT

    last_price = get_symbol_price(symbol, data_client, fallback_prices)
    if last_price is None:
        return None, "market data error"

    qty = int(alloc_amount / last_price)
    if qty < 1:
        reason = "allocation insufficient"
        logger.debug("Skipping %s: %s", symbol, reason)
        return None, reason

    logger.debug("Allocating %d shares of %s at %s", qty, symbol, last_price)
    return (qty, round(last_price, 2)), None

def submit_trades() -> list[dict]:
    df = load_top_candidates(limit=3)
    records = df.to_dict("records") if not df.empty else []
    symbols = [rec.get("symbol") for rec in records if rec.get("symbol")]
    cached_prices = load_cached_prices(symbols)
    calendar_today = trading_client.get_calendar()[0]
    if calendar_today.open is None or calendar_today.close is None:
        logger.warning(
            "No market hours available today. Possibly a holiday or weekend. Skipping."
        )
        return

    clock = trading_client.get_clock()
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    if not clock.is_open:
        if time(4, 0) <= now_et < time(9, 30):
            logger.info("In pre-market session – continuing trade execution.")
        else:
            logger.warning(
                "Market is fully closed (not even in pre-market). Skipping trade execution.")
            for rec in records:
                sym = rec.get("symbol")
                if sym:
                    skip(sym, "SESSION_WINDOW", "Market closed outside allowed trading window.")
            return []
    submitted = 0
    skipped = 0
    positions = trading_client.get_all_positions()
    open_orders = trading_client.get_orders(
        GetOrdersRequest(statuses=[QueryOrderStatus.OPEN])
    )
    position_symbols = {p.symbol for p in positions}
    open_order_symbols = {o.symbol for o in open_orders}
    open_symbols = position_symbols.union(open_order_symbols)
    logger.info(f"Existing or pending symbols: {open_symbols}")

    trade_log_entries = []

    for rec in records:
        sym = rec.get("symbol")
        if not sym:
            continue
        metrics["symbols_processed"] += 1
        entry = {
            "symbol": sym,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "order_status": None,
            "qty": None,
            "entry_price": None,
            "reason_skipped": None,
        }

        if sym in open_symbols:
            logger.info(f"[SKIP] {sym}: Existing position or pending order detected.")
            skipped += 1
            entry["order_status"] = "skipped"
            entry["reason_skipped"] = "existing/pending"

            if sym in position_symbols:
                skip(sym, "EXISTING_POSITION", "Existing position detected.")
                inc("orders_skipped_existing_positions")
            elif sym in open_order_symbols:
                skip(sym, "PENDING_ORDER", "Pending order detected.")
                inc("orders_skipped_pending_orders")
            else:
                skip(sym, "OTHER", "Symbol flagged as already open without context.")

            trade_log_entries.append(entry)
            continue

        alloc, reason = allocate_position(sym, cached_prices)
        if alloc is None:
            skipped += 1
            entry["order_status"] = "skipped"
            entry["reason_skipped"] = reason

            if reason == "market data error":
                logger.warning("Skipping %s: No bars available.", sym)
                skip(sym, "MARKET_DATA", reason)
            elif reason in {"max open trades reached", "allocation insufficient"}:
                logger.warning("Trade skipped for %s: %s", sym, reason)
                skip(sym, "RISK_LIMIT", reason)
            elif reason == "already in open positions":
                logger.warning("Trade skipped for %s: %s", sym, reason)
                skip(sym, "EXISTING_POSITION", reason)
            else:
                logger.warning("Trade skipped for %s: %s", sym, reason)
                skip(sym, "OTHER", reason, raw_reason=reason)

            trade_log_entries.append(entry)
            continue

        qty, entry_price = alloc
        logger.info(
            "Submitting limit buy order for %s, qty=%s, limit=%s | score=%s win_rate=%s net_pnl=%s avg_return=%s",
            sym,
            qty,
            entry_price,
            rec.get("score"),
            rec.get("win_rate"),
            rec.get("net_pnl"),
            rec.get("avg_return"),
        )
        try:
            limit_price = get_symbol_price(sym, data_client, cached_prices)
            if limit_price is None:
                logger.error("Latest price unavailable for %s", sym)
                limit_price = entry_price
            if limit_price is None:
                logger.warning("[SKIP] %s: No valid price available after fallbacks.", sym)
                skipped += 1
                entry["order_status"] = "skipped"
                entry["reason_skipped"] = "no price"
                skip(sym, "MARKET_DATA", "No valid price available after fallbacks.")
                trade_log_entries.append(entry)
                continue
            price = round_price(limit_price)
            price = round(price + 1e-6, 2)

            if DRY_RUN:
                logger.info(
                    "WOULD SUBMIT %s qty=%s limit=%s (dry-run)",
                    sym,
                    qty,
                    price,
                )
                entry["order_status"] = "dry-run"
                entry["qty"] = qty
                entry["entry_price"] = price
                trade_log_entries.append(entry)
                continue

            final_order = submit_order_with_polling(
                trading_client,
                sym,
                qty,
                price,
                side="buy",
            )
            entry["order_status"] = getattr(final_order, "status", "cancelled") if final_order else "cancelled"
            entry["qty"] = qty
            entry["entry_price"] = price
            trade_log_entries.append(entry)

            record_executed_trade(
                sym,
                qty,
                price,
                order_type="limit",
                side="buy",
                order_id=str(final_order.id) if final_order else "",
                order_status=getattr(final_order, "status", "cancelled") if final_order else "cancelled",
            )

            if final_order:
                log_trade_exit(
                    sym,
                    qty,
                    entry_price,
                    price,
                    entry["entry_time"],
                    "",
                    "submitted",
                    "limit",
                    "Pre-market entry",
                    "buy",
                    order_id=str(final_order.id),
                )

            if final_order and getattr(final_order, "status", "") == "filled":
                attach_trailing_stops()
                save_open_positions_csv()
                update_trades_log()

            if final_order:
                submitted += 1
                metrics["orders_submitted"] += 1
            else:
                skipped += 1
                metrics["api_failures"] += 1
        except Exception as e:
            logger.error("Failed to submit buy order for %s: %s", sym, e)
            skipped += 1
            metrics["api_failures"] += 1
    errors = metrics.get("api_failures", 0)
    logger.info(
        "Orders submitted: %d, skipped: %d, errors: %d",
        submitted,
        skipped,
        errors,
    )

    execute_metrics = persist_execute_metrics()

    return trade_log_entries

def attach_trailing_stops():
    positions = get_open_positions()
    for symbol, pos in positions.items():
        # Fetch current open orders and cancel existing trailing stops
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            existing_orders = trading_client.get_orders(filter=request)
        except Exception as exc:
            logger.error("Failed to fetch open orders for %s: %s", symbol, exc)
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "error")
            continue

        for order in existing_orders:
            if getattr(order, "order_type", "") == "trailing_stop":
                try:
                    trading_client.cancel_order_by_id(order.id)
                    logger.info(
                        f"Cancelled existing trailing stop {order.id} for {symbol}"
                    )
                except Exception as cancel_exc:  # pragma: no cover - API errors
                    logger.error(
                        "Failed to cancel trailing stop %s for %s: %s",
                        order.id,
                        symbol,
                        cancel_exc,
                    )

        other_orders = [
            o
            for o in existing_orders
            if getattr(o, "order_type", "") != "trailing_stop"
            and o.status.lower() in ("open", "new", "accepted")
        ]
        if other_orders:
            logger.info("Skipping trailing stop for %s: already has open orders.", symbol)
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "skipped")
            continue

        # Re-check available quantity after cancellations
        try:
            position = trading_client.get_open_position(symbol)
            available_qty = int(getattr(position, "qty_available", 0))
        except Exception as exc:
            logger.error(
                "Failed to fetch position for %s after cancelling trailing stop: %s",
                symbol,
                exc,
            )
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "error")
            continue

        if available_qty <= 0:
            logger.warning(
                f"Insufficient available qty for {symbol}: {available_qty}"
            )
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "skipped")
            continue
        submit_new_trailing_stop(symbol, available_qty, TRAIL_PERCENT)

def daily_exit_check():
    positions = get_open_positions()
    request = GetOrdersRequest()
    orders = trading_client.get_orders(request)
    closed_orders = [
        o
        for o in orders
        if getattr(o, "status", "").lower() in ("filled", "canceled", "rejected")
    ]
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    extended = is_extended_hours(now_et)

    for symbol, pos in positions.items():
        entry_orders = [
            o
            for o in closed_orders
            if o.symbol == symbol and getattr(o.side, "value", o.side) == "buy"
        ]
        if not entry_orders:
            logger.warning("No entry order found for %s, skipping.", symbol)
            continue

        # Filter out orders without a fill timestamp
        valid_entries = [o for o in entry_orders if getattr(o, "filled_at", None)]
        if not valid_entries:
            logger.warning(
                f"No filled entry orders found for symbol {symbol}. Skipping exit check."
            )
            continue  # Move to next symbol
        entry_order = max(valid_entries, key=lambda o: o.filled_at)
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        logger.debug("%s entered on %s, held for %s days", symbol, entry_date, days_held)

        if days_held >= MAX_HOLD_DAYS:
            logger.info("Exiting %s after %s days", symbol, days_held)
            try:
                valid_exit_price = round_price(float(pos.current_price))
                valid_exit_price = round(valid_exit_price + 1e-6, 2)
                available_qty = get_available_qty(symbol)
                qty = min(int(pos.qty), available_qty)
                if qty <= 0:
                    logger.info(
                        "Skipped exit for %s: available qty %s insufficient.",
                        symbol,
                        available_qty,
                    )
                    continue
                session = "extended" if extended else "regular"
                logger.info(
                    "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
                    symbol,
                    pos.qty,
                    valid_exit_price,
                    session,
                )
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=valid_exit_price,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended,
                )
                submit_ts = datetime.utcnow()
                order = trading_client.submit_order(order_request)
                log_exit_submit(symbol, qty, "limit", "max_hold")
                status = poll_order_until_complete(order.id)
                latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                log_exit_final(status, latency_ms)
                logger.info("Order submitted successfully for %s", symbol)
                record_executed_trade(
                    symbol,
                    qty,
                    pos.current_price,
                    order_type="market",
                    side="sell",
                    order_id=str(order.id),
                    order_status=status,
                )
                if status == "filled":
                    save_open_positions_csv()
                    update_trades_log()
                metrics["orders_submitted"] += 1
            except APIError as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = LimitOrderRequest(
                            limit_price=valid_exit_price,
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        submit_ts = datetime.utcnow()
                        order = trading_client.submit_order(retry_req)
                        log_exit_submit(symbol, int(pos.qty), "limit", "max_hold")
                        status = poll_order_until_complete(order.id)
                        latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                        log_exit_final(status, latency_ms)
                        logger.info(
                            "Retry successful with regular hours order for %s",
                            symbol,
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            pos.current_price,
                            order_type="market",
                            side="sell",
                            order_id=str(order.id),
                            order_status=status,
                        )
                        if status == "filled":
                            save_open_positions_csv()
                            update_trades_log()
                        metrics["orders_submitted"] += 1
                        metrics["api_retries"] += 1
                    except APIError as retry_e:
                        logger.error(
                            "Retry order submission failed for %s: %s",
                            symbol,
                            retry_e,
                        )
                        metrics["api_failures"] += 1
            except Exception as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                metrics["api_failures"] += 1
                try:
                    trading_client.close_position(symbol)
                except Exception as exc:  # pragma: no cover - API errors
                    logger.error("Failed to close position %s: %s", symbol, exc)
        elif should_exit_early(symbol, data_client, os.path.join(BASE_DIR, "data", "history_cache")):
            logger.info("Early exit signal for %s", symbol)
            try:
                valid_exit_price = round_price(float(pos.current_price))
                valid_exit_price = round(valid_exit_price + 1e-6, 2)
                available_qty = get_available_qty(symbol)
                qty = min(int(pos.qty), available_qty)
                if qty <= 0:
                    logger.info(
                        "Skipped exit for %s: available qty %s insufficient.",
                        symbol,
                        available_qty,
                    )
                    continue
                now_et = datetime.now(pytz.timezone("US/Eastern")).time()
                extended_now = is_extended_hours(now_et)
                session = "extended" if extended_now else "regular"
                logger.info(
                    "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
                    symbol,
                    pos.qty,
                    valid_exit_price,
                    session,
                )
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=valid_exit_price,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended_now,
                )
                submit_ts = datetime.utcnow()
                order = trading_client.submit_order(order_request)
                log_exit_submit(symbol, qty, "limit", "monitor")
                status = poll_order_until_complete(order.id)
                latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                log_exit_final(status, latency_ms)
                record_executed_trade(
                    symbol,
                    qty,
                    valid_exit_price,
                    order_type="limit",
                    side="sell",
                    order_id=str(order.id),
                    order_status=status,
                )
                if status == "filled":
                    save_open_positions_csv()
                    update_trades_log()
                metrics["orders_submitted"] += 1
            except APIError as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = LimitOrderRequest(
                            limit_price=valid_exit_price,
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        submit_ts = datetime.utcnow()
                        order = trading_client.submit_order(retry_req)
                        log_exit_submit(symbol, int(pos.qty), "limit", "monitor")
                        status = poll_order_until_complete(order.id)
                        latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                        log_exit_final(status, latency_ms)
                        logger.info(
                            "Retry successful with regular hours order for %s", symbol
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            valid_exit_price,
                            order_type="limit",
                            side="sell",
                            order_status=status,
                            order_id=str(order.id),
                        )
                        metrics["orders_submitted"] += 1
                        metrics["api_retries"] += 1
                    except APIError as retry_e:
                        logger.error(
                            "Retry order submission failed for %s: %s", symbol, retry_e
                        )
                        metrics["api_failures"] += 1
            except Exception as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                metrics["api_failures"] += 1
                try:
                    trading_client.close_position(symbol)
                except Exception as exc:  # pragma: no cover - API errors
                    logger.error("Failed to close position %s: %s", symbol, exc)

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the trade execution script."""

    parser = argparse.ArgumentParser(description="Execute Alpaca trades with guardrails")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the market guard determines the market is closed.",
    )
    parser.add_argument(
        "--source",
        default=CANDIDATE_SOURCE_DEFAULT,
        help="Path to candidates CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders; log intent instead",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint for executing the trade pipeline with market guardrails."""

    raw_args = list(argv) if argv is not None else sys.argv[1:]

    try:
        os.chdir(repo_root())
    except Exception:
        pass

    force_flag = "--force" in raw_args
    emit("RUN_START", component=COMPONENT_NAME, force=str(force_flag).lower())

    status = "UNKNOWN"
    clock_source = "unknown"
    err: Optional[str] = None

    try:
        key_id = os.environ.get("ALPACA_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
        secret_key = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        base_url_env = os.environ.get("ALPACA_BASE_URL") or os.environ.get("APCA_API_BASE_URL") or ""
        if not key_id or not secret_key:
            raise KeyError("Missing ALPACA credentials")
        try:
            from alpaca.trading.client import TradingClient

            clock_source = "TradingClient"
            tc = TradingClient(
                key_id,
                secret_key,
                paper="paper" in base_url_env.lower(),
            )
            status = "OPEN" if bool(tc.get_clock().is_open) else "CLOSED"
        except Exception:
            import alpaca_trade_api as a

            clock_source = "RESTv2"
            api = a.REST(
                key_id,
                secret_key,
                base_url_env,
                api_version="v2",
            )
            status = "OPEN" if bool(api.get_clock().is_open) else "CLOSED"
    except Exception as exc:
        err = str(exc)
        status = "UNKNOWN"

    emit(
        "MARKET_GUARD_STATUS",
        component=COMPONENT_NAME,
        status=status,
        clock_source=clock_source,
        force=str(force_flag).lower(),
        error=err or "",
    )

    if status == "CLOSED" and not force_flag:
        logger.warning("Market is closed; aborting trade execution run.")
        metrics["run_aborted_reason"] = "MARKET_CLOSED"
        persist_execute_metrics()
        emit("RUN_ABORT", component=COMPONENT_NAME, reason_code="MARKET_CLOSED")
        emit("RUN_END", component=COMPONENT_NAME, status="aborted")
        return 0

    if status == "CLOSED" and force_flag:
        logger.warning("Market closed but proceeding due to --force flag.")

    args = parse_args(raw_args)
    force_flag = bool(getattr(args, "force", False))
    global DRY_RUN, CANDIDATE_SOURCE_PATH
    DRY_RUN = bool(getattr(args, "dry_run", False))
    CANDIDATE_SOURCE_PATH = str(getattr(args, "source", CANDIDATE_SOURCE_DEFAULT) or CANDIDATE_SOURCE_DEFAULT)
    if DRY_RUN:
        logger.info("Running in dry-run mode; no orders will be submitted.")

    exit_code = 0
    try:
        run_start_time = datetime.utcnow()
        logger.info("Starting pre-market trade execution script")

        try:
            trade_entries = submit_trades()
            if not DRY_RUN:
                attach_trailing_stops()
                daily_exit_check()
                save_open_positions_csv()
                update_trades_log()
            else:
                logger.info("Dry-run mode: skipping trailing stops, exits, and log refresh.")
            if trade_entries and not DRY_RUN:
                try:
                    csv_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
                    existing = (
                        pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
                    )
                    new_df = pd.DataFrame(trade_entries)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_csv(csv_path, index=False)
                    logger.info("Trades log updated successfully.")
                except Exception as exc:
                    logger.error("Failed to append trade log entries: %s", exc)
            logger.info(
                "Metrics - processed: %d, submitted: %d, skipped: %d",
                metrics["symbols_processed"],
                metrics["orders_submitted"],
                metrics["symbols_skipped"],
            )

            if not DRY_RUN:
                history_script = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "fetch_trades_history.py"
                )
                try:
                    subprocess.run(["python", history_script], check=True)
                    logger.info("Historical trades successfully fetched and CSV files updated.")
                except subprocess.CalledProcessError as e:
                    logger.error("Failed to run historical trade script: %s", e)
        except Exception as exc:
            logger.exception("Critical error occurred: %s", exc)
            send_alert(str(exc))
            exit_code = 1
            raise
        finally:
            end_time = datetime.utcnow()
            elapsed_time = end_time - run_start_time
            logger.info("Script finished in %s", elapsed_time)
            logger.info("Pre-market trade execution script complete")
    except Exception:
        emit("RUN_END", component=COMPONENT_NAME, status="error")
        return 1

    emit("RUN_END", component=COMPONENT_NAME, status="ok")
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

