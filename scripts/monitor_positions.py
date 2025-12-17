# monitor_positions.py

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    TrailingStopOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
)

# Support both ``python -m scripts.monitor_positions`` (preferred) and direct invocation.
if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from scripts.utils import fetch_bars_with_cutoff
else:
    from .utils import fetch_bars_with_cutoff

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import logging
import shutil
from tempfile import NamedTemporaryFile


import pytz
from utils.alerts import send_alert

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

logfile = os.path.join(BASE_DIR, "logs", "monitor.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("Monitoring service active.")

STATUS_PATH = Path(BASE_DIR) / "data" / "monitor_status.json"
METRICS_PATH = Path(BASE_DIR) / "data" / "monitor_metrics.json"
MONITOR_STATE_PATH = Path(BASE_DIR) / "data" / "monitor_state.json"
METRIC_KEYS = [
    "stops_attached",
    "stops_missing",
    "stops_tightened",
    "exits_max_hold",
    "exits_signal",
    "api_errors",
    "stop_attach_failed",
    "stop_coverage_pct",
]


def _default_metrics(date_str: str | None = None) -> dict:
    payload = {key: 0 for key in METRIC_KEYS}
    payload["date"] = date_str or datetime.utcnow().date().isoformat()
    return payload


def _load_metrics() -> dict:
    today = datetime.utcnow().date().isoformat()
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text())
            if isinstance(data, dict):
                if data.get("date") != today:
                    data = _default_metrics(today)
                for key in METRIC_KEYS:
                    data.setdefault(key, 0)
                return data
        except Exception:
            logger.warning("[MONITOR] Unable to read monitor metrics; resetting state")
    return _default_metrics(today)


MONITOR_METRICS = _load_metrics()


def _load_monitor_state() -> dict:
    try:
        data = json.loads(MONITOR_STATE_PATH.read_text())
        if isinstance(data, dict):
            data.setdefault("stop_attach", {})
            return data
    except Exception:
        pass
    return {"stop_attach": {}}


def _save_monitor_state(state: dict) -> None:
    try:
        MONITOR_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.error("Unable to persist monitor state: %s", exc)


MONITOR_STATE = _load_monitor_state()


RANKER_EVAL_PATH = Path(BASE_DIR) / "data" / "ranker_eval" / "latest.json"


def _load_ranker_eval() -> dict:
    default_state = {"signal_quality": "MEDIUM", "decile_lift": None}
    try:
        payload = json.loads(RANKER_EVAL_PATH.read_text())
    except Exception as exc:
        logger.warning("[MONITOR] Unable to load ranker eval: %s", exc)
        return default_state

    quality = (payload.get("signal_quality") or "MEDIUM").upper()
    if quality not in {"LOW", "MEDIUM", "HIGH"}:
        quality = "MEDIUM"
    decile_lift = payload.get("decile_lift")
    try:
        decile_lift = float(decile_lift) if decile_lift is not None else None
    except Exception:
        decile_lift = None
    return {"signal_quality": quality, "decile_lift": decile_lift}


def _sanitize_ranker_eval_state(state: dict) -> dict:
    """Return a sanitized ML risk state."""

    return {
        "signal_quality": (state.get("signal_quality") or "MEDIUM").upper(),
        "decile_lift": state.get("decile_lift"),
    }


ML_RISK_STATE = _sanitize_ranker_eval_state(_load_ranker_eval())


def _persist_metrics() -> None:
    payload = dict(MONITOR_METRICS)
    payload["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    METRICS_PATH.write_text(json.dumps(payload, indent=2))


def increment_metric(name: str, delta: int = 1) -> None:
    today = datetime.utcnow().date().isoformat()
    if MONITOR_METRICS.get("date") != today:
        MONITOR_METRICS.clear()
        MONITOR_METRICS.update(_default_metrics(today))
    if name not in METRIC_KEYS:
        return
    MONITOR_METRICS[name] = int(MONITOR_METRICS.get(name, 0)) + int(delta)
    _persist_metrics()


def write_status(
    *,
    status: str = "running",
    positions_count: int = 0,
    trailing_count: int = 0,
    protective_orders_count: int = 0,
    stop_coverage_pct: float | None = None,
) -> None:
    payload = {
        "pid": os.getpid(),
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "positions_count": int(positions_count),
        "open_trailing_stops_count": int(trailing_count),
        "protective_orders_count": int(protective_orders_count),
        "stop_coverage_pct": float(stop_coverage_pct) if stop_coverage_pct is not None else None,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def record_heartbeat(
    positions_count: int,
    trailing_count: int,
    protective_orders_count: int = 0,
    stop_coverage_pct: float | None = None,
    status: str = "running",
) -> None:
    logger.info(
        "[MONITOR_HEARTBEAT] positions=%s trailing_stops=%s protective_orders=%s coverage=%.3f status=%s",
        positions_count,
        trailing_count,
        protective_orders_count,
        float(stop_coverage_pct or 0.0),
        status,
    )
    today = datetime.utcnow().date().isoformat()
    if MONITOR_METRICS.get("date") != today:
        MONITOR_METRICS.clear()
        MONITOR_METRICS.update(_default_metrics(today))
    write_status(
        status=status,
        positions_count=positions_count,
        trailing_count=trailing_count,
        protective_orders_count=protective_orders_count,
        stop_coverage_pct=stop_coverage_pct,
    )
    _persist_metrics()


def is_debug_mode() -> bool:
    return os.getenv("JBRAVO_MONITOR_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def evaluate_exit_signals_for_debug(position: dict, indicators: dict) -> list[str]:
    """Thin wrapper around real exit-evaluation logic without side effects."""

    def _desired_trail_pct(gain: float) -> float:
        if gain >= 10:
            return TRAIL_TIGHTEST_PERCENT
        if gain >= 5:
            return TRAIL_TIGHT_PERCENT
        return TRAIL_START_PERCENT

    normalized = dict(indicators)
    case_map = {
        "ema20": "EMA20",
        "rsi": "RSI",
        "macd": "MACD",
        "macd_signal": "MACD_signal",
        "macd_prev": "MACD_prev",
        "macd_signal_prev": "MACD_signal_prev",
    }
    for lower_key, upper_key in case_map.items():
        if lower_key in normalized and upper_key not in normalized:
            normalized[upper_key] = normalized[lower_key]

    entry_price = float(position.get("entry_price") or position.get("avg_entry_price") or 0)
    close_price = float(normalized.get("close", entry_price))
    gain_pct = (close_price - entry_price) / entry_price * 100 if entry_price else 0

    reasons = []
    if gain_pct >= PARTIAL_GAIN_THRESHOLD:
        reasons.append(f"Partial exit at +{PARTIAL_GAIN_THRESHOLD:.0f}% gain")

    reasons.extend(check_sell_signal(position.get("symbol", ""), normalized))

    trail_pct = _desired_trail_pct(gain_pct)
    reasons.append(f"Trailing stop target {trail_pct:.2f}% (gain {gain_pct:.2f}%)")

    return reasons


def run_debug_smoke_test():
    if not is_debug_mode():
        return

    def simulate_debug_scenario(
        *,
        tag: str,
        symbol: str,
        entry_price: float,
        current_price: float,
        ema20: float,
        rsi: float,
        macd_cross: bool = False,
        pattern: Optional[str] = None,
    ) -> None:
        position = {
            "symbol": symbol,
            "qty": 100,
            "entry_price": entry_price,
            "days_held": 3,
        }

        indicators = {
            "close": current_price,
            "EMA20": ema20,
            "RSI": rsi,
            "MACD": 0.5,
            "MACD_signal": 0.0,
            "MACD_prev": 0.5,
            "MACD_signal_prev": 0.4,
        }

        if macd_cross:
            indicators.update({
                "MACD": -0.5,
                "MACD_signal": 0.5,
                "MACD_prev": 0.5,
                "MACD_signal_prev": -0.2,
            })

        if pattern == "shooting_star":
            open_price = max(current_price, entry_price * 1.05)
            close_price = open_price - 1.0
            indicators.update(
                {
                    "open": open_price,
                    "close": close_price,
                    "high": open_price + 4.0,
                    "low": close_price - 0.1,
                }
            )
        else:
            indicators.setdefault("open", current_price)
            indicators.setdefault("high", current_price)
            indicators.setdefault("low", current_price)

        reasons = evaluate_exit_signals_for_debug(position, indicators)
        logger.info(
            "[DEBUG_SMOKE] Scenario %s (%s): exit reasons=%s",
            tag,
            symbol,
            ";".join(reasons) if reasons else "NO_EXIT",
        )

    logger.info("[DEBUG_SMOKE] Starting monitor_positions smoke test")

    try:
        simulate_debug_scenario(
            tag="PARTIAL_OR_TIERED",
            symbol="DBGGAIN",
            entry_price=100.0,
            current_price=106.0,
            ema20=98.0,
            rsi=65.0,
        )

        simulate_debug_scenario(
            tag="MACD",
            symbol="DBGMACD",
            entry_price=50.0,
            current_price=55.0,
            ema20=52.0,
            rsi=60.0,
            macd_cross=True,
        )

        simulate_debug_scenario(
            tag="PATTERN",
            symbol="DBGPAT",
            entry_price=20.0,
            current_price=21.0,
            ema20=21.5,
            rsi=65.0,
            pattern="shooting_star",
        )

        logger.info("[DEBUG_SMOKE] Completed monitor_positions smoke test")
    except Exception:
        logger.exception("[DEBUG_SMOKE] Smoke test raised unexpectedly")

REQUIRED_ALPACA_ENV = [
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
]


def _manual_load_env(path: Path) -> bool:
    loaded_any = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded_any = True
    return loaded_any


def _load_env_if_needed() -> None:
    missing_before = [k for k in REQUIRED_ALPACA_ENV if not os.getenv(k)]
    if not missing_before:
        logger.info("[STARTUP] Alpaca env already present; skipping .env load")
        return

    logger.info("[STARTUP] Alpaca env missing, attempting to load .env files")

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    candidates = [
        Path.home() / ".config" / "jbravo" / ".env",
        Path(os.environ.get("JBRAVO_HOME", BASE_DIR)) / ".env",
    ]

    for path in candidates:
        if not path.exists():
            continue
        loaded = False
        if load_dotenv:
            loaded = bool(load_dotenv(dotenv_path=path, override=False))
        if not loaded:
            loaded = _manual_load_env(path)
        if loaded:
            logger.info("[STARTUP] Loaded env from %s", path)

    missing_after = [k for k in REQUIRED_ALPACA_ENV if not os.getenv(k)]
    if missing_after:
        logger.error(
            "[FATAL] Missing required Alpaca env vars after .env load: %s. "
            "Ensure ~/.config/jbravo/.env is present or set these in the task environment.",
            ", ".join(missing_after),
        )
        raise SystemExit(1)

    logger.info(
        "[STARTUP] Alpaca env OK (key_id=%s…, base_url=%s)",
        os.getenv("APCA_API_KEY_ID", "")[:4],
        os.getenv("APCA_API_BASE_URL"),
    )


def log_trailing_stop_event(symbol: str, trail_percent: float, order_id: Optional[str], status: str) -> None:
    """Emit the structured trailing-stop attachment log."""

    payload = {
        "symbol": symbol,
        "trail_percent": trail_percent,
        "order_id": order_id,
        "status": status,
    }
    logger.info("TRAILING_STOP_ATTACH %s", payload)


def log_exit_submit(symbol: str, qty: int, order_type: str, reason_code: str, side: str = "sell") -> None:
    """Emit the EXIT_SUBMIT structured log."""

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "reason_code": reason_code,
    }
    logger.info("EXIT_SUBMIT %s", payload)


def log_exit_final(status: str, latency_ms: int) -> None:
    """Emit the EXIT_FINAL structured log."""

    payload = {
        "status": status,
        "latency_ms": latency_ms,
    }
    logger.info("EXIT_FINAL %s", payload)


def log_if_stale(file_path: str, name: str, threshold_minutes: int = 15):
    """Log a warning if ``file_path`` is older than ``threshold_minutes``."""
    if not os.path.exists(file_path):
        logger.warning("%s missing: %s", name, file_path)
        send_alert(f"{name} missing: {file_path}")
        return
    age = datetime.utcnow() - datetime.utcfromtimestamp(os.path.getmtime(file_path))
    if age > timedelta(minutes=threshold_minutes):
        minutes = age.total_seconds() / 60
        msg = f"{name} is stale ({minutes:.1f} minutes old)"
        logger.warning(msg)
        send_alert(msg)


def round_price(value: float) -> float:
    """Return ``value`` rounded to the nearest cent."""
    return round(value + 1e-6, 2)


def wait_for_order_terminal(order_id: str, poll_interval: int = 10, timeout_seconds: int = 600) -> str:
    """Poll until ``order_id`` reaches a terminal state or times out."""

    deadline = datetime.utcnow() + timedelta(seconds=timeout_seconds)
    status = "unknown"
    while datetime.utcnow() <= deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            status_obj = getattr(order, "status", "unknown")
            status = status_obj.value if hasattr(status_obj, "value") else str(status_obj)
        except Exception as exc:
            logger.error("Failed to fetch status for %s: %s", order_id, exc)
            status = "error"
            break

        normalized = status.lower()
        if normalized in {"filled", "canceled", "cancelled", "expired", "rejected"}:
            break

        time.sleep(poll_interval)

    return status


def cancel_order_safe(order_id: str, symbol: str):
    """Attempt to cancel ``order_id`` using available client methods."""
    if hasattr(trading_client, "cancel_order_by_id"):
        try:
            trading_client.cancel_order_by_id(order_id)
            return
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to cancel order %s: %s", order_id, exc)
    elif hasattr(trading_client, "cancel_order"):
        try:
            trading_client.cancel_order(order_id)
            return
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to cancel order %s: %s", order_id, exc)
    if hasattr(trading_client, "close_position"):
        try:
            trading_client.close_position(symbol)
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to close position %s: %s", symbol, exc)


def ensure_column_exists(df: pd.DataFrame, column: str, default=None) -> pd.DataFrame:
    """Ensure ``column`` exists in ``df``; create with ``default`` if missing."""
    if column not in df.columns:
        df[column] = default
        logger.warning("Added missing '%s' column with default %s", column, default)
    return df

# Required columns for open_positions.csv
REQUIRED_COLUMNS = [
    "symbol",
    "qty",
    "avg_entry_price",
    "entry_price",
    "current_price",
    "unrealized_pl",
    "net_pnl",
    "entry_time",
    "days_in_trade",
    "side",
    "order_status",
    "pnl",
    "order_type",
]


open_pos_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
if not os.path.exists(open_pos_path):
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(open_pos_path, index=False)

# No global entry time cache - load the CSV each cycle to preserve original
# entry times even if other processes modify the file.

def get_original_entry_time(existing_df: pd.DataFrame, symbol: str, default_time: str) -> str:
    """Return the previously recorded entry time for ``symbol`` if available."""
    match = existing_df[existing_df["symbol"] == symbol]
    if not match.empty and "entry_time" in match.columns:
        try:
            return match.iloc[0]["entry_time"]
        except Exception:
            pass
    return default_time

executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
trades_log_real_path = os.path.join(BASE_DIR, "data", "trades_log_real.csv")
partial_exit_state_path = os.path.join(BASE_DIR, "data", "partial_exit_state.json")

REAL_TRADE_COLUMNS = [
    "symbol",
    "qty",
    "entry_price",
    "exit_price",
    "entry_time",
    "exit_time",
    "net_pnl",
    "order_status",
    "order_type",
    "exit_reason",
    "side",
    "order_id",
]

TRADE_COLUMNS = [
    "symbol",
    "qty",
    "entry_price",
    "exit_price",
    "entry_time",
    "exit_time",
    "order_status",
    "net_pnl",
    "order_type",
    "exit_reason",
    "side",
]

for path in (executed_trades_path, trades_log_path):
    if not os.path.exists(path):
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(path, index=False)

if not os.path.exists(trades_log_real_path):
    pd.DataFrame(columns=REAL_TRADE_COLUMNS).to_csv(trades_log_real_path, index=False)


def _load_partial_state() -> dict:
    if not os.path.exists(partial_exit_state_path):
        return {}
    try:
        with open(partial_exit_state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    except Exception:
        logger.warning("Failed to read partial exit state; starting fresh.")
    return {}


def _save_partial_state(state: dict) -> None:
    try:
        with open(partial_exit_state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
    except Exception as exc:
        logger.error("Unable to persist partial exit state: %s", exc)


PARTIAL_EXIT_TAKEN = _load_partial_state()
RSI_HIGH_MEMORY: dict[str, dict[str, float]] = {}
DEBUG_STATE: dict[str, object] = {
    "symbol": None,
    "flags": {"partial": False, "macd": False, "pattern": False},
}

LOW_SIGNAL_STOP_COOLDOWN_PATH = Path(BASE_DIR) / "data" / "monitor_stop_cooldowns.json"


def _load_stop_cooldowns() -> dict[str, str]:
    try:
        return json.loads(LOW_SIGNAL_STOP_COOLDOWN_PATH.read_text())
    except Exception:
        return {}


def _save_stop_cooldowns(state: dict[str, str]) -> None:
    try:
        LOW_SIGNAL_STOP_COOLDOWN_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.error("Unable to persist stop cooldowns: %s", exc)


LOW_SIGNAL_STOP_COOLDOWNS = _load_stop_cooldowns()
STOP_ATTACH_COOLDOWN_HOURS = int(os.getenv("STOP_ATTACH_COOLDOWN_HOURS", "24"))


def _parse_iso_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _mark_symbol_protected(symbol: str, order_ids: list[str]) -> None:
    state = MONITOR_STATE.setdefault("stop_attach", {})
    state[symbol] = {
        "last_state": "protected",
        "last_missing_date": None,
        "last_attempt_utc": None,
        "last_seen_order_ids": order_ids,
    }
    _save_monitor_state(MONITOR_STATE)


def _record_stop_missing(symbol: str, today: str) -> bool:
    state = MONITOR_STATE.setdefault("stop_attach", {})
    current = state.get(symbol) or {}
    first_missing_today = current.get("last_missing_date") != today
    state[symbol] = {
        "last_state": "missing",
        "last_missing_date": today,
        "last_attempt_utc": current.get("last_attempt_utc"),
        "last_seen_order_ids": [],
    }
    _save_monitor_state(MONITOR_STATE)
    return first_missing_today


def _can_attempt_stop_attach(symbol: str, now: datetime) -> bool:
    state = MONITOR_STATE.setdefault("stop_attach", {}).get(symbol) or {}
    previous_state = state.get("last_state")
    last_attempt = _parse_iso_datetime(state.get("last_attempt_utc"))

    if previous_state == "protected":
        return True

    if not last_attempt:
        return True

    return now - last_attempt >= timedelta(hours=STOP_ATTACH_COOLDOWN_HOURS)


def _mark_stop_attach_attempt(symbol: str, now: datetime) -> None:
    state = MONITOR_STATE.setdefault("stop_attach", {})
    current = state.get(symbol) or {}
    current.update({
        "last_state": current.get("last_state", "missing"),
        "last_attempt_utc": now.isoformat(),
    })
    state[symbol] = current
    _save_monitor_state(MONITOR_STATE)

_load_env_if_needed()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Constants
SLEEP_INTERVAL = int(os.getenv("MONITOR_SLEEP_INTERVAL", "60"))
OFF_HOUR_SLEEP_INTERVAL = int(os.getenv("MONITOR_OFF_HOUR_SLEEP", "300"))
TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", "4"))
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", "20"))
EASTERN_TZ = pytz.timezone("US/Eastern")

# Trailing stop and position exit configuration
TRAIL_START_PERCENT = float(os.getenv("TRAIL_START_PERCENT", "3"))
TRAIL_TIGHT_PERCENT = float(os.getenv("TRAIL_TIGHT_PERCENT", "2"))
TRAIL_TIGHTEST_PERCENT = float(os.getenv("TRAIL_TIGHTEST_PERCENT", "1"))
GAIN_THRESHOLD_ADJUST = float(os.getenv("GAIN_THRESHOLD_ADJUST", "10"))
PARTIAL_GAIN_THRESHOLD = float(os.getenv("PARTIAL_GAIN_THRESHOLD", "5"))
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "7"))
LOW_SIGNAL_GAIN_THRESHOLD = float(os.getenv("LOW_SIGNAL_GAIN_THRESHOLD", "4"))
LOW_SIGNAL_TIGHTEN_DELTA = float(os.getenv("LOW_SIGNAL_TIGHTEN_DELTA", "0.75"))
LOW_SIGNAL_TIGHTEN_MIN = float(os.getenv("LOW_SIGNAL_TIGHTEN_MIN", "0.5"))
LOW_SIGNAL_TIGHTEN_MAX = float(os.getenv("LOW_SIGNAL_TIGHTEN_MAX", "1.0"))
LOW_SIGNAL_TRAIL_FLOOR = float(os.getenv("LOW_SIGNAL_TRAIL_FLOOR", "1.5"))
LOW_SIGNAL_TIGHTEN_COOLDOWN_HOURS = int(
    os.getenv("LOW_SIGNAL_TIGHTEN_COOLDOWN_HOURS", "24")
)

# Minimum number of historical bars required for indicator calculation
required_bars = 200


def refresh_ml_risk_state() -> dict:
    """Reload the ML signal quality file and store a sanitized version."""

    global ML_RISK_STATE
    ML_RISK_STATE = _sanitize_ranker_eval_state(_load_ranker_eval())
    return ML_RISK_STATE


def _is_low_signal_cooldown(symbol: str, now: datetime) -> bool:
    last_ts = LOW_SIGNAL_STOP_COOLDOWNS.get(symbol)
    if not last_ts:
        return False
    try:
        last_dt = datetime.fromisoformat(last_ts)
    except Exception:
        return False
    return now - last_dt < timedelta(hours=LOW_SIGNAL_TIGHTEN_COOLDOWN_HOURS)


def _mark_low_signal_cooldown(symbol: str, now: datetime) -> None:
    LOW_SIGNAL_STOP_COOLDOWNS[symbol] = now.isoformat()
    _save_stop_cooldowns(LOW_SIGNAL_STOP_COOLDOWNS)


def is_extended_hours(now_et: dt_time) -> bool:
    """Return True if ``now_et`` falls within pre/post market."""
    return dt_time(4, 0) <= now_et < dt_time(9, 30) or now_et >= dt_time(16, 0)


def calculate_days_held(position) -> int:
    """Return the number of days the position has been held."""
    entry_ts = getattr(position, "created_at", None)
    if entry_ts is None:
        return 0
    try:
        entry_dt = pd.to_datetime(entry_ts, utc=True)
    except Exception:
        return 0
    return (datetime.now(timezone.utc) - entry_dt).days


# Fetch current positions
def get_open_positions():
    try:
        positions = trading_client.get_all_positions()
        logger.info(f"Fetched {len(positions)} positions from Alpaca API.")
        return positions
    except Exception as e:
        logger.error("Failed to fetch open positions: %s", e)
        increment_metric("api_errors")
        return []


# Save open positions to CSV for dashboard consumption
def save_positions_csv(positions, csv_path):
    REQUIRED_COLUMNS = [
        "symbol",
        "qty",
        "avg_entry_price",
        "current_price",
        "net_pnl",
        "unrealized_pl",
        "entry_price",
        "entry_time",
        "days_in_trade",
    ]

    if positions:
        positions_df = pd.DataFrame([p.__dict__ for p in positions])
    else:
        positions_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in positions_df.columns:
            positions_df[col] = None

    positions_df["entry_time"] = pd.to_datetime(positions_df["entry_time"])
    positions_df["days_in_trade"] = (datetime.now() - positions_df["entry_time"]).dt.days

    positions_df.to_csv(csv_path, index=False)
    logger.info(f"Positions saved successfully to {csv_path}")


def update_open_positions():
    """Fetch positions, persist to CSV and log any closed trades."""
    csv_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
    if os.path.exists(csv_path):
        existing_positions_df = pd.read_csv(csv_path)
    else:
        existing_positions_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    positions = get_open_positions()
    save_positions_csv(positions, csv_path)

    existing_symbols = set(existing_positions_df.get("symbol", []))
    current_symbols = set(p.symbol for p in positions)
    closed_symbols = existing_symbols - current_symbols

    log_closed_positions(trading_client, closed_symbols, existing_positions_df)

    return positions


def fetch_indicators(symbol):
    """Fetch recent daily bars and compute indicators."""
    try:
        start_date = datetime.now(timezone.utc) - timedelta(days=750)
        bars = fetch_bars_with_cutoff(symbol, start_date, data_client)
        logger.info(f"Successfully fetched bars for {symbol}")
    except Exception as e:
        logger.exception(f"Failed to fetch bars for {symbol}: {e}")
        return None

    logger.debug(
        f"{symbol}: Screener-bar-count={len(bars)}, Monitor-threshold={required_bars}"
    )

    if bars.empty or len(bars) < required_bars:
        logger.warning("Not enough bars for %s indicator calculation", symbol)
        logger.info(
            "Skipping indicator evaluation for %s due to insufficient bars.",
            symbol,
        )
        return None

    bars["SMA9"] = bars["close"].rolling(9).mean()
    bars["EMA20"] = bars["close"].ewm(span=20).mean()

    delta = bars["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    bars["RSI"] = 100 - (100 / (1 + rs))

    ema12 = bars["close"].ewm(span=12, adjust=False).mean()
    ema26 = bars["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    bars["MACD"] = macd
    bars["MACD_signal"] = macd.ewm(span=9, adjust=False).mean()

    last = bars.iloc[-1]
    prev = bars.iloc[-2] if len(bars) >= 2 else last
    return {
        "close": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "SMA9": float(last["SMA9"]),
        "EMA20": float(last["EMA20"]),
        "RSI": float(last["RSI"]),
        "MACD": float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
        "MACD_prev": float(prev["MACD"]),
        "MACD_signal_prev": float(prev["MACD_signal"]),
    }


def is_shooting_star(indicators: dict) -> bool:
    open_price = indicators.get("open")
    close_price = indicators.get("close")
    high_price = indicators.get("high")
    low_price = indicators.get("low")

    if None in {open_price, close_price, high_price, low_price}:
        return False

    real_body = abs(close_price - open_price)
    if real_body == 0:
        return False

    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    is_red = close_price < open_price
    long_upper = upper_shadow > 2 * real_body
    tiny_lower = lower_shadow <= 0.2 * real_body
    return is_red and long_upper and tiny_lower


def check_sell_signal(symbol: str, indicators: dict) -> list:
    """Return list of exit reasons triggered by indicators."""
    price = indicators["close"]
    ema20 = indicators["EMA20"]
    rsi = indicators["RSI"]

    reasons = []
    if price < ema20:
        logger.info("Price %.2f below EMA20 %.2f", price, ema20)
        reasons.append("EMA20 cross")
    if rsi > 70:
        logger.info("RSI %.2f above 70", rsi)
        reasons.append("RSI > 70")

    if indicators.get("MACD") is not None and indicators.get("MACD_signal") is not None:
        macd = indicators["MACD"]
        signal = indicators["MACD_signal"]
        prev_macd = indicators.get("MACD_prev", macd)
        prev_signal = indicators.get("MACD_signal_prev", signal)
        if macd < signal:
            cross_new = prev_macd >= prev_signal
            reasons.append("MACD cross")
            if cross_new:
                logger.info("MACD crossed below signal for %s", symbol)
            else:
                logger.info("MACD remains below signal for %s", symbol)

    if is_shooting_star(indicators):
        reasons.append("Shooting star")

    state = RSI_HIGH_MEMORY.get(symbol, {"price": price, "rsi": rsi})
    if rsi > 70 and price > state.get("price", price) and rsi < state.get("rsi", rsi):
        reasons.append("RSI divergence")
    if price > state.get("price", price) and rsi >= state.get("rsi", rsi):
        RSI_HIGH_MEMORY[symbol] = {"price": price, "rsi": rsi}
    elif symbol not in RSI_HIGH_MEMORY:
        RSI_HIGH_MEMORY[symbol] = state

    return reasons


def apply_debug_overrides(symbol: str, indicators: dict, position, state: dict) -> None:
    """Mutate ``indicators`` to force exit paths once per run when debug is enabled."""

    if not is_debug_mode():
        return

    target_symbol = state.get("symbol") or os.getenv("JBRAVO_MONITOR_DEBUG_SYMBOL", symbol)
    state.setdefault("flags", {"partial": False, "macd": False, "pattern": False})

    if state.get("symbol") is None:
        state["symbol"] = target_symbol

    if symbol != state.get("symbol"):
        return

    flags: dict = state["flags"]

    entry_price = float(getattr(position, "avg_entry_price", 0) or 0)
    if not flags.get("partial") and entry_price > 0:
        indicators["close"] = round_price(entry_price * 1.06)
        indicators["RSI"] = max(indicators.get("RSI", 0), 72)
        logger.info("[DEBUG_SMOKE] Forcing partial exit for %s", symbol)
        flags["partial"] = True
        return

    if not flags.get("macd"):
        signal = indicators.get("MACD_signal", 0)
        indicators["MACD_signal"] = signal
        indicators["MACD"] = signal - 1
        indicators["MACD_prev"] = signal + 1
        indicators["MACD_signal_prev"] = signal
        logger.info("[DEBUG_SMOKE] Forcing MACD cross-down exit for %s", symbol)
        flags["macd"] = True
        return

    if not flags.get("pattern"):
        open_price = indicators.get("open", 100)
        indicators["open"] = open_price
        indicators["close"] = open_price - 1
        indicators["high"] = open_price + 3
        indicators["low"] = indicators["close"] - 0.1
        logger.info("[DEBUG_SMOKE] Forcing shooting-star exit for %s", symbol)
        flags["pattern"] = True


def get_open_orders(symbol):
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    try:
        open_orders = trading_client.get_orders(request)
        logger.info(
            f"Fetched open orders for {symbol}: {len(open_orders)} found.")
        return list(open_orders)
    except Exception as e:
        logger.error("Failed to fetch open orders for %s: %s", symbol, e)
        increment_metric("api_errors")
        return []


def get_trailing_stop_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if getattr(o, "order_type", "") == "trailing_stop":
            return o
    return None


def _normalize_order_side(order) -> str:
    side_value = getattr(order, "side", "")
    if hasattr(side_value, "value"):
        side_value = side_value.value
    return str(side_value).lower()


def _is_protective_order(order, expected_side: str) -> bool:
    order_type = str(getattr(order, "order_type", "")).lower()
    if order_type not in {"trailing_stop", "stop"}:
        return False
    return _normalize_order_side(order) == expected_side


def last_filled_trailing_stop(symbol):
    request = GetOrdersRequest(symbols=[symbol], limit=10)
    try:
        orders = trading_client.get_orders(request)
        closed_orders = [
            o
            for o in orders
            if getattr(o, "status", "").lower() in ("filled", "canceled", "rejected")
        ]
        for o in closed_orders:
            if getattr(o, "order_type", "") == "trailing_stop" and getattr(o, "status", "").lower() == "filled":
                return o
    except Exception as e:
        logger.error("Failed to fetch order history for %s: %s", symbol, e)
        increment_metric("api_errors")
    return None


def count_open_trailing_stops() -> int:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    try:
        orders = trading_client.get_orders(request)
        count = sum(1 for order in orders if getattr(order, "order_type", "") == "trailing_stop")
        logger.info("[MONITOR] Open trailing stops: %s", count)
        return int(count)
    except Exception as exc:
        logger.error("Failed to count trailing stops: %s", exc)
        increment_metric("api_errors")
        return 0


def _compute_stop_price_from_trail(current_price: float, trail_percent: float) -> float:
    return round_price(current_price * (1 + float(trail_percent) / 100))


def log_trade_exit(
    symbol: str,
    qty: float,
    entry_price: float,
    exit_price: str,
    entry_time: str,
    exit_time: str,
    order_status: str,
    order_type: str,
    exit_reason: str,
    side: str,
    order_id: Optional[str] = None,
):
    """Append real trade record to ``trades_log_real.csv``."""

    trade_entry = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "net_pnl": (float(exit_price) - float(entry_price)) * float(qty) if exit_price else None,
        "order_status": order_status,
        "order_type": order_type,
        "exit_reason": exit_reason,
        "side": side,
        "order_id": order_id or "",
    }

    trades_log_df = pd.read_csv(trades_log_real_path)
    trades_log_df = pd.concat([trades_log_df, pd.DataFrame([trade_entry])], ignore_index=True)
    trades_log_df.to_csv(trades_log_real_path, index=False)
    logger.info(
        f"Real trade exit logged: {symbol}, qty={qty}, exit={exit_price}, order_id={order_id or 'N/A'}"
    )


def log_closed_positions(trading_client, closed_symbols, existing_positions_df):
    for symbol in closed_symbols:
        existing_record = existing_positions_df[existing_positions_df["symbol"] == symbol].iloc[0]

        entry_price = existing_record["entry_price"]
        entry_time = existing_record["entry_time"]
        qty = existing_record["qty"]

        request = GetOrdersRequest(symbols=[symbol], statuses=[QueryOrderStatus.CLOSED])
        closed_orders = trading_client.get_orders(request)

        exit_price = closed_orders[0].filled_avg_price if closed_orders else None
        exit_time = closed_orders[0].filled_at.isoformat() if closed_orders else datetime.now().isoformat()

        log_trade_exit(
            symbol,
            qty,
            entry_price,
            exit_price if exit_price is not None else "",
            entry_time,
            exit_time,
            "closed",
            "market",
            "Position closed outside monitor",
            "sell",
        )
        if symbol in PARTIAL_EXIT_TAKEN:
            PARTIAL_EXIT_TAKEN.pop(symbol, None)
            _save_partial_state(PARTIAL_EXIT_TAKEN)


def _determine_position_side(position) -> str:
    side_val = str(getattr(position, "side", "")).lower()
    if side_val in {"long", "short"}:
        return side_val
    try:
        qty_val = float(getattr(position, "qty", 0))
        return "short" if qty_val < 0 else "long"
    except Exception:
        return "long"


def _get_available_qty(position) -> int:
    try:
        qty_available = float(getattr(position, "qty_available", position.qty))
    except Exception:
        qty_available = float(getattr(position, "qty", 0))
    return abs(int(qty_available))


def _attach_long_protective_stop(position, trail_percent: float) -> bool:
    symbol = position.symbol
    qty = _get_available_qty(position)
    logger.info(
        "STOP_ATTACH_ATTEMPT symbol=%s side=long type=trailing_sell trail_pct=%s",
        symbol,
        trail_percent,
    )
    if qty <= 0:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=long type=trailing_sell error=no_available_qty",
            symbol,
        )
        increment_metric("stop_attach_failed")
        return False
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=str(trail_percent),
        )
        order = trading_client.submit_order(order_data=request)
        logger.info(
            "STOP_ATTACH_OK symbol=%s side=long type=trailing_sell order_id=%s",
            symbol,
            getattr(order, "id", None),
        )
        increment_metric("stops_attached")
        return True
    except Exception as exc:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=long type=trailing_sell error=%s",
            symbol,
            exc,
        )
        increment_metric("stop_attach_failed")
        increment_metric("api_errors")
        return False


def _attach_short_protective_stop(position, trail_percent: float) -> bool:
    symbol = position.symbol
    qty = _get_available_qty(position)
    logger.info(
        "STOP_ATTACH_ATTEMPT symbol=%s side=short type=trailing_buy trail_pct=%s",
        symbol,
        trail_percent,
    )
    if qty <= 0:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=short type=trailing_buy error=no_available_qty",
            symbol,
        )
        increment_metric("stop_attach_failed")
        return False
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            trail_percent=str(trail_percent),
        )
        order = trading_client.submit_order(order_data=request)
        logger.info(
            "STOP_ATTACH_OK symbol=%s side=short type=trailing_buy order_id=%s",
            symbol,
            getattr(order, "id", None),
        )
        increment_metric("stops_attached")
        return True
    except Exception as exc:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=short type=trailing_buy error=%s",
            symbol,
            exc,
        )
        increment_metric("stop_attach_failed")
        increment_metric("api_errors")

    stop_price = _compute_stop_price_from_trail(
        float(getattr(position, "current_price", position.avg_entry_price)), trail_percent
    )
    logger.info(
        "STOP_ATTACH_ATTEMPT symbol=%s side=short type=stop_buy trail_pct=%s stop_price=%.2f",
        symbol,
        trail_percent,
        stop_price,
    )
    try:
        request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC,
        )
        order = trading_client.submit_order(order_data=request)
        logger.info(
            "STOP_ATTACH_OK symbol=%s side=short type=stop_buy order_id=%s",
            symbol,
            getattr(order, "id", None),
        )
        increment_metric("stops_attached")
        return True
    except Exception as exc:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=short type=stop_buy error=%s",
            symbol,
            exc,
        )
        increment_metric("stop_attach_failed")
        increment_metric("api_errors")
        return False


def enforce_stop_coverage(positions: list) -> tuple[int, float, int]:
    if not positions:
        MONITOR_METRICS["stop_coverage_pct"] = 0.0
        _persist_metrics()
        return 0, 0.0, 0

    try:
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
    except Exception as exc:
        logger.error("Failed to fetch open orders for stop coverage: %s", exc)
        increment_metric("api_errors")
        MONITOR_METRICS["stop_coverage_pct"] = 0.0
        _persist_metrics()
        return 0, 0.0, 0

    trailing_stops_count = sum(
        1 for order in open_orders if getattr(order, "order_type", "") == "trailing_stop"
    )

    orders_by_symbol: dict[str, list] = {}
    for order in open_orders:
        orders_by_symbol.setdefault(getattr(order, "symbol", ""), []).append(order)

    protected_symbols: set[str] = set()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    for position in positions:
        symbol = position.symbol
        side = _determine_position_side(position)
        expected_side = "sell" if side == "long" else "buy"
        symbol_orders = orders_by_symbol.get(symbol, [])
        protective_orders = [
            order for order in symbol_orders if _is_protective_order(order, expected_side)
        ]

        if protective_orders:
            _mark_symbol_protected(
                symbol, [str(getattr(order, "id", "")) for order in protective_orders]
            )
            protected_symbols.add(symbol)
            continue

        if _record_stop_missing(symbol, today):
            logger.warning("[WARN] STOP_MISSING symbol=%s side=%s", symbol, side)
            increment_metric("stops_missing")

        if not _can_attempt_stop_attach(symbol, now):
            continue

        _mark_stop_attach_attempt(symbol, now)
        attached = (
            _attach_long_protective_stop(position, TRAIL_START_PERCENT)
            if side == "long"
            else _attach_short_protective_stop(position, TRAIL_START_PERCENT)
        )
        if attached:
            protected_symbols.add(symbol)
            _mark_symbol_protected(symbol, [])

    protective_orders_count = len(protected_symbols)
    coverage_pct = (
        protective_orders_count / len(positions) if positions else 0.0
    )
    MONITOR_METRICS["stop_coverage_pct"] = float(coverage_pct)
    _persist_metrics()
    return protective_orders_count, float(coverage_pct), trailing_stops_count


def has_pending_sell_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if o.side == OrderSide.SELL and getattr(o, "order_type", "") != "trailing_stop":
            return True
    return False


def submit_new_trailing_stop(symbol: str, qty: int, trail_percent: float) -> None:
    """Submit a trailing stop order for ``symbol``."""
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=str(trail_percent),
        )
        order = trading_client.submit_order(order_data=request)
        logger.info(
            f"Placed trailing stop for {symbol}: qty={qty}, trail_pct={trail_percent}"
        )
        log_trailing_stop_event(
            symbol,
            float(trail_percent),
            str(getattr(order, "id", None)),
            "submitted",
        )
        increment_metric("stops_attached")
    except Exception as exc:
        logger.error("Failed to submit trailing stop for %s: %s", symbol, exc)
        log_trailing_stop_event(symbol, float(trail_percent), None, "error")
        increment_metric("api_errors")


def manage_trailing_stop(position):
    symbol = position.symbol
    logger.info(f"Evaluating trailing stop for {symbol}")
    if symbol.upper() == "ARQQ":
        # Explicit log entry requested to verify ARQQ trailing stop evaluation
        logger.info("[INFO] Evaluating trailing stop for ARQQ")
    qty = position.qty
    logger.info(f"Evaluating trailing stop for {symbol} – qty: {qty}")
    entry = float(position.avg_entry_price)
    current = float(position.current_price)
    gain_pct = (current - entry) / entry * 100 if entry else 0
    ml_signal_quality = ML_RISK_STATE.get("signal_quality", "MEDIUM")
    if not qty or float(qty) <= 0:
        logger.info(
            f"Skipping trailing stop for {symbol} due to non-positive quantity: {qty}."
        )
        log_trailing_stop_event(symbol, TRAIL_START_PERCENT, None, "skipped")
        return
    try:
        existing_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
    except Exception as exc:
        logger.error("Failed to fetch open orders for %s: %s", symbol, exc)
        existing_orders = []

    trailing_stops = [
        order for order in existing_orders if getattr(order, "order_type", "") == "trailing_stop"
    ]

    if len(trailing_stops) > 1:
        for order in trailing_stops[1:]:
            try:
                trading_client.cancel_order_by_id(order.id)
                logger.info(
                    f"Cancelled redundant trailing-stop order {order.id} for {symbol}"
                )
            except Exception as exc:
                logger.error(
                    "Failed to cancel trailing stop %s for %s: %s",
                    order.id,
                    symbol,
                    exc,
                )
    trailing_stops = trailing_stops[:1]

    if not trailing_stops:
        available_qty = int(getattr(position, "qty_available", 0))
        if available_qty > 0:
            submit_new_trailing_stop(symbol, available_qty, TRAIL_START_PERCENT)
        else:
            logger.warning(f"No available quantity for trailing stop on {symbol}.")
            log_trailing_stop_event(symbol, TRAIL_START_PERCENT, None, "skipped")
            increment_metric("stops_missing")
        return

    trailing_order = trailing_stops[0]

    available_qty = int(getattr(position, "qty_available", 0))
    if available_qty <= 0:
        logger.warning(
            f"Insufficient available qty for {symbol}: {available_qty}"
        )
        log_trailing_stop_event(
            symbol,
            float(getattr(trailing_order, "trail_percent", TRAIL_START_PERCENT)),
            str(getattr(trailing_order, "id", None)),
            "skipped",
        )
        return
    use_qty = available_qty

    logger.debug(
        f"Entry={entry}, Current={current}, Gain={gain_pct:.2f}% for {symbol}."
    )
    logger.info(
        "Existing trailing stop for %s (order id %s, status %s, trail %% %s)",
        symbol,
        trailing_order.id,
        trailing_order.status,
        getattr(trailing_order, "trail_percent", "n/a"),
    )

    def desired_trail_pct(gain: float) -> float:
        if gain >= 10:
            return TRAIL_TIGHTEST_PERCENT
        if gain >= 5:
            return TRAIL_TIGHT_PERCENT
        return TRAIL_START_PERCENT

    current_trail_pct = float(
        getattr(trailing_order, "trail_percent", TRAIL_START_PERCENT) or TRAIL_START_PERCENT
    )
    target_trail_pct = desired_trail_pct(gain_pct)
    effective_target = min(current_trail_pct, target_trail_pct)

    now_utc = datetime.utcnow()
    low_signal_tighten = False
    if (
        ml_signal_quality == "LOW"
        and gain_pct >= LOW_SIGNAL_GAIN_THRESHOLD
        and not _is_low_signal_cooldown(symbol, now_utc)
    ):
        tighten_delta = min(
            LOW_SIGNAL_TIGHTEN_MAX,
            max(LOW_SIGNAL_TIGHTEN_MIN, LOW_SIGNAL_TIGHTEN_DELTA),
        )
        proposed_low_signal_target = max(
            LOW_SIGNAL_TRAIL_FLOOR, current_trail_pct - tighten_delta
        )
        if proposed_low_signal_target + 1e-6 < effective_target:
            effective_target = proposed_low_signal_target
            low_signal_tighten = True
    elif ml_signal_quality == "LOW" and gain_pct >= LOW_SIGNAL_GAIN_THRESHOLD:
        logger.info(
            "[INFO] STOP_TIGHTEN_COOLDOWN signal_quality=LOW symbol=%s last=%s",
            symbol,
            LOW_SIGNAL_STOP_COOLDOWNS.get(symbol),
        )

    if abs(current_trail_pct - effective_target) < 1e-6:
        logger.info(
            "No trailing stop adjustment needed for %s (gain: %.2f%%; trail %.2f%%)",
            symbol,
            gain_pct,
            current_trail_pct,
        )
        return

    try:
        cancel_order_safe(trailing_order.id, symbol)
        reason_detail = (
            "low_signal_profit_lock"
            if low_signal_tighten
            else "+10% gain"
            if effective_target == TRAIL_TIGHTEST_PERCENT
            else "+5% gain"
            if effective_target == TRAIL_TIGHT_PERCENT
            else "default"
        )
        if low_signal_tighten:
            logger.info(
                "[INFO] STOP_TIGHTEN signal_quality=LOW symbol=%s old_trail=%.2f new_trail=%.2f reason=low_signal_profit_lock",
                symbol,
                current_trail_pct,
                effective_target,
            )
        logger.info(
            "Tightening trailing stop for %s from %.2f%% to %.2f%% due to %s",
            symbol,
            current_trail_pct,
            effective_target,
            reason_detail,
        )
        increment_metric("stops_tightened")
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=use_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=str(effective_target),
        )
        trading_client.submit_order(order_data=request)
        logger.info(
            "Adjusted trailing stop for %s from %.2f%% to %.2f%% (gain: %.2f%%).",
            symbol,
            current_trail_pct,
            effective_target,
            gain_pct,
        )
        log_trailing_stop_event(symbol, effective_target, None, "adjusted")
        if low_signal_tighten:
            _mark_low_signal_cooldown(symbol, now_utc)
    except Exception as e:
        logger.error("Failed to adjust trailing stop for %s: %s", symbol, e)
        try:
            trading_client.close_position(symbol)
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to close position %s: %s", symbol, exc)


def check_pending_orders():
    """Log status of open sell orders and remove redundant trailing stops."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(request)
        logger.info(
            f"Fetched open orders for all symbols: {len(open_orders)} found.")

        trailing_stops = [
            o for o in open_orders if getattr(o, "order_type", "") == "trailing_stop"
        ]
        unique_symbols = set()
        for order in trailing_stops:
            if order.symbol in unique_symbols:
                try:
                    trading_client.cancel_order_by_id(order.id)
                    logger.info(
                        f"Cancelled redundant trailing stop {order.id} for {order.symbol}"
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to cancel trailing stop %s for %s: %s",
                        order.id,
                        order.symbol,
                        exc,
                    )
            else:
                unique_symbols.add(order.symbol)

        for order in open_orders:
            try:
                status = trading_client.get_order_by_id(order.id).status
                status = status.value if hasattr(status, "value") else status
                if status in ["submitted", "pending"]:
                    logger.info(
                        f"Pending order {order.id} for {order.symbol} status: {status}."
                    )
            except Exception as exc:
                logger.error("Failed to fetch status for %s: %s", order.id, exc)
    except Exception as exc:
        logger.error("Failed to list open orders: %s", exc)


# Execute sell orders


def submit_sell_market_order(position, reason: str, reason_code: str, qty_override: Optional[int] = None):
    """Submit a limit sell order compatible with extended hours."""
    symbol = position.symbol
    qty = position.qty
    entry_price = float(position.avg_entry_price)
    exit_price = round_price(float(getattr(position, "current_price", entry_price)))
    entry_time = getattr(position, "created_at", datetime.utcnow()).isoformat()
    now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ).time()

    desired_qty = int(qty_override) if qty_override is not None else int(qty)
    if getattr(position, "qty_available", None) and int(position.qty_available) >= desired_qty:
        use_qty = int(position.qty_available)
    else:
        logger.warning(
            f"Insufficient available qty for {symbol}: {getattr(position, 'qty_available', 0)}"
        )
        return

    try:
        try:
            available_qty_attr = int(getattr(position, "qty_available", use_qty))
        except Exception:
            available_qty_attr = int(use_qty)
        if available_qty_attr <= 0:
            logger.warning(
                "Insufficient available qty for %s: %s", symbol, getattr(position, "qty_available", 0)
            )
            return
        try:
            positions = trading_client.get_all_positions()
            existing_symbols = {p.symbol: int(getattr(p, "qty_available", p.qty)) for p in positions}
            available_qty = existing_symbols.get(symbol, available_qty_attr)
        except Exception:
            available_qty = available_qty_attr

        order_qty = min(int(use_qty), available_qty)
        if available_qty < order_qty or order_qty <= 0:
            logger.info(
                f"Skipped selling {symbol}: insufficient available quantity ({available_qty})."
            )
            return
        session = "extended" if is_extended_hours(now_et) else "regular"
        logger.info(
            "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
            symbol,
            order_qty,
            exit_price,
            session,
        )
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=order_qty,
            side="sell",
            type="limit",
            limit_price=exit_price,
            time_in_force="day",
            extended_hours=is_extended_hours(now_et),
        )
        submit_ts = datetime.utcnow()
        order = trading_client.submit_order(order_request)
        log_exit_submit(symbol, order_qty, "limit", reason_code)
        status = wait_for_order_terminal(str(getattr(order, "id", "")))
        latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
        log_exit_final(status, latency_ms)
        logger.info(
            "[EXIT] Limit sell %s qty %s at %.2f due to %s",
            symbol,
            order_qty,
            exit_price,
            reason,
        )
        log_trade_exit(
            symbol,
            float(order_qty),
            entry_price,
            str(exit_price),
            entry_time,
            "",
            status,
            "limit",
            reason,
            "sell",
            order_id=getattr(order, "id", None),
        )
    except Exception as e:
        logger.error("Error submitting sell order for %s: %s", symbol, e)
        increment_metric("api_errors")
        try:
            trading_client.close_position(symbol)
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to close position %s: %s", symbol, exc)


def process_positions_cycle():
    csv_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
    if os.path.exists(csv_path):
        existing_positions_df = pd.read_csv(csv_path)
    else:
        existing_positions_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    ml_state = refresh_ml_risk_state()
    logger.info(
        "[MONITOR] ML risk state: signal_quality=%s decile_lift=%s",
        ml_state.get("signal_quality"),
        ml_state.get("decile_lift"),
    )

    positions = get_open_positions()
    save_positions_csv(positions, csv_path)

    existing_symbols = set(existing_positions_df.get("symbol", []))
    current_symbols = set(p.symbol for p in positions)
    closed_symbols = existing_symbols - current_symbols

    log_closed_positions(trading_client, closed_symbols, existing_positions_df)

    if not positions:
        logger.info("No open positions found.")
        return positions
    for position in positions:
        symbol = position.symbol
        days_held = calculate_days_held(position)
        logger.info("%s held for %d days", symbol, days_held)
        if days_held >= MAX_HOLD_DAYS:
            if has_pending_sell_order(symbol):
                logger.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                try:
                    cancel_order_safe(trailing_order.id, symbol)
                    logger.info(
                        "Cancelled trailing stop for %s due to max hold exit",
                        symbol,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to cancel trailing stop for %s: %s", symbol, e
                    )
            increment_metric("exits_max_hold")
            submit_sell_market_order(position, reason=f"Max Hold {days_held}d", reason_code="max_hold")
            if symbol in PARTIAL_EXIT_TAKEN:
                PARTIAL_EXIT_TAKEN.pop(symbol, None)
                _save_partial_state(PARTIAL_EXIT_TAKEN)
            continue

        indicators = fetch_indicators(symbol)
        if not indicators:
            logger.info(
                "Indicators unavailable for %s. Trailing-stop logic will be skipped.",
                symbol,
            )
            continue

        apply_debug_overrides(symbol, indicators, position, DEBUG_STATE)

        logger.info(
            "Evaluating sell conditions for %s - price: %.2f, EMA20: %.2f, RSI: %.2f",
            symbol,
            indicators["close"],
            indicators["EMA20"],
            indicators["RSI"],
        )

        entry_price = float(position.avg_entry_price)
        gain_pct = (indicators["close"] - entry_price) / entry_price * 100 if entry_price else 0

        if (
            gain_pct >= PARTIAL_GAIN_THRESHOLD
            and not PARTIAL_EXIT_TAKEN.get(symbol)
            and not has_pending_sell_order(symbol)
        ):
            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                cancel_order_safe(trailing_order.id, symbol)
            half_qty = max(1, int(float(position.qty) / 2))
            submit_sell_market_order(
                position,
                reason=f"Partial exit at +{PARTIAL_GAIN_THRESHOLD:.0f}% gain",
                reason_code="partial_gain",
                qty_override=half_qty,
            )
            PARTIAL_EXIT_TAKEN[symbol] = True
            _save_partial_state(PARTIAL_EXIT_TAKEN)
            logger.info("Partial exit recorded for %s; awaiting trailing re-attachment.", symbol)
            continue

        reasons = check_sell_signal(symbol, indicators)
        if reasons:
            if has_pending_sell_order(symbol):
                logger.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                try:
                    cancel_order_safe(trailing_order.id, symbol)
                except Exception as e:
                    logger.error(
                        "Failed to cancel trailing stop for %s: %s", symbol, e
                    )
            reason_text = "; ".join(reasons)
            increment_metric("exits_signal")
            submit_sell_market_order(position, reason=reason_text, reason_code="monitor")
            if symbol in PARTIAL_EXIT_TAKEN:
                PARTIAL_EXIT_TAKEN.pop(symbol, None)
                _save_partial_state(PARTIAL_EXIT_TAKEN)
        else:
            logger.info(f"No sell signal for {symbol}; managing trailing stop.")
            manage_trailing_stop(position)

    return positions


def monitor_positions():
    logger.info("[MONITOR_START] Starting real-time position monitoring (pid=%s)", os.getpid())
    write_status(
        status="starting",
        positions_count=0,
        trailing_count=0,
        protective_orders_count=0,
        stop_coverage_pct=0.0,
    )
    while True:
        positions: list = []
        trailing_stops_count = 0
        protective_orders_count = 0
        stop_coverage_pct = 0.0
        loop_status = "running"
        market_hours = True
        try:
            now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ)
            market_hours = TRADING_START_HOUR <= now_et.hour < TRADING_END_HOUR

            log_if_stale(open_pos_path, "open_positions.csv", threshold_minutes=10)

            if market_hours:
                positions = process_positions_cycle() or []
            else:
                positions = update_open_positions() or []

            protective_orders_count, stop_coverage_pct, trailing_stops_count = enforce_stop_coverage(positions)

            check_pending_orders()

            logger.info("Updated open_positions.csv successfully.")
        except Exception as e:
            logger.exception("Monitoring loop error")
            increment_metric("api_errors")
            loop_status = "error"

        record_heartbeat(
            len(positions or []),
            trailing_stops_count,
            protective_orders_count,
            stop_coverage_pct,
            status=loop_status,
        )

        sleep_time = SLEEP_INTERVAL if market_hours else OFF_HOUR_SLEEP_INTERVAL
        time.sleep(sleep_time)


if __name__ == "__main__":
    logger.info("Starting monitor_positions.py")
    if is_debug_mode():
        run_debug_smoke_test()
    while True:
        try:
            monitor_positions()
        except Exception as e:
            logger.exception("Monitoring loop error")
        time.sleep(300)
