# monitor_positions.py

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
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


def configure_logging() -> None:
    root = logging.getLogger()
    level_name = (os.getenv("MONITOR_LOG_LEVEL") or "INFO").strip().upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    desired_level = level_map.get(level_name, logging.INFO)
    has_file = False
    has_stream = False
    for handler in root.handlers:
        if getattr(handler, "name", "") == "jbravo_monitor_file":
            has_file = True
            handler.setLevel(desired_level)
        if isinstance(handler, logging.FileHandler):
            base = getattr(handler, "baseFilename", "")
            if base and os.path.basename(base).lower() == "monitor.log":
                has_file = True
                handler.setLevel(desired_level)
        if getattr(handler, "name", "") == "jbravo_monitor_stream":
            has_stream = True
            handler.setLevel(desired_level)
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            has_stream = True
            handler.setLevel(desired_level)

    if root.level == logging.NOTSET or root.level > desired_level:
        root.setLevel(desired_level)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s")

    if not has_file:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(desired_level)
        file_handler.name = "jbravo_monitor_file"
        root.addHandler(file_handler)

    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(desired_level)
        stream_handler.name = "jbravo_monitor_stream"
        root.addHandler(stream_handler)

    if not getattr(configure_logging, "_ready_logged", False):
        logging.getLogger(__name__).warning(
            "LOGGING_READY root_level=%s desired=%s",
            logging.getLevelName(root.level),
            logging.getLevelName(desired_level),
        )
        configure_logging._ready_logged = True


configure_logging()
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
    "stop_tighten_skipped",
    "stop_tighten_cooldown",
    "breakeven_tightens",
    "time_decay_tightens",
    "live_price_ok",
    "live_price_fail",
    "db_event_ok",
    "db_event_fail",
]

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value_norm = value.strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    return default


MONITOR_DB_LOGGING_ENV = "MONITOR_ENABLE_DB_LOGGING"

# Monitor DB event vocabulary (aligned with execute_trades where possible).
# New values here are additive and do not change behavior unless DB logging is enabled.
MONITOR_DB_EVENT_TYPES = {
    "sell_submit": "SELL_SUBMIT",
    "sell_fill": "SELL_FILL",
    "sell_cancelled": "SELL_CANCELLED",
    "sell_rejected": "SELL_REJECTED",
    "sell_expired": "SELL_EXPIRED",
    "trail_submit": "TRAIL_SUBMIT",
    "trail_adjust": "TRAIL_ADJUST",
    "trail_cancel": "TRAIL_CANCEL",
}


def db_logging_enabled() -> bool:
    if not env_bool(MONITOR_DB_LOGGING_ENV, default=False):
        return False
    try:
        from scripts import db as db_module
    except Exception as exc:
        logger.warning("DB_LOGGING_DISABLED err=%s", exc)
        return False
    try:
        return bool(db_module.db_enabled())
    except Exception as exc:
        logger.warning("DB_LOGGING_DISABLED err=%s", exc)
        return False


def _sanitize_raw_payload(raw: dict | None) -> dict:
    payload = dict(raw or {})
    payload["source"] = "monitor_positions"
    if "dryrun" in payload:
        payload["dryrun"] = bool(payload["dryrun"])
    try:
        return json.loads(json.dumps(payload, default=str))
    except Exception:
        return {"source": "monitor_positions", "raw": str(payload)}


def _log_db_event(message: str, payload: dict, *, level: str | None = None) -> None:
    rendered = json.dumps(payload, sort_keys=True, default=str)
    if message == "DB_EVENT_OK":
        logger.info("DB_EVENT_OK %s", rendered)
        return
    logger.warning("DB_EVENT_FAIL %s", rendered)


def _exit_summary_from_raw(raw_payload: dict | None) -> dict:
    if not isinstance(raw_payload, dict):
        return {
            "exit_reason_code": None,
            "exit_snapshot_present": False,
            "raw_keys": [],
        }
    exit_snapshot = raw_payload.get("exit_snapshot")
    return {
        "exit_reason_code": raw_payload.get("exit_reason_code"),
        "exit_snapshot_present": exit_snapshot is not None,
        "raw_keys": sorted(raw_payload.keys()),
    }


def db_log_event(
    *,
    event_type: str,
    symbol: str,
    qty: float | int | None,
    order_id: str | None,
    status: str | None,
    event_time: datetime | None = None,
    raw: dict | None = None,
) -> bool:
    if not db_logging_enabled():
        return False

    raw_payload = _sanitize_raw_payload(raw)
    exit_summary = _exit_summary_from_raw(raw_payload)

    event_label = (event_type or "").upper()
    if event_label not in set(MONITOR_DB_EVENT_TYPES.values()):
        fail_payload = {
            "event_type": event_label or "",
            "symbol": symbol,
            "qty": qty,
            "order_id": order_id,
            "status": status,
            "event_time": (event_time or datetime.now(timezone.utc)).isoformat(),
            "err": "invalid_event_type",
            **exit_summary,
        }
        _log_db_event("DB_EVENT_FAIL", fail_payload)
        increment_metric("db_event_fail")
        return False

    event_time_value = event_time or datetime.now(timezone.utc)
    payload = {
        "event_type": event_label,
        "symbol": symbol,
        "qty": qty,
        "order_id": order_id,
        "status": status,
        "event_time": event_time_value.isoformat(),
        **exit_summary,
    }
    try:
        from scripts import db as db_module
    except Exception as exc:
        fail_payload = dict(payload)
        fail_payload["err"] = str(exc)
        _log_db_event("DB_EVENT_FAIL", fail_payload)
        increment_metric("db_event_fail")
        return False
    try:
        ok = db_module.insert_order_event(
            event_type=event_label,
            symbol=symbol,
            qty=qty,
            order_id=order_id,
            status=status,
            event_time=event_time_value,
            raw=raw_payload,
        )
    except Exception as exc:
        fail_payload = dict(payload)
        fail_payload["err"] = str(exc)
        _log_db_event("DB_EVENT_FAIL", fail_payload)
        increment_metric("db_event_fail")
        return False

    if ok is True:
        _log_db_event("DB_EVENT_OK", payload)
        increment_metric("db_event_ok")
        return True

    fail_payload = dict(payload)
    fail_payload["err"] = "insert_failed"
    _log_db_event("DB_EVENT_FAIL", fail_payload)
    increment_metric("db_event_fail")
    return False


def _stringify_value(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def enum_str(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().lower()
    if text.startswith("orderside."):
        return text[len("orderside.") :]
    if text.startswith("ordertype."):
        return text[len("ordertype.") :]
    return text


def order_attr_str(order, attr_names: tuple[str, ...]) -> str:
    for name in attr_names:
        if not hasattr(order, name):
            continue
        try:
            value = getattr(order, name)
        except Exception:
            continue
        if value is None:
            continue
        text = enum_str(value)
        if text:
            return text
    return ""


def _order_timestamp(order) -> datetime | None:
    for field in ("submitted_at", "created_at", "created_at_utc", "updated_at", "updated_at_utc"):
        value = getattr(order, field, None)
        if not value:
            continue
        if isinstance(value, datetime):
            return value
        try:
            ts = pd.to_datetime(value, utc=True)
        except Exception:
            continue
        if ts is None or getattr(ts, "to_pydatetime", None) is None:
            continue
        if pd.isna(ts):
            continue
        try:
            return ts.to_pydatetime()
        except Exception:
            continue
    return None


def _order_sort_key(order) -> tuple[datetime, str]:
    ts = _order_timestamp(order)
    if ts is None:
        ts = datetime.min.replace(tzinfo=timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts, str(getattr(order, "id", "") or "")


def get_protective_trailing_stop_for_symbol(symbol: str, orders: list) -> object | None:
    best_order = None
    best_qty = None
    for order in orders or []:
        if getattr(order, "symbol", None) != symbol:
            continue
        if order_attr_str(order, ("order_type", "type")) != "trailing_stop":
            continue
        if order_attr_str(order, ("side",)) != "sell":
            continue
        status = order_attr_str(order, ("status",))
        if status not in {"open", "new", "accepted"}:
            continue
        qty_value = getattr(order, "qty", None)
        try:
            qty = float(qty_value)
        except (TypeError, ValueError):
            qty = None
        if qty is None:
            if best_order is None:
                best_order = order
            continue
        if best_qty is None or qty > best_qty:
            best_order = order
            best_qty = qty
    return best_order


def is_fully_covered(position, stop_order) -> bool:
    if position is None or stop_order is None:
        return False
    try:
        pos_qty = abs(float(getattr(position, "qty", 0) or 0))
    except (TypeError, ValueError):
        return False
    try:
        stop_qty = float(getattr(stop_order, "qty", 0) or 0)
    except (TypeError, ValueError):
        return False
    return stop_qty + 1e-6 >= pos_qty


def _order_request_summary(order_request) -> dict:
    if order_request is None:
        return {}
    payload = {"order_request": order_request.__class__.__name__}
    for field in (
        "symbol",
        "qty",
        "side",
        "type",
        "order_type",
        "limit_price",
        "stop_price",
        "trail_percent",
        "time_in_force",
        "extended_hours",
    ):
        if hasattr(order_request, field):
            payload[field] = _stringify_value(getattr(order_request, field))
    return payload


def _build_dryrun_payload(context: dict | None, order_request=None) -> dict:
    payload = dict(context or {})
    for key, value in _order_request_summary(order_request).items():
        payload.setdefault(key, value)
    return payload


def _dryrun_order_stub(payload: dict) -> types.SimpleNamespace:
    stub = {
        "id": f"dryrun-{int(time.time() * 1000)}",
        "status": "accepted",
        "dryrun": True,
    }
    for key in ("symbol", "qty", "side", "order_type", "type"):
        if key in payload and payload[key] is not None:
            stub[key] = payload[key]
    return types.SimpleNamespace(**stub)


def broker_submit_order(order_request, context: dict | None = None):
    payload = _build_dryrun_payload(context, order_request)
    if env_bool("MONITOR_DISABLE_SELLS", default=False):
        logger.info("DRYRUN_SUBMIT %s", payload)
        return _dryrun_order_stub(payload)
    return trading_client.submit_order(order_data=order_request)


def broker_cancel_order(order_id: str, context: dict | None = None) -> bool:
    payload = dict(context or {})
    payload.setdefault("order_id", order_id)
    if env_bool("MONITOR_DISABLE_SELLS", default=False):
        logger.info("DRYRUN_CANCEL %s", payload)
        return True
    if hasattr(trading_client, "cancel_order_by_id"):
        trading_client.cancel_order_by_id(order_id)
        return True
    if hasattr(trading_client, "cancel_order"):
        trading_client.cancel_order(order_id)
        return True
    return False


def broker_close_position(symbol: str, context: dict | None = None) -> bool:
    payload = dict(context or {})
    payload.setdefault("symbol", symbol)
    if env_bool("MONITOR_DISABLE_SELLS", default=False):
        logger.info("DRYRUN_CLOSE %s", payload)
        return True
    if hasattr(trading_client, "close_position"):
        trading_client.close_position(symbol)
        return True
    return False


def _default_metrics(date_str: str | None = None) -> dict:
    payload = {key: 0 for key in METRIC_KEYS}
    payload["date"] = date_str or datetime.now(timezone.utc).date().isoformat()
    return payload


def _load_metrics() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
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
    today = datetime.now(timezone.utc).date().isoformat()
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
    today = datetime.now(timezone.utc).date().isoformat()
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


def _build_monitor_config() -> dict:
    return {
        "MONITOR_DISABLE_SELLS": env_bool("MONITOR_DISABLE_SELLS", default=False),
        "MONITOR_ENABLE_LIVE_PRICES": env_bool("MONITOR_ENABLE_LIVE_PRICES", default=False),
        "MONITOR_ENABLE_EXIT_SIGNALS_V2": env_bool("MONITOR_ENABLE_EXIT_SIGNALS_V2", default=False),
        "MONITOR_ENABLE_BREAKEVEN_TIGHTEN": env_bool("MONITOR_ENABLE_BREAKEVEN_TIGHTEN", default=False),
        "MONITOR_ENABLE_TIMEDECAY_TIGHTEN": env_bool("MONITOR_ENABLE_TIMEDECAY_TIGHTEN", default=False),
        "MONITOR_ENABLE_DB_LOGGING": env_bool(MONITOR_DB_LOGGING_ENV, default=False),
        "MONITOR_ENABLE_EXIT_INTELLIGENCE": env_bool(
            "MONITOR_ENABLE_EXIT_INTELLIGENCE", default=False
        ),
    }


def assert_paper_mode() -> None:
    base_url = (os.getenv("APCA_API_BASE_URL") or "").strip()
    if "paper-api.alpaca.markets" not in base_url.lower():
        logger.error(
            "[FATAL] Refusing to run: APCA_API_BASE_URL must be paper (%s)",
            base_url or "missing",
        )
        raise SystemExit(2)


def _kill_switch_triggered(kill_switch_path: Path | None) -> bool:
    if not kill_switch_path:
        return False
    try:
        return kill_switch_path.exists()
    except Exception as exc:
        logger.warning("Kill switch check failed for %s: %s", kill_switch_path, exc)
        return False


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
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        os.path.getmtime(file_path),
        timezone.utc,
    )
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

    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    status = "unknown"
    while datetime.now(timezone.utc) <= deadline:
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


def cancel_order_safe(order_id: str, symbol: str, reason: str | None = None) -> bool:
    """Attempt to cancel ``order_id`` using available client methods."""
    context = {"symbol": symbol, "reason": reason}
    try:
        success = broker_cancel_order(order_id, context)
        if success:
            return True
    except Exception as exc:  # pragma: no cover - API errors
        logger.error("Failed to cancel order %s: %s", order_id, exc)
    try:
        broker_close_position(
            symbol,
            {"symbol": symbol, "reason": reason, "action": "close_position_after_cancel_failure"},
        )
    except Exception as exc:  # pragma: no cover - API errors
        logger.error("Failed to close position %s: %s", symbol, exc)
    return False


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
TIGHTEN_COOLDOWN_PATH = Path(BASE_DIR) / "data" / "tighten_cooldowns.json"


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


def _load_tighten_cooldowns() -> dict[str, str]:
    try:
        return json.loads(TIGHTEN_COOLDOWN_PATH.read_text())
    except Exception:
        return {}


def _save_tighten_cooldowns(state: dict[str, str]) -> None:
    try:
        TIGHTEN_COOLDOWN_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.error("Unable to persist tighten cooldowns: %s", exc)


LOW_SIGNAL_STOP_COOLDOWNS = _load_stop_cooldowns()
TIGHTEN_COOLDOWNS = _load_tighten_cooldowns()
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
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", str(PARTIAL_GAIN_THRESHOLD)))
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "7"))
BREAKEVEN_THRESHOLD_PCT = float(os.getenv("BREAKEVEN_THRESHOLD_PCT", "2.0"))
TIMEDECAY_START_DAYS = int(os.getenv("TIMEDECAY_START_DAYS", "2"))
TIMEDECAY_STEP_PCT = float(os.getenv("TIMEDECAY_STEP_PCT", "0.5"))
MOMENTUM_STALL_PCT = float(os.getenv("MOMENTUM_STALL_PCT", "0.5"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_REVERSAL_THRESHOLD = float(os.getenv("RSI_REVERSAL_THRESHOLD", "50"))
RISK_OFF_SCORE_CUTOFF = float(os.getenv("RISK_OFF_SCORE_CUTOFF", "0.7"))
EXIT_SIGNAL_TIGHTEN_CONFIDENCE = float(os.getenv("EXIT_SIGNAL_TIGHTEN_CONFIDENCE", "0.7"))
EXIT_SIGNAL_TIGHTEN_DELTA = float(os.getenv("EXIT_SIGNAL_TIGHTEN_DELTA", "0.5"))
TIGHTEN_COOLDOWN_MINUTES = int(os.getenv("MONITOR_TIGHTEN_COOLDOWN_MINUTES", "15"))
TIGHTEN_COOLDOWN_EPS_PCT = float(os.getenv("MONITOR_TIGHTEN_COOLDOWN_EPS_PCT", "0.05"))
RSI_REVERSAL_DROP = float(os.getenv("RSI_REVERSAL_DROP", "20"))
RSI_REVERSAL_FLOOR = float(os.getenv("RSI_REVERSAL_FLOOR", "50"))
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


def _mark_tighten_cooldown(symbol: str, now: datetime) -> None:
    TIGHTEN_COOLDOWNS[symbol] = now.isoformat()
    _save_tighten_cooldowns(TIGHTEN_COOLDOWNS)


def compute_target_trail_pct_legacy(
    gain_pct: float,
    days_held: int,
    current_trail_pct: float | None,
) -> tuple[float, str]:
    """Return the tightened trailing stop percent and reason detail (legacy)."""

    if gain_pct >= GAIN_THRESHOLD_ADJUST:
        target = TRAIL_TIGHTEST_PERCENT
    elif gain_pct >= PARTIAL_GAIN_THRESHOLD:
        target = TRAIL_TIGHT_PERCENT
    else:
        target = TRAIL_START_PERCENT

    reason_parts: list[str] = []
    eps = 1e-6

    if env_bool("MONITOR_ENABLE_BREAKEVEN_TIGHTEN", default=False):
        if gain_pct >= BREAKEVEN_THRESHOLD_PCT:
            tightened = min(target, float(gain_pct))
            if tightened + eps < target:
                target = tightened
                reason_parts.append("breakeven_lock")

    if env_bool("MONITOR_ENABLE_TIMEDECAY_TIGHTEN", default=False):
        extra_days = int(days_held) - TIMEDECAY_START_DAYS
        if extra_days > 0:
            reduction = TIMEDECAY_STEP_PCT * extra_days
            tightened = max(TRAIL_TIGHTEST_PERCENT, target - reduction)
            if tightened + eps < target:
                target = tightened
                reason_parts.append("time_decay")

    if current_trail_pct is not None:
        try:
            current_val = float(current_trail_pct)
            if current_val + eps < target:
                target = current_val
        except (TypeError, ValueError):
            pass

    reason_detail = "+".join(reason_parts) if reason_parts else "profit_tier"
    return float(target), reason_detail


def _compute_target_trail_meta(
    position,
    indicators: dict | None,
    exit_signals: list[dict] | None,
) -> tuple[float, list[str]]:
    indicators = indicators or {}
    entry_price = float(getattr(position, "avg_entry_price", 0) or 0)
    current_price = indicators.get("close")
    if current_price is None:
        current_price = getattr(position, "live_price", None)
    if current_price is None:
        current_price = getattr(position, "current_price", entry_price)
    try:
        current_price = float(current_price)
    except (TypeError, ValueError):
        current_price = float(entry_price or 0)

    gain_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0.0
    days_held = calculate_days_held(position)

    if gain_pct >= GAIN_THRESHOLD_ADJUST:
        target = TRAIL_TIGHTEST_PERCENT
    elif gain_pct >= PARTIAL_GAIN_THRESHOLD:
        target = TRAIL_TIGHT_PERCENT
    else:
        target = TRAIL_START_PERCENT

    reasons: list[str] = []
    eps = 1e-6

    if env_bool("MONITOR_ENABLE_BREAKEVEN_TIGHTEN", default=False):
        if gain_pct >= BREAKEVEN_THRESHOLD_PCT:
            tightened = min(target, float(gain_pct))
            if tightened + eps < target:
                target = tightened
                reasons.append("breakeven_lock")

    if env_bool("MONITOR_ENABLE_TIMEDECAY_TIGHTEN", default=False):
        extra_days = int(days_held) - TIMEDECAY_START_DAYS
        if extra_days > 0:
            reduction = TIMEDECAY_STEP_PCT * extra_days
            tightened = max(TRAIL_TIGHTEST_PERCENT, target - reduction)
            if tightened + eps < target:
                target = tightened
                reasons.append("time_decay")

    signal_conf = EXIT_SIGNAL_TIGHTEN_CONFIDENCE
    signal_delta = EXIT_SIGNAL_TIGHTEN_DELTA
    has_high_conf = False
    for signal in exit_signals or []:
        try:
            if float(signal.get("confidence", 0)) >= signal_conf:
                has_high_conf = True
                break
        except (TypeError, ValueError):
            continue
    if has_high_conf and signal_delta > 0:
        tightened = max(TRAIL_TIGHTEST_PERCENT, target - signal_delta)
        if tightened + eps < target:
            target = tightened
            reasons.append("exit_signal")

    return float(target), reasons


def compute_target_trail_pct(position, indicators: dict | None, exit_signals: list[dict] | None) -> float:
    """Return the tightened trailing stop percent (exit intelligence)."""

    target, _reasons = _compute_target_trail_meta(position, indicators, exit_signals)
    return float(target)


def is_extended_hours(now_et: dt_time) -> bool:
    """Return True if ``now_et`` falls within pre/post market."""
    return dt_time(4, 0) <= now_et < dt_time(9, 30) or now_et >= dt_time(16, 0)


def calculate_days_held(position, now_utc: datetime | None = None) -> int:
    """Return the number of days the position has been held."""
    entry_ts = getattr(position, "created_at", None)
    if entry_ts is None:
        return 0
    try:
        entry_dt = pd.to_datetime(entry_ts, utc=True)
    except Exception:
        return 0
    now = now_utc or datetime.now(timezone.utc)
    if getattr(now, "tzinfo", None) is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - entry_dt).days


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


def _alpaca_data_feed() -> Optional[str]:
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "").strip()
    if not feed_env:
        return None
    upper = feed_env.upper()
    if upper in {"IEX", "SIP"}:
        return upper.lower()
    return feed_env


def get_latest_trade_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    if data_client is None:
        logger.error("LIVE_PRICE_FAIL err=data_client_unavailable")
        increment_metric("live_price_fail")
        return {}
    feed = _alpaca_data_feed()
    try:
        if feed:
            request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=feed)
        else:
            request = StockLatestTradeRequest(symbol_or_symbols=symbols)
        latest_trades = data_client.get_stock_latest_trade(request) or {}
        prices: dict[str, float] = {}
        for symbol, trade in latest_trades.items():
            price = getattr(trade, "price", None)
            if price is None:
                continue
            try:
                prices[str(symbol)] = float(price)
            except (TypeError, ValueError):
                continue
        missing = max(0, len(symbols) - len(prices))
        logger.info(
            "LIVE_PRICE_OK symbols=%s used=%s missing=%s",
            len(symbols),
            len(prices),
            missing,
        )
        increment_metric("live_price_ok")
        return prices
    except Exception as exc:
        logger.error("LIVE_PRICE_FAIL err=%s", exc)
        increment_metric("live_price_fail")
        return {}


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
    bars["MACD_hist"] = bars["MACD"] - bars["MACD_signal"]

    last = bars.iloc[-1]
    prev = bars.iloc[-2] if len(bars) >= 2 else last
    return {
        "close": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close_prev": float(prev["close"]),
        "open_prev": float(prev["open"]),
        "high_prev": float(prev["high"]),
        "low_prev": float(prev["low"]),
        "SMA9": float(last["SMA9"]),
        "SMA9_prev": float(prev["SMA9"]),
        "EMA20": float(last["EMA20"]),
        "EMA20_prev": float(prev["EMA20"]),
        "RSI": float(last["RSI"]),
        "RSI_prev": float(prev["RSI"]),
        "MACD": float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
        "MACD_hist": float(last["MACD_hist"]),
        "MACD_prev": float(prev["MACD"]),
        "MACD_signal_prev": float(prev["MACD_signal"]),
        "MACD_hist_prev": float(prev["MACD_hist"]),
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


def is_bearish_engulfing(indicators: dict) -> bool:
    open_prev = indicators.get("open_prev")
    close_prev = indicators.get("close_prev")
    open_price = indicators.get("open")
    close_price = indicators.get("close")

    if None in {open_prev, close_prev, open_price, close_price}:
        return False

    if close_prev <= open_prev:
        return False
    if close_price >= open_price:
        return False

    prev_body = abs(close_prev - open_prev)
    curr_body = abs(close_price - open_price)
    if prev_body == 0 or curr_body == 0:
        return False

    return open_price >= close_prev and close_price <= open_prev


def check_sell_signal(symbol: str, indicators: dict) -> list:
    """Return list of exit reasons triggered by indicators."""
    price = indicators["close"]
    ema20 = indicators["EMA20"]
    rsi = indicators["RSI"]
    shooting_star = is_shooting_star(indicators)
    bear_engulf = is_bearish_engulfing(indicators)

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

    if shooting_star:
        reasons.append("Shooting star")

    state = RSI_HIGH_MEMORY.get(symbol, {"price": price, "rsi": rsi})
    if rsi > 70 and price > state.get("price", price) and rsi < state.get("rsi", rsi):
        reasons.append("RSI divergence")
    if price > state.get("price", price) and rsi >= state.get("rsi", rsi):
        RSI_HIGH_MEMORY[symbol] = {"price": price, "rsi": rsi}
    elif symbol not in RSI_HIGH_MEMORY:
        RSI_HIGH_MEMORY[symbol] = state

    def dedupe_preserve_order(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    baseline_reasons = dedupe_preserve_order(list(reasons))
    reasons = list(baseline_reasons)
    enable_v2 = env_bool("MONITOR_ENABLE_EXIT_SIGNALS_V2", default=False)
    v2_reasons: list[str] = []

    def _num(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rsi_now = _num(indicators.get("RSI"))
    rsi_prev = _num(indicators.get("RSI_prev", rsi_now))
    if rsi_now is not None and rsi_prev is not None:
        if rsi_prev >= 70 and (
            rsi_now <= RSI_REVERSAL_FLOOR
            or (rsi_prev - rsi_now) >= RSI_REVERSAL_DROP
        ):
            v2_reasons.append("RSI reversal")

    close_prev = _num(indicators.get("close_prev", price))
    ema20_prev = _num(indicators.get("EMA20_prev", ema20))
    if close_prev is not None and ema20_prev is not None:
        if price < ema20 and close_prev >= ema20_prev:
            v2_reasons.append("EMA20 cross-down")

    sma9 = _num(indicators.get("SMA9"))
    sma9_prev = _num(indicators.get("SMA9_prev", sma9))
    if sma9 is not None and sma9_prev is not None and ema20_prev is not None:
        if sma9 < ema20 and sma9_prev >= ema20_prev:
            v2_reasons.append("SMA9/EMA20 cross-down")

    macd_val = _num(indicators.get("MACD"))
    macd_signal = _num(indicators.get("MACD_signal"))
    macd_hist = None
    macd_hist_prev = _num(indicators.get("MACD_hist_prev"))
    if macd_val is not None and macd_signal is not None:
        macd_hist = _num(indicators.get("MACD_hist", macd_val - macd_signal))
        if macd_hist_prev is None:
            prev_macd = _num(indicators.get("MACD_prev", macd_val))
            prev_signal = _num(indicators.get("MACD_signal_prev", macd_signal))
            if prev_macd is not None and prev_signal is not None:
                macd_hist_prev = prev_macd - prev_signal
        if macd_hist is not None and macd_hist_prev is not None:
            if (macd_hist < macd_hist_prev and macd_hist_prev > 0) or (
                macd_hist < 0 <= macd_hist_prev
            ):
                v2_reasons.append("MACD fade")

    if shooting_star:
        v2_reasons.append("Shooting star")
    if bear_engulf:
        v2_reasons.append("Bearish engulfing")

    v2_reasons = dedupe_preserve_order(list(v2_reasons))
    if enable_v2:
        for reason in v2_reasons:
            if reason not in reasons:
                reasons.append(reason)
        reasons = dedupe_preserve_order(reasons)
        final_reasons = list(reasons)

        payload = {
            "symbol": symbol,
            "enabled": True,
            "baseline_reasons": list(baseline_reasons),
            "v2_reasons": list(v2_reasons),
            "final_reasons": list(final_reasons),
            "price": _num(price),
            "close_prev": _num(indicators.get("close_prev", price)),
            "rsi_now": rsi_now,
            "rsi_prev": rsi_prev,
            "rsi_reversal_drop": float(RSI_REVERSAL_DROP),
            "rsi_reversal_floor": float(RSI_REVERSAL_FLOOR),
            "shooting_star": bool(shooting_star),
            "bearish_engulfing": bool(bear_engulf),
            "sma9": _num(indicators.get("SMA9")),
            "sma9_prev": _num(indicators.get("SMA9_prev")),
            "ema20": _num(ema20),
            "ema20_prev": _num(indicators.get("EMA20_prev", ema20)),
            "macd": macd_val,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "macd_hist_prev": macd_hist_prev,
        }
        logger.info(
            "EXIT_SIGNAL_V2 %s",
            json.dumps(payload, sort_keys=True),
        )

        reasons = list(final_reasons)

    return reasons


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_risk_score(position, indicators: dict | None) -> float | None:
    indicators = indicators or {}
    for key in ("risk_score", "ml_risk_score", "risk_score_pct"):
        if key in indicators:
            return _safe_float(indicators.get(key))
    value = getattr(position, "risk_score", None)
    return _safe_float(value)


def evaluate_exit_signals(position, indicators, now_utc) -> list[dict]:
    """Return canonical exit signals for the current position."""

    indicators = indicators or {}
    entry_price = _safe_float(getattr(position, "avg_entry_price", 0) or 0) or 0.0
    current_price = _safe_float(indicators.get("close"))
    if current_price is None:
        current_price = _safe_float(getattr(position, "live_price", None))
    if current_price is None:
        current_price = _safe_float(getattr(position, "current_price", entry_price))
    current_price = current_price or 0.0

    gain_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0.0
    days_held = calculate_days_held(position)
    risk_score = _extract_risk_score(position, indicators)

    signals: list[dict] = []

    rsi_now = _safe_float(indicators.get("RSI"))
    rsi_prev = _safe_float(indicators.get("RSI_prev", rsi_now))
    sma9 = _safe_float(indicators.get("SMA9"))
    ema20 = _safe_float(indicators.get("EMA20"))
    if (
        rsi_now is not None
        and rsi_prev is not None
        and sma9 is not None
        and ema20 is not None
        and rsi_prev >= RSI_OVERBOUGHT
        and rsi_now <= RSI_REVERSAL_THRESHOLD
        and sma9 < ema20
    ):
        signals.append(
            {
                "code": "SIGNAL_REVERSAL",
                "confidence": 0.85,
                "detail": f"RSI {rsi_prev:.1f}->{rsi_now:.1f} with SMA9<EMA20",
                "priority": 1,
            }
        )

    high_now = _safe_float(indicators.get("high"))
    high_prev = _safe_float(indicators.get("high_prev", high_now))
    close_prev = _safe_float(indicators.get("close_prev", current_price))
    if high_now and current_price and close_prev is not None and high_prev is not None:
        stall_ratio = MOMENTUM_STALL_PCT / 100.0
        near_high = (high_now - current_price) / high_now <= stall_ratio
        not_advancing = current_price <= close_prev and high_now <= high_prev * (1 + stall_ratio)
        if near_high and not_advancing:
            signals.append(
                {
                    "code": "MOMENTUM_FADE",
                    "confidence": 0.65,
                    "detail": f"stall near high {high_now:.2f}",
                    "priority": 3,
                }
            )

    if days_held > MAX_HOLD_DAYS:
        signals.append(
            {
                "code": "TIME_STOP",
                "confidence": 0.7,
                "detail": f"days_held={days_held} > {MAX_HOLD_DAYS}",
                "priority": 4,
            }
        )

    if gain_pct >= PROFIT_TARGET_PCT:
        signals.append(
            {
                "code": "PROFIT_TARGET_HIT",
                "confidence": 0.6,
                "detail": f"gain_pct={gain_pct:.2f} >= {PROFIT_TARGET_PCT:.2f}",
                "priority": 5,
            }
        )

    if risk_score is not None and risk_score >= RISK_OFF_SCORE_CUTOFF:
        signals.append(
            {
                "code": "RISK_OFF",
                "confidence": 0.8,
                "detail": f"risk_score={risk_score:.2f}",
                "priority": 2,
            }
        )

    return signals


def _select_primary_exit_signal(signals: list[dict]) -> dict | None:
    if not signals:
        return None
    return sorted(
        signals,
        key=lambda item: (item.get("priority", 999), -float(item.get("confidence", 0))),
    )[0]


def _signals_from_legacy_reasons(reasons: list[str]) -> list[dict]:
    if not reasons:
        return []
    detail = "; ".join(reasons)
    return [
        {
            "code": "SIGNAL_REVERSAL",
            "confidence": 0.55,
            "detail": detail,
            "priority": 3,
        }
    ]


def _build_exit_snapshot(
    position,
    indicators: dict | None,
    exit_signals: list[dict] | None,
    trail_pct: float | None,
    risk_score: float | None,
) -> dict:
    indicators = indicators or {}
    entry_price = _safe_float(getattr(position, "avg_entry_price", 0) or 0) or 0.0
    current_price = _safe_float(indicators.get("close"))
    if current_price is None:
        current_price = _safe_float(getattr(position, "live_price", None))
    if current_price is None:
        current_price = _safe_float(getattr(position, "current_price", entry_price))
    current_price = current_price or 0.0

    gain_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0.0
    days_held = calculate_days_held(position)

    max_runup_pct = None
    max_drawdown_pct = None
    high_price = _safe_float(indicators.get("high"))
    low_price = _safe_float(indicators.get("low"))
    if entry_price and high_price is not None:
        max_runup_pct = (high_price - entry_price) / entry_price * 100
    if entry_price and low_price is not None:
        max_drawdown_pct = (low_price - entry_price) / entry_price * 100

    return {
        "gain_pct": round(gain_pct, 4),
        "days_held": int(days_held),
        "max_runup_pct": None if max_runup_pct is None else round(max_runup_pct, 4),
        "max_drawdown_pct": None if max_drawdown_pct is None else round(max_drawdown_pct, 4),
        "signals": list(exit_signals or []),
        "trail_pct": None if trail_pct is None else float(trail_pct),
        "risk_score": None if risk_score is None else float(risk_score),
    }


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
        if order_attr_str(o, ("order_type", "type")) == "trailing_stop":
            return o
    return None


def _normalize_order_side(order) -> str:
    return order_attr_str(order, ("side",))


def _is_protective_order(order, expected_side: str) -> bool:
    side = order_attr_str(order, ("side",))
    order_type = order_attr_str(order, ("order_type", "type"))
    status = order_attr_str(order, ("status",))

    stop_like = "trailing" in order_type or "stop" in order_type
    is_open = any(state in status for state in ["open", "new", "accepted", "held", "partially_filled"])

    normalized_side = "sell" if "sell" in side else "buy" if "buy" in side else ""

    return stop_like and is_open and normalized_side == expected_side


def _normalize_order_status(order) -> str:
    return order_attr_str(order, ("status",))


def _is_open_protective_order(order) -> bool:
    order_type = order_attr_str(order, ("order_type", "type"))
    if "stop" not in order_type and "trailing" not in order_type:
        return False

    status = _normalize_order_status(order)
    if status not in {"open", "new", "accepted", "held", "partially_filled"}:
        return False

    return _normalize_order_side(order) in {"buy", "sell"}


def last_filled_trailing_stop(symbol):
    request = GetOrdersRequest(symbols=[symbol], limit=10)
    try:
        orders = trading_client.get_orders(request)
        closed_orders = [
            o
            for o in orders
            if order_attr_str(o, ("status",)) in ("filled", "canceled", "rejected")
        ]
        for o in closed_orders:
            if (
                order_attr_str(o, ("order_type", "type")) == "trailing_stop"
                and order_attr_str(o, ("status",)) == "filled"
            ):
                return o
    except Exception as e:
        logger.error("Failed to fetch order history for %s: %s", symbol, e)
        increment_metric("api_errors")
    return None


def count_open_trailing_stops() -> int:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    try:
        orders = trading_client.get_orders(request)
        count = sum(
            1
            for order in orders
            if order_attr_str(order, ("order_type", "type")) == "trailing_stop"
        )
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
        order = broker_submit_order(
            request,
            {
                "symbol": symbol,
                "qty": qty,
                "order_type": "trailing_stop",
                "reason": "attach_stop_long",
            },
        )
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_submit"],
            symbol=symbol,
            qty=qty,
            order_id=str(getattr(order, "id", "") or "") or None,
            status=str(getattr(order, "status", "submitted")),
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": "attach_stop_long",
                "order_type": "trailing_stop",
                "trail_percent": float(trail_percent),
                "side": "sell",
                "time_in_force": "gtc",
                "dryrun": bool(getattr(order, "dryrun", False)),
            },
        )
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
    qty_abs = abs(int(float(getattr(position, "qty", 0))))
    logger.info(
        "[INFO] SHORT_STOP_QTY symbol=%s position_qty=%s qty_abs=%s",
        symbol,
        getattr(position, "qty", None),
        qty_abs,
    )
    logger.info(
        "STOP_ATTACH_ATTEMPT symbol=%s side=short type=trailing_buy trail_pct=%s",
        symbol,
        trail_percent,
    )
    if qty_abs <= 0:
        logger.warning(
            "STOP_ATTACH_FAILED symbol=%s side=short type=trailing_buy error=no_available_qty",
            symbol,
        )
        increment_metric("stop_attach_failed")
        return False
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty_abs,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            trail_percent=str(trail_percent),
        )
        order = broker_submit_order(
            request,
            {
                "symbol": symbol,
                "qty": qty_abs,
                "order_type": "trailing_stop",
                "reason": "attach_stop_short_trailing",
            },
        )
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_submit"],
            symbol=symbol,
            qty=qty_abs,
            order_id=str(getattr(order, "id", "") or "") or None,
            status=str(getattr(order, "status", "submitted")),
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": "attach_stop_short_trailing",
                "order_type": "trailing_stop",
                "trail_percent": float(trail_percent),
                "side": "buy",
                "time_in_force": "gtc",
                "dryrun": bool(getattr(order, "dryrun", False)),
            },
        )
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
            qty=qty_abs,
            side=OrderSide.BUY,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC,
        )
        order = broker_submit_order(
            request,
            {
                "symbol": symbol,
                "qty": qty_abs,
                "order_type": "stop",
                "reason": "attach_stop_short_fallback",
            },
        )
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_submit"],
            symbol=symbol,
            qty=qty_abs,
            order_id=str(getattr(order, "id", "") or "") or None,
            status=str(getattr(order, "status", "submitted")),
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": "attach_stop_short_fallback",
                "order_type": "stop",
                "stop_price": stop_price,
                "side": "buy",
                "time_in_force": "gtc",
                "dryrun": bool(getattr(order, "dryrun", False)),
            },
        )
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
    positions = positions or []
    positions_count = len(positions)

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

    orders_by_symbol: dict[str, list] = {}
    trailing_stops_count = 0
    protective_open_orders = []

    for order in open_orders:
        if not _is_open_protective_order(order):
            continue

        protective_open_orders.append(order)

        otype = order_attr_str(order, ("order_type", "type"))
        if otype == "trailing_stop":
            trailing_stops_count += 1

        symbol = getattr(order, "symbol", "")
        if symbol:
            orders_by_symbol.setdefault(symbol, []).append(order)

    open_orders = protective_open_orders

    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    protected_symbols: set[str] = set()
    for position in positions:
        symbol = position.symbol
        side = _determine_position_side(position)
        expected_side = "sell" if side == "long" else "buy"
        symbol_orders = orders_by_symbol.get(symbol, [])
        trailing_stop = get_protective_trailing_stop_for_symbol(symbol, symbol_orders)
        if trailing_stop and is_fully_covered(position, trailing_stop):
            _mark_symbol_protected(
                symbol, [str(getattr(trailing_stop, "id", ""))]
            )
            protected_symbols.add(symbol)
            continue

        protective_orders = [
            order
            for order in symbol_orders
            if _is_protective_order(order, expected_side)
        ]

        if protective_orders:
            _mark_symbol_protected(
                symbol, [str(getattr(order, "id", "")) for order in protective_orders]
            )
            protected_symbols.add(symbol)
            continue

        if _record_stop_missing(symbol, today):
            logger.warning("STOP_MISSING symbol=%s side=%s", symbol, side)
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

    logger.info(
        "[INFO] COVERAGE_DEBUG open_orders=%s trailing_stops=%s protected=%s",
        len(open_orders),
        trailing_stops_count,
        sorted(protected_symbols),
    )

    protective_orders_count = len(protected_symbols)
    coverage_pct = (
        protective_orders_count / positions_count if positions_count else 1.0
    )
    MONITOR_METRICS["stop_coverage_pct"] = float(coverage_pct)
    _persist_metrics()
    return protective_orders_count, float(coverage_pct), trailing_stops_count


def has_pending_sell_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        side = order_attr_str(o, ("side",))
        order_type = order_attr_str(o, ("order_type", "type"))
        if side == "sell" and order_type != "trailing_stop":
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
        order = broker_submit_order(
            request,
            {
                "symbol": symbol,
                "qty": qty,
                "order_type": "trailing_stop",
                "reason": "trailing_stop_submit",
            },
        )
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_submit"],
            symbol=symbol,
            qty=qty,
            order_id=str(getattr(order, "id", "") or "") or None,
            status=str(getattr(order, "status", "submitted")),
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": "trailing_stop_submit",
                "order_type": "trailing_stop",
                "trail_percent": float(trail_percent),
                "side": "sell",
                "time_in_force": "gtc",
                "dryrun": bool(getattr(order, "dryrun", False)),
            },
        )
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


def manage_trailing_stop(position, indicators: dict | None = None, exit_signals: list[dict] | None = None):
    symbol = position.symbol
    logger.info(f"Evaluating trailing stop for {symbol}")
    if symbol.upper() == "ARQQ":
        # Explicit log entry requested to verify ARQQ trailing stop evaluation
        logger.info("Evaluating trailing stop for ARQQ")
    qty = position.qty
    logger.info(f"Evaluating trailing stop for {symbol} - qty: {qty}")
    entry = float(position.avg_entry_price)
    current = float(getattr(position, "live_price", position.current_price))
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
        order
        for order in existing_orders
        if order_attr_str(order, ("order_type", "type")) == "trailing_stop"
    ]

    if len(trailing_stops) > 1:
        for order in trailing_stops[1:]:
            try:
                broker_cancel_order(
                    order.id,
                    {
                        "symbol": symbol,
                        "order_id": getattr(order, "id", None),
                        "reason": "redundant_trailing_stop",
                    },
                )
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

    available_qty = int(getattr(position, "qty_available", 0) or 0)
    try:
        pos_qty_abs = abs(int(float(getattr(position, "qty", 0) or 0)))
    except Exception:
        pos_qty_abs = 0
    if available_qty <= 0:
        logger.info(
            "QTY_RESERVED symbol=%s pos_qty=%s qty_available=%s reason=protected_order",
            symbol,
            pos_qty_abs,
            available_qty,
        )
    use_qty = pos_qty_abs if pos_qty_abs > 0 else available_qty
    if use_qty <= 0:
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

    current_trail_pct = float(
        getattr(trailing_order, "trail_percent", TRAIL_START_PERCENT) or TRAIL_START_PERCENT
    )
    days_held = calculate_days_held(position)
    exit_intel_enabled = env_bool("MONITOR_ENABLE_EXIT_INTELLIGENCE", default=False)
    if exit_intel_enabled:
        target_trail_pct, reason_parts = _compute_target_trail_meta(
            position,
            indicators,
            exit_signals,
        )
        reason_detail = "+".join(reason_parts) if reason_parts else "profit_tier"
    else:
        target_trail_pct, reason_detail = compute_target_trail_pct_legacy(
            gain_pct,
            days_held,
            current_trail_pct,
        )
    effective_target = min(current_trail_pct, target_trail_pct)

    now_utc = datetime.now(timezone.utc)
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
            "STOP_TIGHTEN_COOLDOWN signal_quality=LOW symbol=%s last=%s",
            symbol,
            LOW_SIGNAL_STOP_COOLDOWNS.get(symbol),
        )

    tighten_eps = max(0.0, float(TIGHTEN_COOLDOWN_EPS_PCT))
    if effective_target >= current_trail_pct - tighten_eps:
        logger.info(
            "No trailing stop adjustment needed for %s (gain: %.2f%%; trail %.2f%%)",
            symbol,
            gain_pct,
            current_trail_pct,
        )
        return

    existing_stop = None
    stop_price = getattr(trailing_order, "stop_price", None)
    if stop_price is not None:
        try:
            existing_stop = float(stop_price)
        except (TypeError, ValueError):
            existing_stop = None
    if existing_stop is None:
        hwm = getattr(trailing_order, "hwm", None)
        if hwm is None:
            hwm = getattr(trailing_order, "high_water_mark", None)
        trail_pct_val = getattr(trailing_order, "trail_percent", None)
        try:
            if hwm is not None and trail_pct_val is not None:
                existing_stop = float(hwm) * (1 - float(trail_pct_val) / 100.0)
        except (TypeError, ValueError):
            existing_stop = None

    if existing_stop is not None:
        proposed_initial_stop = current * (1 - effective_target / 100.0)
        if proposed_initial_stop < existing_stop - 0.01:
            payload = {
                "symbol": symbol,
                "reason": "would_lower_stop",
                "existing_stop": round(existing_stop, 4),
                "proposed_stop": round(proposed_initial_stop, 4),
            }
            logger.info(
                "STOP_TIGHTEN_SKIP %s",
                json.dumps(payload, sort_keys=True),
            )
            increment_metric("stop_tighten_skipped")
            return

    cooldown_minutes = max(0, int(TIGHTEN_COOLDOWN_MINUTES))
    last_tighten = _parse_iso_datetime(TIGHTEN_COOLDOWNS.get(symbol))
    if last_tighten is not None:
        if last_tighten.tzinfo is None:
            last_tighten = last_tighten.replace(tzinfo=timezone.utc)
        minutes_since = (now_utc - last_tighten).total_seconds() / 60.0
        if minutes_since < cooldown_minutes:
            logger.info(
                "STOP_TIGHTEN_COOLDOWN symbol=%s last=%s minutes_since=%.2f cooldown_min=%s target=%.4f current=%.4f",
                symbol,
                last_tighten.isoformat(),
                minutes_since,
                cooldown_minutes,
                effective_target,
                current_trail_pct,
            )
            increment_metric("stop_tighten_cooldown")
            return

    try:
        if low_signal_tighten:
            reason_detail = f"{reason_detail}+low_signal"
        payload = {
            "symbol": symbol,
            "from": round(current_trail_pct, 4),
            "to": round(effective_target, 4),
            "gain_pct": round(gain_pct, 4),
            "days_held": int(days_held),
            "reason": reason_detail,
        }
        logger.info(
            "STOP_TIGHTEN %s",
            json.dumps(payload, sort_keys=True),
        )
        canceled = cancel_order_safe(trailing_order.id, symbol, reason="adjust_trailing_stop")
        if not canceled:
            logger.warning(
                "STOP_TIGHTEN_CANCEL_FAILED symbol=%s order_id=%s",
                symbol,
                getattr(trailing_order, "id", None),
            )
            return
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_cancel"],
            symbol=symbol,
            qty=use_qty,
            order_id=str(getattr(trailing_order, "id", "") or "") or None,
            status="canceled",
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": "adjust_trailing_stop",
                "order_type": "trailing_stop",
                "from_trail": round(current_trail_pct, 4),
                "to_trail": round(effective_target, 4),
                "dryrun": env_bool("MONITOR_DISABLE_SELLS", default=False),
            },
        )
        if low_signal_tighten:
            logger.info(
                "STOP_TIGHTEN signal_quality=LOW symbol=%s old_trail=%.2f new_trail=%.2f reason=low_signal_profit_lock",
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
        new_order = broker_submit_order(
            request,
            {
                "symbol": symbol,
                "qty": use_qty,
                "order_type": "trailing_stop",
                "reason": f"tighten_trailing_stop:{reason_detail}",
            },
        )
        db_log_event(
            event_type=MONITOR_DB_EVENT_TYPES["trail_adjust"],
            symbol=symbol,
            qty=use_qty,
            order_id=str(getattr(new_order, "id", "") or "") or None,
            status=str(getattr(new_order, "status", "submitted")),
            event_time=datetime.now(timezone.utc),
            raw={
                "reason": reason_detail,
                "order_type": "trailing_stop",
                "from_trail": round(current_trail_pct, 4),
                "to_trail": round(effective_target, 4),
                "gain_pct": round(gain_pct, 4),
                "days_held": int(days_held),
                "dryrun": bool(getattr(new_order, "dryrun", False)),
            },
        )
        new_order_id = str(getattr(new_order, "id", "") or "") or None
        if "breakeven_lock" in reason_detail:
            increment_metric("breakeven_tightens")
        if "time_decay" in reason_detail:
            increment_metric("time_decay_tightens")
        logger.info(
            "Adjusted trailing stop for %s from %.2f%% to %.2f%% (gain: %.2f%%).",
            symbol,
            current_trail_pct,
            effective_target,
            gain_pct,
        )
        log_trailing_stop_event(symbol, effective_target, new_order_id, "adjusted")
        if low_signal_tighten:
            _mark_low_signal_cooldown(symbol, now_utc)
        _mark_tighten_cooldown(symbol, now_utc)
    except Exception as e:
        logger.error("Failed to adjust trailing stop for %s: %s", symbol, e)
        try:
            broker_close_position(
                symbol,
                {"symbol": symbol, "reason": "adjust_trailing_stop_failure"},
            )
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to close position %s: %s", symbol, exc)


def check_pending_orders():
    """Log status of open sell orders and remove redundant trailing stops."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(request)
        logger.info(
            f"Fetched open orders for all symbols: {len(open_orders)} found.")

        trailing_by_key: dict[tuple[str, str], list] = {}
        for order in open_orders:
            if order_attr_str(order, ("order_type", "type")) != "trailing_stop":
                continue
            side = order_attr_str(order, ("side",))
            if side not in {"sell", "buy"}:
                continue
            symbol = getattr(order, "symbol", "")
            if not symbol:
                continue
            trailing_by_key.setdefault((symbol, side), []).append(order)

        for (symbol, side), orders in trailing_by_key.items():
            if len(orders) <= 1:
                continue
            keep_order = max(orders, key=_order_sort_key)
            keep_id = str(getattr(keep_order, "id", "") or "") or None
            for order in orders:
                if order is keep_order:
                    continue
                cancel_id = str(getattr(order, "id", "") or "") or None
                try:
                    broker_cancel_order(
                        order.id,
                        {
                            "symbol": getattr(order, "symbol", None),
                            "order_id": getattr(order, "id", None),
                            "reason": "redundant_cleanup",
                            "keep_id": keep_id,
                        },
                    )
                    db_log_event(
                        event_type=MONITOR_DB_EVENT_TYPES["trail_cancel"],
                        symbol=symbol,
                        qty=getattr(order, "qty", None),
                        order_id=cancel_id,
                        status="canceled",
                        event_time=datetime.now(timezone.utc),
                        raw={
                            "reason": "redundant_cleanup",
                            "order_type": "trailing_stop",
                            "keep_id": keep_id,
                            "dryrun": env_bool("MONITOR_DISABLE_SELLS", default=False),
                        },
                    )
                    logger.info(
                        "TRAIL_REDUNDANT_CANCEL symbol=%s keep=%s cancel=%s reason=duplicate_open_trailing_stop",
                        symbol,
                        keep_id,
                        cancel_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to cancel trailing stop %s for %s: %s",
                        getattr(order, "id", None),
                        symbol,
                        exc,
                    )

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


def submit_sell_market_order(
    position,
    reason: str,
    reason_code: str,
    qty_override: Optional[int] = None,
    *,
    exit_signals: list[dict] | None = None,
    exit_snapshot: dict | None = None,
):
    """Submit a limit sell order compatible with extended hours."""
    symbol = position.symbol
    qty = position.qty
    entry_price = float(position.avg_entry_price)
    exit_price = round_price(float(getattr(position, "current_price", entry_price)))
    entry_time = getattr(position, "created_at", datetime.now(timezone.utc)).isoformat()
    now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ).time()
    db_enabled = db_logging_enabled()
    db_module = None
    if db_enabled:
        try:
            from scripts import db as db_module
        except Exception as exc:
            logger.warning("DB_LOGGING_IMPORT_FAIL err=%s", exc)
            db_enabled = False
    partial_exit = False
    if qty_override is not None:
        try:
            partial_exit = float(qty_override) < float(qty)
        except Exception:
            partial_exit = True

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
        submit_ts = datetime.now(timezone.utc)
        order = broker_submit_order(
            order_request,
            {
                "symbol": symbol,
                "qty": order_qty,
                "order_type": "limit",
                "reason": reason,
                "reason_code": reason_code,
            },
        )
        order_id_value = str(getattr(order, "id", "") or "") or None
        is_dryrun = bool(
            getattr(order, "dryrun", env_bool("MONITOR_DISABLE_SELLS", default=False))
        )
        if db_enabled:
            db_log_event(
                event_type=MONITOR_DB_EVENT_TYPES["sell_submit"],
                symbol=symbol,
                qty=order_qty,
                order_id=order_id_value,
                status="submitted",
                event_time=submit_ts,
                raw={
                    "reason": reason,
                    "reason_code": reason_code,
                    "exit_reason_code": reason_code,
                    "exit_reason_detail": reason,
                    "limit_price": exit_price,
                    "qty": order_qty,
                    "side": "sell",
                    "order_type": "limit",
                    "extended_hours": bool(is_extended_hours(now_et)),
                    "time_in_force": "day",
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "partial": partial_exit,
                    "dryrun": is_dryrun,
                    "exit_signals": list(exit_signals or []),
                    "exit_snapshot": exit_snapshot,
                },
            )
        log_exit_submit(symbol, order_qty, "limit", reason_code)
        if getattr(order, "dryrun", False):
            status = "dryrun"
            latency_ms = 0
        else:
            status = wait_for_order_terminal(str(getattr(order, "id", "")))
            latency_ms = int((datetime.now(timezone.utc) - submit_ts).total_seconds() * 1000)
        log_exit_final(status, latency_ms)
        final_status = str(status or "")
        status_norm = final_status.lower()
        final_ts = datetime.now(timezone.utc)
        if status_norm == "filled":
            final_event_type = MONITOR_DB_EVENT_TYPES["sell_fill"]
        elif status_norm in {"canceled", "cancelled"}:
            final_event_type = MONITOR_DB_EVENT_TYPES["sell_cancelled"]
        elif status_norm == "expired":
            final_event_type = MONITOR_DB_EVENT_TYPES["sell_expired"]
        else:
            final_event_type = MONITOR_DB_EVENT_TYPES["sell_rejected"]
        if db_enabled:
            db_log_event(
                event_type=final_event_type,
                symbol=symbol,
                qty=order_qty,
                order_id=order_id_value,
                status=final_status,
                event_time=final_ts,
                raw={
                    "reason": reason,
                    "reason_code": reason_code,
                    "exit_reason_code": reason_code,
                    "exit_reason_detail": reason,
                    "limit_price": exit_price,
                    "qty": order_qty,
                    "side": "sell",
                    "order_type": "limit",
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "partial": partial_exit,
                    "dryrun": is_dryrun,
                    "exit_signals": list(exit_signals or []),
                    "exit_snapshot": exit_snapshot,
                },
            )
            if status_norm == "filled" and not partial_exit and not is_dryrun:
                exit_reason = f"{reason_code}:{reason}" if reason_code else str(reason)
                exit_reason = (exit_reason or "").strip()
                if len(exit_reason) > 200:
                    exit_reason = exit_reason[:200]
                if db_module is not None:
                    try:
                        db_module.close_trade_on_sell_fill(
                            symbol,
                            order_id_value,
                            final_ts,
                            exit_price,
                            exit_reason,
                        )
                    except Exception as exc:
                        logger.warning(
                            "DB_TRADE_CLOSE_FAILED symbol=%s order_id=%s err=%s",
                            symbol,
                            order_id_value or "",
                            exc,
                        )
                pnl_value = None
                try:
                    pnl_value = (float(exit_price) - float(entry_price)) * float(order_qty)
                except Exception:
                    pnl_value = None
                executed_payload = {
                    "symbol": symbol,
                    "qty": float(order_qty),
                    "entry_time": entry_time,
                    "entry_price": float(entry_price),
                    "exit_time": final_ts.isoformat(),
                    "exit_price": float(exit_price),
                    "pnl": pnl_value,
                    "net_pnl": pnl_value,
                    "order_id": order_id_value,
                    "status": final_status,
                    "event_type": final_event_type,
                    "side": "sell",
                    "order_type": "limit",
                    "reason": reason,
                    "reason_code": reason_code,
                    "exit_reason_code": reason_code,
                    "exit_reason_detail": reason,
                    "exit_signals": list(exit_signals or []),
                    "exit_snapshot": exit_snapshot,
                }
                if db_module is not None:
                    try:
                        db_module.insert_executed_trade(executed_payload)
                    except Exception as exc:
                        logger.warning(
                            "DB_EXECUTED_TRADE_FAILED symbol=%s order_id=%s err=%s",
                            symbol,
                            order_id_value or "",
                            exc,
                        )
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
            order_id=order_id_value,
        )
    except Exception as e:
        logger.error("Error submitting sell order for %s: %s", symbol, e)
        increment_metric("api_errors")
        try:
            broker_close_position(
                symbol,
                {"symbol": symbol, "reason": reason, "reason_code": reason_code},
            )
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

    use_live = env_bool("MONITOR_ENABLE_LIVE_PRICES", default=False)
    latest_prices: dict[str, float] = {}
    if use_live and positions:
        symbols = sorted({p.symbol for p in positions if getattr(p, "symbol", None)})
        latest_prices = get_latest_trade_prices(symbols)
        missing_count = max(0, len(symbols) - len(latest_prices))
        if missing_count:
            logger.info("LIVE_PRICE_MISSING count=%s", missing_count)

    existing_symbols = set(existing_positions_df.get("symbol", []))
    current_symbols = set(p.symbol for p in positions)
    closed_symbols = existing_symbols - current_symbols

    log_closed_positions(trading_client, closed_symbols, existing_positions_df)

    if not positions:
        logger.info("No open positions found.")
        return positions
    for position in positions:
        symbol = position.symbol
        live_price = latest_prices.get(symbol)
        if live_price is not None:
            try:
                setattr(position, "live_price", float(live_price))
            except Exception:
                pass
        days_held = calculate_days_held(position)
        logger.info("%s held for %d days", symbol, days_held)
        exit_intel_enabled = env_bool("MONITOR_ENABLE_EXIT_INTELLIGENCE", default=False)
        if days_held >= MAX_HOLD_DAYS:
            if has_pending_sell_order(symbol):
                logger.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            trail_pct = None
            if trailing_order:
                try:
                    trail_pct = float(getattr(trailing_order, "trail_percent", None))
                except Exception:
                    trail_pct = None
                try:
                    cancel_order_safe(trailing_order.id, symbol, reason="max_hold_exit")
                    logger.info(
                        "Cancelled trailing stop for %s due to max hold exit",
                        symbol,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to cancel trailing stop for %s: %s", symbol, e
                    )
            increment_metric("exits_max_hold")
            exit_signals = None
            exit_snapshot = None
            if exit_intel_enabled:
                exit_signals = evaluate_exit_signals(position, {}, datetime.now(timezone.utc))
                if not exit_signals:
                    exit_signals = [
                        {
                            "code": "TIME_STOP",
                            "confidence": 0.7,
                            "detail": f"days_held={days_held} > {MAX_HOLD_DAYS}",
                            "priority": 4,
                        }
                    ]
                primary_signal = _select_primary_exit_signal(exit_signals) or exit_signals[0]
                reason_detail = str(primary_signal.get("detail", ""))
                reason_code = str(primary_signal.get("code", "TIME_STOP"))
                exit_snapshot = _build_exit_snapshot(
                    position,
                    {},
                    exit_signals,
                    trail_pct,
                    _extract_risk_score(position, {}),
                )
            else:
                reason_detail = f"Max Hold {days_held}d"
                reason_code = "max_hold"
            submit_sell_market_order(
                position,
                reason=reason_detail,
                reason_code=reason_code,
                exit_signals=exit_signals,
                exit_snapshot=exit_snapshot,
            )
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
        current_price = None
        if live_price is not None:
            current_price = float(live_price)
        else:
            fallback_price = getattr(position, "current_price", None)
            if fallback_price is None:
                fallback_price = indicators.get("close") if indicators else None
            if fallback_price is not None:
                current_price = float(fallback_price)
        if indicators and current_price is not None:
            indicators["close"] = float(current_price)

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

        partial_trigger = PROFIT_TARGET_PCT if exit_intel_enabled else PARTIAL_GAIN_THRESHOLD
        if (
            gain_pct >= partial_trigger
            and not PARTIAL_EXIT_TAKEN.get(symbol)
            and not has_pending_sell_order(symbol)
        ):
            trailing_order = get_trailing_stop_order(symbol)
            trail_pct = None
            if trailing_order:
                try:
                    trail_pct = float(getattr(trailing_order, "trail_percent", None))
                except Exception:
                    trail_pct = None
                cancel_order_safe(trailing_order.id, symbol, reason="partial_exit")
            half_qty = max(1, int(float(position.qty) / 2))
            exit_signals = None
            exit_snapshot = None
            if exit_intel_enabled:
                exit_signals = evaluate_exit_signals(position, indicators, datetime.now(timezone.utc))
                if not any(signal.get("code") == "PROFIT_TARGET_HIT" for signal in exit_signals):
                    exit_signals.append(
                        {
                            "code": "PROFIT_TARGET_HIT",
                            "confidence": 0.6,
                            "detail": f"gain_pct={gain_pct:.2f} >= {partial_trigger:.2f}",
                            "priority": 5,
                        }
                    )
                primary_signal = _select_primary_exit_signal(exit_signals) or exit_signals[0]
                reason_detail = str(primary_signal.get("detail", ""))
                reason_code = str(primary_signal.get("code", "PROFIT_TARGET_HIT"))
                exit_snapshot = _build_exit_snapshot(
                    position,
                    indicators,
                    exit_signals,
                    trail_pct,
                    _extract_risk_score(position, indicators),
                )
            else:
                reason_detail = f"Partial exit at +{PARTIAL_GAIN_THRESHOLD:.0f}% gain"
                reason_code = "partial_gain"
            submit_sell_market_order(
                position,
                reason=reason_detail,
                reason_code=reason_code,
                qty_override=half_qty,
                exit_signals=exit_signals,
                exit_snapshot=exit_snapshot,
            )
            PARTIAL_EXIT_TAKEN[symbol] = True
            _save_partial_state(PARTIAL_EXIT_TAKEN)
            logger.info("Partial exit recorded for %s; awaiting trailing re-attachment.", symbol)
            continue

        if exit_intel_enabled:
            exit_signals = evaluate_exit_signals(position, indicators, datetime.now(timezone.utc))
            actionable_signals = [
                signal
                for signal in exit_signals
                if signal.get("code") not in {"PROFIT_TARGET_HIT", "TIME_STOP"}
            ]
            if actionable_signals:
                if has_pending_sell_order(symbol):
                    logger.info("Sell order already pending for %s", symbol)
                else:
                    trailing_order = get_trailing_stop_order(symbol)
                    trail_pct = None
                    if trailing_order:
                        try:
                            trail_pct = float(getattr(trailing_order, "trail_percent", None))
                        except Exception:
                            trail_pct = None
                        try:
                            cancel_order_safe(trailing_order.id, symbol, reason="signal_exit")
                        except Exception as e:
                            logger.error(
                                "Failed to cancel trailing stop for %s: %s", symbol, e
                            )
                    primary_signal = _select_primary_exit_signal(actionable_signals) or actionable_signals[0]
                    exit_snapshot = _build_exit_snapshot(
                        position,
                        indicators,
                        actionable_signals,
                        trail_pct,
                        _extract_risk_score(position, indicators),
                    )
                    increment_metric("exits_signal")
                    submit_sell_market_order(
                        position,
                        reason=str(primary_signal.get("detail", "")),
                        reason_code=str(primary_signal.get("code", "SIGNAL_REVERSAL")),
                        exit_signals=actionable_signals,
                        exit_snapshot=exit_snapshot,
                    )
                    if symbol in PARTIAL_EXIT_TAKEN:
                        PARTIAL_EXIT_TAKEN.pop(symbol, None)
                        _save_partial_state(PARTIAL_EXIT_TAKEN)
            else:
                logger.info(f"No sell signal for {symbol}; managing trailing stop.")
                manage_trailing_stop(position, indicators=indicators, exit_signals=exit_signals)
        else:
            reasons = check_sell_signal(symbol, indicators)
            if reasons:
                if has_pending_sell_order(symbol):
                    logger.info("Sell order already pending for %s", symbol)
                    continue

                trailing_order = get_trailing_stop_order(symbol)
                if trailing_order:
                    try:
                        cancel_order_safe(trailing_order.id, symbol, reason="signal_exit")
                    except Exception as e:
                        logger.error(
                            "Failed to cancel trailing stop for %s: %s", symbol, e
                        )
                reason_text = "; ".join(reasons)
                increment_metric("exits_signal")
                submit_sell_market_order(
                    position,
                    reason=reason_text,
                    reason_code="monitor",
                    exit_signals=None,
                    exit_snapshot=None,
                )
                if symbol in PARTIAL_EXIT_TAKEN:
                    PARTIAL_EXIT_TAKEN.pop(symbol, None)
                    _save_partial_state(PARTIAL_EXIT_TAKEN)
            else:
                logger.info(f"No sell signal for {symbol}; managing trailing stop.")
                manage_trailing_stop(position)

    return positions


def monitor_positions(*, run_once: bool = False, kill_switch_path: Path | None = None) -> str:
    logger.info("[MONITOR_START] Starting real-time position monitoring (pid=%s)", os.getpid())
    write_status(
        status="starting",
        positions_count=0,
        trailing_count=0,
        protective_orders_count=0,
        stop_coverage_pct=0.0,
    )
    while True:
        logger.info("CYCLE_START pid=%s", os.getpid())
        if _kill_switch_triggered(kill_switch_path):
            logger.warning("Kill switch detected at %s; exiting monitor loop", kill_switch_path)
            write_status(
                status="stopped",
                positions_count=0,
                trailing_count=0,
                protective_orders_count=0,
                stop_coverage_pct=0.0,
            )
            return "killed"
        positions: list = []
        trailing_stops_count = 0
        protective_orders_count = 0
        stop_coverage_pct = 0.0
        loop_status = "running"
        market_hours = True
        try:
            now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ)
            market_hours = TRADING_START_HOUR <= now_et.hour < TRADING_END_HOUR
            logger.info(
                "MARKET_WINDOW now_et=%s market_hours=%s start=%s end=%s",
                now_et.isoformat(),
                market_hours,
                TRADING_START_HOUR,
                TRADING_END_HOUR,
            )

            log_if_stale(open_pos_path, "open_positions.csv", threshold_minutes=10)

            if market_hours:
                logger.info("CYCLE_PATH process_positions_cycle")
                positions = process_positions_cycle() or []
            else:
                logger.info("CYCLE_PATH update_open_positions")
                if env_bool("MONITOR_ENABLE_LIVE_PRICES", default=False):
                    logger.info("LIVE_PRICE_SKIP reason=off_hours")
                positions = update_open_positions() or []

            protective_orders_count, stop_coverage_pct, trailing_stops_count = enforce_stop_coverage(positions)

            check_pending_orders()

            logger.info("Updated open_positions.csv successfully.")
            loop_status = "ok"
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
        cycle_rc = 0 if loop_status == "ok" else 1
        logger.info("CYCLE_END rc=%s status=%s", cycle_rc, loop_status)

        if run_once:
            return loop_status

        sleep_time = SLEEP_INTERVAL if market_hours else OFF_HOUR_SLEEP_INTERVAL
        time.sleep(sleep_time)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monitor positions and manage exits.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single monitor cycle and exit.",
    )
    parser.add_argument(
        "--kill-switch",
        default=None,
        help="Path to kill switch file; if present, exit gracefully.",
    )
    args = parser.parse_args(argv)

    assert_paper_mode()

    config = _build_monitor_config()
    config["once"] = bool(args.once)
    config["kill_switch"] = args.kill_switch
    logger.info("MONITOR_CONFIG %s", json.dumps(config, sort_keys=True))

    kill_switch_path = None
    if args.kill_switch:
        kill_switch_path = Path(args.kill_switch).expanduser().resolve()

    logger.info("Starting monitor_positions.py")
    if is_debug_mode():
        run_debug_smoke_test()

    if args.once:
        try:
            status = monitor_positions(run_once=True, kill_switch_path=kill_switch_path)
            if status not in {"ok", "killed"}:
                raise SystemExit(1)
        except Exception:
            logger.exception("Monitoring loop error")
            raise SystemExit(1)
        return

    while True:
        try:
            result = monitor_positions(run_once=False, kill_switch_path=kill_switch_path)
            if result in {"killed", "once"}:
                return
        except Exception:
            logger.exception("Monitoring loop error")
        if _kill_switch_triggered(kill_switch_path):
            write_status(
                status="stopped",
                positions_count=0,
                trailing_count=0,
                protective_orders_count=0,
                stop_coverage_pct=0.0,
            )
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
