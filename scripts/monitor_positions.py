# monitor_positions.py

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone, time as dt_time
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


from dotenv import load_dotenv
import pytz
from utils.alerts import send_alert

dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

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

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca credentials")

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

# Minimum number of historical bars required for indicator calculation
required_bars = 200


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
        if macd < signal and prev_macd >= prev_signal:
            reasons.append("MACD cross")

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


def get_open_orders(symbol):
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    try:
        open_orders = trading_client.get_orders(request)
        logger.info(
            f"Fetched open orders for {symbol}: {len(open_orders)} found.")
        return list(open_orders)
    except Exception as e:
        logger.error("Failed to fetch open orders for %s: %s", symbol, e)
        return []


def get_trailing_stop_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if getattr(o, "order_type", "") == "trailing_stop":
            return o
    return None


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
    return None


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
    except Exception as exc:
        logger.error("Failed to submit trailing stop for %s: %s", symbol, exc)
        log_trailing_stop_event(symbol, float(trail_percent), None, "error")


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

    if abs(current_trail_pct - effective_target) < 1e-6:
        logger.info(
            "No trailing stop adjustment needed for %s (gain: %.2f%%)",
            symbol,
            gain_pct,
        )
        return

    try:
        cancel_order_safe(trailing_order.id, symbol)
        reason_detail = (
            "+10% gain" if effective_target == TRAIL_TIGHTEST_PERCENT else "+5% gain"
            if effective_target == TRAIL_TIGHT_PERCENT
            else "default"
        )
        logger.info(
            "Tightening trailing stop for %s from %.2f%% to %.2f%% due to %s",
            symbol,
            current_trail_pct,
            effective_target,
            reason_detail,
        )
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

    positions = get_open_positions()
    save_positions_csv(positions, csv_path)

    existing_symbols = set(existing_positions_df.get("symbol", []))
    current_symbols = set(p.symbol for p in positions)
    closed_symbols = existing_symbols - current_symbols

    log_closed_positions(trading_client, closed_symbols, existing_positions_df)

    if not positions:
        logger.info("No open positions found.")
        return
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
            submit_sell_market_order(position, reason=reason_text, reason_code="monitor")
            if symbol in PARTIAL_EXIT_TAKEN:
                PARTIAL_EXIT_TAKEN.pop(symbol, None)
                _save_partial_state(PARTIAL_EXIT_TAKEN)
        else:
            logger.info(f"No sell signal for {symbol}; managing trailing stop.")
            manage_trailing_stop(position)


def monitor_positions():
    logger.info("Starting real-time position monitoring...")
    while True:
        try:
            now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ)
            market_hours = TRADING_START_HOUR <= now_et.hour < TRADING_END_HOUR

            log_if_stale(open_pos_path, "open_positions.csv", threshold_minutes=10)

            if market_hours:
                process_positions_cycle()
            else:
                update_open_positions()

            check_pending_orders()

            logger.info("Updated open_positions.csv successfully.")
        except Exception as e:
            logger.exception("Monitoring loop error")

        sleep_time = SLEEP_INTERVAL if market_hours else OFF_HOUR_SLEEP_INTERVAL
        time.sleep(sleep_time)


if __name__ == "__main__":
    logger.info("Starting monitor_positions.py")
    print("Starting monitor_positions.py")
    while True:
        try:
            monitor_positions()
        except Exception as e:
            logger.exception("Monitoring loop error")
        time.sleep(300)
