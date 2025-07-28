# monitor_positions.py

import os
import sys
import time
from datetime import datetime, timedelta, timezone, time as dt_time
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    TrailingStopOrderRequest,
    LimitOrderRequest,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils import fetch_bars_with_cutoff
from utils.logger_utils import init_logging
import shutil
from tempfile import NamedTemporaryFile


from dotenv import load_dotenv
import pytz
import requests

dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

logger = init_logging(__name__, "monitor.log")
logger.info("Monitoring service active.")


def send_alert(message: str):
    """Send alert message to webhook if configured."""
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)


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

executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")

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
TRAIL_START_PERCENT = float(os.getenv("TRAIL_START_PERCENT", "5"))
TRAIL_TIGHT_PERCENT = float(os.getenv("TRAIL_TIGHT_PERCENT", "3"))
GAIN_THRESHOLD_ADJUST = float(os.getenv("GAIN_THRESHOLD_ADJUST", "10"))
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
        return trading_client.get_all_positions()
    except Exception as e:
        logger.error("Failed to fetch open positions: %s", e)
        return []


# Save open positions to CSV for dashboard consumption
def save_positions_csv(positions):
    csv_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
    try:
        existing_positions_df = (
            pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        )

        active_symbols = set()
        rows = []
        for p in positions:
            try:
                qty = float(p.qty)
            except Exception:
                qty = 0
            if qty <= 0:
                logger.warning(
                    "Skipping %s due to non-positive quantity: %s", p.symbol, qty
                )
                continue
            active_symbols.add(p.symbol)
            if (
                not existing_positions_df.empty
                and p.symbol in existing_positions_df.get("symbol", []).values
            ):
                try:
                    entry_time = (
                        existing_positions_df.loc[
                            existing_positions_df["symbol"] == p.symbol,
                            "entry_time",
                        ].values[0]
                    )
                except Exception:
                    entry_time = (
                        getattr(p, "created_at", datetime.utcnow()).isoformat()
                    )
            else:
                entry_attr = getattr(p, "created_at", None)
                if entry_attr is not None:
                    entry_time = (
                        entry_attr.isoformat()
                        if hasattr(entry_attr, "isoformat")
                        else str(entry_attr)
                    )
                else:
                    entry_time = datetime.utcnow().isoformat()
            updated_entry = {
                "symbol": p.symbol,
                "qty": qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "unrealized_pl": p.unrealized_pl,
                "entry_price": p.avg_entry_price,
                "entry_time": entry_time,
                "side": getattr(p, "side", "long"),
                "order_status": "open",
                "net_pnl": p.unrealized_pl,
                "pnl": p.unrealized_pl,
                "order_type": getattr(p, "order_type", "market"),
            }

            rows.append(updated_entry)

        columns = [
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

        df = pd.DataFrame(rows)

        df['side'] = df.get('side', 'long')
        df['order_status'] = df.get('order_status', 'open')
        df['net_pnl'] = df.get('unrealized_pl', 0.0)
        df['pnl'] = df['net_pnl']
        df['order_type'] = df.get('order_type', 'limit')
        df['days_in_trade'] = (
            pd.Timestamp.now() - pd.to_datetime(df['entry_time'])
        ).dt.days
        df = df[columns]

        removed = []
        if not existing_positions_df.empty and "symbol" in existing_positions_df.columns:
            removed = sorted(set(existing_positions_df["symbol"]) - active_symbols)
        if removed:
            msg = f"Removing inactive positions: {', '.join(removed)}"
            logger.info(msg)
            send_alert(msg)

        if df.empty:
            df = pd.DataFrame(columns=columns)
        tmp = NamedTemporaryFile(
            "w", delete=False, dir=os.path.dirname(csv_path), newline=""
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        shutil.move(tmp.name, csv_path)
        logger.debug("Saved open positions to %s", csv_path)
        logger.info("Updated open_positions.csv successfully.")
    except Exception as e:
        logger.error("Failed to save positions CSV: %s", e)


def update_open_positions():
    """Fetch positions and persist to CSV."""
    positions = get_open_positions()
    save_positions_csv(positions)


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
    return {
        "close": float(last["close"]),
        "SMA9": float(last["SMA9"]),
        "EMA20": float(last["EMA20"]),
        "RSI": float(last["RSI"]),
        "MACD": float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
    }


def check_sell_signal(indicators) -> list:
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
    return reasons


def get_open_orders(symbol):
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    try:
        open_orders = trading_client.get_orders(filter=request)
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
):
    """Append a standardized trade record to CSV files."""
    row = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "order_status": order_status,
        "net_pnl": 0.0,
        "order_type": order_type,
        "exit_reason": exit_reason,
        "side": "sell",
    }
    for path in (executed_trades_path, trades_log_path):
        try:
            pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)
        except Exception as exc:
            logger.error("Failed to log trade to %s: %s", path, exc)


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
        trading_client.submit_order(order_data=request)
        logger.info(
            f"Placed trailing stop for {symbol}: qty={qty}, trail_pct={trail_percent}"
        )
    except Exception as exc:
        logger.error("Failed to submit trailing stop for %s: %s", symbol, exc)


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
        return
    desired_qty = int(qty)
    if getattr(position, "qty_available", None) and int(position.qty_available) >= desired_qty:
        use_qty = int(position.qty_available)
    else:
        logger.warning(
            f"Insufficient available qty for {symbol}: {getattr(position, 'qty_available', 0)}"
        )
        return

    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        existing_orders = trading_client.get_orders(filter=request)
        logger.info(
            f"Fetched open orders for {symbol}: {len(existing_orders)} found.")
        for order in existing_orders:
            if getattr(order, "order_type", "") == "trailing_stop":
                trading_client.cancel_order_by_id(order.id)
                logger.info(
                    f"Cancelled existing trailing-stop order {order.id} for {symbol}"
                )
    except Exception as exc:
        logger.error("Failed to cancel existing trailing stop for %s: %s", symbol, exc)
    try:
        refreshed = trading_client.get_open_position(symbol)
        available_qty = int(getattr(refreshed, "qty_available", 0))
    except Exception as exc:
        logger.error(
            "Failed to fetch position for %s after cancelling trailing stop: %s",
            symbol,
            exc,
        )
        return

    if available_qty > 0:
        submit_new_trailing_stop(symbol, available_qty, TRAIL_START_PERCENT)
    else:
        logger.warning(
            f"No available quantity for trailing stop on {symbol} after cancelling."
        )
        return

    logger.debug(
        f"Entry={entry}, Current={current}, Gain={gain_pct:.2f}% for {symbol}."
    )
    logger.info(f"Checking existing orders for {symbol}.")
    trailing_order = get_trailing_stop_order(symbol)
    if not trailing_order:
        logger.info(f"No active trailing stop for {symbol}.")
        last_filled = last_filled_trailing_stop(symbol)
        if last_filled:
            logger.info(
                "Previous trailing stop filled for %s at %s",
                symbol,
                getattr(last_filled, "filled_avg_price", "n/a"),
            )
        # Trailing stop already placed above
        return
    else:
        logger.info(
            "Existing trailing stop for %s (order id %s, status %s, trail %% %s)",
            symbol,
            trailing_order.id,
            trailing_order.status,
            getattr(trailing_order, "trail_percent", "n/a"),
        )

    if gain_pct > GAIN_THRESHOLD_ADJUST:
        new_trail = str(TRAIL_TIGHT_PERCENT)
        try:
            cancel_order_safe(trailing_order.id, symbol)
            logger.info(
                f"Placing trailing stop for {symbol}: qty={qty}, side=SELL, trail_pct={new_trail}"
            )
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=use_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=new_trail,
            )
            trading_client.submit_order(order_data=request)
            logger.info(
                "Adjusted trailing stop for %s from %s%% to %s%% (gain: %.2f%%).",
                symbol,
                TRAIL_START_PERCENT,
                TRAIL_TIGHT_PERCENT,
                gain_pct,
            )
        except Exception as e:
            logger.error("Failed to adjust trailing stop for %s: %s", symbol, e)
            try:
                trading_client.close_position(symbol)
            except Exception as exc:  # pragma: no cover - API errors
                logger.error("Failed to close position %s: %s", symbol, exc)
    else:
        logger.info(
            "No trailing stop adjustment needed for %s (gain: %.2f%%)",
            symbol,
            gain_pct,
        )


def check_pending_orders():
    """Log status of any open sell orders."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(filter=request)
        logger.info(
            f"Fetched open orders for all symbols: {len(open_orders)} found.")
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


def submit_sell_market_order(position, reason: str):
    """Submit a limit sell order compatible with extended hours."""
    symbol = position.symbol
    qty = position.qty
    entry_price = float(position.avg_entry_price)
    exit_price = round_price(float(getattr(position, "current_price", entry_price)))
    entry_time = getattr(position, "created_at", datetime.utcnow()).isoformat()
    now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ).time()

    desired_qty = int(qty)
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
        trading_client.submit_order(order_request)
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
            "submitted",
            "limit",
            reason,
        )
    except Exception as e:
        logger.error("Error submitting sell order for %s: %s", symbol, e)
        try:
            trading_client.close_position(symbol)
        except Exception as exc:  # pragma: no cover - API errors
            logger.error("Failed to close position %s: %s", symbol, exc)


def process_positions_cycle():
    positions = get_open_positions()
    save_positions_csv(positions)
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

            submit_sell_market_order(position, reason=f"Max Hold {days_held}d")
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

        reasons = check_sell_signal(indicators)
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
            submit_sell_market_order(position, reason=reason_text)
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
