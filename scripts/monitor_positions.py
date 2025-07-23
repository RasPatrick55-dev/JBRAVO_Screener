# monitor_positions.py

import os
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, TrailingStopOrderRequest
from logging.handlers import RotatingFileHandler
from utils import fetch_bars_with_cutoff
import shutil
from tempfile import NamedTemporaryFile


class InfoRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that logs when a rollover occurs."""

    def doRollover(self):
        super().doRollover()
        logging.info("Log rotated due to size limit.")


from dotenv import load_dotenv
import logging
import pytz
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "monitor.log")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
handler = InfoRotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5)
handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)


def send_alert(message: str):
    """Send alert message to webhook if configured."""
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logging.error("Failed to send alert: %s", exc)


def log_if_stale(file_path: str, name: str, threshold_minutes: int = 15):
    """Log a warning if ``file_path`` is older than ``threshold_minutes``."""
    if not os.path.exists(file_path):
        logging.warning("%s missing: %s", name, file_path)
        send_alert(f"{name} missing: {file_path}")
        return
    age = datetime.utcnow() - datetime.utcfromtimestamp(os.path.getmtime(file_path))
    if age > timedelta(minutes=threshold_minutes):
        minutes = age.total_seconds() / 60
        msg = f"{name} is stale ({minutes:.1f} minutes old)"
        logging.warning(msg)
        send_alert(msg)


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
    logging.error("Missing Alpaca API credentials.")
    raise SystemExit(1)

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
        logging.error("Failed to fetch open positions: %s", e)
        return []


# Save open positions to CSV for dashboard consumption
def save_positions_csv(positions):
    csv_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
    try:
        existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

        active_symbols = set()
        rows = []
        for p in positions:
            try:
                qty = float(p.qty)
            except Exception:
                qty = 0
            if qty <= 0:
                logging.warning(
                    "Skipping %s due to non-positive quantity: %s", p.symbol, qty
                )
                continue
            active_symbols.add(p.symbol)
            rows.append(
                {
                    "symbol": p.symbol,
                    "qty": qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "unrealized_pl": p.unrealized_pl,
                    "entry_price": p.avg_entry_price,
                    "entry_time": getattr(p, "created_at", datetime.utcnow()).isoformat(),
                    "side": getattr(p, "side", "long"),
                    "order_status": "open",
                    "net_pnl": p.unrealized_pl,
                    "pnl": p.unrealized_pl,
                    "order_type": getattr(p, "order_type", "market"),
                }
            )

        columns = [
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "entry_price",
            "entry_time",
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
        df = df[columns]

        removed = []
        if not existing.empty and "symbol" in existing.columns:
            removed = sorted(set(existing["symbol"]) - active_symbols)
        if removed:
            msg = f"Removing inactive positions: {', '.join(removed)}"
            logging.info(msg)
            send_alert(msg)

        if df.empty:
            df = pd.DataFrame(columns=columns)
        tmp = NamedTemporaryFile(
            "w", delete=False, dir=os.path.dirname(csv_path), newline=""
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        shutil.move(tmp.name, csv_path)
        logging.debug("Saved open positions to %s", csv_path)
        logging.info("Updated open_positions.csv successfully.")
    except Exception as e:
        logging.error("Failed to save positions CSV: %s", e)


def fetch_indicators(symbol):
    """Fetch recent daily bars and compute indicators."""
    try:
        start = (datetime.now(timezone.utc) - timedelta(days=750)).strftime("%Y-%m-%dT%H:%M:%SZ")
        bars = fetch_bars_with_cutoff(symbol, start, TimeFrame.Day, data_client).df
    except Exception as e:
        logging.error("Failed to fetch bars for %s: %s", symbol, e)
        return None

    logging.debug(
        f"{symbol}: Screener-bar-count={len(bars)}, Monitor-threshold={required_bars}"
    )

    if bars.empty or len(bars) < required_bars:
        logging.warning("Not enough bars for %s indicator calculation", symbol)
        logging.info(
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
        logging.info("Price %.2f below EMA20 %.2f", price, ema20)
        reasons.append("EMA20 cross")
    if rsi > 70:
        logging.info("RSI %.2f above 70", rsi)
        reasons.append("RSI > 70")
    return reasons


def get_open_orders(symbol):
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    try:
        return trading_client.get_orders(filter=request)
    except Exception as e:
        logging.error("Failed to fetch open orders for %s: %s", symbol, e)
        return []


def get_trailing_stop_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if getattr(o, "order_type", "") == "trailing_stop":
            return o
    return None


def last_filled_trailing_stop(symbol):
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=10
    )
    try:
        orders = trading_client.get_orders(filter=request)
        for o in orders:
            if getattr(o, "order_type", "") == "trailing_stop" and o.status == "filled":
                return o
    except Exception as e:
        logging.error("Failed to fetch order history for %s: %s", symbol, e)
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
            logging.error("Failed to log trade to %s: %s", path, exc)


def has_pending_sell_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if o.side == OrderSide.SELL and getattr(o, "order_type", "") != "trailing_stop":
            return True
    return False


def manage_trailing_stop(position):
    symbol = position.symbol
    logging.info(f"Evaluating trailing stop for {symbol}")
    if symbol.upper() == "ARQQ":
        # Explicit log entry requested to verify ARQQ trailing stop evaluation
        logging.info("[INFO] Evaluating trailing stop for ARQQ")
    qty = position.qty
    logging.info(f"Evaluating trailing stop for {symbol} – qty: {qty}")
    entry = float(position.avg_entry_price)
    current = float(position.current_price)
    gain_pct = (current - entry) / entry * 100 if entry else 0
    if not qty or float(qty) <= 0:
        logging.info(
            f"Skipping trailing stop for {symbol} due to non-positive quantity: {qty}."
        )
        return
    logging.debug(
        f"Entry={entry}, Current={current}, Gain={gain_pct:.2f}% for {symbol}."
    )
    logging.info(f"Checking existing orders for {symbol}.")
    trailing_order = get_trailing_stop_order(symbol)
    if not trailing_order:
        logging.info(f"No active trailing stop for {symbol}.")
        last_filled = last_filled_trailing_stop(symbol)
        if last_filled:
            logging.info(
                "Previous trailing stop filled for %s at %s",
                symbol,
                getattr(last_filled, "filled_avg_price", "n/a"),
            )
        logging.info(
            f"Placing trailing stop for {symbol}: qty={qty}, side=SELL, trail_pct={TRAIL_START_PERCENT}"
        )
        try:
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=position.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=str(TRAIL_START_PERCENT),
            )
            trading_client.submit_order(order_data=request)
            logging.info(f"Placed new trailing stop for {symbol} at {TRAIL_START_PERCENT}%.")
        except Exception as e:
            logging.error("Failed to create trailing stop for %s: %s", symbol, e)
        return
    else:
        logging.info(
            "Existing trailing stop for %s (order id %s, status %s, trail %% %s)",
            symbol,
            trailing_order.id,
            trailing_order.status,
            getattr(trailing_order, "trail_percent", "n/a"),
        )
        logging.info(f"Skipping creation of new trailing stop for {symbol}.")

    if gain_pct > GAIN_THRESHOLD_ADJUST:
        new_trail = str(TRAIL_TIGHT_PERCENT)
        try:
            trading_client.cancel_order(trailing_order.id)
            logging.info(
                f"Placing trailing stop for {symbol}: qty={qty}, side=SELL, trail_pct={new_trail}"
            )
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=position.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=new_trail,
            )
            trading_client.submit_order(order_data=request)
            logging.info(
                "Adjusted trailing stop for %s from %s%% to %s%% (gain: %.2f%%).",
                symbol,
                TRAIL_START_PERCENT,
                TRAIL_TIGHT_PERCENT,
                gain_pct,
            )
        except Exception as e:
            logging.error("Failed to adjust trailing stop for %s: %s", symbol, e)
    else:
        logging.info(
            "No trailing stop adjustment needed for %s (gain: %.2f%%)",
            symbol,
            gain_pct,
        )


# Execute sell orders


def submit_sell_market_order(position, reason: str):
    """Submit a market sell order and log the attempt."""
    symbol = position.symbol
    qty = position.qty
    entry_price = float(position.avg_entry_price)
    entry_time = getattr(position, "created_at", datetime.utcnow()).isoformat()
    try:
        trading_client.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )
        logging.info("[EXIT] Market sell %s qty %s due to %s", symbol, qty, reason)
        log_trade_exit(
            symbol,
            float(qty),
            entry_price,
            "",
            entry_time,
            "",
            "submitted",
            "market",
            reason,
        )
    except Exception as e:
        logging.error("Error submitting sell order for %s: %s", symbol, e)


def process_positions_cycle():
    positions = get_open_positions()
    save_positions_csv(positions)
    if not positions:
        logging.info("No open positions found.")
        return
    for position in positions:
        symbol = position.symbol
        days_held = calculate_days_held(position)
        logging.info("%s held for %d days", symbol, days_held)
        if days_held >= MAX_HOLD_DAYS:
            if has_pending_sell_order(symbol):
                logging.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                try:
                    trading_client.cancel_order(trailing_order.id)
                    logging.info(
                        "Cancelled trailing stop for %s due to max hold exit",
                        symbol,
                    )
                except Exception as e:
                    logging.error(
                        "Failed to cancel trailing stop for %s: %s", symbol, e
                    )

            submit_sell_market_order(position, reason=f"Max Hold {days_held}d")
            continue

        indicators = fetch_indicators(symbol)
        if not indicators:
            logging.info(
                "Indicators unavailable for %s. Trailing-stop logic will be skipped.",
                symbol,
            )
            continue

        logging.info(
            "Evaluating sell conditions for %s - price: %.2f, EMA20: %.2f, RSI: %.2f",
            symbol,
            indicators["close"],
            indicators["EMA20"],
            indicators["RSI"],
        )

        reasons = check_sell_signal(indicators)
        if reasons:
            if has_pending_sell_order(symbol):
                logging.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                try:
                    trading_client.cancel_order(trailing_order.id)
                except Exception as e:
                    logging.error(
                        "Failed to cancel trailing stop for %s: %s", symbol, e
                    )
            reason_text = "; ".join(reasons)
            submit_sell_market_order(position, reason=reason_text)
        else:
            logging.info(f"No sell signal for {symbol}; managing trailing stop.")
            manage_trailing_stop(position)


def monitor_positions():
    logging.info("Starting real-time position monitoring...")
    try:
        save_positions_csv(get_open_positions())
    except Exception as e:
        logging.error("Initial position fetch failed: %s", e)

    in_market = False
    off_hours_written = False
    try:
        while True:
            now_et = datetime.now(pytz.utc).astimezone(EASTERN_TZ)
            market_hours = TRADING_START_HOUR <= now_et.hour < TRADING_END_HOUR

            log_if_stale(open_pos_path, "open_positions.csv")

            if market_hours and not in_market:
                logging.info("Entering market hours")
                in_market = True
                off_hours_written = False
            elif not market_hours and in_market:
                logging.info("Exiting market hours")
                in_market = False

            if market_hours:
                try:
                    process_positions_cycle()
                except Exception as e:
                    logging.error("Error during monitoring cycle: %s", e)
                sleep_time = SLEEP_INTERVAL
            else:
                if not off_hours_written:
                    current_positions = get_open_positions()
                    save_positions_csv(current_positions)
                    if current_positions:
                        logging.info("Persisted open positions to CSV.")
                    else:
                        logging.info("No open positions found; CSV cleared.")
                    off_hours_written = True
                sleep_time = OFF_HOUR_SLEEP_INTERVAL

            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")


if __name__ == "__main__":
    logging.info("Starting monitor_positions.py")
    print("Starting monitor_positions.py")
    try:
        monitor_positions()
    except Exception as e:
        logging.error(f"Script error: {e}")
