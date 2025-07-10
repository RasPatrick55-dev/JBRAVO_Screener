# monitor_positions.py

import os
import time
from datetime import datetime
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import QueryOrderStatus, OrderSide
from alpaca.trading.requests import GetOrdersRequest
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import logging
import pytz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'monitor.log')

LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=5)
handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)

open_pos_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
if not os.path.exists(open_pos_path):
    pd.DataFrame(
        columns=['symbol', 'qty', 'avg_entry_price', 'current_price',
                 'unrealized_pl', 'entry_price', 'entry_time']
    ).to_csv(open_pos_path, index=False)

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
EASTERN_TZ = pytz.timezone('US/Eastern')

# Fetch current positions
def get_open_positions():
    try:
        return trading_client.get_all_positions()
    except Exception as e:
        logging.error("Failed to fetch open positions: %s", e)
        return []

# Save open positions to CSV for dashboard consumption
def save_positions_csv(positions):
    csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
    try:
        data = [
            {
                'symbol': p.symbol,
                'qty': p.qty,
                'avg_entry_price': p.avg_entry_price,
                'current_price': p.current_price,
                'unrealized_pl': p.unrealized_pl,
                'entry_price': p.avg_entry_price,
                'entry_time': getattr(p, 'created_at', datetime.utcnow()).isoformat(),
            }
            for p in positions
        ]
        df = pd.DataFrame(
            data,
            columns=[
                'symbol', 'qty', 'avg_entry_price', 'current_price',
                'unrealized_pl', 'entry_price', 'entry_time'
            ]
        )
        if df.empty:
            df = pd.DataFrame(columns=[
                'symbol', 'qty', 'avg_entry_price', 'current_price',
                'unrealized_pl', 'entry_price', 'entry_time'
            ])
        df.to_csv(csv_path, index=False)
        logging.debug("Saved open positions to %s", csv_path)
        logging.info("Updated open_positions.csv successfully.")
    except Exception as e:
        logging.error("Failed to save positions CSV: %s", e)

def fetch_indicators(symbol):
    """Fetch recent daily bars and compute indicators."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=50
        )
        bars = data_client.get_stock_bars(request).df
    except Exception as e:
        logging.error("Failed to fetch bars for %s: %s", symbol, e)
        return None

    if bars.empty or len(bars) < 26:
        logging.warning("Not enough bars for %s indicator calculation", symbol)
        return None

    bars['SMA9'] = bars['close'].rolling(9).mean()
    bars['EMA20'] = bars['close'].ewm(span=20).mean()

    delta = bars['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    bars['RSI'] = 100 - (100 / (1 + rs))

    ema12 = bars['close'].ewm(span=12, adjust=False).mean()
    ema26 = bars['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    bars['MACD'] = macd
    bars['MACD_signal'] = macd.ewm(span=9, adjust=False).mean()

    last = bars.iloc[-1]
    return {
        'close': float(last['close']),
        'SMA9': float(last['SMA9']),
        'EMA20': float(last['EMA20']),
        'RSI': float(last['RSI']),
        'MACD': float(last['MACD']),
        'MACD_signal': float(last['MACD_signal']),
    }


def check_sell_signal(indicators) -> bool:
    price = indicators['close']
    ema20 = indicators['EMA20']
    rsi = indicators['RSI']
    macd = indicators['MACD']
    macd_signal = indicators['MACD_signal']

    sell_by_ma = price < ema20
    sell_by_rsi = rsi < 50
    sell_by_macd = macd < macd_signal

    if sell_by_ma and sell_by_rsi and sell_by_macd:
        return True
    return False


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
        if getattr(o, 'order_type', '') == 'trailing_stop':
            return o
    return None


def last_filled_trailing_stop(symbol):
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=10)
    try:
        orders = trading_client.get_orders(filter=request)
        for o in orders:
            if (
                getattr(o, 'order_type', '') == 'trailing_stop'
                and o.status == 'filled'
            ):
                return o
    except Exception as e:
        logging.error("Failed to fetch order history for %s: %s", symbol, e)
    return None


def has_pending_sell_order(symbol):
    orders = get_open_orders(symbol)
    for o in orders:
        if o.side == OrderSide.SELL and getattr(o, 'order_type', '') != 'trailing_stop':
            return True
    return False


def manage_trailing_stop(position):
    symbol = position.symbol
    qty = position.qty
    logging.info(f"Evaluating trailing stop for {symbol} – qty: {qty}")
    entry = float(position.avg_entry_price)
    current = float(position.current_price)
    gain_pct = (current - entry) / entry * 100 if entry else 0

    logging.info(f"Checking existing orders for {symbol}.")
    trailing_order = get_trailing_stop_order(symbol)
    if not trailing_order:
        logging.info(f"No active trailing stop for {symbol}.")
        last_filled = last_filled_trailing_stop(symbol)
        if last_filled:
            logging.info(
                "Previous trailing stop filled for %s at %s",
                symbol,
                getattr(last_filled, 'filled_avg_price', 'n/a'),
            )
        try:
            trading_client.submit_order(
                symbol=symbol,
                qty=position.qty,
                side='sell',
                type='trailing_stop',
                trail_percent='5',
                time_in_force='gtc'
            )
            logging.info(f"Placed new trailing stop for {symbol}.")
        except Exception as e:
            logging.error("Failed to create trailing stop for %s: %s", symbol, e)
        return
    else:
        logging.info(
            f"Trailing stop already exists for {symbol} (order id {trailing_order.id})."
        )

    if gain_pct > 10:
        new_trail = '3'
        try:
            trading_client.cancel_order(trailing_order.id)
            trading_client.submit_order(
                symbol=symbol,
                qty=position.qty,
                side='sell',
                type='trailing_stop',
                trail_percent=new_trail,
                time_in_force='gtc'
            )
            logging.info(
                "Adjusted trailing stop for %s from 5% to %s%% (gain: %.2f%%).",
                symbol,
                new_trail,
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

def submit_sell_market_order(symbol, qty):
    """Submit a market sell order with error handling."""
    try:
        trading_client.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        logging.info(
            "[SELL SIGNAL] Selling %s (qty %s) due to strategy exit conditions.",
            symbol, qty
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
        indicators = fetch_indicators(symbol)
        if not indicators:
            continue

        logging.info(
            "Evaluating sell conditions for %s - price: %.2f, EMA20: %.2f, RSI: %.2f",
            symbol,
            indicators['close'],
            indicators['EMA20'],
            indicators['RSI'],
        )

        if check_sell_signal(indicators):
            if has_pending_sell_order(symbol):
                logging.info("Sell order already pending for %s", symbol)
                continue

            trailing_order = get_trailing_stop_order(symbol)
            if trailing_order:
                try:
                    trading_client.cancel_order(trailing_order.id)
                except Exception as e:
                    logging.error("Failed to cancel trailing stop for %s: %s", symbol, e)

            submit_sell_market_order(symbol, position.qty)
        else:
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
                    save_positions_csv([])
                    off_hours_written = True
                sleep_time = OFF_HOUR_SLEEP_INTERVAL

            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")

if __name__ == '__main__':
    logging.info("Starting monitor_positions.py")
    print("Starting monitor_positions.py")
    try:
        monitor_positions()
    except Exception as e:
        logging.error(f"Script error: {e}")

