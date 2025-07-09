# monitor_positions.py

import os
import time
from datetime import datetime
import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import logging
import pytz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'monitor.log')
logging.basicConfig(
    handlers=[RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")
alpaca = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Constants
SLEEP_INTERVAL = int(os.getenv("MONITOR_SLEEP_INTERVAL", "60"))
OFF_HOUR_SLEEP_INTERVAL = int(os.getenv("MONITOR_OFF_HOUR_SLEEP", "300"))
TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", "4"))
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", "20"))
EASTERN_TZ = pytz.timezone('US/Eastern')

# Fetch current positions
def get_open_positions():
    try:
        return alpaca.list_positions()
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
                'current_price': p.current_price,
                'unrealized_pl': p.unrealized_pl,
            }
            for p in positions
        ]
        pd.DataFrame(data).to_csv(csv_path, index=False)
        logging.debug("Saved open positions to %s", csv_path)
    except Exception as e:
        logging.error("Failed to save positions CSV: %s", e)

# Check sell signals (close < 9 SMA or 20 EMA)
def sell_signal(symbol):
    try:
        bars = alpaca.get_bars(symbol, TimeFrame.Day, limit=20).df
    except Exception as e:
        logging.error("Failed to fetch bars for %s: %s", symbol, e)
        return False
    if bars.empty or len(bars) < 20:
        return False

    bars['sma9'] = bars['close'].rolling(9).mean()
    bars['ema20'] = bars['close'].ewm(span=20).mean()

    last_close = bars['close'].iloc[-1]
    last_sma9 = bars['sma9'].iloc[-1]
    last_ema20 = bars['ema20'].iloc[-1]

    if last_close < last_sma9 or last_close < last_ema20:
        logging.warning(
            "Sell signal detected for %s: close=%s, SMA9=%s, EMA20=%s",
            symbol, last_close, last_sma9, last_ema20
        )
        return True

    return False

# Execute sell orders

def close_position_order(symbol, qty, side):
    try:
        alpaca.submit_order(
            symbol=symbol,
            qty=str(qty),
            side=side,
            type='market',
            time_in_force='day',
            extended_hours=True,
        )
        logging.info("%s order submitted for %s, qty=%s", side.capitalize(), symbol, qty)
    except Exception as e:
        logging.error("Failed to submit %s order for %s: %s", side, symbol, e)

def process_positions_cycle():
    positions = get_open_positions()
    save_positions_csv(positions)
    for position in positions:
        symbol = position.symbol
        quantity = abs(float(position.qty))
        if sell_signal(symbol):
            side = 'sell' if float(position.qty) > 0 else 'buy'
            close_position_order(symbol, quantity, side)


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
    monitor_positions()

