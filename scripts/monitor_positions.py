# monitor_positions.py

import os
import time
from datetime import datetime, timezone
import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

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
SLEEP_INTERVAL = 60  # Check every 60 seconds

# Fetch current positions
def get_open_positions():
    return alpaca.list_positions()

# Save open positions to CSV for dashboard consumption
def save_positions_csv(positions):
    data = []
    for p in positions:
        data.append({
            'symbol': p.symbol,
            'qty': p.qty,
            'current_price': p.current_price,
            'unrealized_pl': p.unrealized_pl
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
    df.to_csv(csv_path, index=False)
    logging.debug("Saved open positions to %s", csv_path)

# Check sell signals (close < 9 SMA or 20 EMA)
def sell_signal(symbol):
    bars = alpaca.get_bars(symbol, TimeFrame.Day, limit=20).df
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

def submit_sell_order(symbol, qty):
    try:
        alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day',
            extended_hours=True
        )
        logging.info("Sell order submitted for %s, qty=%s", symbol, qty)
    except Exception as e:
        logging.error("Failed to submit sell order for %s: %s", symbol, e)

# Continuous monitoring from pre-market to after-hours
def monitor_positions():
    logging.info("Starting real-time position monitoring...")
    while True:
        now = datetime.now(timezone.utc)
        current_hour = now.astimezone().hour
        current_minute = now.astimezone().minute

        # Market hours from 4 AM to 8 PM EST
        if 4 <= current_hour < 20:
            positions = get_open_positions()
            save_positions_csv(positions)
            for position in positions:
                symbol = position.symbol
                qty = abs(int(position.qty))

                if sell_signal(symbol):
                    submit_sell_order(symbol, qty)
        else:
            logging.info("Outside trading hours, monitoring paused.")

        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    monitor_positions()

