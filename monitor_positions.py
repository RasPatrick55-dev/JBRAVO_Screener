# monitor_positions.py

import os
import time
from datetime import datetime, timezone
import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")
alpaca = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Constants
SLEEP_INTERVAL = 60  # Check every 60 seconds

# Fetch current positions
def get_open_positions():
    return alpaca.list_positions()

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
        print(f"[ALERT] Sell signal detected for {symbol}: close={last_close}, SMA9={last_sma9}, EMA20={last_ema20}")
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
        print(f"[INFO] Sell order submitted for {symbol}, qty={qty}")
    except Exception as e:
        print(f"[ERROR] Failed to submit sell order for {symbol}: {e}")

# Continuous monitoring from pre-market to after-hours
def monitor_positions():
    print("[INFO] Starting real-time position monitoring...")
    while True:
        now = datetime.now(timezone.utc)
        current_hour = now.astimezone().hour
        current_minute = now.astimezone().minute

        # Market hours from 4 AM to 8 PM EST
        if 4 <= current_hour < 20:
            positions = get_open_positions()
            for position in positions:
                symbol = position.symbol
                qty = abs(int(position.qty))

                if sell_signal(symbol):
                    submit_sell_order(symbol, qty)
        else:
            print("[INFO] Outside trading hours, monitoring paused.")

        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    monitor_positions()
