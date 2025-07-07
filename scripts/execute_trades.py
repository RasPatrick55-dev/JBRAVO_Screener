# execute_trades.py updated for pre-market trading (3% allocation, top 3 symbols, 3% trailing stop)

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")
alpaca = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Constants
MAX_OPEN_TRADES = 4
ALLOC_PERCENT = 0.03  # Changed allocation to 3%
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7

# Read top candidates (top 3 symbols only)
try:
    df = pd.read_csv('top_candidates.csv')
    df = df.sort_values('score', ascending=False).head(3)
    print("[INFO] Loaded top_candidates.csv successfully")
except Exception as e:
    print(f"[ERROR] Failed to read CSV: {e}")
    exit()

def get_buying_power():
    acc = alpaca.get_account()
    return float(acc.buying_power)

def get_open_positions():
    positions = alpaca.list_positions()
    return {p.symbol: p for p in positions}

def allocate_position(symbol):
    open_positions = get_open_positions()
    if symbol in open_positions or len(open_positions) >= MAX_OPEN_TRADES:
        print(f"[DEBUG] Skipping {symbol}: already trading or max trades reached")
        return None

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT
    bars = alpaca.get_bars(symbol, TimeFrame.Day, limit=1).df

    if bars.empty:
        print(f"[DEBUG] No bars available for {symbol}")
        return None

    last_close = bars['close'].iloc[-1]
    qty = int(alloc_amount / last_close)
    if qty < 1:
        print(f"[DEBUG] Allocation insufficient for {symbol}")
        return None

    print(f"[DEBUG] Allocating {qty} shares of {symbol} at {last_close}")
    return qty, round(last_close, 2)

def submit_trades():
    for _, row in df.iterrows():
        sym = row.symbol
        alloc = allocate_position(sym)
        if not alloc:
            continue

        qty, entry_price = alloc
        print(f"[INFO] Submitting limit buy order for {sym}, qty={qty}, limit={entry_price}")
        try:
            alpaca.submit_order(
                symbol=sym,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force='day',
                limit_price=entry_price,
                extended_hours=True  # enable pre-market trading
            )
        except Exception as e:
            print(f"[ERROR] Failed to submit buy order for {sym}: {e}")

def attach_trailing_stops():
    positions = get_open_positions()
    for symbol, pos in positions.items():
        orders = alpaca.list_orders(status='open', symbols=[symbol])
        has_trail = any(o.order_type == 'trailing_stop' for o in orders)
        if has_trail:
            print(f"[DEBUG] Trailing stop already active for {symbol}")
            continue

        print(f"[INFO] Creating trailing stop for {symbol}, qty={pos.qty}")
        try:
            alpaca.submit_order(
                symbol=symbol,
                qty=pos.qty,
                side='sell',
                type='trailing_stop',
                trail_percent=TRAIL_PERCENT,
                time_in_force='gtc'
            )
        except Exception as e:
            print(f"[ERROR] Failed to create trailing stop for {symbol}: {e}")

def daily_exit_check():
    positions = get_open_positions()
    orders = alpaca.list_orders(status='closed')

    for symbol, pos in positions.items():
        entry_orders = [o for o in orders if o.symbol == symbol and o.side == 'buy']
        if not entry_orders:
            print(f"[WARN] No entry order found for {symbol}, skipping.")
            continue

        entry_order = sorted(entry_orders, key=lambda o: o.filled_at, reverse=True)[0]
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        print(f"[DEBUG] {symbol} entered on {entry_date}, held for {days_held} days")

        if days_held >= MAX_HOLD_DAYS:
            print(f"[INFO] Exiting {symbol} after {days_held} days")
            try:
                alpaca.submit_order(
                    symbol=symbol,
                    qty=pos.qty,
                    side='sell',
                    type='market',
                    time_in_force='day',
                    extended_hours=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to close {symbol}: {e}")

if __name__ == '__main__':
    print("[INFO] Starting pre-market trade execution script")
    submit_trades()
    attach_trailing_stops()
    daily_exit_check()
    print("[INFO] Pre-market trade execution script complete")

