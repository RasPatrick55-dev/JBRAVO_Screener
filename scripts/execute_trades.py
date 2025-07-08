# execute_trades.py updated for pre-market trading (3% allocation, top 3 symbols, 3% trailing stop)

import os
from logging.handlers import RotatingFileHandler
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
log_path = os.path.join(BASE_DIR, 'logs', 'execute_trades.log')
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
MAX_OPEN_TRADES = 4
ALLOC_PERCENT = 0.03  # Changed allocation to 3%
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7

# Read top candidates (top 3 symbols only)
try:
    csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    df = pd.read_csv(csv_path)
    df = df.sort_values('score', ascending=False).head(3)
    logging.info("Loaded %s successfully", csv_path)
except Exception as e:
    logging.error("Failed to read CSV: %s", e)
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
        logging.debug("Skipping %s: already trading or max trades reached", symbol)
        return None

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT
    bars = alpaca.get_bars(symbol, TimeFrame.Day, limit=1).df

    if bars.empty:
        logging.debug("No bars available for %s", symbol)
        return None

    last_close = bars['close'].iloc[-1]
    qty = int(alloc_amount / last_close)
    if qty < 1:
        logging.debug("Allocation insufficient for %s", symbol)
        return None

    logging.debug("Allocating %d shares of %s at %s", qty, symbol, last_close)
    return qty, round(last_close, 2)

def submit_trades():
    for _, row in df.iterrows():
        sym = row.symbol
        alloc = allocate_position(sym)
        if not alloc:
            continue

        qty, entry_price = alloc
        logging.info("Submitting limit buy order for %s, qty=%s, limit=%s", sym, qty, entry_price)
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
            logging.error("Failed to submit buy order for %s: %s", sym, e)

def attach_trailing_stops():
    positions = get_open_positions()
    for symbol, pos in positions.items():
        orders = alpaca.list_orders(status='open', symbols=[symbol])
        has_trail = any(o.order_type == 'trailing_stop' for o in orders)
        if has_trail:
            logging.debug("Trailing stop already active for %s", symbol)
            continue

        logging.info("Creating trailing stop for %s, qty=%s", symbol, pos.qty)
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
            logging.error("Failed to create trailing stop for %s: %s", symbol, e)

def daily_exit_check():
    positions = get_open_positions()
    orders = alpaca.list_orders(status='closed')

    for symbol, pos in positions.items():
        entry_orders = [o for o in orders if o.symbol == symbol and o.side == 'buy']
        if not entry_orders:
            logging.warning("No entry order found for %s, skipping.", symbol)
            continue

        entry_order = sorted(entry_orders, key=lambda o: o.filled_at, reverse=True)[0]
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        logging.debug("%s entered on %s, held for %s days", symbol, entry_date, days_held)

        if days_held >= MAX_HOLD_DAYS:
            logging.info("Exiting %s after %s days", symbol, days_held)
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
                logging.error("Failed to close %s: %s", symbol, e)

if __name__ == '__main__':
    logging.info("Starting pre-market trade execution script")
    submit_trades()
    attach_trailing_stops()
    daily_exit_check()
    logging.info("Pre-market trade execution script complete")

