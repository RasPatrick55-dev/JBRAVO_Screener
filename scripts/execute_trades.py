# execute_trades.py
# Dynamically execute limit buys for the highest ranked candidates.
# Trailing stops and max hold logic manage risk on open trades.

import os
import subprocess
from logging.handlers import RotatingFileHandler
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import TrailingStopOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
log_path = os.path.join(BASE_DIR, 'logs', 'execute_trades.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

error_handler = RotatingFileHandler(error_log_path, maxBytes=2_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[
        RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5),
        error_handler,
    ],
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

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
MAX_OPEN_TRADES = 4
ALLOC_PERCENT = 0.03  # Changed allocation to 3%
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7

# Candidate selection happens dynamically from ``top_candidates.csv``.


# Ensure executed trades file exists
exec_trades_path = os.path.join(BASE_DIR, 'data', 'executed_trades.csv')
if not os.path.exists(exec_trades_path):
    pd.DataFrame(
        columns=[
            'symbol',
            'qty',
            'entry_price',
            'exit_price',
            'entry_time',
            'exit_time',
            'order_status',
            'net_pnl',
            'order_type',
        ]
    ).to_csv(exec_trades_path, index=False)

# Ensure open positions file exists
open_pos_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
if not os.path.exists(open_pos_path):
    pd.DataFrame(
        columns=['symbol', 'qty', 'avg_entry_price', 'current_price', 'unrealized_pl', 'entry_price', 'entry_time']
    ).to_csv(open_pos_path, index=False)

def get_buying_power():
    acc = trading_client.get_account()
    return float(acc.buying_power)

def get_open_positions():
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}

def load_top_candidates() -> pd.DataFrame:
    """Load ranked candidates and return only the top slots available."""
    csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    try:
        df = pd.read_csv(csv_path)
        if 'score' in df.columns:
            df.sort_values('score', ascending=False, inplace=True)
        slots = max(MAX_OPEN_TRADES - len(get_open_positions()), 0)
        df = df.head(slots)
        logging.info("Loaded %s successfully", csv_path)
        return df
    except Exception as exc:
        logging.error("Failed to read %s: %s", csv_path, exc)
        return pd.DataFrame(columns=['symbol'])

def save_open_positions_csv():
    """Fetch current open positions from Alpaca and save to CSV."""
    try:
        positions = trading_client.get_all_positions()
        data = []
        for p in positions:
            data.append({
                'symbol': p.symbol,
                'qty': p.qty,
                'avg_entry_price': p.avg_entry_price,
                'current_price': p.current_price,
                'unrealized_pl': p.unrealized_pl,
                'entry_price': p.avg_entry_price,
                # Safely convert timestamp to ISO format only when present
                'entry_time': (
                    ts.isoformat() if (ts := getattr(p, 'created_at', None)) is not None else 'N/A'
                )
            })

        df = pd.DataFrame(data, columns=[
            'symbol', 'qty', 'avg_entry_price', 'current_price',
            'unrealized_pl', 'entry_price', 'entry_time']
        )
        if df.empty:
            df = pd.DataFrame(columns=[
                'symbol', 'qty', 'avg_entry_price', 'current_price',
                'unrealized_pl', 'entry_price', 'entry_time'
            ])

        csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
        df.to_csv(csv_path, index=False)
        logging.info("Saved open positions to %s", csv_path)
    except Exception as e:
        logging.error("Failed to save open positions: %s", e)

def update_trades_log():
    """Fetch recent closed orders from Alpaca and save to trades_log.csv."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100)
        orders = trading_client.get_orders(filter=request)
        records = []
        for order in orders:
            entry_price = order.filled_avg_price if order.side.value == 'buy' else ''
            exit_price = order.filled_avg_price if order.side.value == 'sell' else ''

            if order.side.value == 'buy':
                ts = order.filled_at
                entry_time = ts.isoformat() if ts is not None else 'N/A'
            else:
                entry_time = ''

            if order.side.value == 'sell':
                ts = order.filled_at
                exit_time = ts.isoformat() if ts is not None else 'N/A'
            else:
                exit_time = ''

            qty = float(order.filled_qty or 0)
            pnl = (float(exit_price) - float(entry_price)) * qty if exit_price and entry_price else 0.0

            records.append({
                'symbol': order.symbol,
                'qty': qty,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'order_status': order.status.value,
                'net_pnl': pnl,
                'order_type': getattr(order, 'order_type', ''),
            })

        df = pd.DataFrame(records, columns=[
            'symbol',
            'qty',
            'entry_price',
            'exit_price',
            'entry_time',
            'exit_time',
            'order_status',
            'net_pnl',
            'order_type',
        ])
        csv_path = os.path.join(BASE_DIR, 'data', 'trades_log.csv')
        df.to_csv(csv_path, index=False)
        logging.info("Saved trades log to %s", csv_path)
    except Exception as e:
        logging.error("Failed to update trades log: %s", e)

def record_executed_trade(symbol, entry_price, order_type, order_status="submitted"):
    """Append executed trade details to CSV using the unified schema."""
    csv_path = os.path.join(BASE_DIR, 'data', 'executed_trades.csv')
    row = {
        'symbol': symbol,
        'qty': 0,
        'entry_price': entry_price,
        'exit_price': '',
        'entry_time': datetime.now(timezone.utc).isoformat(),
        'exit_time': '',
        'order_status': order_status,
        'net_pnl': 0.0,
        'order_type': order_type,
    }
    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

def allocate_position(symbol):
    open_positions = get_open_positions()
    if symbol in open_positions or len(open_positions) >= MAX_OPEN_TRADES:
        logging.debug("Skipping %s: already trading or max trades reached", symbol)
        return None

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
    bars = data_client.get_stock_bars(request).df

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
    df = load_top_candidates()
    for _, row in df.iterrows():
        sym = row.symbol
        alloc = allocate_position(sym)
        if not alloc:
            continue

        qty, entry_price = alloc
        logging.info("Submitting limit buy order for %s, qty=%s, limit=%s", sym, qty, entry_price)
        try:
            order = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=entry_price,
                extended_hours=True
            )
            trading_client.submit_order(order)
            record_executed_trade(sym, entry_price, order_type='limit')
        except Exception as e:
            logging.error("Failed to submit buy order for %s: %s", sym, e)

def attach_trailing_stops():
    positions = get_open_positions()
    for symbol, pos in positions.items():
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = trading_client.get_orders(filter=request)
        has_trail = any(o.order_type == 'trailing_stop' for o in orders)
        if has_trail:
            logging.debug("Trailing stop already active for %s", symbol)
            continue

        logging.info("Creating trailing stop for %s, qty=%s", symbol, pos.qty)
        try:
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=pos.qty,
                side=OrderSide.SELL,
                trail_percent=TRAIL_PERCENT,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(request)
            record_executed_trade(symbol, pos.avg_entry_price, order_type='trailing_stop')
        except Exception as e:
            logging.error("Failed to create trailing stop for %s: %s", symbol, e)

def daily_exit_check():
    positions = get_open_positions()
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED)
    orders = trading_client.get_orders(filter=request)

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
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True
                )
                trading_client.submit_order(order)
                record_executed_trade(symbol, pos.current_price, order_type='market')
            except Exception as e:
                logging.error("Failed to close %s: %s", symbol, e)

if __name__ == '__main__':
    logging.info("Starting pre-market trade execution script")
    submit_trades()
    attach_trailing_stops()
    daily_exit_check()
    save_open_positions_csv()
    update_trades_log()
    logging.info("Pre-market trade execution script complete")

    # Run historical trades script after executing trades
    history_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fetch_trades_history.py')
    try:
        subprocess.run(["python", history_script], check=True)
        logging.info("Historical trades successfully fetched and CSV files updated.")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run historical trade script: %s", e)

