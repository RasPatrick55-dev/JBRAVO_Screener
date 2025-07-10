from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
from datetime import datetime, timezone
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_float(val, default=0.0):
    """Safely convert values to float with a default on failure."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

load_dotenv()
client = TradingClient(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    paper=True,
)

all_orders: list = []
end = pd.Timestamp.utcnow().isoformat()

while True:
    req = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        until=end,
        limit=500,
        direction='desc',
    )
    chunk = client.get_orders(filter=req)
    if not chunk:
        break
    all_orders.extend(chunk)
    end = chunk[-1].submitted_at.isoformat()

# Normalize all filled_at timestamps to timezone-aware UTC datetimes to avoid
# offset-naive vs offset-aware comparison issues when sorting.
for order in all_orders:
    if order.filled_at is not None and order.filled_at.tzinfo is None:
        order.filled_at = order.filled_at.replace(tzinfo=timezone.utc)

open_positions = {}

orders_with_filled = [o for o in all_orders if o.filled_at is not None]
orders_sorted = sorted(orders_with_filled, key=lambda o: o.filled_at)
records = []

for order in orders_sorted:
    side = order.side.value
    symbol = order.symbol
    avg_price = safe_float(order.filled_avg_price)
    qty = safe_float(order.filled_qty)

    entry_price = ''
    exit_price = ''
    entry_time = ''
    exit_time = ''

    if side == 'buy':
        entry_price = avg_price
        entry_time = order.filled_at.isoformat() if order.filled_at else ''
        open_positions[symbol] = {
            'price': avg_price,
            'qty': qty,
            'time': entry_time,
        }
        pnl = 0.0
    elif side == 'sell':
        exit_price = avg_price
        exit_time = order.filled_at.isoformat() if order.filled_at else ''
        if symbol in open_positions:
            entry_info = open_positions.pop(symbol)
            entry_price = entry_info['price']
            entry_time = entry_info['time']
            pnl = (avg_price - entry_info['price']) * entry_info['qty']
        else:
            pnl = 0.0
    else:
        pnl = 0.0

    records.append(
        {
            'id': order.id,
            'symbol': symbol,
            'side': side,
            'filled_qty': qty,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'order_status': order.status.value if order.status else 'unknown',
            'pnl': pnl,
        }
    )

df = pd.DataFrame(records).drop_duplicates('id')
data_dir = os.path.join(BASE_DIR, 'data')

cols = [
    'id',
    'symbol',
    'side',
    'filled_qty',
    'entry_price',
    'exit_price',
    'entry_time',
    'exit_time',
    'order_status',
    'pnl',
]

df[cols].to_csv(os.path.join(data_dir, 'trades_log.csv'), index=False)

executed_trades = df[df['filled_qty'] > 0]
executed_trades[cols].to_csv(
    os.path.join(data_dir, 'executed_trades.csv'),
    index=False,
)
