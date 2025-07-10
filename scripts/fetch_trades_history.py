from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
import pandas as pd
import os


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

records = []
for order in all_orders:
    data = getattr(order, 'raw_data', order.model_dump())

    filled_qty = safe_float(data.get('filled_qty', 0))
    avg_fill_price = safe_float(data.get('filled_avg_price'))
    limit_price = safe_float(data.get('limit_price'))

    if data['side'] == 'buy':
        entry_price = avg_fill_price if avg_fill_price else limit_price
        entry_time = data.get('filled_at') or data.get('submitted_at')
    else:
        entry_price = None
        entry_time = None

    current_price = safe_float(
        data.get('filled_avg_price') or data.get('limit_price')
    )
    pnl = (
        (current_price - (entry_price or 0)) * filled_qty if entry_price else 0
    )

    records.append(
        {
            **data,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'pnl': pnl,
            'order_status': data['status'],
        }
    )

df = pd.DataFrame(records).drop_duplicates('id')
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

trades_log_cols = [
    'id',
    'symbol',
    'side',
    'filled_qty',
    'filled_avg_price',
    'submitted_at',
    'filled_at',
    'entry_price',
    'entry_time',
    'pnl',
    'status',
]
df[trades_log_cols].to_csv(
    os.path.join(data_dir, 'trades_log.csv'), index=False
)

executed_trades = df[df['filled_qty'].astype(float) > 0]
executed_trades_cols = [
    'id',
    'symbol',
    'side',
    'filled_qty',
    'filled_avg_price',
    'submitted_at',
    'filled_at',
    'entry_price',
    'entry_time',
    'order_status',
]
executed_trades[executed_trades_cols].to_csv(
    os.path.join(data_dir, 'executed_trades.csv'),
    index=False,
)
