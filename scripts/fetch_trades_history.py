from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
import pandas as pd, os

load_dotenv()
client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)

all_orders = []
end = pd.Timestamp.utcnow().isoformat()

while True:
    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, until=end, limit=500, direction='desc')
    chunk = client.get_orders(filter=req)
    if not chunk: break
    all_orders.extend(chunk)
    end = chunk[-1].submitted_at.isoformat()

# ``Order`` objects from alpaca-py used to expose a ``_raw`` attribute. In
# newer versions this was renamed to ``raw_data`` and the official way to
# obtain a dictionary representation is via ``dict()``.  To remain compatible
# with either version we check for ``raw_data`` and fall back to ``dict()``.
records = [getattr(order, "raw_data", order.dict()) for order in all_orders]
df = pd.DataFrame(records).drop_duplicates("id")
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
df.to_csv(os.path.join(data_dir, 'trades_log.csv'), index=False)
executed = df[df['filled_qty'].astype(float) > 0]
executed.to_csv(os.path.join(data_dir, 'executed_trades.csv'), index=False)
