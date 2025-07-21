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

try:  # Compatibility with alpaca-py and alpaca-trade-api
    from alpaca_trade_api.rest import APIError  # type: ignore
except Exception:  # pragma: no cover - fallback import
    from alpaca.common.exceptions import APIError  # type: ignore
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from exit_signals import should_exit_early

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

logging.info("Trade execution script started.")

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
            "symbol",
            "qty",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "order_status",
            "net_pnl",
            "order_type",
            "side",
        ]
    ).to_csv(exec_trades_path, index=False)

# Ensure open positions file exists
open_pos_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
if not os.path.exists(open_pos_path):
    pd.DataFrame(
        columns=[
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "entry_price",
            "entry_time",
            "side",
            "order_status",
            "net_pnl",
            "pnl",
            "order_type",
        ]
    ).to_csv(open_pos_path, index=False)

def get_buying_power():
    acc = trading_client.get_account()
    return float(acc.buying_power)

def get_open_positions():
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}



def load_top_candidates() -> pd.DataFrame:
    """Load ranked candidates from ``top_candidates.csv`` and return the
    top entries based on ``MAX_OPEN_TRADES``.
    """

    top_candidates_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
    expected_columns = [
        "symbol",
        "score",
        "win_rate",
        "net_pnl",
        "trades",
        "wins",
        "losses",
        "avg_return",
    ]

    try:
        candidates_df = pd.read_csv(top_candidates_path)
        assert all(
            col in candidates_df.columns for col in expected_columns
        ), "Missing columns in top_candidates.csv"

        candidates_df.sort_values("score", ascending=False, inplace=True)
        selected_candidates = candidates_df.head(MAX_OPEN_TRADES)
        symbols_list = selected_candidates['symbol'].tolist()
        logging.info("Loaded %s successfully", top_candidates_path)
        logging.info("Candidate symbols loaded: %s", symbols_list)
        return selected_candidates
    except Exception as exc:
        logging.error("Failed to read %s: %s", top_candidates_path, exc)
        return pd.DataFrame(columns=expected_columns)

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
                ),
                'side': getattr(p, 'side', 'long'),
                'order_status': 'open',
                'net_pnl': p.unrealized_pl,
                'pnl': p.unrealized_pl,
                'order_type': getattr(p, 'order_type', 'market'),
            })

        columns = [
            'symbol',
            'qty',
            'avg_entry_price',
            'current_price',
            'unrealized_pl',
            'entry_price',
            'entry_time',
            'side',
            'order_status',
            'net_pnl',
            'pnl',
            'order_type',
        ]

        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(columns=columns)

        df['side'] = df.get('side', 'long')
        df['order_status'] = df.get('order_status', 'open')
        df['net_pnl'] = df.get('unrealized_pl', 0.0)
        df['pnl'] = df['net_pnl']
        df['order_type'] = df.get('order_type', 'limit')
        df = df[columns]

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

            records.append(
                {
                    "symbol": order.symbol,
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "order_status": order.status.value,
                    "net_pnl": pnl,
                    "pnl": pnl,
                    "order_type": getattr(order, "order_type", ""),
                    "side": order.side.value,
                }
            )

        df = pd.DataFrame(
            records,
            columns=[
                "symbol",
                "qty",
                "entry_price",
                "exit_price",
                "entry_time",
                "exit_time",
                "order_status",
                "net_pnl",
                "pnl",
                "order_type",
                "side",
            ],
        )
        csv_path = os.path.join(BASE_DIR, 'data', 'trades_log.csv')
        df.to_csv(csv_path, index=False)
        logging.info("Saved trades log to %s", csv_path)
    except Exception as e:
        logging.error("Failed to update trades log: %s", e)

def record_executed_trade(
    symbol, qty, entry_price, order_type, side, order_status="submitted"
):
    """Append executed trade details to CSV using the unified schema."""

    csv_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
    row = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": "",
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "exit_time": "",
        "order_status": order_status,
        "net_pnl": 0.0,
        "order_type": order_type,
        "side": side,
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

def allocate_position(symbol):
    open_positions = get_open_positions()
    if symbol in open_positions:
        reason = "already in open positions"
        logging.debug("Skipping %s: %s", symbol, reason)
        return None, reason
    if len(open_positions) >= MAX_OPEN_TRADES:
        reason = "max open trades reached"
        logging.debug("Skipping %s: %s", symbol, reason)
        return None, reason

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        reason = "no bars available"
        logging.debug("Skipping %s: %s", symbol, reason)
        return None, reason

    last_close = bars['close'].iloc[-1]
    qty = int(alloc_amount / last_close)
    if qty < 1:
        reason = "allocation insufficient"
        logging.debug("Skipping %s: %s", symbol, reason)
        return None, reason

    logging.debug("Allocating %d shares of %s at %s", qty, symbol, last_close)
    return (qty, round(last_close, 2)), None

def submit_trades():
    df = load_top_candidates()
    submitted = 0
    skipped = 0
    for _, row in df.iterrows():
        sym = row.symbol
        alloc, reason = allocate_position(sym)
        if alloc is None:
            skipped += 1
            logging.warning("Trade skipped for %s: %s", sym, reason)
            continue

        qty, entry_price = alloc
        logging.info(
            "Submitting limit buy order for %s, qty=%s, limit=%s | score=%s win_rate=%s net_pnl=%s avg_return=%s",
            sym,
            qty,
            entry_price,
            row.score,
            row.win_rate,
            row.net_pnl,
            row.avg_return,
        )
        try:
            order = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=entry_price,
                extended_hours=True,
            )
            trading_client.submit_order(order)
            logging.info("Order submitted successfully for %s", sym)
            record_executed_trade(sym, qty, entry_price, order_type="limit", side="buy")
            attach_trailing_stops()
            submitted += 1
        except APIError as e:
            logging.error("Order submission error for %s: %s", sym, e)
            if "extended hours order must be DAY limit orders" in str(e):
                try:
                    retry_order = LimitOrderRequest(
                        symbol=sym,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        limit_price=entry_price,
                        extended_hours=False,
                    )
                    trading_client.submit_order(retry_order)
                    logging.info(
                        "Retry successful with regular hours order for %s", sym
                    )
                    record_executed_trade(
                        sym, qty, entry_price, order_type="limit", side="buy"
                    )
                    attach_trailing_stops()
                    submitted += 1
                except APIError as retry_e:
                    logging.error(
                        "Retry order submission failed for %s: %s", sym, retry_e
                    )
                    skipped += 1
            else:
                skipped += 1
        except Exception as e:
            logging.error("Failed to submit buy order for %s: %s", sym, e)
            skipped += 1
    logging.info("Orders submitted: %d, skipped: %d", submitted, skipped)

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
            record_executed_trade(
                symbol,
                int(pos.qty),
                pos.avg_entry_price,
                order_type="trailing_stop",
                side="sell",
            )
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

        # Filter out orders without a fill timestamp
        valid_entries = [o for o in entry_orders if getattr(o, "filled_at", None)]
        if not valid_entries:
            logging.warning(
                f"No filled entry orders found for symbol {symbol}. Skipping exit check."
            )
            continue  # Move to next symbol
        entry_order = max(valid_entries, key=lambda o: o.filled_at)
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        logging.debug("%s entered on %s, held for %s days", symbol, entry_date, days_held)

        if days_held >= MAX_HOLD_DAYS:
            logging.info("Exiting %s after %s days", symbol, days_held)
            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True,
                )
                trading_client.submit_order(order_request)
                logging.info("Order submitted successfully for %s", symbol)
                record_executed_trade(
                    symbol,
                    int(pos.qty),
                    pos.current_price,
                    order_type="market",
                    side="sell",
                )
            except APIError as e:
                logging.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = MarketOrderRequest(
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        trading_client.submit_order(retry_req)
                        logging.info(
                            "Retry successful with regular hours order for %s",
                            symbol,
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            pos.current_price,
                            order_type="market",
                            side="sell",
                        )
                    except APIError as retry_e:
                        logging.error(
                            "Retry order submission failed for %s: %s",
                            symbol,
                            retry_e,
                        )
            except Exception as e:
                logging.error("Order submission error for %s: %s", symbol, e)
        elif should_exit_early(symbol, data_client, os.path.join(BASE_DIR, "data", "history_cache")):
            logging.info("Early exit signal for %s", symbol)
            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True,
                )
                trading_client.submit_order(order_request)
                record_executed_trade(
                    symbol,
                    int(pos.qty),
                    pos.current_price,
                    order_type="market",
                    side="sell",
                )
            except APIError as e:
                logging.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = MarketOrderRequest(
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        trading_client.submit_order(retry_req)
                        logging.info(
                            "Retry successful with regular hours order for %s", symbol
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            pos.current_price,
                            order_type="market",
                            side="sell",
                        )
                    except APIError as retry_e:
                        logging.error(
                            "Retry order submission failed for %s: %s", symbol, retry_e
                        )
            except Exception as e:
                logging.error("Order submission error for %s: %s", symbol, e)

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

