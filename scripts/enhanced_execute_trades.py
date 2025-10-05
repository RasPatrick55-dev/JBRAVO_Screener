"""
Enhanced trade execution script with early exit logic.

This script wraps the original trade execution workflow and adds an
additional daily exit check that consults momentum‑based exit signals
defined in ``exit_signals.should_exit_early``.  Positions will be
liquidated not only when the seven‑day holding limit is reached but
also when the price breaks below the 20‑period EMA, the RSI exceeds
70, or the MACD histogram turns negative【514888614236530†L246-L284】.

The remainder of the behaviour – candidate selection, limit order
placement, trailing stop attachment, logging and CSV maintenance –
mirrors the existing ``execute_trades.py`` script in this repository.

Note that for clarity and brevity this version omits some of the
auxiliary error handling and logging found in the production script.  It
should be treated as a reference implementation; you can integrate
``should_exit_early`` into your existing execution script rather than
replacing it outright.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
import time

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import TrailingStopOrderRequest
from typing import Optional

from utils import write_csv_atomic
from exit_signals import should_exit_early

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# Logging setup
log_path = os.path.join(BASE_DIR, 'logs', 'enhanced_execute_trades.log')
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
    raise ValueError("Missing Alpaca credentials")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Constants controlling risk
MAX_OPEN_TRADES = 4
ALLOC_PERCENT = 0.03  # 3% of available buying power per trade
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7


def log_trailing_stop_event(symbol: str, trail_percent: float, order_id: Optional[str], status: str) -> None:
    payload = {
        "symbol": symbol,
        "trail_percent": trail_percent,
        "order_id": order_id,
        "status": status,
    }
    logging.info("TRAILING_STOP_ATTACH %s", payload)


def log_exit_submit(symbol: str, qty: int, order_type: str, reason_code: str, side: str = "sell") -> None:
    payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "reason_code": reason_code,
    }
    logging.info("EXIT_SUBMIT %s", payload)


def log_exit_final(status: str, latency_ms: int) -> None:
    payload = {
        "status": status,
        "latency_ms": latency_ms,
    }
    logging.info("EXIT_FINAL %s", payload)


# Ensure executed trades and open positions CSVs exist
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


def record_executed_trade(
    symbol: str,
    qty: int,
    price: float,
    order_type: str,
    side: str,
    exit_time: Optional[datetime] = None,
) -> None:
    """Append a single executed trade to executed_trades.csv.

    The ``exit_time`` parameter can be left as None for entry orders; it
    will be filled in when the position is closed.
    """
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": price if side == 'buy' else None,
        "exit_price": price if side == 'sell' else None,
        "entry_time": now if side == 'buy' else None,
        "exit_time": now if side == 'sell' else None,
        "order_status": 'filled',
        "net_pnl": 0.0,
        "order_type": order_type,
        "side": side,
    }
    try:
        df = pd.read_csv(exec_trades_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(exec_trades_path, index=False)
    except Exception as exc:
        logging.error("Failed to record executed trade for %s: %s", symbol, exc)


def wait_for_order_terminal(order_id: str, poll_interval: int = 10, timeout_seconds: int = 600) -> str:
    """Poll the Alpaca API until ``order_id`` hits a terminal status."""

    deadline = datetime.utcnow() + timedelta(seconds=timeout_seconds)
    status = "unknown"
    while datetime.utcnow() <= deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            status_obj = getattr(order, "status", "unknown")
            status = status_obj.value if hasattr(status_obj, "value") else str(status_obj)
        except Exception as exc:
            logging.error("Failed to fetch status for %s: %s", order_id, exc)
            status = "error"
            break

        normalized = status.lower()
        if normalized in {"filled", "canceled", "cancelled", "expired", "rejected"}:
            break

        time.sleep(poll_interval)

    return status


def get_buying_power() -> float:
    acc = trading_client.get_account()
    return float(acc.buying_power)


def get_open_positions() -> dict:
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}


def load_top_candidates() -> pd.DataFrame:
    """Load the ranked candidates and return up to ``MAX_OPEN_TRADES`` rows."""
    top_candidates_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
    expected_columns = ["symbol", "score"]
    try:
        candidates_df = pd.read_csv(top_candidates_path)
        assert all(col in candidates_df.columns for col in expected_columns), "Missing columns in top_candidates.csv"
        candidates_df.sort_values("score", ascending=False, inplace=True)
        return candidates_df.head(MAX_OPEN_TRADES)
    except Exception as exc:
        logging.error("Failed to load top candidates: %s", exc)
        return pd.DataFrame(columns=expected_columns)


def submit_trades() -> None:
    """Place limit buy orders for the top ranked symbols."""
    candidates = load_top_candidates()
    if candidates.empty:
        logging.info("No candidates available for trading.")
        return
    buying_power = get_buying_power()
    alloc_dollars = buying_power * ALLOC_PERCENT
    logging.info("Buying power: %s, allocation per trade: %s", buying_power, alloc_dollars)
    for _, row in candidates.iterrows():
        sym = row["symbol"]
        try:
            last_price = trading_client.get_stock_latest_trade(sym).price
        except AttributeError:
            logging.warning(
                "Method get_stock_latest_trade not available for %s", sym
            )
            continue
        qty = max(int(alloc_dollars / last_price), 1)
        entry_price = last_price  # You can adjust limit pricing here
        logging.info("Submitting buy order for %s: qty=%s @ %s", sym, qty, entry_price)
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
            record_executed_trade(sym, qty, entry_price, order_type="limit", side="buy")
        except Exception as exc:
            logging.error("Failed to submit buy order for %s: %s", sym, exc)


def attach_trailing_stops() -> None:
    """Attach a trailing stop order to each open position lacking one."""
    positions = get_open_positions()
    for symbol, pos in positions.items():
        request = GetOrdersRequest(status="open", symbols=[symbol])
        orders = trading_client.get_orders(filter=request)
        if any(o.order_type == 'trailing_stop' for o in orders):
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "skipped")
            continue
        try:
            trail_req = TrailingStopOrderRequest(
                symbol=symbol,
                qty=pos.qty,
                side=OrderSide.SELL,
                trail_percent=TRAIL_PERCENT,
                time_in_force=TimeInForce.GTC
            )
            order = trading_client.submit_order(trail_req)
            record_executed_trade(symbol, int(pos.qty), pos.avg_entry_price, order_type="trailing_stop", side="sell")
            logging.info("Attached trailing stop to %s", symbol)
            log_trailing_stop_event(symbol, TRAIL_PERCENT, str(getattr(order, "id", None)), "submitted")
        except Exception as exc:
            logging.error("Failed to attach trailing stop for %s: %s", symbol, exc)
            log_trailing_stop_event(symbol, TRAIL_PERCENT, None, "error")


def daily_exit_check() -> None:
    """Close positions that hit the max holding period or trigger early exit signals."""
    positions = get_open_positions()
    request = GetOrdersRequest(status="closed")
    orders = trading_client.get_orders(filter=request)
    for symbol, pos in positions.items():
        entry_orders = [o for o in orders if o.symbol == symbol and o.side == 'buy']
        if not entry_orders:
            continue
        valid_entries = [o for o in entry_orders if getattr(o, "filled_at", None)]
        if not valid_entries:
            continue
        entry_order = max(valid_entries, key=lambda o: o.filled_at)
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days
        # Time‑based exit
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
                submit_ts = datetime.utcnow()
                order = trading_client.submit_order(order_request)
                log_exit_submit(symbol, int(pos.qty), "market", "max_hold")
                status = wait_for_order_terminal(str(getattr(order, "id", "")))
                latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                log_exit_final(status, latency_ms)
                record_executed_trade(symbol, int(pos.qty), pos.current_price, order_type="market", side="sell")
            except Exception as exc:
                logging.error("Failed to submit time‑based exit for %s: %s", symbol, exc)
            continue
        # Early exit based on momentum
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # imported here to avoid unused import if not needed
            data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
            cache_dir = os.path.join(BASE_DIR, 'data', 'history_cache')
            if should_exit_early(symbol, data_client, cache_dir):
                logging.info("Early exit signal triggered for %s", symbol)
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True,
                )
                submit_ts = datetime.utcnow()
                order = trading_client.submit_order(order_request)
                log_exit_submit(symbol, int(pos.qty), "market", "monitor")
                status = wait_for_order_terminal(str(getattr(order, "id", "")))
                latency_ms = int((datetime.utcnow() - submit_ts).total_seconds() * 1000)
                log_exit_final(status, latency_ms)
                record_executed_trade(symbol, int(pos.qty), pos.current_price, order_type="market", side="sell")
        except Exception as exc:
            logging.error("Error during early exit check for %s: %s", symbol, exc)


def update_trades_log() -> None:
    """Placeholder for trades log updates (mirrors production script)."""
    # In the original implementation, this function aggregates filled
    # orders from Alpaca and appends them to a persistent CSV.  To keep
    # this reference implementation concise, we omit those details.
    pass


def save_open_positions_csv() -> None:
    """Save current open positions to open_positions.csv."""
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
            'symbol', 'qty', 'avg_entry_price', 'current_price', 'unrealized_pl',
            'entry_price', 'entry_time', 'side', 'order_status', 'net_pnl',
            'pnl', 'order_type'
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
        df.to_csv(open_pos_path, index=False)
        logging.info("Saved open positions to %s", open_pos_path)
    except Exception as exc:
        logging.error("Failed to save open positions: %s", exc)


if __name__ == '__main__':
    logging.info("Starting pre‑market trade execution script")
    submit_trades()
    attach_trailing_stops()
    daily_exit_check()
    save_open_positions_csv()
    update_trades_log()
    logging.info("Pre‑market trade execution script complete")