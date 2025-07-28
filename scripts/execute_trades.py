# execute_trades.py
# Dynamically execute limit buys for the highest ranked candidates.
# Trailing stops and max hold logic manage risk on open trades.

import os
import sys

# Explicitly insert the project root at the front of sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
import logging
import pandas as pd
import json
from datetime import datetime, timedelta, timezone, time
import pytz
from time import sleep
from decimal import Decimal, ROUND_HALF_UP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    GetOrdersRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.requests import StockBarsRequest
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
# Alerting
import requests
# Import from the top-level utils package explicitly
from utils.logger_utils import init_logging

# Import scripts/utils explicitly
from scripts.utils import cache_bars

# Explicit import from scripts directory
from scripts.exit_signals import should_exit_early

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
logger = init_logging(__name__, 'execute_trades.log')
start_time = datetime.utcnow()
logger.info("Trade execution script started.")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(message: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca credentials")

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Constants
# Maximum number of concurrent open trades allowed
MAX_OPEN_TRADES = 10
ALLOC_PERCENT = 0.03  # Changed allocation to 3%
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7

# Directory used for caching market history
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "history_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Runtime metrics recorded for dashboard display
metrics = {
    "symbols_processed": 0,
    "orders_submitted": 0,
    "symbols_skipped": 0,
    "api_retries": 0,
    "api_failures": 0,
}
metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")


def round_price(value: float) -> float:
    """Return ``value`` rounded to the nearest cent."""
    return round(value + 1e-6, 2)


def get_latest_price(symbol: str) -> float | None:
    """Return the most recent trade price for ``symbol``."""
    if hasattr(trading_client, "get_stock_latest_trade"):
        try:
            trade = trading_client.get_stock_latest_trade(symbol)
            return float(trade.price)
        except Exception as exc:  # pragma: no cover - API errors
            logger.warning("get_stock_latest_trade failed for %s: %s", symbol, exc)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now(timezone.utc) - timedelta(minutes=5),
            end=datetime.now(timezone.utc),
            feed="sip",
        )
        bars = data_client.get_stock_bars(req).df
        if not bars.empty:
            return float(bars["close"].iloc[-1])
    except Exception as exc:  # pragma: no cover - API errors
        logger.error("Fallback price fetch failed for %s: %s", symbol, exc)
    return None


def is_extended_hours(now_et: time) -> bool:
    """Return True if ``now_et`` falls in extended trading hours."""
    return time(4, 0) <= now_et < time(9, 30) or now_et >= time(16, 0)

# Candidate selection happens dynamically from ``top_candidates.csv``.


# Ensure executed trades file exists
exec_trades_path = os.path.join(BASE_DIR, 'data', 'executed_trades.csv')
if not os.path.exists(exec_trades_path):
    pd.DataFrame(
        columns=[
            "order_id",
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
            "days_in_trade",
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


def get_available_qty(symbol: str) -> int:
    """Return the available quantity for ``symbol`` or 0 on error."""
    try:
        positions = trading_client.get_all_positions()
        for p in positions:
            if p.symbol == symbol:
                return int(getattr(p, "qty_available", p.qty))
    except Exception as exc:
        logger.error("Failed retrieving available qty for %s: %s", symbol, exc)
    return 0



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
        logger.info("Loaded %s successfully", top_candidates_path)
        logger.info("Candidate symbols loaded: %s", symbols_list)
        return selected_candidates
    except Exception as exc:
        logger.error("Failed to read %s: %s", top_candidates_path, exc)
        return pd.DataFrame(columns=expected_columns)

def save_open_positions_csv():
    """Fetch current open positions from Alpaca and save to CSV."""
    try:
        positions = trading_client.get_all_positions()
        csv_path = os.path.join(BASE_DIR, 'data', 'open_positions.csv')
        existing_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

        def get_entry_time(sym: str, default: str) -> str:
            if not existing_df.empty and sym in existing_df.get('symbol', []).values:
                try:
                    return existing_df.loc[existing_df['symbol'] == sym, 'entry_time'].iloc[0]
                except Exception:
                    return default
            return default

        data = []
        for p in positions:
            ts = getattr(p, 'created_at', None)
            entry_iso = ts.isoformat() if ts is not None else 'N/A'
            entry_dt = pd.to_datetime(get_entry_time(p.symbol, entry_iso), utc=True, errors='coerce')
            days_in_trade = (pd.Timestamp.utcnow() - entry_dt).days if not pd.isna(entry_dt) else 0
            data.append({
                'symbol': p.symbol,
                'qty': p.qty,
                'avg_entry_price': p.avg_entry_price,
                'current_price': p.current_price,
                'unrealized_pl': p.unrealized_pl,
                'entry_price': p.avg_entry_price,
                'entry_time': get_entry_time(p.symbol, entry_iso),
                'days_in_trade': days_in_trade,
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
            'days_in_trade',
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
        logger.info("Saved open positions to %s", csv_path)
    except Exception as e:
        logger.error("Failed to save open positions: %s", e)

def update_trades_log():
    """Fetch recent closed orders from Alpaca and save to trades_log.csv."""
    try:
        request = GetOrdersRequest(limit=100)
        orders = trading_client.get_orders(request)
        closed_orders = [
            o
            for o in orders
            if getattr(o, "status", "").lower() in ("filled", "canceled", "rejected")
        ]
        records = []
        for order in closed_orders:
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

            status_val = (
                order.status.value if hasattr(order.status, "value") else order.status
            )
            records.append(
                {
                    "symbol": order.symbol,
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "order_status": status_val,
                    "net_pnl": pnl,
                    "pnl": pnl,
                    "order_type": getattr(order, "order_type", ""),
                    "side": getattr(order.side, "value", order.side),
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
        logger.info("Saved trades log to %s", csv_path)
    except Exception as e:
        logger.error("Failed to update trades log: %s", e)

def record_executed_trade(
    symbol,
    qty,
    entry_price,
    order_type,
    side,
    order_id="",
    order_status="submitted",
):
    """Append executed trade details to CSV using the unified schema."""

    csv_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
    row = {
        "order_id": order_id,
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


def submit_new_trailing_stop(symbol: str, qty: int, trail_percent: float) -> None:
    """Submit a trailing stop sell order for ``symbol``."""
    try:
        position = trading_client.get_open_position(symbol)
        entry_price = getattr(position, "avg_entry_price", 0)
    except Exception:
        entry_price = 0
    try:
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            trail_percent=trail_percent,
            time_in_force=TimeInForce.GTC,
        )
        order = trading_client.submit_order(request)
        record_executed_trade(
            symbol,
            qty,
            entry_price,
            order_type="trailing_stop",
            side="sell",
            order_id=str(order.id),
        )
        logger.info(
            "Placed trailing stop for %s: qty=%s, trail_pct=%s",
            symbol,
            qty,
            trail_percent,
        )
    except Exception as exc:
        logger.error("Failed to submit trailing stop for %s: %s", symbol, exc)


def poll_order_until_complete(order_id: str) -> str:
    """Poll Alpaca for ``order_id`` until it is filled or cancelled."""
    while True:
        try:
            order = trading_client.get_order_by_id(order_id)
            status = order.status.value if hasattr(order.status, "value") else order.status
        except Exception as exc:
            logger.error("Failed fetching status for %s: %s", order_id, exc)
            status = "unknown"
            break
        if status in {"filled", "canceled", "expired"}:
            break
        logger.info("Polling order %s, status: %s", order_id, status)
        sleep(10)
    logger.info("Order %s completed with status %s", order_id, status)
    return status

def allocate_position(symbol):
    open_positions = get_open_positions()
    if symbol in open_positions:
        reason = "already in open positions"
        logger.debug("Skipping %s: %s", symbol, reason)
        return None, reason
    if len(open_positions) >= MAX_OPEN_TRADES:
        reason = "max open trades reached"
        logger.info(
            "Skipped %s due to reaching max open trades (%s).",
            symbol,
            MAX_OPEN_TRADES,
        )
        return None, reason

    buying_power = get_buying_power()
    alloc_amount = buying_power * ALLOC_PERCENT
    start = datetime.now(timezone.utc) - timedelta(hours=1, minutes=16)
    last_close = None
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start,
            end=datetime.now(timezone.utc) - timedelta(minutes=16),
        )
        bars = data_client.get_stock_bars(req).df
        if bars.empty:
            logger.warning(
                "No bars available for %s from %s to %s. Retrying with previous day's close.",
                symbol,
                start,
                (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
            )
            prev_close = get_latest_price(symbol)
            if prev_close is None:
                logger.error("Fallback price fetch failed for %s", symbol)
            else:
                last_close = float(prev_close)
                bars = pd.DataFrame([{"close": prev_close}])
                logger.info("Using previous close price for %s: %s", symbol, prev_close)
        else:
            last_close = float(bars['close'].iloc[-1])
    except Exception as e:
        logger.error("Error fetching bars for %s: %s", symbol, e)
        return None, "market data error"

    if last_close is None:
        return None, "market data error"

    qty = int(alloc_amount / last_close)
    if qty < 1:
        reason = "allocation insufficient"
        logger.debug("Skipping %s: %s", symbol, reason)
        return None, reason

    logger.debug("Allocating %d shares of %s at %s", qty, symbol, last_close)
    return (qty, round(last_close, 2)), None

def submit_trades():
    df = load_top_candidates()
    calendar_today = trading_client.get_calendar()[0]
    if calendar_today.open is None or calendar_today.close is None:
        logger.warning(
            "No market hours available today. Possibly a holiday or weekend. Skipping."
        )
        return

    clock = trading_client.get_clock()
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    if not clock.is_open:
        if time(4, 0) <= now_et < time(9, 30):
            logger.info("In pre-market session – continuing trade execution.")
        else:
            logger.warning(
                "Market is fully closed (not even in pre-market). Skipping trade execution.")
            return
    submitted = 0
    skipped = 0
    existing_positions = trading_client.get_all_positions()
    existing_symbols = [p.symbol for p in existing_positions]

    for _, row in df.iterrows():
        sym = row.symbol
        metrics["symbols_processed"] += 1
        if sym in existing_symbols:
            logger.info("Skipped %s: position already exists.", sym)
            skipped += 1
            metrics["symbols_skipped"] += 1
            continue
        alloc, reason = allocate_position(sym)
        if alloc is None:
            skipped += 1
            metrics["symbols_skipped"] += 1
            if reason == "market data error":
                logger.warning("Skipping %s: No bars available.", sym)
            else:
                logger.warning("Trade skipped for %s: %s", sym, reason)
            continue

        qty, entry_price = alloc
        logger.info(
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
            limit_price = get_latest_price(sym)
            if limit_price is None:
                logger.error("Latest price unavailable for %s", sym)
                limit_price = entry_price
            if limit_price is None:
                logger.warning("[SKIPPED] %s: reason=market data unavailable", sym)
                skipped += 1
                metrics["symbols_skipped"] += 1
                continue
            price = round_price(limit_price)
            price = round(price + 1e-6, 2)
            now_et = datetime.now(pytz.timezone("US/Eastern")).time()
            session = "extended" if is_extended_hours(now_et) else "regular"
            logger.info(
                "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
                sym,
                qty,
                price,
                session,
            )
            order_request = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY,
                type="limit",
                limit_price=price,
                time_in_force=TimeInForce.DAY,
                extended_hours=is_extended_hours(now_et),
            )
            order = trading_client.submit_order(order_request)
            status = poll_order_until_complete(order.id)
            record_executed_trade(
                sym,
                qty,
                price,
                order_type="limit",
                side="buy",
                order_id=str(order.id),
                order_status=status,
            )
            if status == "filled":
                attach_trailing_stops()
                save_open_positions_csv()
                update_trades_log()
            submitted += 1
            metrics["orders_submitted"] += 1
        except APIError as e:
            if "extended hours order must be DAY limit orders" in str(e):
                logger.warning("Retrying %s without extended_hours flag.", sym)
                try:
                    retry_price = round_price(entry_price)
                    retry_price = round(retry_price + 1e-6, 2)
                    retry_order = LimitOrderRequest(
                        symbol=sym,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        limit_price=retry_price,
                        extended_hours=False,
                    )
                    order = trading_client.submit_order(retry_order)
                    status = poll_order_until_complete(order.id)
                    logger.info(
                        "Retry successful with regular hours order for %s", sym
                    )
                    record_executed_trade(
                        sym,
                        qty,
                        retry_price,
                        order_type="limit",
                        side="buy",
                        order_id=str(order.id),
                        order_status=status,
                    )
                    if status == "filled":
                        attach_trailing_stops()
                        save_open_positions_csv()
                        update_trades_log()
                    submitted += 1
                    metrics["orders_submitted"] += 1
                    metrics["api_retries"] += 1
                except APIError as retry_e:
                    logger.error(
                        "Retry order submission failed for %s: %s", sym, retry_e
                    )
                    skipped += 1
                    metrics["api_failures"] += 1
            else:
                logger.error("Order failed for %s: %s", sym, e)
                skipped += 1
                metrics["api_failures"] += 1
        except Exception as e:
            logger.error("Failed to submit buy order for %s: %s", sym, e)
            skipped += 1
            metrics["api_failures"] += 1
    errors = metrics.get("api_failures", 0)
    logger.info(
        "Orders submitted: %d, skipped: %d, errors: %d",
        submitted,
        skipped,
        errors,
    )

def attach_trailing_stops():
    positions = get_open_positions()
    for symbol, pos in positions.items():
        # Fetch current open orders and cancel existing trailing stops
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            existing_orders = trading_client.get_orders(filter=request)
        except Exception as exc:
            logger.error("Failed to fetch open orders for %s: %s", symbol, exc)
            continue

        for order in existing_orders:
            if getattr(order, "order_type", "") == "trailing_stop":
                try:
                    trading_client.cancel_order_by_id(order.id)
                    logger.info(
                        "Cancelled existing trailing-stop order %s for %s",
                        order.id,
                        symbol,
                    )
                except Exception as cancel_exc:  # pragma: no cover - API errors
                    logger.error(
                        "Failed to cancel trailing stop %s for %s: %s",
                        order.id,
                        symbol,
                        cancel_exc,
                    )

        other_orders = [
            o
            for o in existing_orders
            if getattr(o, "order_type", "") != "trailing_stop"
            and o.status.lower() in ("open", "new", "accepted")
        ]
        if other_orders:
            logger.info("Skipping trailing stop for %s: already has open orders.", symbol)
            continue

        # Re-check available quantity after cancellations
        try:
            position = trading_client.get_open_position(symbol)
            available_qty = int(getattr(position, "qty_available", 0))
        except Exception as exc:
            logger.error(
                "Failed to fetch position for %s after cancelling trailing stop: %s",
                symbol,
                exc,
            )
            continue

        if available_qty > 0:
            submit_new_trailing_stop(symbol, available_qty, TRAIL_PERCENT)
        else:
            logger.warning(
                "No available quantity for trailing stop on %s after cancelling.",
                symbol,
            )

def daily_exit_check():
    positions = get_open_positions()
    request = GetOrdersRequest()
    orders = trading_client.get_orders(request)
    closed_orders = [
        o
        for o in orders
        if getattr(o, "status", "").lower() in ("filled", "canceled", "rejected")
    ]
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    extended = is_extended_hours(now_et)

    for symbol, pos in positions.items():
        entry_orders = [
            o
            for o in closed_orders
            if o.symbol == symbol and getattr(o.side, "value", o.side) == "buy"
        ]
        if not entry_orders:
            logger.warning("No entry order found for %s, skipping.", symbol)
            continue

        # Filter out orders without a fill timestamp
        valid_entries = [o for o in entry_orders if getattr(o, "filled_at", None)]
        if not valid_entries:
            logger.warning(
                f"No filled entry orders found for symbol {symbol}. Skipping exit check."
            )
            continue  # Move to next symbol
        entry_order = max(valid_entries, key=lambda o: o.filled_at)
        entry_date = entry_order.filled_at.date()
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        logger.debug("%s entered on %s, held for %s days", symbol, entry_date, days_held)

        if days_held >= MAX_HOLD_DAYS:
            logger.info("Exiting %s after %s days", symbol, days_held)
            try:
                valid_exit_price = round_price(float(pos.current_price))
                valid_exit_price = round(valid_exit_price + 1e-6, 2)
                available_qty = get_available_qty(symbol)
                qty = min(int(pos.qty), available_qty)
                if qty <= 0:
                    logger.info(
                        "Skipped exit for %s: available qty %s insufficient.",
                        symbol,
                        available_qty,
                    )
                    continue
                session = "extended" if extended else "regular"
                logger.info(
                    "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
                    symbol,
                    pos.qty,
                    valid_exit_price,
                    session,
                )
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    type="limit",
                    limit_price=valid_exit_price,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended,
                )
                order = trading_client.submit_order(order_request)
                status = poll_order_until_complete(order.id)
                logger.info("Order submitted successfully for %s", symbol)
                record_executed_trade(
                    symbol,
                    qty,
                    pos.current_price,
                    order_type="market",
                    side="sell",
                    order_id=str(order.id),
                    order_status=status,
                )
                if status == "filled":
                    save_open_positions_csv()
                    update_trades_log()
                metrics["orders_submitted"] += 1
            except APIError as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = LimitOrderRequest(
                            type="limit",
                            limit_price=valid_exit_price,
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        order = trading_client.submit_order(retry_req)
                        status = poll_order_until_complete(order.id)
                        logger.info(
                            "Retry successful with regular hours order for %s",
                            symbol,
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            pos.current_price,
                            order_type="market",
                            side="sell",
                            order_id=str(order.id),
                            order_status=status,
                        )
                        if status == "filled":
                            save_open_positions_csv()
                            update_trades_log()
                        metrics["orders_submitted"] += 1
                        metrics["api_retries"] += 1
                    except APIError as retry_e:
                        logger.error(
                            "Retry order submission failed for %s: %s",
                            symbol,
                            retry_e,
                        )
                        metrics["api_failures"] += 1
            except Exception as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                metrics["api_failures"] += 1
                try:
                    trading_client.close_position(symbol)
                except Exception as exc:  # pragma: no cover - API errors
                    logger.error("Failed to close position %s: %s", symbol, exc)
        elif should_exit_early(symbol, data_client, os.path.join(BASE_DIR, "data", "history_cache")):
            logger.info("Early exit signal for %s", symbol)
            try:
                valid_exit_price = round_price(float(pos.current_price))
                valid_exit_price = round(valid_exit_price + 1e-6, 2)
                available_qty = get_available_qty(symbol)
                qty = min(int(pos.qty), available_qty)
                if qty <= 0:
                    logger.info(
                        "Skipped exit for %s: available qty %s insufficient.",
                        symbol,
                        available_qty,
                    )
                    continue
                now_et = datetime.now(pytz.timezone("US/Eastern")).time()
                extended_now = is_extended_hours(now_et)
                session = "extended" if extended_now else "regular"
                logger.info(
                    "[SUBMIT] Order for %s: qty=%s, price=%s, session=%s",
                    symbol,
                    pos.qty,
                    valid_exit_price,
                    session,
                )
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    type="limit",
                    limit_price=valid_exit_price,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended_now,
                )
                order = trading_client.submit_order(order_request)
                status = poll_order_until_complete(order.id)
                record_executed_trade(
                    symbol,
                    qty,
                    valid_exit_price,
                    order_type="limit",
                    side="sell",
                    order_status=status,
                )
                if status == "filled":
                    save_open_positions_csv()
                    update_trades_log()
                metrics["orders_submitted"] += 1
            except APIError as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                if "extended hours order must be DAY limit orders" in str(e):
                    try:
                        retry_req = LimitOrderRequest(
                            type="limit",
                            limit_price=valid_exit_price,
                            symbol=symbol,
                            qty=pos.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=False,
                        )
                        trading_client.submit_order(retry_req)
                        logger.info(
                            "Retry successful with regular hours order for %s", symbol
                        )
                        record_executed_trade(
                            symbol,
                            int(pos.qty),
                            valid_exit_price,
                            order_type="limit",
                            side="sell",
                        )
                        metrics["orders_submitted"] += 1
                        metrics["api_retries"] += 1
                    except APIError as retry_e:
                        logger.error(
                            "Retry order submission failed for %s: %s", symbol, retry_e
                        )
                        metrics["api_failures"] += 1
            except Exception as e:
                logger.error("Order submission error for %s: %s", symbol, e)
                metrics["api_failures"] += 1
                try:
                    trading_client.close_position(symbol)
                except Exception as exc:  # pragma: no cover - API errors
                    logger.error("Failed to close position %s: %s", symbol, exc)

if __name__ == '__main__':
    logger.info("Starting pre-market trade execution script")
    try:
        submit_trades()
        attach_trailing_stops()
        daily_exit_check()
        save_open_positions_csv()
        update_trades_log()
        # Persist metrics for dashboard
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            logger.info("Execution metrics saved to %s", metrics_path)
        except Exception as exc:
            logger.error("Failed to save execution metrics: %s", exc)

        logger.info(
            "Metrics - processed: %d, submitted: %d, skipped: %d",
            metrics["symbols_processed"],
            metrics["orders_submitted"],
            metrics["symbols_skipped"],
        )

        # Run historical trades script after executing trades
        history_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fetch_trades_history.py')
        try:
            subprocess.run(["python", history_script], check=True)
            logger.info("Historical trades successfully fetched and CSV files updated.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to run historical trade script: %s", e)
    except Exception as exc:
        logger.exception("Critical error occurred: %s", exc)
        send_alert(str(exc))
    finally:
        end_time = datetime.utcnow()
        elapsed_time = end_time - start_time
        logger.info("Script finished in %s", elapsed_time)
        logger.info("Pre-market trade execution script complete")

