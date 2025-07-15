import os
import logging
from logging.handlers import RotatingFileHandler
import shutil
import sqlite3
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

logger = logging.getLogger("update_dashboard_data")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "data_update.log"), maxBytes=2_000_000, backupCount=5
)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
DB_PATH = os.path.join(DATA_DIR, "dashboard.db")

OPEN_POSITIONS_CSV = os.path.join(DATA_DIR, "open_positions.csv")
TRADES_LOG_CSV = os.path.join(DATA_DIR, "trades_log.csv")
EXECUTED_TRADES_CSV = os.path.join(DATA_DIR, "executed_trades.csv")


def send_alert(message: str):
    """Send alert via webhook if configured."""
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)


def log_if_stale(file_path: str, name: str, threshold_minutes: int = 15):
    """Log a warning if ``file_path`` is older than ``threshold_minutes``."""
    if not os.path.exists(file_path):
        logger.warning("%s missing: %s", name, file_path)
        send_alert(f"{name} missing: {file_path}")
        return
    age = datetime.utcnow() - datetime.utcfromtimestamp(os.path.getmtime(file_path))
    if age > timedelta(minutes=threshold_minutes):
        minutes = age.total_seconds() / 60
        msg = f"{name} is stale ({minutes:.1f} minutes old)"
        logger.warning(msg)
        send_alert(msg)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    avg_entry_price REAL,
                    current_price REAL,
                    unrealized_pl REAL,
                    entry_price REAL,
                    entry_time TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades_log (
                    symbol TEXT,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    order_status TEXT,
                    net_pnl REAL,
                    order_type TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS executed_trades (
                    symbol TEXT,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    order_status TEXT,
                    net_pnl REAL,
                    order_type TEXT
            )"""
        )


def write_csv_atomic(df: pd.DataFrame, dest: str):
    tmp = NamedTemporaryFile("w", delete=False, dir=DATA_DIR, newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, dest)


def update_open_positions():
    try:
        columns = [
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "entry_price",
            "entry_time",
        ]

        positions = trading_client.get_all_positions()
        rows = [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "entry_price": float(p.avg_entry_price),
                "entry_time": getattr(p, "created_at", datetime.utcnow()).isoformat(),
            }
            for p in positions
        ]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            df = pd.DataFrame(columns=columns)
        write_csv_atomic(df, OPEN_POSITIONS_CSV)
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql("open_positions", conn, if_exists="replace", index=False)
        logger.info("Updated open_positions.csv successfully.")
    except Exception as e:
        logger.exception("Failed to update open_positions.csv due to %s", e)


def fetch_all_orders(limit=500):
    """Retrieve the full order history from Alpaca."""
    orders = []
    end = None
    while True:
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=limit, until=end, direction="desc"
        )
        chunk = trading_client.get_orders(filter=req)
        if not chunk:
            break
        orders.extend(chunk)
        if len(chunk) < limit:
            break
        end = chunk[-1].submitted_at.isoformat()
    return orders


def update_order_history():
    """Update executed_trades.csv and trades_log.csv from order history."""
    try:
        orders = [o for o in fetch_all_orders() if o.filled_at is not None]
        orders.sort(key=lambda o: o.filled_at)

        open_positions = {}
        records = []

        for order in orders:
            side = order.side.value
            symbol = order.symbol
            qty = float(order.filled_qty or 0)
            price = float(order.filled_avg_price or 0)

            entry_price = ""
            exit_price = ""
            entry_time = ""
            exit_time = ""
            pnl = 0.0

            if side == "buy":
                entry_price = price
                entry_time = order.filled_at.isoformat()
                open_positions[symbol] = {
                    "price": price,
                    "qty": qty,
                    "time": entry_time,
                }
            elif side == "sell":
                exit_price = price
                exit_time = order.filled_at.isoformat()
                if symbol in open_positions:
                    info = open_positions.pop(symbol)
                    entry_price = info["price"]
                    entry_time = info["time"]
                    pnl = (price - info["price"]) * info["qty"]

            records.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "order_status": order.status.value if order.status else "unknown",
                    "net_pnl": pnl,
                    "order_type": getattr(order, "order_type", ""),
                }
            )

        cols = [
            "symbol",
            "qty",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "order_status",
            "net_pnl",
            "order_type",
        ]
        df = pd.DataFrame(records, columns=cols)

        if df.empty:
            df = pd.DataFrame(columns=cols)

        write_csv_atomic(df, TRADES_LOG_CSV)
        executed_df = df[df["qty"] > 0][cols]
        write_csv_atomic(executed_df, EXECUTED_TRADES_CSV)

        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql("trades_log", conn, if_exists="replace", index=False)
            executed_df.to_sql(
                "executed_trades", conn, if_exists="replace", index=False
            )

        logger.info("Updated trades_log.csv and executed_trades.csv successfully.")
    except Exception as e:
        logger.exception("Failed to update order history due to %s", e)


def update_metrics_summary():
    """Compute and persist summary trading metrics."""
    try:
        if not os.path.exists(TRADES_LOG_CSV):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(TRADES_LOG_CSV)

        if df.empty or "net_pnl" not in df.columns:
            summary_df = pd.DataFrame(
                [
                    {
                        "Total Trades": 0,
                        "Total Wins": 0,
                        "Total Losses": 0,
                        "Win Rate (%)": 0.0,
                        "Total Net PnL": 0.0,
                        "Average Return per Trade": 0.0,
                    }
                ]
            )
        else:
            total_trades = len(df)
            wins = len(df[df["net_pnl"] > 0])
            losses = total_trades - wins
            win_rate = (wins / total_trades) * 100 if total_trades else 0.0
            total_pnl = df["net_pnl"].sum()
            avg_return = df["net_pnl"].mean()
            summary_df = pd.DataFrame(
                [
                    {
                        "Total Trades": total_trades,
                        "Total Wins": wins,
                        "Total Losses": losses,
                        "Win Rate (%)": round(win_rate, 2),
                        "Total Net PnL": round(total_pnl, 2),
                        "Average Return per Trade": round(avg_return, 2),
                    }
                ]
            )

        write_csv_atomic(summary_df, os.path.join(DATA_DIR, "metrics_summary.csv"))
        with sqlite3.connect(DB_PATH) as conn:
            summary_df.to_sql("metrics_summary", conn, if_exists="replace", index=False)
        logger.info("Updated metrics_summary.csv successfully.")
    except Exception as e:
        logger.exception("Failed to update metrics_summary.csv due to %s", e)


def validate_open_positions():
    """Cross-check open_positions.csv against live Alpaca positions."""
    try:
        live = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
        if not os.path.exists(OPEN_POSITIONS_CSV):
            logger.warning("open_positions.csv not found for validation")
            return
        df = pd.read_csv(OPEN_POSITIONS_CSV)
        csv_map = {row["symbol"]: float(row["qty"]) for _, row in df.iterrows()}
        mismatches = []
        for symbol, qty in csv_map.items():
            live_qty = live.get(symbol)
            if live_qty is None:
                mismatches.append(f"{symbol} not in Alpaca")
            elif float(live_qty) != float(qty):
                mismatches.append(f"{symbol} qty csv={qty} live={live_qty}")
        for symbol, qty in live.items():
            if symbol not in csv_map:
                mismatches.append(f"{symbol} missing from CSV")
        if mismatches:
            msg = "Open positions discrepancy detected: " + "; ".join(mismatches)
            logger.warning(msg)
            send_alert(msg)
    except Exception as e:
        logger.exception("Failed to validate open positions: %s", e)


if __name__ == "__main__":
    init_db()
    update_open_positions()
    update_order_history()
    update_metrics_summary()
    validate_open_positions()
    for pth, name in [
        (OPEN_POSITIONS_CSV, "open_positions.csv"),
        (TRADES_LOG_CSV, "trades_log.csv"),
        (EXECUTED_TRADES_CSV, "executed_trades.csv"),
    ]:
        log_if_stale(pth, name)
    logger.info("Dashboard data refresh complete")
