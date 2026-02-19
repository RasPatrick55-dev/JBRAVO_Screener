import os
import logging
from logging.handlers import RotatingFileHandler
import shutil
import sqlite3
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from dotenv import load_dotenv
from utils.alerts import send_alert
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger("update_dashboard_data")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "update_dashboard_data.log"),
    maxBytes=2_000_000,
    backupCount=5,
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
ACCOUNT_EQUITY_CSV = os.path.join(DATA_DIR, "account_equity.csv")


def load_csv_with_pnl(path: str, fallback_col: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV ensuring a ``pnl`` column is present."""
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if "pnl" not in df.columns:
        if fallback_col and fallback_col in df.columns:
            df["pnl"] = df[fallback_col]
        elif "net_pnl" in df.columns:
            df["pnl"] = df["net_pnl"]
        else:
            df["pnl"] = 0.0
    return df


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


def fetch_account_equity() -> None:
    """Fetch recent account equity history and save to CSV."""
    try:
        client = TradingClient(API_KEY, API_SECRET, paper=True)
        request = GetPortfolioHistoryRequest(period="3M", timeframe="1D")
        history = client.get_portfolio_history(request)
        equity = history.equity
        df_equity = pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.now(), periods=len(equity)),
                "equity": equity,
            }
        )
        write_csv_atomic(ACCOUNT_EQUITY_CSV, df_equity)
        logger.info("Updated account_equity.csv with %s rows", len(df_equity))
    except Exception as exc:
        logger.exception("Failed to fetch account equity: %s", exc)


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
                    entry_time TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    net_pnl REAL,
                    order_status TEXT,
                    order_type TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades_log (
                    symbol TEXT,
                    qty REAL,
                    avg_entry_price REAL,
                    current_price REAL,
                    unrealized_pl REAL,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    net_pnl REAL,
                    order_status TEXT,
                    order_type TEXT,
                    side TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS executed_trades (
                    symbol TEXT,
                    qty REAL,
                    avg_entry_price REAL,
                    current_price REAL,
                    unrealized_pl REAL,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    net_pnl REAL,
                    order_status TEXT,
                    order_type TEXT,
                    side TEXT
            )"""
        )


def write_csv_atomic(path: str, df: pd.DataFrame):
    tmp = NamedTemporaryFile("w", delete=False, dir=DATA_DIR, newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, path)


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
            "side",
            "order_status",
            "net_pnl",
            "pnl",
            "order_type",
        ]

        positions = trading_client.get_all_positions()
        existing_df = (
            pd.read_csv(OPEN_POSITIONS_CSV)
            if os.path.exists(OPEN_POSITIONS_CSV)
            else pd.DataFrame()
        )

        def get_entry_time(sym: str, default: str) -> str:
            if not existing_df.empty and sym in existing_df.get("symbol", []).values:
                try:
                    return existing_df.loc[existing_df["symbol"] == sym, "entry_time"].iloc[0]
                except Exception:
                    return default
            return default

        rows = []
        for p in positions:
            try:
                ts = getattr(p, "created_at", None)
                default_time = ts.isoformat() if ts is not None else datetime.utcnow().isoformat()
                rows.append(
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "unrealized_pl": float(p.unrealized_pl),
                        "entry_price": float(p.avg_entry_price),
                        "entry_time": get_entry_time(p.symbol, default_time),
                        "side": getattr(p, "side", "long"),
                        "order_status": "open",
                        "net_pnl": float(p.unrealized_pl),
                        "pnl": float(p.unrealized_pl),
                        "order_type": getattr(p, "order_type", "limit"),
                    }
                )
            except Exception as exc:
                logger.error("Error processing position %s: %s", getattr(p, "symbol", ""), exc)

        logger.info("Fetched %s open positions", len(rows))

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=columns)
        df["side"] = df.get("side", "long")
        df["order_status"] = df.get("order_status", "open")
        df["net_pnl"] = df.get("unrealized_pl", 0.0)
        df["pnl"] = df["net_pnl"]
        df["order_type"] = df.get("order_type", "limit")
        df = df[columns]
        write_csv_atomic(OPEN_POSITIONS_CSV, df)
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
        req = GetOrdersRequest(status="all", limit=limit, until=end, direction="desc")
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
        logger.info("Fetched %s orders from Alpaca", len(orders))

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
                    "order_id": str(order.id),
                    "symbol": symbol,
                    "qty": qty,
                    "avg_entry_price": entry_price,
                    "current_price": exit_price or entry_price,
                    "unrealized_pl": 0.0,
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "exit_price": exit_price,
                    "exit_time": exit_time,
                    "net_pnl": pnl,
                    "pnl": pnl,
                    "order_status": order.status.value if order.status else "unknown",
                    "order_type": getattr(order, "order_type", ""),
                    "side": side,
                }
            )

        cols = [
            "order_id",
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "entry_price",
            "entry_time",
            "exit_price",
            "exit_time",
            "net_pnl",
            "pnl",
            "order_status",
            "order_type",
            "side",
        ]
        df = pd.DataFrame(records, columns=cols)

        if df.empty:
            df = pd.DataFrame(columns=cols)

        write_csv_atomic(TRADES_LOG_CSV, df)
        executed_df = df[df["qty"] > 0][cols]
        write_csv_atomic(EXECUTED_TRADES_CSV, executed_df)

        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql("trades_log", conn, if_exists="replace", index=False)
            executed_df.to_sql("executed_trades", conn, if_exists="replace", index=False)

        logger.info(
            "Updated trades_log.csv with %s records and executed_trades.csv with %s records.",
            len(df),
            len(executed_df),
        )
    except Exception as e:
        logger.exception("Failed to update order history due to %s", e)


def update_metrics_summary():
    """Compute and persist summary trading metrics."""
    try:
        if not os.path.exists(TRADES_LOG_CSV):
            df = pd.DataFrame()
        else:
            df = load_csv_with_pnl(TRADES_LOG_CSV)

        if "pnl" not in df.columns:
            df["pnl"] = df.get("net_pnl", 0.0)

        if df.empty:
            summary_df = pd.DataFrame(
                [
                    {
                        "total_trades": 0,
                        "net_pnl": 0.0,
                        "win_rate": 0.0,
                        "expectancy": 0.0,
                        "profit_factor": 0.0,
                        "max_drawdown": 0.0,
                    }
                ]
            )
        else:
            total_trades = len(df)
            net_pnl = df["pnl"].sum()
            win_rate = (df["pnl"] > 0).mean() * 100
            expectancy = df["pnl"].mean()
            profits = df[df["pnl"] > 0]["pnl"].sum()
            losses = df[df["pnl"] < 0]["pnl"].sum()
            profit_factor = profits / abs(losses) if losses != 0 else float("inf")
            cumulative = df["pnl"].cumsum()
            max_drawdown = (cumulative - cumulative.cummax()).min() if not cumulative.empty else 0.0

            summary_df = pd.DataFrame(
                [
                    {
                        "total_trades": int(total_trades),
                        "net_pnl": round(net_pnl, 2),
                        "win_rate": round(win_rate, 2),
                        "expectancy": round(expectancy, 2),
                        "profit_factor": round(profit_factor, 2)
                        if profit_factor != float("inf")
                        else profit_factor,
                        "max_drawdown": round(max_drawdown, 2),
                    }
                ]
            )

        write_csv_atomic(os.path.join(DATA_DIR, "metrics_summary.csv"), summary_df)
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
        df = load_csv_with_pnl(OPEN_POSITIONS_CSV, fallback_col="unrealized_pl")
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


def update_order_status_in_csv(order_id: str, status: str) -> None:
    """Update the order status for ``order_id`` in executed_trades.csv."""
    if not os.path.exists(EXECUTED_TRADES_CSV):
        return
    try:
        df = pd.read_csv(EXECUTED_TRADES_CSV)
        if "order_id" not in df.columns:
            return
        mask = df["order_id"] == str(order_id)
        if mask.any():
            df.loc[mask, "order_status"] = status
            write_csv_atomic(EXECUTED_TRADES_CSV, df)
    except Exception as exc:
        logger.error("Failed updating CSV for order %s: %s", order_id, exc)


def update_pending_orders():
    """Poll open orders and refresh their status in CSV."""
    try:
        open_orders = trading_client.get_orders(filter=GetOrdersRequest(status="open"))
        for order in open_orders:
            current = trading_client.get_order_by_id(order.id).status
            logger.info("Order %s for %s currently %s", order.id, order.symbol, current)
            if current in ["filled", "canceled", "expired", "rejected"]:
                update_order_status_in_csv(order.id, current)
    except Exception as exc:
        logger.error("Failed updating open order statuses: %s", exc)


if __name__ == "__main__":
    init_db()
    update_open_positions()
    update_order_history()
    update_pending_orders()
    fetch_account_equity()
    update_metrics_summary()
    validate_open_positions()
    for pth, name in [
        (OPEN_POSITIONS_CSV, "open_positions.csv"),
        (TRADES_LOG_CSV, "trades_log.csv"),
        (EXECUTED_TRADES_CSV, "executed_trades.csv"),
    ]:
        log_if_stale(pth, name)
    logger.info("Dashboard data refresh complete")
