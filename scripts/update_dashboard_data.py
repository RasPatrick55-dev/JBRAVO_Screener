import os
import logging
from logging.handlers import RotatingFileHandler
import shutil
import sqlite3
from datetime import datetime
from tempfile import NamedTemporaryFile

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, ActivityType
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, '.env'))

logger = logging.getLogger('update_dashboard_data')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'data_update.log'), maxBytes=2_000_000, backupCount=5
)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
DB_PATH = os.path.join(DATA_DIR, 'dashboard.db')


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    avg_entry_price REAL,
                    current_price REAL,
                    unrealized_pl REAL,
                    entry_time TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades_log (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    transaction_time TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS executed_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    filled_qty REAL,
                    status TEXT,
                    submitted_at TEXT,
                    filled_at TEXT
            )"""
        )


def write_csv_atomic(df: pd.DataFrame, dest: str):
    tmp = NamedTemporaryFile('w', delete=False, dir=DATA_DIR, newline='')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, dest)


def update_open_positions():
    try:
        positions = trading_client.get_all_positions()
        rows = [
            {
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'unrealized_pl': float(p.unrealized_pl),
                'entry_time': getattr(p, 'created_at', datetime.utcnow()).isoformat(),
            }
            for p in positions
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['symbol', 'qty', 'avg_entry_price', 'current_price', 'unrealized_pl', 'entry_time'])
        path = os.path.join(DATA_DIR, 'open_positions.csv')
        write_csv_atomic(df, path)
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql('open_positions', conn, if_exists='replace', index=False)
        logger.info('Updated open_positions.csv successfully.')
    except Exception as e:
        logger.exception('Failed to update open_positions.csv due to %s', e)


def update_trades_log():
    try:
        activities = trading_client.get_activities(activity_types=[ActivityType.FILL])
        records = [
            {
                'id': a.id,
                'symbol': a.symbol,
                'side': a.side,
                'qty': float(getattr(a, 'qty', 0)),
                'price': float(getattr(a, 'price', 0)),
                'transaction_time': a.transaction_time.isoformat(),
            }
            for a in activities
        ]
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['id', 'symbol', 'side', 'qty', 'price', 'transaction_time'])
        path = os.path.join(DATA_DIR, 'trades_log.csv')
        write_csv_atomic(df, path)
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql('trades_log', conn, if_exists='replace', index=False)
        logger.info('Updated trades_log.csv successfully.')
    except Exception as e:
        logger.exception('Failed to update trades_log.csv due to %s', e)


def update_executed_trades():
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)
        orders = trading_client.get_orders(filter=request)
        rows = [
            {
                'id': o.id,
                'symbol': o.symbol,
                'side': o.side.value,
                'filled_qty': float(o.filled_qty or 0),
                'status': o.status.value,
                'submitted_at': o.submitted_at.isoformat() if o.submitted_at else '',
                'filled_at': o.filled_at.isoformat() if o.filled_at else '',
            }
            for o in orders
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['id', 'symbol', 'side', 'filled_qty', 'status', 'submitted_at', 'filled_at'])
        path = os.path.join(DATA_DIR, 'executed_trades.csv')
        write_csv_atomic(df, path)
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql('executed_trades', conn, if_exists='replace', index=False)
        logger.info('Updated executed_trades.csv successfully.')
    except Exception as e:
        logger.exception('Failed to update executed_trades.csv due to %s', e)


if __name__ == '__main__':
    init_db()
    update_open_positions()
    update_trades_log()
    update_executed_trades()
    logger.info('Dashboard data refresh complete')

