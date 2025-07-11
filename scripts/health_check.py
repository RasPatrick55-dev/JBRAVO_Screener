import os
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('health_check')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOG_DIR, 'health_check.log'), maxBytes=2_000_000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)

FILES = [
    'trades_log.csv',
    'executed_trades.csv',
    'metrics_summary.csv',
    'open_positions.csv',
]

THRESHOLD_MINUTES = int(os.getenv('HEALTHCHECK_THRESHOLD', '15'))


def file_age_minutes(path: str) -> float:
    if not os.path.exists(path):
        return float('inf')
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) / 60


def run_check():
    alerts = []
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        age = file_age_minutes(path)
        if age == float('inf'):
            alerts.append(f"{name} missing")
        elif age > THRESHOLD_MINUTES:
            alerts.append(f"{name} stale ({int(age)} min old)")
    if alerts:
        logger.warning('Data health issue: %s', ' | '.join(alerts))
    else:
        logger.info('All CSV files fresh')


if __name__ == '__main__':
    run_check()
