import os
import time
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'data_freshness.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

critical_files = [
    (os.path.join(BASE_DIR, 'data', 'top_candidates.csv'), 90),
    (os.path.join(BASE_DIR, 'data', 'open_positions.csv'), 90),
    (os.path.join(BASE_DIR, 'data', 'trades_log.csv'), 90),
]

current_time = time.time()

for file_path, max_age in critical_files:
    if not os.path.exists(file_path) or (
        current_time - os.path.getmtime(file_path)
    ) / 60 > max_age:
        logging.warning(f"{file_path} is stale or missing.")
    else:
        logging.info(f"{file_path} is current.")

