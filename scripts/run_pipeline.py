# run_pipeline.py
import subprocess
import os
import logging
import shutil
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import requests
import pandas as pd

from .utils import write_csv_atomic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'pipeline.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

error_handler = RotatingFileHandler(error_log_path, maxBytes=2_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[
        RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5),
        error_handler,
    ],
    level=logging.INFO,
    format='%(asctime)s UTC [%(levelname)s] %(message)s'
)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(msg: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": msg}, timeout=5)
    except Exception as exc:
        logging.error("Failed to send alert: %s", exc)

def run_step(step_name, command):
    logging.info(f"Starting {step_name}...")
    try:
        subprocess.run(command, check=True)
        logging.info(f"Completed {step_name} successfully.")
    except subprocess.CalledProcessError as e:
        logging.error("ERROR in %s: %s", step_name, e)
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        logging.error("Unexpected failure in %s: %s", step_name, e)
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise

if __name__ == "__main__":
    logging.info("Pipeline execution started.")
    steps = [
        (
            "Screener",
            ["python", "/home/RasPatrick/jbravo_screener/scripts/screener.py"],
        ),
        (
            "Backtest",
            ["python", "/home/RasPatrick/jbravo_screener/scripts/backtest.py"],
        ),
        (
            "Metrics Calculation",
            ["python", "/home/RasPatrick/jbravo_screener/scripts/metrics.py"],
        ),
    ]
    for name, cmd in steps:
        try:
            run_step(name, cmd)
        except Exception:
            logging.error("Step %s failed", name)
            send_alert(f"Pipeline halted at step {name}")
            break

    # Update latest_candidates.csv with the newest results
    source_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    target_path = os.path.join(BASE_DIR, 'data', 'latest_candidates.csv')
    if os.path.exists(source_path):
        try:
            df = pd.read_csv(source_path)
            write_csv_atomic(df, target_path)
            logging.info(
                "Updated latest_candidates.csv at %s",
                datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            )
        except Exception as exc:
            logging.error("Failed to update latest_candidates.csv: %s", exc)
            send_alert(f"Failed to update latest candidates: {exc}")
    else:
        msg = "top_candidates.csv not found; latest_candidates.csv was not updated."
        logging.error(msg)
        send_alert(msg)

    logging.info("Pipeline execution complete.")
