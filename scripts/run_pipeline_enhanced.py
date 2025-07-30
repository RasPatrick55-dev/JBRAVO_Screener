"""
Enhanced pipeline runner that calls the new screener implementation.

This script mirrors the behaviour of ``run_pipeline.py`` but replaces the
first step to invoke ``enhanced_screener.py`` instead of the original
``screener.py``.  All other steps remain unchanged.  Use this runner
on PythonAnywhere or locally to execute the updated screening logic
alongside your existing backtests and metrics calculations.
"""

import os
import sys
import subprocess
import logging
import shutil
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import requests
import pandas as pd

from utils import write_csv_atomic, logger_utils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'pipeline_enhanced.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

logger = logger_utils.init_logging(__name__, 'pipeline_enhanced.log')
error_handler = RotatingFileHandler(error_log_path, maxBytes=2_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(msg: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": msg}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)


def run_step(step_name, command):
    start_time = datetime.utcnow()
    logger.info("Starting %s", step_name)
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        duration = datetime.utcnow() - start_time
        logger.info("%s stdout:\n%s", step_name, result.stdout)
        logger.info("%s stderr:\n%s", step_name, result.stderr)
        logger.info("Completed %s successfully in %s", step_name, duration)
    except subprocess.CalledProcessError as e:
        duration = datetime.utcnow() - start_time
        logger.error("%s failed with exit %d after %s", step_name, e.returncode, duration)
        logger.error("%s stdout:\n%s", step_name, e.stdout)
        logger.error("%s stderr:\n%s", step_name, e.stderr)
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        duration = datetime.utcnow() - start_time
        logger.exception("Unexpected failure in %s after %s: %s", step_name, duration, e)
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise


if __name__ == "__main__":
    logger.info("Enhanced pipeline execution started.")
    steps = [
        (
            "Enhanced Screener",
            ["python", "/home/RasPatrick/jbravo_screener/scripts/enhanced_screener.py"],
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
            logger.error("Step %s failed", name)
            send_alert(f"Pipeline halted at step {name}")
            break
    # Copy latest results into latest_candidates.csv
    source_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    target_path = os.path.join(BASE_DIR, 'data', 'latest_candidates.csv')
    if os.path.exists(source_path):
        try:
            df = pd.read_csv(source_path)
            write_csv_atomic(target_path, df)
            logger.info(
                "Updated latest_candidates.csv at %s",
                datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.error("Failed to update latest_candidates.csv: %s", exc)
            send_alert(f"Failed to update latest candidates: {exc}")
    else:
        msg = "top_candidates.csv not found; latest_candidates.csv was not updated."
        logger.error(msg)
        send_alert(msg)
    logger.info("Enhanced pipeline execution complete.")
