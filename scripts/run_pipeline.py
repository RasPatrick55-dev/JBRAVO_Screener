# run_pipeline.py
# Scheduled to run nightly around 9:00 PM Central Standard Time
# after market close to process the most recent data.
import os
import sys

# Ensure the script runs from the repository root regardless of where it is
# invoked from.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)

# Add project root to sys.path so that sibling packages (like scripts) can be imported
sys.path.insert(0, BASE_DIR)

import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import requests
import pandas as pd

from utils import write_csv_atomic

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

log_path = os.path.join(BASE_DIR, "logs", "pipeline.log")
error_log_path = os.path.join(BASE_DIR, "logs", "error.log")

error_handler = RotatingFileHandler(error_log_path, maxBytes=2_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5), error_handler],
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s",
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
    start_time = datetime.utcnow()
    logging.info("Starting %s at %s", step_name, start_time.isoformat())
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        end_time = datetime.utcnow()
        duration = end_time - start_time
        logging.info("%s stdout:\n%s", step_name, result.stdout)
        logging.info("%s stderr:\n%s", step_name, result.stderr)
        logging.info("Completed %s successfully in %s", step_name, duration)
    except subprocess.CalledProcessError as e:
        end_time = datetime.utcnow()
        duration = end_time - start_time
        logging.error("%s failed with exit %d after %s", step_name, e.returncode, duration)
        logging.error("%s stdout:\n%s", step_name, e.stdout)
        logging.error("%s stderr:\n%s", step_name, e.stderr)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} error: {e}\n")
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        end_time = datetime.utcnow()
        duration = end_time - start_time
        logging.error("Unexpected failure in %s after %s: %s", step_name, duration, e)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} exception: {e}\n")
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise

if __name__ == "__main__":
    logging.info("Pipeline execution started.")

    try:
        run_step("Screener", [sys.executable, "scripts/screener.py"])
    except Exception:
        logging.error("Screener step failed")
        sys.exit(1)

    steps = [
        (
            "Backtest",
            [sys.executable, "scripts/backtest.py"],
        ),
        (
            "Metrics Calculation",
            [sys.executable, "scripts/metrics.py"],
        ),
    ]

    for name, cmd in steps:
        try:
            run_step(name, cmd)
            if name == "Backtest":
                backtest_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")
                if os.path.exists(backtest_path):
                    logging.info("Backtest results written to %s", backtest_path)
                else:
                    logging.error("Expected backtest results at %s not found", backtest_path)
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
