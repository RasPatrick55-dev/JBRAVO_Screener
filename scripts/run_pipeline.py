# run_pipeline.py
import subprocess
import os
import logging
import shutil
from datetime import datetime
from logging.handlers import RotatingFileHandler

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
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def run_step(step_name, command):
    logging.info(f"Starting {step_name}...")
    try:
        subprocess.run(command, check=True)
        logging.info(f"Completed {step_name} successfully.")
    except subprocess.CalledProcessError as e:
        logging.error("ERROR in %s: %s", step_name, e)
        raise
    except Exception as e:
        logging.error("Unexpected failure in %s: %s", step_name, e)
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

    # Update latest_candidates.csv with the newest results
    source_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    target_path = os.path.join(BASE_DIR, 'data', 'latest_candidates.csv')
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            logging.info(
                "Updated latest_candidates.csv at %s",
                datetime.utcnow().isoformat(),
            )
        except Exception as exc:
            logging.error("Failed to update latest_candidates.csv: %s", exc)
    else:
        logging.error(
            "top_candidates.csv not found; latest_candidates.csv was not updated."
        )

    logging.info("Pipeline execution complete.")
