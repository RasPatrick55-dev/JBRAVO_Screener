# run_pipeline.py
import subprocess
import os
import logging
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
        ("Screener", ["python", "scripts/screener.py"]),
        ("Backtest", ["python", "scripts/backtest.py"]),
        ("Metrics Calculation", ["python", "scripts/metrics.py"]),
    ]
    for name, cmd in steps:
        try:
            run_step(name, cmd)
        except Exception:
            logging.error("Step %s failed", name)
    logging.info("Pipeline execution complete.")
