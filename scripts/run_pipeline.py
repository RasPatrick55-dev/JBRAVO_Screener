# run_pipeline.py
import subprocess
import os
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(BASE_DIR, 'logs', 'pipeline_log.txt')

logging.basicConfig(
    handlers=[RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def run_step(step_name, command):
    logging.info(f"Starting {step_name}...")
    try:
        subprocess.run(command, check=True)
        logging.info(f"Completed {step_name} successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR in {step_name}: {e}")
        raise

if __name__ == "__main__":
    logging.info("Pipeline execution started.")
    run_step("Screener", ["python", "scripts/screener.py"])
    run_step("Backtest", ["python", "scripts/backtest.py"])
    run_step("Metrics Calculation", ["python", "scripts/metrics.py"])
    logging.info("Pipeline execution complete.")
