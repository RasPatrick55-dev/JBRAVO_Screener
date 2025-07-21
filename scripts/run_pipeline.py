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
from datetime import datetime, timezone
from utils import logger_utils
import json
import requests
import pandas as pd

from utils import write_csv_atomic


logger = logger_utils.init_logging(__name__, "pipeline.log")
start_time = datetime.utcnow()
logger.info("Pipeline execution started")
error_log_path = os.path.join(BASE_DIR, "logs", "error.log")

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
    logger.info("Starting %s at %s", step_name, start_time.isoformat())
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
        logger.info("%s stdout:\n%s", step_name, result.stdout)
        logger.info("%s stderr:\n%s", step_name, result.stderr)
        logger.info("Completed %s successfully in %s", step_name, duration)
    except subprocess.CalledProcessError as e:
        end_time = datetime.utcnow()
        duration = end_time - start_time
        logger.error("%s failed with exit %d after %s", step_name, e.returncode, duration)
        logger.error("%s stdout:\n%s", step_name, e.stdout)
        logger.error("%s stderr:\n%s", step_name, e.stderr)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} error: {e}\n")
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        end_time = datetime.utcnow()
        duration = end_time - start_time
        logger.error("Unexpected failure in %s after %s: %s", step_name, duration, e)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} exception: {e}\n")
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise

if __name__ == "__main__":

    try:
        run_step("Screener", [sys.executable, "scripts/screener.py"])
    except Exception:
        logger.error("Screener step failed")
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
                    logger.info("Backtest results written to %s", backtest_path)
                else:
                    logger.error("Expected backtest results at %s not found", backtest_path)
        except Exception:
            logger.error("Step %s failed", name)
            send_alert(f"Pipeline halted at step {name}")
            break

    # Update latest_candidates.csv with the newest results
    source_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    target_path = os.path.join(BASE_DIR, 'data', 'latest_candidates.csv')
    if os.path.exists(source_path):
        try:
            df = pd.read_csv(source_path)
            write_csv_atomic(df, target_path)
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

    # Summaries from generated artifacts
    screener_processed = "N/A"
    screener_skipped = "N/A"
    scored_path = os.path.join(BASE_DIR, "data", "scored_candidates.csv")
    if os.path.exists(scored_path):
        try:
            screener_processed = len(pd.read_csv(scored_path))
        except Exception:
            screener_processed = "error"

    backtest_tested = "N/A"
    backtest_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")
    if os.path.exists(backtest_path):
        try:
            backtest_tested = len(pd.read_csv(backtest_path))
        except Exception:
            backtest_tested = "error"

    metrics_file = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
    total_trades = win_rate = net_pnl = "N/A"
    if os.path.exists(metrics_file):
        try:
            mdf = pd.read_csv(metrics_file)
            if not mdf.empty:
                last = mdf.iloc[-1]
                total_trades = int(last.get("Total Trades", 0))
                win_rate = round(last.get("Win Rate (%)", 0), 2)
                net_pnl = round(last.get("Total Net PnL", 0), 2)
        except Exception:
            pass

    exec_metrics = {
        "orders_submitted": "N/A",
        "orders_skipped": "N/A",
        "api_failures": "N/A",
    }
    exec_metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")
    if os.path.exists(exec_metrics_path):
        try:
            with open(exec_metrics_path) as f:
                exec_metrics = json.load(f)
        except Exception:
            pass

    end_time = datetime.utcnow()
    total_duration = end_time - start_time

    logger.info("Pipeline Summary:")
    logger.info(
        "Screener: %s processed, %s skipped",
        screener_processed,
        screener_skipped,
    )
    logger.info(
        "Backtest: %s tested, %s skipped",
        backtest_tested,
        "N/A",
    )
    logger.info(
        "Metrics: %s trades, win rate %s%%, net PnL $%s",
        total_trades,
        win_rate,
        net_pnl,
    )
    logger.info(
        "Execution: %s orders submitted, %s skipped, %s API errors",
        exec_metrics.get("orders_submitted", "N/A"),
        exec_metrics.get("symbols_skipped", "N/A"),
        exec_metrics.get("api_failures", "N/A"),
    )
    logger.info("Total Pipeline Duration: %s", total_duration)
    logger.info("Pipeline execution complete.")
