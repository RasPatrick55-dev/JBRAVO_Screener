# run_pipeline.py
# Scheduled to run nightly around 9:00 PM Central Standard Time
# after market close to process the most recent data.
import os
import sys
import traceback

from pathlib import Path

from utils.telemetry import RunSentinel, repo_root, log_event as telemetry_log_event

BASE_DIR = repo_root()
os.chdir(BASE_DIR)

# Add project root to sys.path so that sibling packages (like scripts) can be imported
sys.path.insert(0, str(BASE_DIR))

import subprocess
import logging
from datetime import datetime, timezone
from utils import logger_utils
import json
import requests
import pandas as pd
from utils import write_csv_atomic


logger = logger_utils.init_logging(__name__, "pipeline.log")
error_log_path = os.path.join(BASE_DIR, "logs", "error.log")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


COMPONENT_NAME = "pipeline"


def log_event(event: dict) -> None:
    payload = {"component": COMPONENT_NAME}
    payload.update(event)
    telemetry_log_event(payload)


def send_alert(msg: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": msg}, timeout=5)
    except Exception as exc:
        logger.error("Failed to send alert: %s", exc)

def run_step(step_name, command):
    start_time = datetime.now(timezone.utc)
    logger.info("Starting %s at %s", step_name, start_time.isoformat())
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info("%s stdout:\n%s", step_name, result.stdout)
        logger.info("%s stderr:\n%s", step_name, result.stderr)
        logger.info("Completed %s successfully in %s", step_name, duration)
    except subprocess.CalledProcessError as e:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.error("%s failed with exit %d after %s", step_name, e.returncode, duration)
        logger.error("%s stdout:\n%s", step_name, e.stdout)
        logger.error("%s stderr:\n%s", step_name, e.stderr)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} error: {e}\n")
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.error("Unexpected failure in %s after %s: %s", step_name, duration, e)
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{end_time.isoformat()} {step_name} exception: {e}\n")
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise


def main() -> int:
    start_time = datetime.now(timezone.utc)
    logger.info("Pipeline execution started")
    exit_code = 0
    summarize = True

    screener_processed = "N/A"
    screener_skipped = "N/A"
    backtest_tested = "N/A"
    total_trades = "N/A"
    win_rate = "N/A"
    net_pnl = "N/A"
    exec_metrics = {
        "orders_submitted": "N/A",
        "orders_skipped": "N/A",
        "api_failures": "N/A",
    }

    try:
        with RunSentinel(component=COMPONENT_NAME) as rs:
            log_event({"event": "PIPELINE_START"})
            try:
                try:
                    run_step("Screener", [sys.executable, "scripts/screener.py"])
                except Exception:
                    logger.error("Screener step failed")
                    summarize = False
                    exit_code = 1
                    return exit_code

                steps = [
                    ("Backtest", [sys.executable, "scripts/backtest.py"]),
                    ("Metrics Calculation", [sys.executable, "scripts/metrics.py"]),
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
                        if name == "Metrics Calculation":
                            metrics_summary_file = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
                            if Path(metrics_summary_file).exists():
                                logger.info(
                                    "Metrics summary file exists and is confirmed updated at: %s",
                                    metrics_summary_file,
                                )
                            else:
                                logger.error(
                                    "Metrics summary file missing after metrics calculation step: %s",
                                    metrics_summary_file,
                                )
                    except Exception:
                        logger.error("Step %s failed", name)
                        send_alert(f"Pipeline halted at step {name}")
                        exit_code = 1
                        break

                from scripts.execute_trades import main as exec_main

                exec_result = exec_main()
                if isinstance(exec_result, int) and exec_result != 0:
                    exit_code = exec_result

                source_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
                target_path = os.path.join(BASE_DIR, 'data', 'latest_candidates.csv')
                if os.path.exists(source_path):
                    try:
                        df = pd.read_csv(source_path)
                        write_csv_atomic(target_path, df)
                        logger.info(
                            "Updated latest_candidates.csv at %s",
                            datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception as exc:
                        logger.error("Failed to update latest_candidates.csv: %s", exc)
                        send_alert(f"Failed to update latest candidates: {exc}")
                else:
                    msg = "top_candidates.csv not found; latest_candidates.csv was not updated."
                    logger.error(msg)
                    send_alert(msg)

                scored_path = os.path.join(BASE_DIR, "data", "scored_candidates.csv")
                if os.path.exists(scored_path):
                    try:
                        screener_processed = len(pd.read_csv(scored_path))
                    except Exception:
                        screener_processed = "error"

                backtest_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")
                if os.path.exists(backtest_path):
                    try:
                        backtest_tested = len(pd.read_csv(backtest_path))
                    except Exception:
                        backtest_tested = "error"

                metrics_file = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
                if os.path.exists(metrics_file):
                    try:
                        mdf = pd.read_csv(metrics_file)
                        if not mdf.empty:
                            last = mdf.iloc[-1]
                            total_trades = int(last.get("total_trades", 0))
                            win_rate = round(last.get("win_rate", 0), 2)
                            net_pnl = round(last.get("net_pnl", 0), 2)
                    except Exception:
                        pass

                exec_metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")
                if os.path.exists(exec_metrics_path):
                    try:
                        with open(exec_metrics_path) as f:
                            exec_metrics = json.load(f)
                    except Exception:
                        pass
            except BaseException:
                log_event({"event": "PIPELINE_ERROR", "traceback": traceback.format_exc()})
                raise
            finally:
                end_time = datetime.now(timezone.utc)
                total_duration = end_time - start_time
                if summarize:
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
                log_event({"event": "PIPELINE_END"})
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)
    except Exception:
        if exit_code == 0:
            exit_code = 1
        return exit_code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
