"""
Enhanced pipeline runner that calls the new screener implementation.

This script mirrors the behaviour of ``run_pipeline.py`` but replaces the
first step to invoke ``enhanced_screener.py`` instead of the original
``screener.py``.  All other steps remain unchanged.  Use this runner
on PythonAnywhere or locally to execute the updated screening logic
alongside your existing backtests and metrics calculations.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import run_pipeline
from utils import write_csv_atomic
from utils.alerts import send_alert

BASE_DIR = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("pipeline")


def _load_metrics_payload(base_dir: Path) -> dict[str, object]:
    metrics_path = base_dir / "data" / "screener_metrics.json"
    try:
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        LOG.exception("PIPELINE_SUMMARY_READ_FAILED path=%s", metrics_path)
    return {}


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:
        LOG.exception("PIPELINE_ROW_COUNT_FAILED path=%s", path)
        return 0


def _emit_tokens(base_dir: Path, *, steps: str, started: float, rc: int) -> None:
    metrics_payload = _load_metrics_payload(base_dir)
    top_candidates = base_dir / "data" / "top_candidates.csv"
    rows = int(metrics_payload.get("rows", 0) or _count_rows(top_candidates))
    symbols_in = int(metrics_payload.get("symbols_in", 0) or 0)
    symbols_with_bars = int(metrics_payload.get("symbols_with_bars") or 0)
    bars_rows_total = metrics_payload.get("bars_rows_total")
    source = metrics_payload.get("latest_source") or metrics_payload.get("source") or "enhanced"
    LOG.info("[INFO] FALLBACK_CHECK rows_out=%s source=%s", rows, source)
    summary_parts = [
        "[INFO] PIPELINE_SUMMARY",
        f"symbols_in={symbols_in}",
        f"with_bars={symbols_with_bars}",
        f"rows={rows}",
        "fetch_secs=0.000",
        "feature_secs=0.000",
        "rank_secs=0.000",
        "gate_secs=0.000",
    ]
    if bars_rows_total is not None:
        summary_parts.append(f"bars_rows_total={bars_rows_total}")
    summary_parts.append(f"source={source}")
    LOG.info(" ".join(summary_parts))
    duration = datetime.now(timezone.utc).timestamp() - started
    LOG.info("[INFO] PIPELINE_END rc=%s duration=%.1fs", rc, duration)


def run_step(step_name, command):
    start_time = datetime.utcnow()
    LOG.info("Starting %s", step_name)
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        duration = datetime.utcnow() - start_time
        LOG.info("%s stdout:\n%s", step_name, result.stdout)
        LOG.info("%s stderr:\n%s", step_name, result.stderr)
        LOG.info("Completed %s successfully in %s", step_name, duration)
    except subprocess.CalledProcessError as e:
        duration = datetime.utcnow() - start_time
        LOG.error("%s failed with exit %d after %s", step_name, e.returncode, duration)
        LOG.error("%s stdout:\n%s", step_name, e.stdout)
        LOG.error("%s stderr:\n%s", step_name, e.stderr)
        send_alert(f"Pipeline step {step_name} failed: {e}")
        raise
    except Exception as e:
        duration = datetime.utcnow() - start_time
        LOG.exception("Unexpected failure in %s after %s: %s", step_name, duration, e)
        send_alert(f"Pipeline step {step_name} exception: {e}")
        raise


if __name__ == "__main__":
    run_pipeline.configure_logging()
    LOG.info("[INFO] ENV_LOADED files=[]")
    started = datetime.now(timezone.utc).timestamp()
    LOG.info("[INFO] PIPELINE_START steps=%s", "enhanced_screener,backtest,metrics")
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
    rc = 0
    for name, cmd in steps:
        try:
            run_step(name, cmd)
        except Exception:
            rc = 1
            LOG.error("Step %s failed", name)
            send_alert(f"Pipeline halted at step {name}")
            break
    source_path = BASE_DIR / "data" / "top_candidates.csv"
    target_path = BASE_DIR / "data" / "latest_candidates.csv"
    if source_path.exists():
        try:
            df = pd.read_csv(source_path)
            write_csv_atomic(target_path, df)
            LOG.info(
                "Updated latest_candidates.csv at %s",
                datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            )
        except Exception as exc:
            LOG.error("Failed to update latest_candidates.csv: %s", exc)
            send_alert(f"Failed to update latest candidates: {exc}")
    else:
        msg = "top_candidates.csv not found; latest_candidates.csv was not updated."
        LOG.error(msg)
        send_alert(msg)
    _emit_tokens(BASE_DIR, steps="enhanced_screener,backtest,metrics", started=started, rc=rc)
