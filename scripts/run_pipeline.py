import argparse
import json
import logging
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Iterable, Optional

from utils.env import load_env


LOG = logging.getLogger("pipeline")
LOG_PATH = pathlib.Path("logs") / "pipeline.log"
LATEST_HEADER = "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown\n"


def configure_logging() -> None:
    """Configure console and file logging for the pipeline."""
    if getattr(configure_logging, "_configured", False):  # pragma: no cover - defensive
        return

    fmt = "%(asctime)s - pipeline - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(logging.Formatter(fmt))
    LOG.addHandler(file_handler)
    LOG.propagate = True

    configure_logging._configured = True  # type: ignore[attr-defined]


def run_cmd(cmd: list[str], name: str) -> int:
    """Run ``cmd`` while logging start/end markers and outputs."""
    LOG.info("[INFO] START %s: %s", name, " ".join(shlex.quote(part) for part in cmd))
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    LOG.info("[INFO] END %s (rc=%s, %.1fs)", name, result.returncode, duration)
    if result.stdout:
        LOG.info("[INFO] %s stdout:\n%s", name, result.stdout.strip())
    if result.stderr:
        LOG.info("[INFO] %s stderr:\n%s", name, result.stderr.strip())
    return result.returncode


def screener_cmd() -> list[str]:
    base = [
        sys.executable,
        "-m",
        "scripts.screener",
        "--mode",
        "screener",
        "--feed",
        "iex",
    ]
    extra = shlex.split(os.environ.get("JBR_SCREENER_ARGS", "")) if os.environ.get("JBR_SCREENER_ARGS") else []
    return base + extra


def copy_latest_candidates() -> None:
    src = pathlib.Path("data/top_candidates.csv")
    dst = pathlib.Path("data/latest_candidates.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dst.write_bytes(src.read_bytes())
        try:
            size = dst.stat().st_size
        except FileNotFoundError:  # pragma: no cover - race condition safeguard
            size = 0
        LOG.info("[INFO] refreshed latest_candidates.csv (size=%s)", size)
    else:
        if not dst.exists():
            dst.write_text(LATEST_HEADER)
            LOG.warning(
                "[INFO] top_candidates.csv missing; created header-only latest_candidates.csv"
            )
        else:
            LOG.warning("[INFO] top_candidates.csv missing; retained existing latest_candidates.csv")


def write_metrics_summary() -> None:
    """Write a minimal metrics_summary.csv for the dashboard overview."""
    summary_path = pathlib.Path("data/metrics_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "last_run_utc": "",
        "symbols_in": 0,
        "symbols_with_bars": 0,
        "bars_rows_total": 0,
        "rows": 0,
    }

    metrics_path = pathlib.Path("data/screener_metrics.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if isinstance(metrics, dict):
                for key in meta:
                    meta[key] = metrics.get(key, meta[key])
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOG.warning("[INFO] could not parse screener_metrics.json: %s", exc)

    summary_path.write_text(
        "last_run_utc,symbols_in,with_bars,bars_rows,candidates\n"
        f"{meta['last_run_utc']},{meta['symbols_in']},{meta['symbols_with_bars']},{meta['bars_rows_total']},{meta['rows']}\n"
    )
    LOG.info("[INFO] wrote metrics_summary.csv")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JBRAVO pipeline")
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated list of steps to run (default env PIPE_STEPS or screener,backtest,metrics)",
    )
    parser.add_argument(
        "--reload-web",
        choices=("true", "false"),
        default="true",
        help="Reload the web application when the pipeline finishes (default: true)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def determine_steps(requested: Optional[str]) -> list[str]:
    raw = requested or os.environ.get("PIPE_STEPS", "screener,backtest,metrics")
    steps = [step.strip().lower() for step in raw.split(",") if step.strip()]
    return steps or ["screener", "backtest", "metrics"]


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_env()
    args = parse_args(argv)
    configure_logging()

    steps = determine_steps(args.steps)
    LOG.info("[INFO] PIPELINE_START steps=%s", steps)

    rc = 0
    start = time.time()

    if "screener" in steps:
        rc_scr = run_cmd(screener_cmd(), "SCREENER")
        if rc_scr != 0:
            LOG.error("[INFO] SCREENER failed rc=%s; continuing to write minimal artifacts", rc_scr)
            rc = rc_scr if rc == 0 else rc
        copy_latest_candidates()
        write_metrics_summary()

    if "backtest" in steps:
        rc_bt = run_cmd([sys.executable, "-m", "scripts.backtest"], "BACKTEST")
        if rc_bt != 0 and rc == 0:
            rc = rc_bt

    if "metrics" in steps:
        rc_mx = run_cmd([sys.executable, "-m", "scripts.metrics"], "METRICS")
        if rc_mx != 0 and rc == 0:
            rc = rc_mx

    duration = time.time() - start
    LOG.info("[INFO] PIPELINE_END rc=%s duration=%.1fs", rc, duration)

    if args.reload_web.lower() == "true":
        try:
            domain = os.environ.get("PYTHONANYWHERE_DOMAIN")
            if domain:
                subprocess.run(["pa_reload_webapp", domain], check=False)
                LOG.info("[INFO] AUTO-RELOAD webapp requested for %s", domain)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOG.warning("[INFO] AUTO-RELOAD failed: %s", exc)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
