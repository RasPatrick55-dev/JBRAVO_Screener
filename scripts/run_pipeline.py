import argparse
import json
import logging
import csv
import os
import pathlib
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from shutil import copyfile
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from utils.env import load_env


LOG = logging.getLogger("pipeline")
LOG_PATH = pathlib.Path("logs") / "pipeline.log"
EVENTS_PATH = pathlib.Path("logs") / "execute_events.jsonl"
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


def execute_cmd() -> list[str]:
    base = [sys.executable, "-m", "scripts.execute_trades"]
    extra = shlex.split(os.environ.get("JBR_EXEC_ARGS", "")) if os.environ.get("JBR_EXEC_ARGS") else []
    return base + extra


def emit(event: str, **payload: Any) -> None:
    """Append a structured event to ``logs/execute_events.jsonl``."""

    record: dict[str, Any] = {"event": str(event), **payload}
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())

    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _merge_metrics_defaults(base: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    defaults: Mapping[str, Any] = {
        "last_run_utc": "",
        "rows": 0,
        "symbols_in": 0,
        "symbols_with_bars": 0,
        "bars_rows_total": 0,
        "http": {"429": 0, "404": 0, "empty_pages": 0},
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "universe_prefix_counts": {},
        "timings": {},
    }

    for key, value in defaults.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if isinstance(node, MutableMapping):
                for nested_key, nested_value in value.items():
                    node.setdefault(nested_key, nested_value)
            else:  # pragma: no cover - defensive fallback
                base[key] = dict(value)
        else:
            base.setdefault(key, value)
    return base


def refresh_latest_candidates() -> None:
    """Refresh ``latest_candidates.csv`` and ensure metrics scaffolding exists."""

    src = os.path.join("data", "top_candidates.csv")
    dst = os.path.join("data", "latest_candidates.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(src):
        copyfile(src, dst)
        try:
            size = os.path.getsize(dst)
        except OSError:  # pragma: no cover - defensive guard
            size = 0
        LOG.info("[INFO] refreshed latest_candidates.csv (size=%s)", size)
    else:
        pathlib.Path(dst).write_text(LATEST_HEADER, encoding="utf-8")
        LOG.info("[INFO] top_candidates.csv missing; wrote header-only latest_candidates.csv")

    metrics_path = pathlib.Path("data/screener_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing: MutableMapping[str, Any] = {}
    if metrics_path.exists():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8")) or {}
            if isinstance(parsed, dict):
                existing.update(parsed)
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOG.warning("[INFO] could not parse screener_metrics.json: %s", exc)
    merged = _merge_metrics_defaults(existing)
    merged["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    metrics_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    write_metrics_summary()


def copy_latest_candidates() -> None:  # pragma: no cover - backward compatibility alias
    refresh_latest_candidates()


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
        f"{meta['last_run_utc']}"
        f",{int(meta['symbols_in'] or 0)}"
        f",{int(meta['symbols_with_bars'] or 0)}"
        f",{int(meta['bars_rows_total'] or 0)}"
        f",{int(meta['rows'] or 0)}\n"
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
    raw = requested or os.environ.get("PIPE_STEPS", "screener,execute,backtest,metrics")
    steps = [step.strip().lower() for step in raw.split(",") if step.strip()]
    return steps or ["screener", "backtest", "metrics"]


def latest_candidates_has_rows(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if any(field.strip() for field in row):
                    return True
    except Exception as exc:  # pragma: no cover - defensive parsing
        LOG.warning("[INFO] failed to inspect %s: %s", path, exc)
    return False


def run_execute_step(cmd: list[str]) -> int:
    latest_path = pathlib.Path("data") / "latest_candidates.csv"
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    start = time.time()
    LOG.info("[INFO] START EXECUTE %s", cmd_str)
    if not latest_candidates_has_rows(latest_path):
        duration = time.time() - start
        LOG.info("[INFO] EXECUTE SKIPPED: NO CANDIDATES")
        LOG.info("[INFO] END EXECUTE rc=0 duration=%.1fs", duration)
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    LOG.info("[INFO] END EXECUTE rc=%s duration=%.1fs", result.returncode, duration)
    if result.stdout:
        LOG.info("[INFO] EXECUTE stdout:\n%s", result.stdout.strip())
    if result.stderr:
        LOG.info("[INFO] EXECUTE stderr:\n%s", result.stderr.strip())
    return result.returncode


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_env()
    args = parse_args(argv)
    configure_logging()

    steps = determine_steps(args.steps)
    LOG.info("[INFO] PIPELINE_START steps=%s", steps)

    rc = 0
    start = time.time()
    artifacts_written = False
    latest_refreshed = False

    if "screener" in steps:
        rc_scr = run_cmd(screener_cmd(), "SCREENER")
        if rc_scr != 0:
            LOG.error("[INFO] SCREENER failed rc=%s; continuing to write minimal artifacts", rc_scr)
            rc = rc_scr if rc == 0 else rc
        try:
            refresh_latest_candidates()
            artifacts_written = True
            latest_refreshed = True
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOG.error("[INFO] failed to refresh screener artifacts after SCREENER step: %s", exc)

    if "execute" in steps:
        if not latest_refreshed:
            try:
                refresh_latest_candidates()
                latest_refreshed = True
            except Exception as exc:  # pragma: no cover - defensive safeguard
                LOG.error("[INFO] failed to refresh artifacts before EXECUTE step: %s", exc)
        rc_exec = run_execute_step(execute_cmd())
        if rc_exec != 0 and rc == 0:
            rc = rc_exec

    if "backtest" in steps:
        rc_bt = run_cmd([sys.executable, "-m", "scripts.backtest"], "BACKTEST")
        if rc_bt != 0 and rc == 0:
            rc = rc_bt

    if "metrics" in steps:
        rc_mx = run_cmd([sys.executable, "-m", "scripts.metrics"], "METRICS")
        if rc_mx != 0 and rc == 0:
            rc = rc_mx

    if not artifacts_written:
        try:
            refresh_latest_candidates()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOG.error("[INFO] final artifact refresh failed: %s", exc)

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
