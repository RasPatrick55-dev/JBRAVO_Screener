import argparse
import json
import logging
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Iterable, Optional

from utils.env import load_env
from utils.io_utils import atomic_write_bytes

load_env()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
LOG_PATH = ROOT / "logs" / "pipeline.log"


def _parse_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


RATE_LIMIT_ALERT_THRESHOLD = _parse_int_env("RATE_LIMIT_ALERT_THRESHOLD", 10)
HTTP_EMPTY_ALERT_THRESHOLD = _parse_int_env("HTTP_EMPTY_ALERT_THRESHOLD", 25)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%Y-%m-%d %H:%M:%S %(message)s")

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


LOGGER = _configure_logger()


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - unexpected I/O or JSON issues
        LOGGER.error("Failed to load JSON from %s: %s", path, exc)
        return {}
    if isinstance(data, dict):
        return data
    LOGGER.error("JSON payload at %s is not an object; ignoring.", path)
    return {}


def _write_json_dict(path: Path, payload: Dict[str, Any]) -> None:
    try:
        encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        atomic_write_bytes(path, encoded)
    except Exception as exc:  # pragma: no cover - unexpected I/O issues
        LOGGER.error("Failed to persist JSON to %s: %s", path, exc)


def check_screener_alerts(base_dir: Path) -> None:
    metrics_path = base_dir / "data" / "screener_metrics.json"
    metrics = _load_json_dict(metrics_path)
    if not metrics:
        LOGGER.warning("Screener metrics unavailable at %s; skipping alerts.", metrics_path)
        return

    alerts: list[str] = []

    bars_rows_total = _coerce_int(metrics.get("bars_rows_total"))
    symbols_with_bars = _coerce_int(metrics.get("symbols_with_bars"))
    if bars_rows_total == 0 or symbols_with_bars == 0:
        alerts.append(
            "Screener returned no bars (symbols_with_bars=%d, bars_rows_total=%d)."
            % (symbols_with_bars, bars_rows_total)
        )

    rate_limited = _coerce_int(metrics.get("rate_limited"))
    if rate_limited > RATE_LIMIT_ALERT_THRESHOLD:
        alerts.append(
            "Rate limit hits elevated (%d > %d)."
            % (rate_limited, RATE_LIMIT_ALERT_THRESHOLD)
        )

    http_empty_batches = _coerce_int(metrics.get("http_empty_batches"))
    if http_empty_batches > HTTP_EMPTY_ALERT_THRESHOLD:
        alerts.append(
            "Empty HTTP batches elevated (%d > %d)."
            % (http_empty_batches, HTTP_EMPTY_ALERT_THRESHOLD)
        )

    state_path = base_dir / "data" / "last_alert.json"
    state = _load_json_dict(state_path)
    rows_zero_streak = _coerce_int(state.get("rows_zero_streak"))
    rows = _coerce_int(metrics.get("rows"))
    if rows == 0:
        rows_zero_streak += 1
        last_run_utc = str(metrics.get("last_run_utc") or _now_utc_iso())
        state["last_zero_rows_utc"] = last_run_utc
        if rows_zero_streak >= 2:
            alerts.append(
                "Screener produced zero candidates for %d consecutive runs."
                % rows_zero_streak
            )
    else:
        rows_zero_streak = 0
        state.pop("last_zero_rows_utc", None)

    state["rows_zero_streak"] = rows_zero_streak
    state["last_rows_value"] = rows
    state["last_seen_metrics_run_utc"] = str(metrics.get("last_run_utc") or "")
    state["last_updated_utc"] = _now_utc_iso()
    _write_json_dict(state_path, state)

    for message in alerts:
        LOGGER.warning("ALERT: %s", message)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JBRAVO pipeline")
    parser.add_argument(
        "--steps",
        default="screener,backtest,metrics",
        help="Comma-separated list of steps to run (default: screener,backtest,metrics)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def emit(evt, **kvs):
    cmd = [sys.executable, "-m", "bin.emit_event", evt]
    for k, v in kvs.items():
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, check=False)


def _run_step(
    name: str,
    module: str,
    success_event: str,
    error_event: str,
    cwd: Path,
    extra_args: Optional[list[str]] = None,
) -> bool:
    print(f"Starting {name} step...")
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        emit(error_event, component="pipeline", returncode=str(result.returncode))
        print(f"{name} step failed with exit code {result.returncode}.")
        return False

    emit(success_event, component="pipeline")
    print(f"{name} step completed.")
    return True


def refresh_latest_candidates() -> None:
    logger = LOGGER
    src, dst = "data/top_candidates.csv", "data/latest_candidates.csv"
    if os.path.exists(src):
        try:
            copyfile(src, dst)
        except Exception as exc:  # pragma: no cover - copy failures are unexpected
            logger.error("Failed to refresh %s from %s: %s", dst, src, exc)
            emit(
                "LATEST_COPY_FAILED",
                component="pipeline",
                error=str(exc).replace(" ", "_"),
            )
        else:
            logger.info("Refreshed %s from %s.", dst, src)
            emit("LATEST_UPDATED", component="pipeline")
    else:
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "a", encoding="utf-8"):
                pass
        except Exception as exc:  # pragma: no cover - directory issues are unexpected
            logger.error("Failed to create placeholder for %s: %s", dst, exc)
        logger.warning("Missing %s; created/kept %s to clear staleness.", src, dst)
        emit("TOP_MISSING", component="pipeline")

    metrics_path = "data/screener_metrics.json"
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh) or {}
        else:
            metrics = {}
        metrics["last_run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh)
        logger.info("Updated screener metrics last_run_utc.")
    except Exception as exc:  # pragma: no cover - metrics update should succeed
        logger.exception("Could not update last_run_utc: %s", exc)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    requested_steps = [
        step.strip().lower()
        for step in str(args.steps or "").split(",")
        if step.strip()
    ]
    if not requested_steps:
        requested_steps = ["screener", "backtest", "metrics"]

    root = repo_root()
    os.chdir(root)
    emit("PIPELINE_START", component="pipeline")

    exit_code = 0
    try:
        last_step_success = True

        for step in requested_steps:
            if step == "screener":
                screener_ok = _run_step(
                    "Screener",
                    "scripts.screener",
                    "SCREENER_SUCCESS",
                    "SCREENER_ERROR",
                    root,
                    extra_args=[
                        "--universe",
                        "alpaca-active",
                        "--days",
                        "750",
                        "--feed",
                        "iex",
                    ],
                )
                last_step_success = screener_ok
                if screener_ok:
                    refresh_latest_candidates()
                    check_screener_alerts(root)
                else:
                    exit_code = 1
            elif step == "backtest":
                if last_step_success:
                    backtest_ok = _run_step(
                        "Backtest",
                        "scripts.backtest",
                        "BACKTEST_SUCCESS",
                        "BACKTEST_ERROR",
                        root,
                    )
                    last_step_success = backtest_ok
                    if not backtest_ok:
                        exit_code = 1
                else:
                    print("Skipping Backtest step because a previous step failed.")
                    exit_code = 1
                    last_step_success = False
            elif step == "metrics":
                if last_step_success:
                    metrics_ok = _run_step(
                        "Metrics",
                        "scripts.metrics",
                        "METRICS_SUCCESS",
                        "METRICS_ERROR",
                        root,
                    )
                    last_step_success = metrics_ok
                    if not metrics_ok:
                        exit_code = 1
                else:
                    print("Skipping Metrics step because a previous step failed.")
                    exit_code = 1
                    last_step_success = False
            else:
                print(f"Unknown step '{step}' requested; skipping.")
    except Exception as e:
        emit("PIPELINE_ERROR", component="pipeline", error=str(e).replace(" ", "_"))
        raise
    finally:
        emit("PIPELINE_END", component="pipeline")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
