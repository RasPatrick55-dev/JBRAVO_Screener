import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from shutil import copyfile
from typing import Iterable, Optional

from utils.env import load_env

load_env()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    src = Path("data/top_candidates.csv")
    dst = Path("data/latest_candidates.csv")
    logger = logging.getLogger(__name__)
    if src.exists():
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
        logger.warning("Top candidates file %s not found; latest not updated.", src)
        emit("TOP_MISSING", component="pipeline")


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
