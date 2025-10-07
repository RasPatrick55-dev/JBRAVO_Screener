import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

from utils import write_csv_atomic


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def emit(evt, **kvs):
    cmd = [sys.executable, "-m", "bin.emit_event", evt]
    for k, v in kvs.items():
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, check=False)


def _run_step(name: str, module: str, success_event: str, error_event: str, cwd: Path) -> bool:
    print(f"Starting {name} step...")
    result = subprocess.run([sys.executable, "-m", module], cwd=cwd)
    if result.returncode != 0:
        emit(error_event, component="pipeline", returncode=str(result.returncode))
        print(f"{name} step failed with exit code {result.returncode}.")
        return False

    emit(success_event, component="pipeline")
    print(f"{name} step completed.")
    return True


def main() -> int:
    root = repo_root()
    os.chdir(root)
    emit("PIPELINE_START", component="pipeline")

    exit_code = 0
    try:
        screener_ok = _run_step(
            "Screener",
            "scripts.screener",
            "SCREENER_SUCCESS",
            "SCREENER_ERROR",
            root,
        )

        if screener_ok:
            top_path = root / "data" / "top_candidates.csv"
            latest_path = root / "data" / "latest_candidates.csv"
            if top_path.exists():
                try:
                    df = pd.read_csv(top_path)
                    write_csv_atomic(str(latest_path), df)
                    emit(
                        "LATEST_UPDATED",
                        component="pipeline",
                        rows=str(len(df)),
                    )
                    print(
                        f"Copied {len(df)} rows from top_candidates.csv to latest_candidates.csv."
                    )
                except Exception as copy_exc:
                    emit(
                        "LATEST_COPY_FAILED",
                        component="pipeline",
                        error=str(copy_exc).replace(" ", "_"),
                    )
            else:
                emit("TOP_MISSING", component="pipeline")
                print(
                    "top_candidates.csv not found after screener run; latest_candidates.csv left untouched."
                )
        else:
            exit_code = 1

        backtest_ok = False
        if screener_ok:
            backtest_ok = _run_step(
                "Backtest",
                "scripts.backtest",
                "BACKTEST_SUCCESS",
                "BACKTEST_ERROR",
                root,
            )
            if not backtest_ok:
                exit_code = 1
        else:
            print("Skipping Backtest step because Screener failed.")

        metrics_ok = False
        if backtest_ok:
            metrics_ok = _run_step(
                "Metrics",
                "scripts.metrics",
                "METRICS_SUCCESS",
                "METRICS_ERROR",
                root,
            )
            if not metrics_ok:
                exit_code = 1
        else:
            if screener_ok:
                print("Skipping Metrics step because Backtest failed.")
            else:
                print("Skipping Metrics step because Screener failed.")

    except Exception as e:
        emit("PIPELINE_ERROR", component="pipeline", error=str(e).replace(" ", "_"))
        raise
    finally:
        emit("PIPELINE_END", component="pipeline")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
