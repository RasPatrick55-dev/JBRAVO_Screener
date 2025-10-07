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


def main():
    root = repo_root()
    os.chdir(root)
    emit("PIPELINE_START", component="pipeline")

    try:
        # ---- screener step ----
        result = subprocess.run(
            [sys.executable, "scripts/screener.py"],
            cwd=root,
        )

        if result.returncode != 0:
            emit(
                "SCREENER_ERROR",
                component="pipeline",
                returncode=str(result.returncode),
            )
            print("Screener step failed (non-zero exit); dashboard will show stale.")
        else:
            emit("SCREENER_SUCCESS", component="pipeline")
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

        # ---- executor import ----
        try:
            from scripts.execute_trades import main as exec_main
            emit("IMPORT_SUCCESS", component="execute_trades")
        except Exception as e:
            emit("IMPORT_FAILURE", component="execute_trades", error=str(e).replace(" ", "_"))
            raise

        # ---- run executor (will log its own events once imports work) ----
        exec_main()

    except Exception as e:
        emit("PIPELINE_ERROR", component="pipeline", error=str(e).replace(" ", "_"))
        raise
    finally:
        emit("PIPELINE_END", component="pipeline")


if __name__ == "__main__":
    main()
