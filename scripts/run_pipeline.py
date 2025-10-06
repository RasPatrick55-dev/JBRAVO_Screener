import os, sys, subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def emit(evt, **kvs):
    cmd = [sys.executable, "-m", "bin.emit_event", evt]
    for k, v in kvs.items():
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, check=False)


def main():
    os.chdir(repo_root())
    emit("PIPELINE_START", component="pipeline")

    try:
        # ---- screener step (wrap to log error instead of hard crash) ----
        try:
            # from scripts.screener import run as run_screener
            # run_screener()
            pass  # leave for now; we'll fix screener in step 4
        except Exception as e:
            emit("SCREENER_ERROR", component="pipeline", error=str(e).replace(" ", "_"))

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
