import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def emit(evt, **kvs):
    cmd = [sys.executable, "-m", "bin.emit_event", evt]
    for k, v in kvs.items():
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, check=False)


COMPONENT_NAME = "pipeline"


def log_event(event: dict) -> None:
    from utils.telemetry import log_event as telemetry_log_event

    payload = {"component": COMPONENT_NAME}
    payload.update(event)
    telemetry_log_event(payload)


def main():
    os.chdir(repo_root())
    emit("PIPELINE_START", component="pipeline")

    try:
        # ---- screener step ----
        # (call your screener cli / function here)
        # if you currently import a module that fails, wrap in try/except:
        try:
            # example:
            # from scripts.screener import run as run_screener
            # run_screener()
            pass
        except Exception as e:
            emit("SCREENER_ERROR", component="pipeline", error=str(e).replace(" ", "_"))
            # do NOT exit yet; proceed to executor after logging error

        # ---- executor step ----
        try:
            from scripts.execute_trades import main as exec_main
            emit("IMPORT_SUCCESS", component="execute_trades")
        except Exception as e:
            emit("IMPORT_FAILURE", component="execute_trades", error=str(e).replace(" ", "_"))
            raise

        exec_main()

    except Exception as e:
        emit("PIPELINE_ERROR", component="pipeline", error=str(e).replace(" ", "_"))
        raise
    finally:
        emit("PIPELINE_END", component="pipeline")


if __name__ == "__main__":
    main()
