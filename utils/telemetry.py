import os
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    """Resolve the repository root regardless of the current working directory."""

    here = Path(__file__).resolve()
    return here.parents[1]


def events_path() -> Path:
    path = repo_root() / "data" / "execute_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_version() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return sha
    except Exception:
        return os.environ.get("JBRAVO_VERSION", "unknown")


def log_event(ev: dict):
    ev.setdefault("ts", utcnow())
    ev.setdefault("component", "unknown")
    path = events_path()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")
    if os.environ.get("JBRAVO_DEBUG_EVENT_PATH") == "1":
        print(f"[telemetry] wrote event to: {path}", file=sys.stderr)


class RunSentinel:
    """Context manager that guarantees RUN_START/RUN_END emission."""

    def __init__(self, component: str, force: bool = False, extra: dict | None = None):
        self.component = component
        self.force = force
        self.extra = extra or {}
        self.version = get_version()

    def __enter__(self):
        payload = {
            "event": "RUN_START",
            "component": self.component,
            "version": self.version,
            "force": self.force,
            **self.extra,
        }
        log_event(payload)
        return self

    def guard(self, status: str, **kvs):
        log_event(
            {
                "event": "MARKET_GUARD_STATUS",
                "component": self.component,
                "status": status,
                "version": self.version,
                **kvs,
            }
        )

    def __exit__(self, _exc_type, exc, _tb):
        log_event(
            {
                "event": "RUN_END",
                "component": self.component,
                "version": self.version,
                "status": "error" if exc else "ok",
                "error": None if not exc else str(exc),
            }
        )
        return False


emit_event = log_event
