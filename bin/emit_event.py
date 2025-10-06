import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    here = Path(__file__).resolve()
    # bin -> repo_root
    return here.parents[1]


def events_path() -> Path:
    path = repo_root() / "data" / "execute_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def emit(event: dict) -> None:
    event.setdefault("ts", utcnow())
    with open(events_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    # Usage: python -m bin.emit_event EVENT_NAME key=val ...
    event_name = sys.argv[1] if len(sys.argv) > 1 else "BOOTSTRAP_EVENT"
    kvs = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kvs[key] = value
    emit({"event": event_name, **kvs})
