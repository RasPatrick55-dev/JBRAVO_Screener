import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    # resolve: /repo/bin/emit_event.py → /repo
    return Path(__file__).resolve().parents[1]


def events_path() -> Path:
    p = repo_root() / "data" / "execute_events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def emit(ev: dict):
    ev.setdefault("ts", utcnow())
    with open(events_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


if __name__ == "__main__":
    # Usage: python -m bin.emit_event EVENT key=val key2=val2 ...
    name = sys.argv[1] if len(sys.argv) > 1 else "BOOTSTRAP_EVENT"
    kvs = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kvs[k] = v
    emit({"event": name, **kvs})
