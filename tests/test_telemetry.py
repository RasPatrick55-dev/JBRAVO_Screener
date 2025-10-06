import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.telemetry as telemetry


@pytest.fixture(autouse=True)
def _ensure_alpaca_env(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test_key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_run_sentinel_writes_start_end(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "events_path", lambda: events_path)
    monkeypatch.setattr(telemetry, "get_version", lambda: "test-version")

    with telemetry.RunSentinel(component="unit-test", force=True, extra={"env": "paper"}) as rs:
        rs.guard(status="OPEN", source="manual")

    events = _read_events(events_path)
    assert [event["event"] for event in events] == [
        "RUN_START",
        "MARKET_GUARD_STATUS",
        "RUN_END",
    ]

    run_start, guard, run_end = events
    assert run_start["component"] == "unit-test"
    assert run_start["force"] is True
    assert run_start["env"] == "paper"

    assert guard["status"] == "OPEN"
    assert guard["component"] == "unit-test"
    assert guard["source"] == "manual"
    assert guard["version"] == "test-version"

    assert run_end["component"] == "unit-test"
    assert run_end["status"] == "ok"
    assert run_end["error"] is None


def test_import_sentinel(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "events_path", lambda: events_path)
    monkeypatch.setattr(telemetry, "get_version", lambda: "test-version")
    monkeypatch.setenv("JBRAVO_IMPORT_SENTINEL", "1")

    module = importlib.import_module("scripts.execute_trades")
    importlib.reload(module)

    events = _read_events(events_path)
    assert any(event["event"] == "IMPORT_SENTINEL" for event in events)

    monkeypatch.delenv("JBRAVO_IMPORT_SENTINEL", raising=False)
