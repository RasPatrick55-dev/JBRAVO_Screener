import os
import sys
import json
from pathlib import Path
from datetime import datetime
from importlib import import_module

import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

dummy_indicators = types.ModuleType("indicators")
dummy_indicators.rsi = lambda series: series
dummy_indicators.macd = lambda series: (series, series, series)
sys.modules.setdefault("indicators", dummy_indicators)

os.environ.setdefault("APCA_API_KEY_ID", "test_key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test_secret")


@pytest.fixture(scope="module")
def execute_trades_module():
    return import_module("scripts.execute_trades")


def test_log_event_appends_valid_json(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path = tmp_path / "execute_events.jsonl"
    monkeypatch.setattr(module, "EVENTS_LOG_PATH", events_path)

    module.log_event({"event": "first"})
    module.log_event({"event": "second", "value": 2})

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["event"] == "first"
    assert second["event"] == "second"
    assert second["value"] == 2

    assert first["run_id"] == second["run_id"] == module.run_id

    datetime.fromisoformat(first["timestamp"])
    datetime.fromisoformat(second["timestamp"])

    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
