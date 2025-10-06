import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.telemetry as telemetry

os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "test")

from scripts import run_pipeline


def test_log_event_appends_and_valid_json(tmp_path, monkeypatch):
    events_path = tmp_path / "execute_events.jsonl"
    monkeypatch.setattr(telemetry, "events_path", lambda: events_path)
    monkeypatch.setattr(telemetry, "get_version", lambda: "test-version")

    run_pipeline.log_event({"event": "FIRST"})
    run_pipeline.log_event({"event": "SECOND"})

    assert events_path.exists()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_event = json.loads(lines[0])
    second_event = json.loads(lines[1])

    assert first_event["event"] == "FIRST"
    assert second_event["event"] == "SECOND"
    assert first_event["component"] == second_event["component"] == "pipeline"
    assert "ts" in first_event
