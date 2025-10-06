import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "test")

from scripts import run_pipeline


def test_log_event_appends_and_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "BASE_DIR", tmp_path)

    log_file = tmp_path / "data" / "execute_events.jsonl"

    run_pipeline.log_event({"event": "FIRST"})
    run_pipeline.log_event({"event": "SECOND"})

    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_event = json.loads(lines[0])
    second_event = json.loads(lines[1])

    assert first_event["event"] == "FIRST"
    assert second_event["event"] == "SECOND"
    assert "timestamp" in first_event
