import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "test")
os.environ.setdefault("ALPACA_KEY_ID", os.environ["APCA_API_KEY_ID"])
os.environ.setdefault("ALPACA_SECRET_KEY", os.environ["APCA_API_SECRET_KEY"])
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

from scripts import run_pipeline


def _stub_emit(monkeypatch, events_path: Path):
    def fake_emit(evt: str, **kvs):
        payload = {"event": evt, **kvs}
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    monkeypatch.setattr(run_pipeline, "emit", fake_emit)


def test_log_event_appends_and_valid_json(tmp_path, monkeypatch):
    events_path = tmp_path / "execute_events.jsonl"
    _stub_emit(monkeypatch, events_path)

    run_pipeline.emit("FIRST", component="pipeline")
    run_pipeline.emit("SECOND", component="pipeline")

    assert events_path.exists()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_event = json.loads(lines[0])
    second_event = json.loads(lines[1])

    assert first_event == {"event": "FIRST", "component": "pipeline", "ts": first_event["ts"]}
    assert second_event == {"event": "SECOND", "component": "pipeline", "ts": second_event["ts"]}
    assert first_event["component"] == second_event["component"] == "pipeline"
    assert "ts" in first_event


def test_pipeline_refresh_latest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "top_candidates.csv").write_text("symbol\nAAPL\n", encoding="utf-8")
    metrics_path = data_dir / "screener_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    def fake_exists(path: str) -> bool:
        return (tmp_path / path).exists()

    copied: dict[str, tuple[str, str]] = {}

    def fake_copy(src: str, dst: str) -> None:
        copied["call"] = (src, dst)
        dest_path = tmp_path / dst
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("symbol\nAAPL\n", encoding="utf-8")

    monkeypatch.setattr(run_pipeline.os.path, "exists", fake_exists)
    monkeypatch.setattr(run_pipeline, "copyfile", fake_copy)

    run_pipeline.refresh_latest_candidates()

    assert copied.get("call") == ("data/top_candidates.csv", "data/latest_candidates.csv")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "last_run_utc" in metrics and metrics["last_run_utc"]
