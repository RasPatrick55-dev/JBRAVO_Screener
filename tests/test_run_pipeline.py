import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "test")
os.environ.setdefault("ALPACA_KEY_ID", os.environ["APCA_API_KEY_ID"])
os.environ.setdefault("ALPACA_SECRET_KEY", os.environ["APCA_API_SECRET_KEY"])
os.environ.setdefault("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

from scripts import run_pipeline


pytestmark = [pytest.mark.alpaca_optional, pytest.mark.slow]


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
    (data_dir / "top_candidates.csv").write_text(
        "symbol,score,close\nAAPL,5,123.45\n",
        encoding="utf-8",
    )
    metrics_path = data_dir / "screener_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    def fake_exists(path: str) -> bool:
        return (tmp_path / path).exists()

    copied: dict[str, tuple[str, str]] = {}

    def fake_copy(src: str, dst: str) -> None:
        copied["call"] = (str(src), str(dst))
        dest_path = tmp_path / dst
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("symbol\nAAPL\n", encoding="utf-8")

    monkeypatch.setattr(run_pipeline.os.path, "exists", fake_exists)
    monkeypatch.setattr(run_pipeline, "copyfile", fake_copy)

    run_pipeline.refresh_latest_candidates()

    call = copied.get("call")
    assert call is not None
    src, dst = call
    expected_src = tmp_path / "data" / "top_candidates.csv"
    expected_dst = tmp_path / "data" / "latest_candidates.csv"
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.is_absolute():
        src_path = tmp_path / src_path
    if not dst_path.is_absolute():
        dst_path = tmp_path / dst_path
    assert src_path == expected_src
    assert dst_path == expected_dst
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "last_run_utc" in metrics and metrics["last_run_utc"]
    assert metrics.get("http") == {"429": 0, "404": 0, "empty_pages": 0}
    assert metrics.get("cache") == {"batches_hit": 0, "batches_miss": 0}
    assert metrics.get("universe_prefix_counts") == {}
    assert "timings" in metrics
    assert metrics.get("status") == "ok"
    assert metrics.get("auth_missing") == []


def test_run_step_streams_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "PROJECT_ROOT", tmp_path)
    cmd = [
        sys.executable,
        "-c",
        "import sys; [sys.stdout.write('x'*1000+'\\n') for _ in range(1500)]",
    ]
    rc, _ = run_pipeline.run_step("loud", cmd, timeout=5)

    log_path = tmp_path / "logs" / "step.loud.out"
    assert rc == 0
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "START loud" in content
    assert "END loud" in content
    assert "x" * 1000 in content
