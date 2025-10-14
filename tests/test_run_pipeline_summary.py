import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_pipeline


@pytest.fixture(autouse=True)
def _set_creds(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test")
    monkeypatch.setenv("ALPACA_KEY_ID", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.mark.alpaca_optional
def test_pipeline_summary_zero_candidates(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(run_pipeline, "load_env", lambda: None)
    monkeypatch.setattr(run_pipeline, "configure_logging", lambda: None)
    monkeypatch.setattr(run_pipeline, "assert_alpaca_creds", lambda: {"id": "ok"})
    monkeypatch.setattr(run_pipeline, "_record_health", lambda stage: {})
    monkeypatch.setattr(run_pipeline, "run_cmd", lambda cmd, name: 0)
    monkeypatch.setattr(run_pipeline, "emit_metric", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "write_metrics_summary", lambda **kwargs: None)

    latest_path = data_dir / "latest_candidates.csv"

    metrics_payload = {
        "symbols_in": 15,
        "symbols_with_bars": 12,
        "rows": 0,
        "timings": {
            "fetch_secs": 1.1,
            "feature_secs": 2.2,
            "rank_secs": 3.3,
            "gates_secs": 4.4,
        },
    }

    def fake_refresh() -> dict[str, object]:
        latest_path.write_text(run_pipeline.LATEST_HEADER, encoding="utf-8")
        return json.loads(json.dumps(metrics_payload))

    monkeypatch.setattr(run_pipeline, "refresh_latest_candidates", fake_refresh)

    def fake_execute(cmd, *, candidate_rows=None):
        raise AssertionError("execute step should be skipped when rows=0")

    monkeypatch.setattr(run_pipeline, "run_execute_step", fake_execute)

    caplog.set_level(logging.INFO, logger="pipeline")
    rc = run_pipeline.main(["--steps", "screener,execute", "--reload-web", "false"])

    assert rc == 0

    messages = [record.getMessage() for record in caplog.records]
    summary_lines = [msg for msg in messages if "PIPELINE_SUMMARY" in msg]
    assert summary_lines, "PIPELINE_SUMMARY log missing"
    summary_line = summary_lines[-1]
    assert "rows=0" in summary_line
    assert "fetch_secs=1.10" in summary_line
    assert "feature_secs=2.20" in summary_line
    assert "rank_secs=3.30" in summary_line
    assert "gate_secs=4.40" in summary_line
    assert "step_rcs=" in summary_line

    skip_lines = [msg for msg in messages if "EXECUTE_SKIP_NO_CANDIDATES" in msg]
    assert skip_lines == ["[INFO] EXECUTE_SKIP_NO_CANDIDATES rows=0"]
