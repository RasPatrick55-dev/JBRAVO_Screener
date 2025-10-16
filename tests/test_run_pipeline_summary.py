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
    monkeypatch.setattr(run_pipeline, "BASE_DIR", tmp_path)

    monkeypatch.setattr(run_pipeline, "load_env", lambda *a, **k: ([], []))
    monkeypatch.setattr(run_pipeline, "configure_logging", lambda: None)
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
        metrics_path = data_dir / "screener_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
        return json.loads(json.dumps(metrics_payload))

    monkeypatch.setattr(run_pipeline, "refresh_latest_candidates", fake_refresh)

    caplog.set_level(logging.INFO, logger="pipeline")
    rc = run_pipeline.main(["--steps", "screener", "--reload-web", "false"])

    assert rc == 0

    messages = [record.getMessage() for record in caplog.records]
    summary_lines = [msg for msg in messages if "PIPELINE_SUMMARY" in msg]
    assert summary_lines, "PIPELINE_SUMMARY log missing"
    summary_line = summary_lines[-1]
    assert "rows=0" in summary_line
    assert "fetch_secs=1.1" in summary_line
    assert "feature_secs=2.2" in summary_line
    assert "rank_secs=3.3" in summary_line
    assert "gate_secs=4.4" in summary_line
    assert any("HEALTH trading_ok=" in msg for msg in messages)
    assert any("PIPELINE_END" in msg for msg in messages)
