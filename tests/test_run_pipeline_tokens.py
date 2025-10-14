import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_pipeline


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test")
    monkeypatch.setenv("ALPACA_KEY_ID", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.mark.alpaca_optional
def test_pipeline_emits_tokens(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pipeline, "BASE_DIR", tmp_path)
    monkeypatch.setattr(run_pipeline, "load_env", lambda: None)
    monkeypatch.setattr(run_pipeline, "configure_logging", lambda: None)
    monkeypatch.setattr(run_pipeline, "assert_alpaca_creds", lambda: {"id": "ok"})
    monkeypatch.setattr(run_pipeline, "_record_health", lambda stage: {})
    monkeypatch.setattr(run_pipeline, "run_cmd", lambda cmd, name: 0)
    monkeypatch.setattr(run_pipeline, "run_execute_step", lambda cmd, candidate_rows=None: 0)
    monkeypatch.setattr(run_pipeline, "emit_metric", lambda *a, **k: None)

    def fake_refresh() -> dict[str, object]:
        metrics = {
            "symbols_in": 5,
            "symbols_with_bars": 4,
            "rows": 2,
            "timings": {
                "fetch_secs": 0.1,
                "feature_secs": 0.2,
                "rank_secs": 0.3,
                "gates_secs": 0.4,
            },
        }
        latest_path = data_dir / "latest_candidates.csv"
        latest_path.write_text(run_pipeline.LATEST_HEADER, encoding="utf-8")
        metrics_path = data_dir / "screener_metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return json.loads(json.dumps(metrics))

    monkeypatch.setattr(run_pipeline, "refresh_latest_candidates", fake_refresh)

    def fake_fallback(*args, **kwargs):
        return 1, "test_source"

    monkeypatch.setattr(run_pipeline, "ensure_min_candidates", fake_fallback)

    caplog.set_level(logging.INFO, logger="pipeline")

    rc = run_pipeline.main(["--steps", "metrics", "--reload-web", "false"])

    assert rc == 0

    messages = [record.getMessage() for record in caplog.records]
    assert any("PIPELINE_SUMMARY" in msg for msg in messages), messages
    assert any("FALLBACK_CHECK" in msg for msg in messages), messages
