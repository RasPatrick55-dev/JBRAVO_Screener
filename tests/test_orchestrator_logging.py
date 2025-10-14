import json
import logging
import sys
from pathlib import Path

import pandas as pd
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
def test_fallback_and_summary_logged_once(tmp_path: Path, monkeypatch, caplog):
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
    monkeypatch.setattr(run_pipeline, "emit_metric", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "write_metrics_summary", lambda **kwargs: None)

    metrics_payload = {
        "symbols_in": 5,
        "symbols_with_bars": 4,
        "rows": 0,
        "timings": {
            "fetch_secs": 1.5,
            "feature_secs": 2.5,
            "rank_secs": 3.5,
            "gates_secs": 4.5,
        },
    }

    def fake_refresh() -> dict[str, object]:
        latest_path = data_dir / "latest_candidates.csv"
        latest_path.write_text(run_pipeline.LATEST_HEADER, encoding="utf-8")
        metrics_path = data_dir / "screener_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
        return json.loads(json.dumps(metrics_payload))

    monkeypatch.setattr(run_pipeline, "refresh_latest_candidates", fake_refresh)

    scored_path = data_dir / "scored_candidates.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "now",
                "symbol": "XYZ",
                "score": 9.9,
                "exchange": "NASDAQ",
                "close": 10.0,
                "volume": 2_500_000,
                "universe_count": 100,
                "score_breakdown": "{}",
                "adv20": 2_500_000,
            }
        ]
    ).to_csv(scored_path, index=False)

    caplog.set_level(logging.INFO, logger="pipeline")
    rc = run_pipeline.main(["--steps", "screener", "--reload-web", "false"])
    assert rc == 0

    messages = [record.getMessage() for record in caplog.records]
    fallback_lines = [msg for msg in messages if "FALLBACK_CHECK" in msg]
    summary_lines = [msg for msg in messages if "PIPELINE_SUMMARY" in msg]
    assert len(fallback_lines) == 1
    assert len(summary_lines) == 1
    assert "rows=" in summary_lines[0]
