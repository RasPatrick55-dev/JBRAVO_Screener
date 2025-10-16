import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_pipeline


@pytest.mark.alpaca_optional
def test_pipeline_logs_summary_and_end(tmp_path: Path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    top_path = data_dir / "top_candidates.csv"
    top_path.write_text("symbol,score\n", encoding="utf-8")
    scored_path = data_dir / "scored_candidates.csv"
    pd.DataFrame(
        [
            {
                "symbol": "ZZZ",
                "score": 4.2,
                "exchange": "NYSE",
                "close": 12.5,
                "volume": 150_000,
                "universe count": 5,
                "score breakdown": "{}",
            }
        ]
    ).to_csv(scored_path, index=False)

    metrics_path = data_dir / "screener_metrics.json"
    metrics_payload = {
        "symbols_in": 10,
        "symbols_with_bars": 9,
        "rows": 0,
        "timings": {
            "fetch_secs": 0.11,
            "feature_secs": 0.22,
            "rank_secs": 0.33,
            "gate_secs": 0.44,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    log_path = logs_dir / "pipeline.log"

    monkeypatch.setattr(run_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_pipeline, "LOG_PATH", log_path)
    monkeypatch.setattr(run_pipeline, "SCREENER_METRICS_PATH", metrics_path)
    monkeypatch.setattr(run_pipeline, "LATEST_CANDIDATES", data_dir / "latest_candidates.csv")
    monkeypatch.setattr(run_pipeline, "TOP_CANDIDATES", top_path)
    test_logger = logging.getLogger("pipeline-test")
    test_logger.handlers = []
    monkeypatch.setattr(run_pipeline, "LOG", test_logger)

    caplog.set_level(logging.INFO, logger="pipeline-test")

    def fake_run_step(name, cmd, timeout=None):
        if name == "screener":
            # ensure metrics payload exists even if overwritten
            metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
        return 0, 0.5

    monkeypatch.setattr(run_pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(run_pipeline, "_reload_dashboard", lambda enabled: None)
    monkeypatch.setattr(run_pipeline, "configure_logging", lambda: None)
    monkeypatch.setattr(run_pipeline, "load_env", lambda *a, **k: ([], []))

    with pytest.raises(SystemExit) as exit_info:
        run_pipeline.main(["--steps", "screener,metrics", "--reload-web", "false"])

    assert exit_info.value.code == 0

    messages = [record.getMessage() for record in caplog.records]
    assert any("FALLBACK_CHECK" in msg for msg in messages)
    assert any("PIPELINE_SUMMARY" in msg for msg in messages)
    assert any("HEALTH trading_ok=" in msg for msg in messages)
    assert any("PIPELINE_END" in msg for msg in messages)
    assert any(
        "source=fallback" in msg or "source=fallback:scored" in msg for msg in messages
    )

    latest = pd.read_csv(data_dir / "latest_candidates.csv")
    assert not latest.empty
    assert latest.iloc[0]["source"].startswith("fallback")
