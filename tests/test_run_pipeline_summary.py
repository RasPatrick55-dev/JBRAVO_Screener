import json
import logging

import pytest

from scripts import run_pipeline


@pytest.mark.alpaca_optional
def test_pipeline_summary_emitted(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    top_csv = data_dir / "top_candidates.csv"
    top_csv.write_text(
        "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown\n"
        "2024-05-01T12:00:00Z,AAPL,3.2,NASDAQ,175.0,1000000,50,{}\n",
        encoding="utf-8",
    )
    metrics_path = data_dir / "screener_metrics.json"
    metrics_path.write_text(
        json.dumps({"symbols_in": 5, "symbols_with_bars": 4, "rows": 0}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONANYWHERE_DOMAIN", "")

    monkeypatch.setattr(run_pipeline, "load_env", lambda: None)
    monkeypatch.setattr(run_pipeline, "assert_alpaca_creds", lambda: {"mode": "paper"})
    monkeypatch.setattr(run_pipeline, "_record_health", lambda stage: {})

    called_steps: list[str] = []

    def fake_run_cmd(cmd, name):
        called_steps.append(name)
        return 0

    monkeypatch.setattr(run_pipeline, "run_cmd", fake_run_cmd)

    emitted: list[tuple[str, int]] = []

    def fake_emit_metric(name, value):
        emitted.append((name, value))

    monkeypatch.setattr(run_pipeline, "emit_metric", fake_emit_metric)

    caplog.set_level(logging.INFO, logger="pipeline")

    rc = run_pipeline.main(["--steps", "screener"])
    assert rc == 0
    assert called_steps == ["SCREENER"]
    assert emitted == [("CANDIDATE_ROWS", 1)]

    summary_line = ""
    for record in caplog.records:
        message = record.getMessage()
        if "PIPELINE_SUMMARY" in message:
            summary_line = message
    assert summary_line, "PIPELINE_SUMMARY log not found"
    assert "symbols_in=5" in summary_line
    assert "with_bars=4" in summary_line
    assert "rows=1" in summary_line
    assert "durations={\"screener\":" in summary_line
    assert "step_rcs={\"screener\":0}" in summary_line

    latest_path = data_dir / "latest_candidates.csv"
    assert latest_path.exists()
