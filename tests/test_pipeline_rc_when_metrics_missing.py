import json
import logging

import pandas as pd
import pytest

from scripts import fallback_candidates, metrics, run_pipeline


@pytest.mark.alpaca_optional
def test_pipeline_rc_zero_when_metrics_missing_trades(tmp_path, monkeypatch, caplog):
    base_dir = tmp_path
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    top_path = data_dir / "top_candidates.csv"
    top_frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "AAPL",
                "score": 1.23,
                "exchange": "NASDAQ",
                "close": 150.0,
                "volume": 1_000_000,
                "universe_count": 100,
                "score_breakdown": "{}",
                "entry_price": 150.0,
                "adv20": 2_000_000.0,
                "atrp": 0.02,
                "source": "screener",
            }
        ]
    )
    top_frame.to_csv(top_path, index=False)

    scored_path = data_dir / "scored_candidates.csv"
    scored_frame = pd.DataFrame(top_frame)
    scored_frame.to_csv(scored_path, index=False)

    screener_metrics_path = data_dir / "screener_metrics.json"
    screener_metrics_path.write_text(
        json.dumps(
            {
                "symbols_in": 10,
                "symbols_with_bars": 8,
                "rows": 1,
                "timings": {"fetch_secs": 0.1, "feature_secs": 0.2},
            }
        ),
        encoding="utf-8",
    )

    backtest_path = data_dir / "backtest_results.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "net_pnl": 0.0,
                "win_rate": 0.0,
                "trades": 0,
            }
        ]
    ).to_csv(backtest_path, index=False)

    pipeline_log_path = logs_dir / "pipeline.log"

    monkeypatch.setattr(run_pipeline, "PROJECT_ROOT", base_dir)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_pipeline, "LOG_PATH", pipeline_log_path)
    monkeypatch.setattr(run_pipeline, "SCREENER_METRICS_PATH", screener_metrics_path)
    monkeypatch.setattr(run_pipeline, "LATEST_CANDIDATES", data_dir / "latest_candidates.csv")
    monkeypatch.setattr(run_pipeline, "TOP_CANDIDATES", top_path)

    monkeypatch.setattr(fallback_candidates, "PROJECT_ROOT", base_dir)
    monkeypatch.setattr(fallback_candidates, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        fallback_candidates,
        "LATEST_CANDIDATES",
        data_dir / "latest_candidates.csv",
    )
    monkeypatch.setattr(fallback_candidates, "TOP_CANDIDATES", top_path, raising=False)
    monkeypatch.setattr(fallback_candidates, "SCORED_CANDIDATES", scored_path)

    test_logger = logging.getLogger("pipeline-test")
    test_logger.handlers = []
    monkeypatch.setattr(run_pipeline, "LOG", test_logger)

    caplog.set_level(logging.INFO, logger="pipeline-test")

    monkeypatch.setattr(metrics, "BASE_DIR", str(base_dir))
    monkeypatch.setattr(metrics, "logfile", str(logs_dir / "metrics.log"), raising=False)

    def fake_run_step(name, cmd, timeout=None):
        if name == "metrics":
            rc = metrics.main()
            return rc, 0.1
        return 0, 0.1

    monkeypatch.setattr(run_pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(run_pipeline, "_reload_dashboard", lambda enabled: None)
    monkeypatch.setattr(run_pipeline, "configure_logging", lambda: None)
    monkeypatch.setattr(run_pipeline, "load_env", lambda *a, **k: ([], []))

    with pytest.raises(SystemExit) as exit_info:
        run_pipeline.main(["--steps", "screener,metrics", "--reload-web", "false"])

    assert exit_info.value.code == 0

    summary_path = data_dir / "metrics_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert summary.iloc[0]["total_trades"] == 0

    messages = [record.getMessage() for record in caplog.records]
    assert any("[INFO] PIPELINE_END rc=0" in message for message in messages)
    assert any("[INFO] PIPELINE_SUMMARY" in message for message in messages)
