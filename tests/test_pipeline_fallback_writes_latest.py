import json
from pathlib import Path

import pandas as pd

from scripts import run_pipeline


def test_pipeline_fallback_writes_latest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pipeline, "BASE_DIR", tmp_path)
    monkeypatch.setattr(run_pipeline, "LOG_PATH", logs_dir / "pipeline.log")

    def fake_check_call(cmd, cwd=None):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "symbol": "FALL",
                    "score": 1.0,
                    "exchange": "NYSE",
                    "close": 10.0,
                    "volume": 1_000,
                    "universe_count": 1,
                    "score_breakdown": json.dumps({}),
                    "entry_price": 10.0,
                    "adv20": 500_000,
                    "atrp": 0.01,
                    "source": "fallback",
                }
            ]
        )
        frame.to_csv(data_dir / "top_candidates.csv", index=False)
        return 0

    monkeypatch.setattr(run_pipeline.subprocess, "check_call", fake_check_call)

    rows = run_pipeline._maybe_fallback(tmp_path)
    assert rows >= 1

    latest = pd.read_csv(data_dir / "latest_candidates.csv")
    assert latest.columns.tolist() == list(run_pipeline.LATEST_COLUMNS)
    assert latest.iloc[0]["symbol"] == "FALL"
