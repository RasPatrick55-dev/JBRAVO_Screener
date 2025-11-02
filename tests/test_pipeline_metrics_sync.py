import json
from pathlib import Path

import pytest

from scripts import run_pipeline


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.alpaca_optional
def test_write_complete_screener_metrics_backfills_from_log(tmp_path):
    base_dir = tmp_path
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    metrics_path = data_dir / "screener_metrics.json"
    metrics_path.write_text(json.dumps({"last_run_utc": "2024-05-20T12:00:00Z"}), encoding="utf-8")

    log_lines = [
        "2024-05-20T11:59:59Z [INFO] PIPELINE_START steps=screener,metrics",
        "2024-05-20T12:00:02Z [INFO] PIPELINE_SUMMARY symbols_in=25 with_bars=20 rows=9 fetch_secs=1.0 feature_secs=2.0 rank_secs=3.0 gate_secs=4.0 bars_rows_total=450 source=screener",
        "2024-05-20T12:00:03Z [INFO] PIPELINE_END rc=0 duration=4.0s",
    ]
    _write(logs_dir / "pipeline.log", "\n".join(log_lines))

    top_candidates = data_dir / "top_candidates.csv"
    top_candidates.write_text("symbol,score\nAAPL,1.0\nMSFT,2.0\n", encoding="utf-8")

    result = run_pipeline.write_complete_screener_metrics(base_dir)

    assert result["symbols_in"] == 25
    assert result["symbols_with_bars"] == 20
    assert result["rows"] == 9
    assert result["bars_rows_total"] == 450
    assert result["last_run_utc"] == "2024-05-20T12:00:00Z"

    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written == result


@pytest.mark.alpaca_optional
def test_write_complete_screener_metrics_counts_rows_when_missing(tmp_path):
    base_dir = tmp_path
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    metrics_path = data_dir / "screener_metrics.json"
    metrics_path.write_text(json.dumps({}), encoding="utf-8")

    top_candidates = data_dir / "top_candidates.csv"
    top_candidates.write_text(
        "symbol,score\nAAPL,1.0\nMSFT,2.0\nNVDA,3.0\n", encoding="utf-8"
    )

    result = run_pipeline.write_complete_screener_metrics(base_dir)

    assert result["rows"] == 3
    assert result["last_run_utc"]
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == result
