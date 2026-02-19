import json
from pathlib import Path

import pandas as pd
import pytest

from tests._data_io_helpers import reload_data_io


def _write_top_candidates(data_dir: Path, rows: int = 2) -> None:
    df = pd.DataFrame([{"symbol": f"SYM{i}"} for i in range(rows)])
    df.to_csv(data_dir / "top_candidates.csv", index=False)


def test_screener_health_backcompat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    (logs_dir / "pipeline.log").write_text(
        "2024-01-01 [INFO] PIPELINE_END rc=0\n", encoding="utf-8"
    )
    _write_top_candidates(data_dir, rows=3)

    legacy_metrics = {
        "symbols_in": 100,
        "symbols_with_bars": 80,
        "bars_rows_total": 2500,
        "rows": 1,
        "latest_source": "fallback",
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(legacy_metrics), encoding="utf-8")
    data_io = reload_data_io(monkeypatch, tmp_path)

    legacy_snapshot = data_io.screener_health()
    assert legacy_snapshot["symbols_with_bars_fetch"] == 80
    assert legacy_snapshot["bars_rows_total_fetch"] == 2500
    assert legacy_snapshot["rows_final"] == 3
    assert legacy_snapshot["source"] == "fallback"

    split_metrics = {
        "symbols_in": 120,
        "symbols_with_bars_fetch": 90,
        "symbols_with_bars_post": 45,
        "bars_rows_total_fetch": 4000,
        "bars_rows_total_post": 2000,
        "rows": 5,
        "latest_source": "screener",
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(split_metrics), encoding="utf-8")

    split_snapshot = data_io.screener_health()
    assert split_snapshot["symbols_with_bars_fetch"] == 90
    assert split_snapshot["symbols_with_bars_post"] == 45
    assert split_snapshot["bars_rows_total_fetch"] == 4000
    assert split_snapshot["bars_rows_total_post"] == 2000
    assert split_snapshot["source"] == "screener"
