from pathlib import Path

import pandas as pd
import pytest

from tests._data_io_helpers import reload_data_io


def test_metrics_summary_snapshot_prefers_data_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    df = pd.DataFrame(
        [
            {
                "profit_factor": 1.1,
                "expectancy": 0.5,
                "win_rate": 50.0,
                "net_pnl": 1000.0,
                "max_drawdown": -200.0,
                "sharpe": 0.8,
                "sortino": 1.2,
                "last_run_utc": "2024-01-01T00:00:00+00:00",
            },
            {
                "profit_factor": 1.8,
                "expectancy": 0.9,
                "win_rate": 65.0,
                "net_pnl": 2500.0,
                "max_drawdown": -150.0,
                "sharpe": 1.1,
                "sortino": 1.6,
                "last_run_utc": "2024-01-02T00:00:00+00:00",
            },
        ]
    )
    df.to_csv(data_dir / "metrics_summary.csv", index=False)

    data_io = reload_data_io(monkeypatch, tmp_path)
    snapshot = data_io.metrics_summary_snapshot()

    assert snapshot["profit_factor"] == 1.8
    assert snapshot["expectancy"] == 0.9
    assert snapshot["net_pnl"] == 2500.0
    assert snapshot["last_run_utc"] == "2024-01-02T00:00:00+00:00"


def test_metrics_summary_snapshot_missing_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    data_io = reload_data_io(monkeypatch, tmp_path)

    snapshot = data_io.metrics_summary_snapshot()
    assert snapshot == {}
