
import pandas as pd
import pytest

from scripts import metrics


def test_metrics_handles_missing_trades(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    backtest_path = data_dir / "backtest_results.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "net_pnl": 1.0,
                "win_rate": 50.0,
                "trades": 1,
            }
        ]
    ).to_csv(backtest_path, index=False)

    monkeypatch.setattr(metrics, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(metrics, "logfile", str(logs_dir / "metrics.log"), raising=False)

    metrics.main()

    summary_path = data_dir / "metrics_summary.csv"
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert set(metrics.REQUIRED_COLUMNS) == set(summary.columns)
    assert not summary.empty
    row = summary.iloc[0]
    assert row["total_trades"] == 0
    assert row["net_pnl"] == pytest.approx(0.0)
    assert row["win_rate"] == pytest.approx(0.0)
    assert row["expectancy"] == pytest.approx(0.0)
    assert row["profit_factor"] == pytest.approx(0.0)
    assert row["max_drawdown"] == pytest.approx(0.0)
