import pandas as pd
import pytest

from dashboards.dashboard_app import load_csv


pytestmark = pytest.mark.alpaca_optional


def test_load_csv_promotes_net_pnl_to_pnl(tmp_path):
    csv_path = tmp_path / "trades_log.csv"
    pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "net_pnl": 5.0,
                "entry_time": "2024-01-01T00:00:00Z",
                "exit_time": "2024-01-02T00:00:00Z",
            }
        ]
    ).to_csv(csv_path, index=False)

    df, alert = load_csv(csv_path, required_columns=["pnl"])

    assert alert is None
    assert "pnl" in df.columns
    assert "net_pnl" in df.columns
    assert df.loc[0, "pnl"] == df.loc[0, "net_pnl"] == 5.0
