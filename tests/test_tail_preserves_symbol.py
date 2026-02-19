import pandas as pd
import pytest

from scripts.utils.frame_guards import ensure_symbol_column


@pytest.mark.alpaca_optional
def test_tail_preserves_symbol():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3 + ["MSFT"] * 3,
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
                utc=True,
            ),
            "open": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5, 6],
            "close": [1, 2, 3, 4, 5, 6],
            "volume": [10] * 6,
        }
    )
    out = df.sort_values("timestamp").groupby("symbol", as_index=False, group_keys=False).tail(2)
    out = ensure_symbol_column(out)
    assert "symbol" in out.columns
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert len(out) == 4
