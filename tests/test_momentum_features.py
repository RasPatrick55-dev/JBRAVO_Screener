import pandas as pd
import pytest

from scripts.features import add_wk52_and_rs


@pytest.mark.alpaca_optional
def test_wk52_and_rs_columns_added():
    periods = 70
    df = pd.DataFrame(
        {
            "symbol": ["A"] * periods,
            "date": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "close": list(range(10, 10 + periods)),
        }
    )
    spy = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "close": list(range(100, 100 + periods)),
            "symbol": ["SPY"] * periods,
        }
    )
    out = add_wk52_and_rs(df.copy(), spy)
    assert {"wk52_high", "wk52_prox", "rs20_slope"}.issubset(out.columns)
    assert out["wk52_high"].notna().any()
