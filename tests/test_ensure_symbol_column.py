import pandas as pd
import pytest

from scripts.utils.frame_guards import ensure_symbol_column


pytestmark = pytest.mark.alpaca_optional


def test_symbol_x_y_rescue():
    df = pd.DataFrame({"symbol_x": ["aapl", "msft"], "close": [1, 2]})
    df2 = ensure_symbol_column(df)
    assert "symbol" in df2.columns and set(df2["symbol"]) == {"AAPL", "MSFT"}


def test_index_rescue():
    df = pd.DataFrame({"close": [1, 2]})
    df.index.name = "symbol"
    df.index = ["aapl", "msft"]
    df2 = ensure_symbol_column(df)
    assert "symbol" in df2.columns and set(df2["symbol"]) == {"AAPL", "MSFT"}
