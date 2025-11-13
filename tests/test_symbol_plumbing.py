import pandas as pd
import pytest

from scripts.screener import _to_symbol_list


@pytest.mark.alpaca_optional
def test_to_symbol_list_df_and_list():
    df = pd.DataFrame({"symbol": [" aapl ", "MSFT", None, "msft", ""]})
    assert _to_symbol_list(df) == ["AAPL", "MSFT"]
    assert _to_symbol_list(["spy", " SPY ", None]) == ["SPY"]
