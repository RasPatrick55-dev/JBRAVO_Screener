import pandas as pd
import pytest

from scripts.screener import _is_common_stock_like


@pytest.mark.alpaca_optional
def test_is_common_stock_like_filters_funds_and_warrants():
    rows = [
        {
            "symbol": "SPY",
            "exchange": "ARCA",
            "asset_type": "EQUITY",
            "name": "SPDR S&P 500 ETF Trust",
        },
        {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "asset_type": "COMMON",
            "name": "Apple Inc.",
        },
        {
            "symbol": "XYZW",
            "exchange": "NASDAQ",
            "asset_type": "COMMON",
            "name": "XYZ Corp Warrant",
        },
    ]
    df = pd.DataFrame(rows)
    keep = df[df.apply(_is_common_stock_like, axis=1)]
    assert list(keep["symbol"]) == ["AAPL"]
