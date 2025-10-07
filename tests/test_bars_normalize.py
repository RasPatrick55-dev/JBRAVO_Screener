import pandas as pd

import pytest

from scripts.screener import normalize_bars_to_df


pytestmark = pytest.mark.alpaca_optional


def test_normalize_handles_multiindex_and_http():
    idx = pd.MultiIndex.from_tuples(
        [("AAPL", "2024-10-01"), ("MSFT", "2024-10-01")],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
        },
        index=idx,
    )

    class Obj:
        pass

    obj = Obj()
    obj.df = df

    out = normalize_bars_to_df(obj)
    assert list(sorted(out.columns)) == sorted(
        ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )
    assert out["symbol"].tolist() == ["AAPL", "MSFT"]

    raw = [
        {"S": "AAPL", "t": "2024-10-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 10}
    ]
    out2 = normalize_bars_to_df(raw)
    assert list(out2.columns) == [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert out2.loc[0, "symbol"] == "AAPL"
