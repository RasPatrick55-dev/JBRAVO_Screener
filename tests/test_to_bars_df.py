import pandas as pd
import pytest

from scripts.utils.dataframe_utils import BARS_COLUMNS, to_bars_df


pytestmark = pytest.mark.alpaca_optional


def test_to_bars_df_sdk_multiindex():
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
        def __init__(self, frame):
            self.df = frame
            self.data = {"AAPL": [], "MSFT": []}

    out = to_bars_df(Obj(df))
    assert list(out.columns) == BARS_COLUMNS
    assert out["symbol"].tolist() == ["AAPL", "MSFT"]


def test_to_bars_df_handles_dict_of_lists():
    class Bar:
        def __init__(self, symbol):
            self.timestamp = "2024-10-01T00:00:00Z"
            self.open = 1
            self.high = 2
            self.low = 0.5
            self.close = 1.5
            self.volume = 100

    class Container:
        def __init__(self):
            self.data = {"aapl": [Bar("AAPL")], "msft": []}

    out = to_bars_df(Container())
    assert list(out.columns) == BARS_COLUMNS
    assert out.shape == (1, 7)
    assert out.loc[0, "symbol"] == "AAPL"


def test_to_bars_df_http_renames():
    payload = [
        {"S": "msft", "t": "2024-10-01T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 200}
    ]
    out = to_bars_df(payload)
    assert list(out.columns) == BARS_COLUMNS
    assert out.loc[0, "symbol"] == "MSFT"


def test_to_bars_df_handles_dataframe_with_named_index():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1000],
        },
        index=pd.MultiIndex.from_tuples([("spy", "2024-10-01")], names=["symbol", "timestamp"]),
    )

    out = to_bars_df(df)
    assert list(out.columns) == BARS_COLUMNS
    assert out.loc[0, "symbol"] == "SPY"
