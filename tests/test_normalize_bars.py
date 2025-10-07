import pandas as pd

from scripts.utils.normalize import to_bars_df


def test_normalize_http_payload_infers_symbol():
    payload = [
        {"S": "aapl", "t": "2024-01-01T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000}
    ]
    out = to_bars_df(payload)
    assert list(out.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert out.loc[0, "symbol"] == "AAPL"


def test_normalize_sdk_multiindex_resets_symbol():
    idx = pd.MultiIndex.from_product([["msft"], pd.date_range("2024-01-01", periods=2)], names=[None, None])
    df = pd.DataFrame({"o": [1, 2], "h": [2, 3], "l": [0.5, 1], "c": [1.5, 2.5], "v": [100, 200]}, index=idx)

    class Resp:
        def __init__(self, frame):
            self.df = frame

    out = to_bars_df(Resp(df))
    assert not out.empty
    assert "symbol" in out.columns
    assert out["symbol"].unique().tolist() == ["MSFT"]
