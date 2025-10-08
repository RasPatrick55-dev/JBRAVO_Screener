from scripts.utils.normalize import to_bars_df


def test_flatten_dict_of_lists():
    raw = {"bars": {
        "AAPL": [{"t": "2024-01-02T00:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}],
        "MSFT": [{"t": "2024-01-02T00:00:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 20}],
    }}
    flattened = []
    for sym, arr in raw["bars"].items():
        for bar in arr:
            flattened.append({**bar, "S": sym})
    df = to_bars_df(flattened)
    assert "symbol" in df.columns
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    assert df["open"].dtype.kind in "f"
    dtype_str = str(df["timestamp"].dtype)
    assert dtype_str.endswith("UTC]") or dtype_str.endswith("[UTC]")
