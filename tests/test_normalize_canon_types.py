from scripts.utils.normalize import to_bars_df


def test_canon_types():
    flat = [
        {
            "symbol": "AAPL",
            "timestamp": "2024-01-02T00:00:00Z",
            "open": "1",
            "high": "2",
            "low": "1",
            "close": "2",
            "volume": "10",
        }
    ]
    df = to_bars_df(flat)
    assert list(df.columns) == [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df["open"].dtype.kind == "f"
    assert str(df["timestamp"].dtype).endswith("[UTC]")
