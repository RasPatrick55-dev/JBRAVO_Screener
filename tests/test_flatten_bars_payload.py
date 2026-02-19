from scripts.utils.http_alpaca import _flatten_to_canonical


def test_flatten_dict_of_lists():
    data = {
        "bars": {
            "AAPL": [{"t": "2024-01-02T00:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}],
            "MSFT": [{"t": "2024-01-02T00:00:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 20}],
        }
    }
    flat = _flatten_to_canonical(data)
    assert len(flat) == 2
    assert {"symbol", "timestamp", "open", "high", "low", "close", "volume"}.issubset(
        flat[0].keys()
    )


def test_flatten_list_of_dicts():
    data = {
        "bars": [{"S": "SPY", "t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5}]
    }
    flat = _flatten_to_canonical(data)
    assert len(flat) == 1 and flat[0]["symbol"] == "SPY"
