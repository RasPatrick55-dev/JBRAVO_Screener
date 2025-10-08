from scripts.utils.http_alpaca import _flatten_to_canonical


def test_dict_of_lists_to_canon():
    data = {
        "bars": {
            "AAPL": [
                {"t": "2024-01-02T00:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}
            ],
            "MSFT": [
                {"t": "2024-01-02T00:00:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 20}
            ],
        }
    }
    out = _flatten_to_canonical(data)
    assert len(out) == 2 and set(out[0].keys()) == {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_list_of_dicts_to_canon():
    data = {
        "bars": [
            {"S": "SPY", "t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5}
        ]
    }
    out = _flatten_to_canonical(data)
    assert len(out) == 1 and out[0]["symbol"] == "SPY"
