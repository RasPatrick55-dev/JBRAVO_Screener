import pytest

from scripts.utils.http_alpaca import fetch_bars_http
from scripts.utils.normalize import BARS_COLUMNS, to_bars_df


@pytest.mark.alpaca_optional
def test_market_data_host(monkeypatch):
    captured_urls: list[str] = []

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"bars": [], "next_page_token": None}

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, params=None, timeout=None):
        captured_urls.append(url)
        return DummyResponse()

    monkeypatch.delenv("APCA_DATA_API_BASE_URL", raising=False)
    monkeypatch.setattr("scripts.utils.http_alpaca.requests.get", fake_get)

    bars, metrics = fetch_bars_http(["AAPL"], "2024-01-01", "2024-01-02", sleep_s=0)

    assert captured_urls == ["https://data.alpaca.markets/v2/stocks/bars"]
    assert bars == []
    assert metrics["http_empty_batches"] == 1
    assert metrics["rate_limited"] == 0


def test_normalize_http():
    payload = [
        {"S": "spy", "t": "2024-01-01T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        {"S": "msft", "t": "2024-01-02T00:00:00Z", "o": 2, "h": 3, "l": 1.5, "c": 2.5, "v": 20},
    ]

    df = to_bars_df(payload)

    assert list(df.columns) == BARS_COLUMNS
    assert df.shape == (2, len(BARS_COLUMNS))
    assert df.loc[0, "symbol"] == "SPY"
    assert df.loc[1, "symbol"] == "MSFT"
