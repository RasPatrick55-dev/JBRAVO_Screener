import pytest
import requests

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
    monkeypatch.setattr("time.sleep", lambda *_, **__: None)

    bars, metrics = fetch_bars_http(["AAPL"], "2024-01-01", "2024-01-02")

    assert captured_urls == ["https://data.alpaca.markets/v2/stocks/bars"]
    assert bars == []
    assert metrics["http_empty_batches"] == 1
    assert metrics["rate_limit_hits"] == 0


def _make_response(status_code, payload=None):
    class DummyResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if 400 <= self.status_code < 600:
                raise requests.HTTPError(f"status={self.status_code}")

    return DummyResponse(status_code, payload or {})


def test_fetch_bars_http_paginates(monkeypatch):
    pages = [
        _make_response(
            200,
            {
                "bars": {
                    "AAPL": [
                        {
                            "t": "2024-01-02T00:00:00Z",
                            "o": 1,
                            "h": 2,
                            "l": 0.5,
                            "c": 1.5,
                            "v": 10,
                        }
                    ]
                },
                "next_page_token": "tok",
            },
        ),
        _make_response(
            200,
            {
                "bars": {
                    "AAPL": [
                        {
                            "t": "2024-01-03T00:00:00Z",
                            "o": 2,
                            "h": 3,
                            "l": 1.5,
                            "c": 2.5,
                            "v": 20,
                        }
                    ]
                },
            },
        ),
    ]
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return pages.pop(0)

    monkeypatch.setattr("scripts.utils.http_alpaca.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda *_, **__: None)

    bars, stats = fetch_bars_http(["aapl"], "2024-01-02", "2024-01-02", batch=10)

    assert len(bars) == 2
    assert stats["requests"] == 2
    assert stats["rows"] == 2
    assert stats["pages"] == 2
    assert calls[0]["symbols"] == "AAPL"
    assert "page_token" not in calls[0]
    assert calls[1]["page_token"] == "tok"


def test_fetch_bars_http_429_backoff(monkeypatch):
    responses = [
        _make_response(429),
        _make_response(
            200,
            {
                "bars": [
                    {
                        "S": "MSFT",
                        "t": "2024-01-02T00:00:00Z",
                        "o": 1,
                        "h": 2,
                        "l": 0.5,
                        "c": 1.5,
                        "v": 10,
                    }
                ]
            },
        ),
    ]
    sleep_calls: list[float] = []

    def fake_sleep(value):
        sleep_calls.append(value)

    def fake_get(url, headers=None, params=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr("scripts.utils.http_alpaca.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", fake_sleep)

    bars, stats = fetch_bars_http(["msft"], "2024-01-02", "2024-01-02", batch=5)

    assert len(bars) == 1
    assert stats["rate_limit_hits"] == 1
    assert stats["retries"] >= 1
    assert any(call > 0 for call in sleep_calls)


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
