import logging

import pytest

from scripts import execute_trades


pytestmark = pytest.mark.alpaca_optional


def test_trading_auth_failure_exits(monkeypatch, caplog):
    monkeypatch.setenv("APCA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("APCA_DATA_API_BASE_URL", "https://data.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")

    class FakeResponse:
        status_code = 401

    def fake_get(url, headers, timeout):  # pragma: no cover - simple stub
        return FakeResponse()

    monkeypatch.setattr(execute_trades.requests, "get", fake_get)

    caplog.set_level(logging.ERROR, logger="execute_trades")
    with pytest.raises(SystemExit) as exc:
        execute_trades._ensure_trading_auth(
            "https://paper-api.alpaca.markets", {"status": "ok"}
        )

    assert exc.value.code == 2
    assert "TRADING_AUTH_FAILED" in caplog.text
