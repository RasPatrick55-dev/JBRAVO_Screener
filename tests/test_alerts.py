import logging
from typing import Any

import pytest

from utils import alerts


def test_send_alert_disabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.delenv("ALERT_WEBHOOK", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    caplog.set_level(logging.INFO, logger="utils.alerts")

    alerts.send_alert("Disabled test", {"foo": "bar"})

    assert "Alerts disabled via ALERTS_ENABLED" in caplog.text


def test_send_alert_enabled_missing_webhook(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.delenv("ALERT_WEBHOOK", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    caplog.set_level(logging.INFO, logger="utils.alerts")

    alerts.send_alert("Missing webhook test")

    assert "ALERT_WEBHOOK not set; would send alert" in caplog.text


def test_send_alert_with_webhook_success(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    class DummyResponse:
        ok = True
        status_code = 200

    def fake_post(url: str, json: dict[str, Any], timeout: int = 5):  # type: ignore[override]
        called["url"] = url
        called["json"] = json
        called["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("ALERT_WEBHOOK", "https://example.com/webhook")
    monkeypatch.setattr(alerts.requests, "post", fake_post)

    alerts.send_alert("Webhook success", {"alpha": 1})

    assert called["url"] == "https://example.com/webhook"
    assert "Webhook success" in called["json"].get("text", "")
    assert called["json"].get("context") == {"alpha": 1}
    assert called["timeout"] == 5
