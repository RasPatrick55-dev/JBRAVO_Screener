"""Lightweight alert helper for webhook notifications.

Alerts are best-effort: missing configuration or network failures should never
crash callers. Set ``ALERTS_ENABLED=false`` to disable alerts locally.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional

import requests

LOGGER = logging.getLogger("alerts")


def _is_enabled() -> bool:
    value = str(os.getenv("ALERTS_ENABLED", "true")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _webhook_url() -> str | None:
    return os.getenv("ALERT_WEBHOOK") or os.getenv("ALERT_WEBHOOK_URL")


def _coerce_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not isinstance(context, Mapping):
        return payload
    for key, value in context.items():
        if value is None:
            continue
        try:
            json.dumps({"_": value})
            payload[str(key)] = value
        except TypeError:
            payload[str(key)] = str(value)
    return payload


def send_alert(message: str, context: Optional[Mapping[str, Any]] = None) -> None:
    """Send an alert message to the configured webhook if enabled.

    The payload is Slack-compatible and intentionally compact. Missing
    configuration or delivery failures are logged as warnings and never raise.
    """

    if not _is_enabled():
        LOGGER.debug("Alerts disabled via ALERTS_ENABLED")
        return

    webhook = _webhook_url()
    if not webhook:
        LOGGER.warning("ALERT_WEBHOOK not configured; skipping alert: %s", message)
        return

    context_payload = _coerce_context(context)
    summary = ", ".join(f"{k}={v}" for k, v in context_payload.items())
    text = message if not summary else f"{message} | {summary}"
    payload = {"text": text}
    if context_payload:
        payload["context"] = context_payload

    try:
        requests.post(webhook, json=payload, timeout=5)
    except Exception as exc:  # pragma: no cover - defensive network guard
        LOGGER.warning("Failed to send alert: %s", exc)

