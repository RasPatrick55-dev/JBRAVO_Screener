from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.models import TradeAccount

from scripts import fetch_account_snapshot as snapshots
from scripts.utils.alpaca_account import account_equity

pytestmark = pytest.mark.alpaca_optional


@pytest.mark.parametrize(
    ("account", "expected"),
    [
        ({"equity": "1250.25"}, 1250.25),
        ({"equity": "1250.25", "portfolio_value": "900"}, 1250.25),
        ({"equity": "0", "portfolio_value": "900"}, 0.0),
        ({"equity": "-10.5"}, -10.5),
        ({"portfolio_value": "900"}, 900.0),
        ({"equity": None, "portfolio_value": "900"}, 900.0),
        ({"equity": "NaN", "portfolio_value": "900"}, 900.0),
        ({"equity": "invalid"}, None),
        ({"equity": "Infinity"}, None),
        ({}, None),
        (SimpleNamespace(equity="1250.25"), 1250.25),
    ],
)
def test_account_equity_compatibility(account, expected):
    assert account_equity(account) == expected


def test_modern_account_does_not_read_deprecated_property():
    class Account:
        equity = "0"

        @property
        def portfolio_value(self):
            raise AssertionError("deprecated SDK field accessed")

    assert account_equity(Account()) == 0.0


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_real_sdk_account_without_portfolio_value():
    # Both old and new SDK models support construction from the current API shape.
    account = TradeAccount(
        id="00000000-0000-0000-0000-000000000001",
        account_number="PAPER123",
        status="ACTIVE",
        equity="1234.50",
        cash="234.50",
    )
    assert account_equity(account) == 1234.5


def test_snapshot_maps_current_equity_and_preserves_raw():
    raw = {"id": "paper-account", "equity": "1250.25", "buying_power": "800"}
    payload = snapshots._build_insert_payload(raw)
    assert payload["equity"] == 1250.25
    assert payload["portfolio_value"] == 1250.25
    assert payload["daytrading_buying_power"] is None
    assert json.loads(payload["raw"]) == raw
    assert payload["taken_at"].tzinfo is not None


def test_snapshot_preserves_legacy_fields_without_substituting_buying_power():
    payload = snapshots._build_insert_payload(
        {"portfolio_value": "900", "daytrading_buying_power": "0", "buying_power": "800"}
    )
    assert payload["equity"] == payload["portfolio_value"] == 900.0
    assert payload["daytrading_buying_power"] == 0.0


def test_snapshot_persists_compatible_columns_in_postgres(monkeypatch, null_logger):
    conn = MagicMock()
    monkeypatch.setattr(snapshots.db, "get_db_conn", lambda: conn)
    payload = snapshots._build_insert_payload({"equity": "1250.25"})
    assert snapshots._persist_snapshot(payload, null_logger)
    sql, params = conn.cursor.return_value.__enter__.return_value.execute.call_args.args
    assert "INSERT INTO alpaca_account_snapshots" in sql
    assert "CAST(%(raw)s AS JSONB)" in sql
    assert params["equity"] == params["portfolio_value"] == 1250.25
    assert params["daytrading_buying_power"] is None
    conn.__exit__.assert_called_once_with(None, None, None)
    conn.close.assert_called_once()


def test_snapshot_without_postgres_reports_failure(monkeypatch, null_logger):
    monkeypatch.setattr(snapshots.db, "get_db_conn", lambda: None)
    assert not snapshots._persist_snapshot({"equity": 1250.25}, null_logger)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets.example.com",
        "http://paper-api.alpaca.markets",
    ],
)
def test_snapshot_rejects_non_paper_urls_before_request(base_url, null_logger):
    session = MagicMock()
    assert snapshots._fetch_snapshot(session, base_url, null_logger) is None
    session.get.assert_not_called()


def test_snapshot_fetch_uses_only_paper_endpoint(monkeypatch, null_logger):
    monkeypatch.setattr(snapshots, "_alpaca_headers", lambda: {"test": "header"})
    session = MagicMock()
    session.get.return_value.json.return_value = {"equity": "1250.25"}
    result = snapshots._fetch_snapshot(session, "https://paper-api.alpaca.markets/", null_logger)
    assert result == {"equity": "1250.25"}
    session.get.assert_called_once_with(
        "https://paper-api.alpaca.markets/v2/account",
        headers={"test": "header"},
        timeout=15,
        allow_redirects=False,
    )


def test_snapshot_rejects_non_object_response(monkeypatch, null_logger):
    monkeypatch.setattr(snapshots, "_alpaca_headers", lambda: {})
    session = MagicMock()
    session.get.return_value.json.return_value = []
    assert snapshots._fetch_snapshot(session, "https://paper-api.alpaca.markets", null_logger) is None
