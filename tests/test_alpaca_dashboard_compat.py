from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest


def _prepare_dashboard_data(base: Path) -> None:
    data_dir = base / "data"
    logs_dir = base / "logs"
    reports_dir = base / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "symbols_in": 12,
        "symbols_with_bars_fetch": 10,
        "bars_rows_total_fetch": 240,
        "rows": 3,
        "last_run_utc": "2024-01-01T00:00:00Z",
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(metrics))

    pd.DataFrame(
        [
            {"symbol": "AAA"},
            {"symbol": "BBB"},
            {"symbol": "CCC"},
        ]
    ).to_csv(data_dir / "top_candidates.csv", index=False)

    conn_payload = {"trading_ok": True, "data_ok": True}
    (data_dir / "connection_health.json").write_text(json.dumps(conn_payload))

    (logs_dir / "pipeline.log").write_text("2024-01-01 PIPELINE_END rc=0\n")

def _reload_dashboard_app(monkeypatch: pytest.MonkeyPatch, base: Path):
    monkeypatch.setenv("JBRAVO_HOME", str(base))
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("JBR_EXEC_PAPER", "1")

    import alpaca.trading.client as alpaca_client

    class _DummyTradingClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(alpaca_client, "TradingClient", _DummyTradingClient)

    sys.modules.pop("dashboards.data_io", None)
    sys.modules.pop("dashboards.dashboard_app", None)
    module = importlib.import_module("dashboards.dashboard_app")
    monkeypatch.setattr(
        module, "_alpaca_rest_get_json", Mock(side_effect=AssertionError("Unexpected broker read"))
    )
    return module

@pytest.mark.alpaca_optional
def test_account_sdk_compatibility_without_deprecated_property(monkeypatch, tmp_path):
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    class Account:
        equity = "1234.50"
        cash = "234.50"
        buying_power = "500"

        @property
        def portfolio_value(self):
            raise AssertionError("deprecated SDK field accessed")

    client = Mock()
    client.get_account.return_value = Account()
    monkeypatch.setattr(module, "trading_client", client)
    row = module._account_latest_alpaca()
    assert row["equity"] == row["portfolio_value"] == 1234.5

    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    client.reset_mock()
    assert module._account_latest_alpaca() == {}
    client.get_account.assert_not_called()

@pytest.mark.alpaca_optional
def test_account_db_summary_prefers_equity_to_legacy_alias(monkeypatch, tmp_path):
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_account_latest_db",
        lambda: {
            "equity": "1234.50",
            "portfolio_value": "9999",
            "cash": "234.50",
            "buying_power": "500",
            "source": "v_account_latest",
        },
    )
    summary, source = module._account_summary_from_db()
    assert summary["equity"] == 1234.5
    assert summary["open_positions_value"] == 1000.0
    assert source == "postgres:v_account_latest"

@pytest.mark.alpaca_optional
def test_persisted_account_read_keeps_postgres_authoritative(monkeypatch, tmp_path):
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_db_fetch_one",
        lambda _: {"equity": 1234.5, "portfolio_value": 1234.5, "cash": 234.5},
    )
    remote = Mock(side_effect=AssertionError("DB account must not query Alpaca"))
    monkeypatch.setattr(module, "_account_latest_alpaca", remote)
    row = module._account_latest_db()
    assert row["equity"] == 1234.5
    assert row["source"] == "v_account_latest"
    remote.assert_not_called()

@pytest.mark.alpaca_optional
@pytest.mark.parametrize("source", ["postgres", "file-fallback"])
@pytest.mark.parametrize("count", [0, None, 5])
def test_execute_summary_metrics_never_query_or_merge_broker_counts(
    monkeypatch, tmp_path, source, count
):
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    metrics = {
        "last_run_utc": "2026-02-20T16:05:36+00:00",
        "candidates": 6,
        "in_window": False,
        "submitted": count,
        "filled": count,
        "rejected": count,
        "result_pl_usd": 0.0,
    }
    db_read = Mock(return_value=(metrics, "postgres:executor_runs"))
    file_read = Mock(side_effect=AssertionError("PostgreSQL must take precedence"))
    if source == "file-fallback":
        db_read.return_value = (None, "postgres:none")
        file_read = Mock(return_value=(metrics, "file:metrics"))
    broker_read = Mock(side_effect=AssertionError("Metrics must not query broker counts"))
    monkeypatch.setattr(module, "_execute_summary_from_metrics_table", db_read)
    monkeypatch.setattr(module, "_execute_summary_from_metrics_file", file_read)
    monkeypatch.setattr(module, "_execute_summary_counts_from_alpaca", broker_read)
    payload = module._execute_summary_payload()
    assert payload["source"] == source
    assert payload["submitted"] == (count or 0)
    assert payload["filled"] == (count or 0)
    assert payload["rejected"] == (count or 0)
    broker_read.assert_not_called()
    if source == "postgres":
        file_read.assert_not_called()

@pytest.mark.alpaca_optional
def test_execute_summary_without_metrics_blocks_live_broker_fallback(monkeypatch, tmp_path):
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setattr(module, "_execute_summary_from_metrics_table", lambda: (None, "db:none"))
    monkeypatch.setattr(module, "_execute_summary_from_metrics_file", lambda: (None, "file:none"))
    monkeypatch.setattr(module, "_execute_last_run_from_logs", lambda: "2026-02-20T16:05:36+00:00")
    monkeypatch.setattr(module, "_execute_latest_candidates_count", lambda: 0)
    monkeypatch.setattr(module, "_execute_result_pl_from_db", lambda _: None)
    rest = Mock(side_effect=AssertionError("Live broker request attempted"))
    monkeypatch.setattr(module, "_alpaca_rest_get_json", rest)
    payload = module._execute_summary_payload()
    assert payload["source"] == "local-fallback"
    assert payload["submitted"] == payload["filled"] == payload["rejected"] == 0
    assert "paper_mode_required" in payload["source_detail"]
    rest.assert_not_called()

@pytest.mark.alpaca_optional
def test_execute_summary_uses_alpaca_fallback_only_when_no_metrics_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    monkeypatch.setattr(
        module, "_execute_summary_from_metrics_table", lambda: (None, "postgres:none")
    )
    monkeypatch.setattr(
        module, "_execute_summary_from_metrics_file", lambda: (None, "file:missing")
    )
    monkeypatch.setattr(
        module,
        "_execute_summary_counts_from_alpaca",
        lambda _after, _until=None: (
            {"submitted": 3, "filled": 2, "rejected": 1},
            "alpaca:/v2/orders",
        ),
    )
    monkeypatch.setattr(module, "_execute_latest_candidates_count", lambda: 6)
    monkeypatch.setattr(module, "_execute_result_pl_from_db", lambda _dt: 42.0)
    monkeypatch.setattr(module, "_execute_last_run_from_logs", lambda: "2026-02-20T16:05:36+00:00")
    monkeypatch.setattr(module, "_execute_now_in_window", lambda: False)

    payload = module._execute_summary_payload()

    assert payload["source"] == "alpaca-fallback"
    assert payload["submitted"] == 3
    assert payload["filled"] == 2
    assert payload["rejected"] == 1
    assert payload["candidates"] == 6

