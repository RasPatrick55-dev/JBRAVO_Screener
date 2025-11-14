from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

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
    return module


def test_connection_badge_color(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    assert module.connection_badge_color({"trading_ok": True, "data_ok": True}) == "success"
    assert module.connection_badge_color({"trading_ok": False, "data_ok": True}) == "danger"


def test_api_health_matches_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    expected = module.load_screener_health()
    client = module.app.server.test_client()
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = json.loads(response.data.decode("utf-8"))
    assert payload == expected
