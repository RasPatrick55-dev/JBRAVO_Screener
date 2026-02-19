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


@pytest.mark.alpaca_optional
def test_trades_leaderboard_modes_sorting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    now = pd.Timestamp.now(tz="UTC")
    mocked_trades = pd.DataFrame(
        [
            {"symbol": "AAA", "realized_pnl": 12.0, "sort_ts": now - pd.Timedelta(days=2)},
            {"symbol": "BBB", "realized_pnl": 40.0, "sort_ts": now - pd.Timedelta(days=1)},
            {"symbol": "CCC", "realized_pnl": -7.0, "sort_ts": now - pd.Timedelta(days=3)},
            {"symbol": "DDD", "realized_pnl": -30.0, "sort_ts": now - pd.Timedelta(hours=8)},
            {"symbol": "EEE", "realized_pnl": 5.0, "sort_ts": now - pd.Timedelta(hours=4)},
        ]
    )

    monkeypatch.setattr(
        module,
        "_load_trades_analytics_frame",
        lambda: (mocked_trades.copy(), "postgres", "trades"),
    )
    monkeypatch.setattr(module, "_record_trades_api_request", lambda **_: True)

    client = module.app.server.test_client()

    winners_response = client.get("/api/trades/leaderboard?range=all&mode=winners&limit=10")
    assert winners_response.status_code == 200
    winners_payload = winners_response.get_json()
    assert winners_payload.get("mode") == "winners"
    winners_rows = winners_payload.get("rows") or []
    assert winners_rows
    winner_pls = [float(row.get("pl") or 0.0) for row in winners_rows]
    assert all(pl > 0 for pl in winner_pls)
    assert winner_pls == sorted(winner_pls, reverse=True)
    assert all({"rank", "symbol", "pl"}.issubset(set(row.keys())) for row in winners_rows)

    losers_response = client.get("/api/trades/leaderboard?range=all&mode=losers&limit=10")
    assert losers_response.status_code == 200
    losers_payload = losers_response.get_json()
    assert losers_payload.get("mode") == "losers"
    losers_rows = losers_payload.get("rows") or []
    assert losers_rows
    loser_pls = [float(row.get("pl") or 0.0) for row in losers_rows]
    assert all(pl < 0 for pl in loser_pls)
    assert loser_pls == sorted(loser_pls)
    assert all({"rank", "symbol", "pl"}.issubset(set(row.keys())) for row in losers_rows)

    invalid_response = client.get("/api/trades/leaderboard?range=all&mode=not-valid&limit=10")
    assert invalid_response.status_code == 400
    invalid_payload = invalid_response.get_json()
    assert invalid_payload.get("ok") is False
    assert invalid_payload.get("error") == "invalid_mode"


@pytest.mark.alpaca_optional
def test_trades_leaderboard_limit_honors_requested_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    now = pd.Timestamp.now(tz="UTC")
    mocked_trades = pd.DataFrame(
        [
            {
                "symbol": f"SYM{index:02d}",
                "realized_pnl": float(100 - index),
                "sort_ts": now - pd.Timedelta(minutes=index),
            }
            for index in range(30)
        ]
    )

    monkeypatch.setattr(
        module,
        "_load_trades_analytics_frame",
        lambda: (mocked_trades.copy(), "postgres", "trades"),
    )
    monkeypatch.setattr(module, "_record_trades_api_request", lambda **_: True)

    client = module.app.server.test_client()
    requested_limit = 20
    available_winners = (
        mocked_trades.groupby("symbol", dropna=False)["realized_pnl"].sum().gt(0).sum()
    )
    response = client.get(f"/api/trades/leaderboard?range=all&mode=winners&limit={requested_limit}")

    assert response.status_code == 200
    payload = response.get_json()
    rows = payload.get("rows") or []
    assert payload.get("mode") == "winners"
    assert payload.get("range") == "all"
    assert payload.get("limit") == requested_limit
    assert len(rows) == min(requested_limit, int(available_winners))
    assert len(rows) > 10
    assert rows[-1].get("rank") == len(rows)

    pls = [float(row.get("pl") or 0.0) for row in rows]
    assert pls == sorted(pls, reverse=True)


@pytest.mark.alpaca_optional
def test_trades_latest_serializes_nan_as_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    now = pd.Timestamp.now(tz="UTC")
    mocked_trades = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "entry_time": now - pd.Timedelta(days=1),
                "exit_time": now,
                "qty": 10,
                "entry_price": float("nan"),
                "exit_price": float("nan"),
                "realized_pnl": float("nan"),
                "sort_ts": now,
            }
        ]
    )

    monkeypatch.setattr(
        module,
        "_load_trades_analytics_frame",
        lambda: (mocked_trades.copy(), "postgres", "trades"),
    )
    monkeypatch.setattr(module, "_record_trades_api_request", lambda **_: True)

    client = module.app.server.test_client()
    response = client.get("/api/trades/latest?limit=25")

    assert response.status_code == 200
    raw_payload = response.get_data(as_text=True)
    assert "NaN" not in raw_payload

    payload = response.get_json()
    rows = payload.get("rows") or []
    assert len(rows) == 1
    assert rows[0].get("avgEntryPrice") == 0.0
    assert rows[0].get("priceSold") == 0.0
    assert rows[0].get("totalPL") == 0.0
