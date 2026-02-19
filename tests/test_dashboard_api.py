from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
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


def _account_point(ts: str, equity: float) -> dict[str, object]:
    dt = datetime.fromisoformat(ts)
    return {"t": dt.isoformat(), "equity": float(equity), "_dt": dt}


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


@pytest.mark.alpaca_optional
def test_api_logs_tail_default_and_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_dashboard_data(tmp_path)
    logs_dir = tmp_path / "logs"
    lines = [f"2024-01-01 00:00:{idx:02d} [INFO] monitor line {idx}" for idx in range(120)]
    (logs_dir / "monitor.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    module = _reload_dashboard_app(monkeypatch, tmp_path)
    client = module.app.server.test_client()

    tail_response = client.get("/api/logs/monitor.log?tail=55")
    assert tail_response.status_code == 200
    tail_lines = [line for line in tail_response.get_data(as_text=True).splitlines() if line.strip()]
    assert len(tail_lines) == 55
    assert tail_lines[0].endswith("monitor line 65")
    assert tail_lines[-1].endswith("monitor line 119")

    full_response = client.get("/api/logs/monitor.log?full=1")
    assert full_response.status_code == 200
    full_lines = [line for line in full_response.get_data(as_text=True).splitlines() if line.strip()]
    assert len(full_lines) == 120


@pytest.mark.alpaca_optional
def test_account_performance_uses_live_equity_for_daily_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    points = [
        _account_point("2026-02-18T01:00:00+00:00", 1000.0),
        _account_point("2026-02-19T01:00:00+00:00", 1100.0),
    ]

    monkeypatch.setattr(module, "_fetch_account_portfolio_points", lambda **_: (points, "ok"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_db", lambda **_: ([], "db_empty"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_csv", lambda **_: ([], "csv_missing"))

    def _mock_rest(path: str, **_: object):
        if path == "/v2/account":
            return {"equity": "1110.0", "cash": "900.0", "buying_power": "0.0"}, "ok"
        return {}, "request_failed"

    monkeypatch.setattr(module, "_alpaca_rest_get_json", _mock_rest)

    client = module.app.server.test_client()
    response = client.get("/api/account/performance?range=d")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload is not None
    total = payload.get("accountTotal") or {}
    assert float(total.get("equity") or 0.0) == pytest.approx(1110.0)
    assert float(total.get("netChangeUsd") or 0.0) == pytest.approx(10.0)
    assert total.get("equityBasis") == "live"
    assert total.get("performanceBasis") == "live_vs_close_baselines"

    rows = payload.get("rows") or []
    by_period = {str((row or {}).get("period") or "").lower(): row for row in rows}
    daily = by_period.get("daily") or {}
    assert float(daily.get("netChangeUsd") or 0.0) == pytest.approx(10.0)


@pytest.mark.alpaca_optional
def test_account_summary_and_performance_total_share_live_equity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    points = [
        _account_point("2026-02-18T01:00:00+00:00", 1000.0),
        _account_point("2026-02-19T01:00:00+00:00", 1100.0),
    ]

    monkeypatch.setattr(module, "_fetch_account_portfolio_points", lambda **_: (points, "ok"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_db", lambda **_: ([], "db_empty"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_csv", lambda **_: ([], "csv_missing"))
    monkeypatch.setattr(module, "_open_positions_value_from_alpaca", lambda: 250.0)

    def _mock_rest(path: str, **_: object):
        if path == "/v2/account":
            return {"equity": "1110.0", "cash": "900.0", "buying_power": "0.0"}, "ok"
        return {}, "request_failed"

    monkeypatch.setattr(module, "_alpaca_rest_get_json", _mock_rest)

    client = module.app.server.test_client()
    summary_resp = client.get("/api/account/summary")
    perf_resp = client.get("/api/account/performance?range=all")
    assert summary_resp.status_code == 200
    assert perf_resp.status_code == 200

    summary = summary_resp.get_json() or {}
    performance = perf_resp.get_json() or {}
    total = performance.get("accountTotal") or {}

    assert float(summary.get("equity") or 0.0) == pytest.approx(1110.0)
    assert float(total.get("equity") or 0.0) == pytest.approx(float(summary.get("equity") or 0.0))
    assert total.get("equityBasis") == "live"


@pytest.mark.alpaca_optional
def test_account_performance_falls_back_to_last_close_when_live_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)

    points = [
        _account_point("2026-02-18T01:00:00+00:00", 1000.0),
        _account_point("2026-02-19T01:00:00+00:00", 1100.0),
    ]

    monkeypatch.setattr(module, "_fetch_account_portfolio_points", lambda **_: (points, "ok"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_db", lambda **_: ([], "db_empty"))
    monkeypatch.setattr(module, "_account_portfolio_points_from_csv", lambda **_: ([], "csv_missing"))
    monkeypatch.setattr(module, "_alpaca_rest_get_json", lambda *_, **__: ({}, "request_failed"))

    client = module.app.server.test_client()
    response = client.get("/api/account/performance?range=d")
    assert response.status_code == 200

    payload = response.get_json() or {}
    total = payload.get("accountTotal") or {}
    assert float(total.get("equity") or 0.0) == pytest.approx(1100.0)
    assert float(total.get("netChangeUsd") or 0.0) == pytest.approx(100.0)
    assert total.get("equityBasis") == "last_close"
