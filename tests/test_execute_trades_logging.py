import os
import sys
import json
from pathlib import Path
from datetime import datetime
from importlib import import_module

import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

dummy_indicators = types.ModuleType("indicators")
dummy_indicators.rsi = lambda series: series
dummy_indicators.macd = lambda series: (series, series, series)
sys.modules.setdefault("indicators", dummy_indicators)

os.environ.setdefault("APCA_API_KEY_ID", "test_key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test_secret")
os.environ.setdefault("ALPACA_KEY_ID", os.environ["APCA_API_KEY_ID"])
os.environ.setdefault("ALPACA_SECRET_KEY", os.environ["APCA_API_SECRET_KEY"])
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.fixture(scope="module")
def execute_trades_module():
    return import_module("scripts.execute_trades")


def _install_emit_stub(module, monkeypatch, events_path: Path):
    def fake_emit(evt: str, **kvs):
        payload = {"event": evt, **kvs}
        payload.setdefault("ts", module.utcnow())
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    monkeypatch.setattr(module, "emit", fake_emit)


def test_log_event_appends_valid_json(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path = tmp_path / "execute_events.jsonl"
    _install_emit_stub(module, monkeypatch, events_path)

    module.log_event({"event": "first"})
    module.log_event({"event": "second", "value": 2})

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["event"] == "first"
    assert second["event"] == "second"
    assert second["value"] == "2"

    assert first["component"] == second["component"] == "execute_trades"

    assert "ts" in first and "ts" in second
    datetime.fromisoformat(first["ts"])
    datetime.fromisoformat(second["ts"])

    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second


def test_metrics_backward_compat(execute_trades_module):
    module = execute_trades_module

    original_metrics = module.metrics.copy()
    original_latencies = list(module.order_latencies_ms)
    original_backoff = module.retry_backoff_ms_total
    try:
        for key in module.metrics:
            module.metrics[key] = 0

        module.metrics.update(
            {
                "symbols_processed": 11,
                "orders_submitted": 5,
                "symbols_skipped": 7,
                "api_failures": 2,
                "api_retries": 3,
                "orders_skipped_existing_position": 4,
                "orders_skipped_pending_order": 5,
                "orders_skipped_risk_limit": 6,
                "orders_skipped_market_data": 7,
                "orders_skipped_session_window": 1,
                "orders_skipped_duplicate_candidate": 2,
                "orders_skipped_other": 9,
            }
        )
        module.order_latencies_ms[:] = [100, 120, 140, 200]
        module.retry_backoff_ms_total = 1500

        snapshot = module.build_execute_metrics_snapshot()

        assert snapshot["symbols_processed"] == 11
        assert snapshot["orders_submitted"] == 5
        assert snapshot["symbols_skipped"] == 7
        assert snapshot["api_failures"] == 2
        assert snapshot["api_retries"] == 3

        assert snapshot["orders_skipped_existing_positions"] == 4
        assert snapshot["orders_skipped_pending_orders"] == 5
        assert snapshot["orders_skipped_risk_limits"] == 6
        assert snapshot["orders_skipped_market_data"] == 7
        assert snapshot["orders_skipped_session"] == 1
        assert snapshot["orders_skipped_duplicate"] == 2
        assert snapshot["orders_skipped_other"] == 9

        assert snapshot["order_latency_ms_p50"] == 130
        assert snapshot["order_latency_ms_p95"] == 191
        assert snapshot["retry_backoff_ms_sum"] == 1500
    finally:
        module.metrics.update(original_metrics)
        module.order_latencies_ms[:] = original_latencies
        module.retry_backoff_ms_total = original_backoff


def test_event_json_schema(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path = tmp_path / "execute_events.jsonl"
    _install_emit_stub(module, monkeypatch, events_path)

    module.log_order_submit_event(
        symbol="AAPL",
        side="buy",
        qty=10,
        limit_price=123.45,
        attempt=1,
        session="pre-market",
    )
    module.log_order_final_event(
        symbol="AAPL",
        attempt=1,
        status="filled",
        latency_ms=120,
        filled_avg_price=123.4,
    )
    module.log_retry_event(
        phase="submit",
        symbol="AAPL",
        attempt=2,
        reason_code="API_LIMIT",
        backoff_ms=500,
    )
    module.log_api_error_event("submit", "AAPL", "boom")
    module.skip("AAPL", "RISK_LIMIT", "Risk guardrail tripped", threshold="max")

    events = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]

    assert all(event.get("component") == "execute_trades" and "ts" in event for event in events)

    submit_event = events[0]
    assert submit_event["event"] == "ORDER_SUBMIT"
    for key in ("symbol", "side", "qty", "limit_price", "attempt", "session"):
        assert key in submit_event

    final_event = events[1]
    assert final_event["event"] == "ORDER_FINAL"
    for key in ("symbol", "attempt", "status", "latency_ms"):
        assert key in final_event
    assert final_event["filled_avg_price"] == "123.4"

    retry_event = events[2]
    assert retry_event["event"] == "RETRY"
    for key in ("phase", "symbol", "attempt", "reason_code", "backoff_ms"):
        assert key in retry_event

    error_event = events[3]
    assert error_event["event"] == "API_ERROR"
    for key in ("phase", "symbol", "message"):
        assert key in error_event

    skip_event = events[4]
    assert skip_event["event"] == "CANDIDATE_SKIPPED"
    for key in ("symbol", "reason_code", "reason_text", "kvs"):
        assert key in skip_event


def _setup_market_guard_test(module, monkeypatch, tmp_path):
    events_path = tmp_path / "execute_events.jsonl"
    metrics_path = tmp_path / "data" / "execute_metrics.json"

    _install_emit_stub(module, monkeypatch, events_path)
    monkeypatch.setattr(module, "metrics_path", str(metrics_path))
    monkeypatch.setattr(module, "metrics", module.metrics.copy())
    monkeypatch.setattr(module, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("ALPACA_KEY_ID", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("APCA_API_KEY_ID", "test_key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test_secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    return events_path, metrics_path


def _stub_execution_pipeline(module, monkeypatch, *, submit_return=None):
    submit_calls = {"count": 0}

    def submit_stub():
        submit_calls["count"] += 1
        return [] if submit_return is None else submit_return

    monkeypatch.setattr(module, "submit_trades", submit_stub)
    monkeypatch.setattr(module, "attach_trailing_stops", lambda: None)
    monkeypatch.setattr(module, "daily_exit_check", lambda: None)
    monkeypatch.setattr(module, "save_open_positions_csv", lambda: None)
    monkeypatch.setattr(module, "update_trades_log", lambda: None)

    return submit_calls


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _install_trading_modules(
    monkeypatch,
    *,
    trading_open: bool | None = None,
    trading_error: Exception | None = None,
    rest_open: bool | None = None,
    rest_error: Exception | None = None,
):
    trading_pkg = types.ModuleType("alpaca.trading")
    client_mod = types.ModuleType("alpaca.trading.client")

    class DummyClock:
        def __init__(self, is_open: bool | None):
            self.is_open = is_open

    class DummyTradingClient:
        def __init__(self, *args, **kwargs):
            if trading_error:
                raise trading_error
            self._clock = DummyClock(trading_open)

        def get_clock(self):
            if trading_error:
                raise trading_error
            return self._clock

    client_mod.TradingClient = DummyTradingClient
    trading_pkg.client = client_mod
    alpaca_pkg = types.ModuleType("alpaca")
    alpaca_pkg.trading = trading_pkg

    monkeypatch.setitem(sys.modules, "alpaca", alpaca_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_mod)

    if rest_open is not None or rest_error is not None:
        rest_mod = types.ModuleType("alpaca_trade_api")

        class DummyRestClient:
            def __init__(self, *args, **kwargs):
                if rest_error:
                    raise rest_error
                self._clock = DummyClock(rest_open)

            def get_clock(self):
                if rest_error:
                    raise rest_error
                return self._clock

        rest_mod.REST = DummyRestClient
        monkeypatch.setitem(sys.modules, "alpaca_trade_api", rest_mod)


def test_guard_open_always_emitted(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path, _ = _setup_market_guard_test(module, monkeypatch, tmp_path)
    submit_calls = _stub_execution_pipeline(module, monkeypatch)

    _install_trading_modules(monkeypatch, trading_open=True)

    exit_code = module.main([])
    assert exit_code == 0
    assert submit_calls["count"] == 1, "Execution pipeline should run when market is open"

    events = _read_events(events_path)
    assert [event["event"] for event in events] == [
        "RUN_START",
        "MARKET_GUARD_STATUS",
        "RUN_END",
    ]

    run_start = events[0]
    assert run_start["event"] == "RUN_START"
    assert run_start["component"] == "execute_trades"
    assert run_start["force"] == "false"

    guard = events[1]
    assert guard["component"] == "execute_trades"
    assert guard["status"] == "OPEN"
    assert guard["clock_source"] == "TradingClient"
    assert guard["force"] == "false"
    assert guard.get("error") == ""

    run_end = events[-1]
    assert run_end["event"] == "RUN_END"
    assert run_end["component"] == "execute_trades"
    assert run_end["status"] == "ok"
    assert run_end.get("error") is None


def test_guard_closed_aborts(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path, metrics_path = _setup_market_guard_test(module, monkeypatch, tmp_path)

    def fail_submit_trades():
        raise AssertionError("submit_trades should not run when market is closed")

    monkeypatch.setattr(module, "submit_trades", fail_submit_trades)
    monkeypatch.setattr(module, "attach_trailing_stops", lambda: None)
    monkeypatch.setattr(module, "daily_exit_check", lambda: None)
    monkeypatch.setattr(module, "save_open_positions_csv", lambda: None)
    monkeypatch.setattr(module, "update_trades_log", lambda: None)
    _install_trading_modules(monkeypatch, trading_open=False)

    exit_code = module.main([])
    assert exit_code == 0

    assert metrics_path.exists()
    metrics_payload = json.loads(metrics_path.read_text("utf-8"))
    assert metrics_payload.get("run_aborted_reason") == "MARKET_CLOSED"

    events = _read_events(events_path)
    assert [event["event"] for event in events] == [
        "RUN_START",
        "MARKET_GUARD_STATUS",
        "RUN_ABORT",
        "RUN_END",
    ]

    guard = events[1]
    assert guard["component"] == "execute_trades"
    assert guard["status"] == "CLOSED"
    assert guard["clock_source"] == "TradingClient"
    assert guard["force"] == "false"
    assert guard.get("error") == ""

    abort_event = events[2]
    assert abort_event["event"] == "RUN_ABORT"
    assert abort_event["component"] == "execute_trades"
    assert abort_event["reason_code"] == "MARKET_CLOSED"

    run_end = events[-1]
    assert run_end["status"] == "aborted"
    assert run_end["component"] == "execute_trades"
    assert run_end.get("error") is None


def test_guard_closed_force_runs(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path, _ = _setup_market_guard_test(module, monkeypatch, tmp_path)
    submit_calls = _stub_execution_pipeline(module, monkeypatch)

    _install_trading_modules(monkeypatch, trading_open=False)

    exit_code = module.main(["--force"])
    assert exit_code == 0
    assert submit_calls["count"] == 1, "Execution pipeline should run with --force"

    events = _read_events(events_path)
    assert [event["event"] for event in events] == [
        "RUN_START",
        "MARKET_GUARD_STATUS",
        "RUN_END",
    ]

    guard = events[1]
    assert guard["component"] == "execute_trades"
    assert guard["status"] == "CLOSED"
    assert guard["clock_source"] == "TradingClient"
    assert guard["force"] == "true"
    assert guard.get("error") == ""

    assert all(event["event"] != "RUN_ABORT" for event in events)

    run_end = events[-1]
    assert run_end["status"] == "ok"
    assert run_end["component"] == "execute_trades"
    assert run_end.get("error") is None


def test_guard_unknown_failopen(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path, _ = _setup_market_guard_test(module, monkeypatch, tmp_path)
    submit_calls = _stub_execution_pipeline(module, monkeypatch)

    error = RuntimeError("clock boom")
    _install_trading_modules(
        monkeypatch,
        trading_error=error,
        rest_error=error,
        rest_open=None,
    )

    exit_code = module.main([])
    assert exit_code == 0
    assert submit_calls["count"] == 1, "Execution pipeline should continue on guard failure"

    events = _read_events(events_path)
    assert [event["event"] for event in events] == [
        "RUN_START",
        "MARKET_GUARD_STATUS",
        "RUN_END",
    ]

    guard_event = events[1]
    assert guard_event["component"] == "execute_trades"
    assert guard_event["status"] == "UNKNOWN"
    assert guard_event["clock_source"] == "RESTv2"
    assert guard_event["force"] == "false"
    assert "clock boom" in guard_event.get("error", "")

    run_end = events[-1]
    assert run_end["status"] == "ok"
    assert run_end["component"] == "execute_trades"
    assert run_end.get("error") is None
