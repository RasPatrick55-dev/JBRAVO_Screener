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


@pytest.fixture(scope="module")
def execute_trades_module():
    return import_module("scripts.execute_trades")


def test_log_event_appends_valid_json(tmp_path, monkeypatch, execute_trades_module):
    module = execute_trades_module
    events_path = tmp_path / "execute_events.jsonl"
    monkeypatch.setattr(module, "EVENTS_LOG_PATH", events_path)

    module.log_event({"event": "first"})
    module.log_event({"event": "second", "value": 2})

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["event"] == "first"
    assert second["event"] == "second"
    assert second["value"] == 2

    assert first["run_id"] == second["run_id"] == module.run_id

    datetime.fromisoformat(first["timestamp"])
    datetime.fromisoformat(second["timestamp"])

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
    monkeypatch.setattr(module, "EVENTS_LOG_PATH", events_path)

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

    assert all("run_id" in event and "timestamp" in event for event in events)

    submit_event = events[0]
    assert submit_event["event"] == "ORDER_SUBMIT"
    for key in ("symbol", "side", "qty", "limit_price", "attempt", "session"):
        assert key in submit_event

    final_event = events[1]
    assert final_event["event"] == "ORDER_FINAL"
    for key in ("symbol", "attempt", "status", "latency_ms"):
        assert key in final_event
    assert final_event["filled_avg_price"] == 123.4

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
