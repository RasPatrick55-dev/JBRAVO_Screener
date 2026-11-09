"""Offline display GET regressions: no app import, credentials or external calls."""
from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import Mock

from flask import Flask, Response, g, jsonify, request
import pytest


pytestmark = pytest.mark.alpaca_optional
APP = Path(__file__).resolve().parents[1] / "dashboards/dashboard_app.py"
TREE = ast.parse(APP.read_text(encoding="utf-8"))


def helpers(*names, **overrides):
    namespace = dict(Any=Any, Mapping=Mapping, Response=Response, g=g, request=request,
                     math=math, time=SimpleNamespace(perf_counter=Mock()))
    nodes = []
    for original in TREE.body:
        if isinstance(original, ast.FunctionDef) and original.name in names:
            node = deepcopy(original)
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), namespace)
    namespace.update(overrides)
    return namespace


def test_display_telemetry_never_connects_or_inserts():
    database = Mock()
    database.get_db_conn.side_effect = AssertionError("display telemetry connected")
    namespace = helpers("_record_trades_api_request", db=database)
    assert namespace["_record_trades_api_request"](
        endpoint="/api/trades/stats", params={"range": "all"},
        source="postgres", rows_returned=5,
    ) is False
    assert database.mock_calls == []


@pytest.mark.parametrize("route", ["stats", "leaderboard", "latest"])
@pytest.mark.parametrize("ready", [False, True])
def test_display_routes_keep_financial_payload_without_telemetry(route, ready):
    database = Mock()
    database.get_db_conn.side_effect = AssertionError("telemetry must not connect")
    rows = [{"symbol": "SYNTHETIC", "totalPL": None}] if ready else []
    policy = {"policy": {"policy_revision": 6}}
    frame = SimpleNamespace(attrs=dict(trade_policy=policy,
        policy_validation_status="VALIDATED_POLICY", excluded_count=1,
        unresolved_trade_ids=[] if ready else [110, 125, 131, 188],
        accounting_input_status="COMPLETE" if ready else "INCOMPLETE"))
    namespace = helpers("_record_trades_api_request", f"api_trades_{route}",
        db=database, _json_no_store=deepcopy,
        _load_trades_analytics_frame=lambda: (frame, "postgres", "validated-performance"
            if ready else "trades-db-held-unvalidated-performance"),
        _trades_api_db_ready=lambda *_: ready,
        _parse_trades_range=lambda *_args, **_kwargs: "all",
        _parse_positive_int=lambda *_args, **_kwargs: 25,
        _TRADES_RANGE_ORDER=["all"],
        _build_range_metrics=lambda *_: rows[0],
        _build_leaderboard_rows=lambda *_args, **_kwargs: rows,
        _build_latest_trades_rows=lambda *_args, **_kwargs: rows)
    app = Flask(__name__)
    with app.test_request_context(f"/api/trades/{route}"):
        # Compare with the historical successful telemetry acknowledgement.
        actual = namespace[f"api_trades_{route}"]()
        namespace["_record_trades_api_request"] = lambda **_: True
        baseline = namespace[f"api_trades_{route}"]()
    assert actual == dict(baseline, recorded_to_db=False)
    assert actual["ok"] is ready
    assert actual["accounting_complete"] is False
    assert actual["recovery_complete"] is False
    assert actual["trade_policy"] == policy
    assert actual["performance_status"] == (
        "VALIDATED_PERFORMANCE" if ready else "HELD_UNVALIDATED_PERFORMANCE")
    assert actual["rows"] == rows
    assert database.mock_calls == []


def timing_app(values):
    namespace = helpers("_api_request_started", "_api_server_timing")
    namespace["time"].perf_counter.side_effect = values
    app = Flask(__name__)
    app.before_request(namespace["_api_request_started"])
    app.after_request(namespace["_api_server_timing"])
    app.add_url_rule("/api/example", view_func=lambda: jsonify(ok=True, rows=[]))
    app.add_url_rule("/", endpoint="root", view_func=lambda: Response("static"))
    app.add_url_rule("/api/stream", endpoint="stream", view_func=lambda:
                     Response(iter(["data: synthetic\n\n"]), mimetype="text/event-stream"))
    return app, namespace


def test_timing_is_fixed_numeric_and_preserves_body_and_headers():
    app, _ = timing_app([100, 101.2345])
    response = app.test_client().get("/api/example?secret=not-for-header")
    assert response.headers["Server-Timing"] == "web_handler;dur=1234.500"
    assert response.get_json() == {"ok": True, "rows": []}
    assert response.content_type == "application/json"
    assert "secret" not in response.headers["Server-Timing"]


def test_timing_is_request_scoped():
    app, _ = timing_app([100, 101, 200, 200.25])
    client = app.test_client()
    assert client.get("/api/example").headers["Server-Timing"] == "web_handler;dur=1000.000"
    assert client.get("/api/example").headers["Server-Timing"] == "web_handler;dur=250.000"


def test_static_and_streaming_responses_have_no_misleading_timing():
    app, namespace = timing_app([100])
    client = app.test_client()
    assert "Server-Timing" not in client.get("/").headers
    namespace["time"].perf_counter.assert_not_called()
    assert "Server-Timing" not in client.get("/api/stream").headers
    namespace["time"].perf_counter.assert_called_once()


@pytest.mark.parametrize("end", [float("nan"), float("inf"), float("-inf"), 99])
def test_invalid_clock_does_not_emit_timing(end):
    app, _ = timing_app([100, end])
    response = app.test_client().get("/api/example")
    assert "Server-Timing" not in response.headers
    assert response.status_code == 200


def test_timing_without_request_start_is_omitted():
    namespace = helpers("_api_server_timing")
    app = Flask(__name__)
    with app.test_request_context("/api/example"):
        response = namespace["_api_server_timing"](Response("{}"))
    assert "Server-Timing" not in response.headers
    namespace["time"].perf_counter.assert_not_called()


def test_flask_hooks_are_registered():
    for name, decorator in [("_api_request_started", "before_request"),
                            ("_api_server_timing", "after_request")]:
        node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                    and node.name == name)
        assert [ast.unparse(value) for value in node.decorator_list] == [f"server.{decorator}"]


def test_retired_state_stream_is_empty_no_store_without_loading_sources():
    factory = Mock(side_effect=AssertionError("retired stream loaded sources"))
    stream = Mock(side_effect=AssertionError("retired stream created a generator"))
    database = Mock(side_effect=AssertionError("retired stream queried database"))
    provider = Mock(side_effect=AssertionError("retired stream queried provider"))
    namespace = helpers("api_execute_state_stream", _execute_state_payload=factory,
        _execute_sse_response=stream, db=database, _alpaca_rest_get_json=provider,
        _pythonanywhere_api_get_json=provider)
    app = Flask(__name__)
    app.add_url_rule("/api/execute/state/stream",
                     view_func=namespace["api_execute_state_stream"])
    response = app.test_client().get("/api/execute/state/stream")
    assert response.status_code == 204
    assert response.data == b""
    assert response.headers["Cache-Control"] == "no-store, max-age=0"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["Expires"] == "0"
    for dependency in (factory, stream, database, provider):
        assert dependency.mock_calls == []


def test_finite_state_route_still_loads_exact_payload_once():
    payload = {"ok": True, "summary": {"performance_status": "HELD_OR_UNKNOWN"},
               "orders": {"rows": []}, "trailing_stops": {"rows": []}}
    factory = Mock(return_value=payload)
    stream = Mock(side_effect=AssertionError("finite route created a stream"))
    namespace = helpers("api_execute_state", "_json_no_store", jsonify=jsonify,
        _execute_state_payload=factory, _execute_sse_response=stream)
    app = Flask(__name__)
    app.add_url_rule("/api/execute/state", view_func=namespace["api_execute_state"])
    response = app.test_client().get("/api/execute/state")
    assert response.status_code == 200
    assert response.get_json() == payload
    assert "no-store" in response.headers["Cache-Control"]
    factory.assert_called_once_with()
    stream.assert_not_called()


def test_retirement_keeps_exact_existing_route_registration():
    node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                and node.name == "api_execute_state_stream")
    assert [ast.unparse(value) for value in node.decorator_list] == [
        "server.route('/api/execute/state/stream')"]
