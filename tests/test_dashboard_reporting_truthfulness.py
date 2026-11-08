"""Offline dashboard regression tests using synthetic data only.

Extract reviewed helpers/handlers without importing the Dash application, loading
credentials, starting services, or calling a broker/database/provider endpoint.
Run with --noconftest and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import math
from pathlib import Path
import re
import shlex
from types import SimpleNamespace
from typing import Any, Mapping, Optional
from unittest.mock import Mock
from urllib.parse import urlsplit

import pandas as pd
import pytest

pytestmark = pytest.mark.alpaca_optional
APP = Path(__file__).resolve().parents[1] / "dashboards/dashboard_app.py"
TREE = ast.parse(APP.read_text(encoding="utf-8"))
NOW = datetime(2030, 1, 2, 12, tzinfo=timezone.utc)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


def helpers(*names, **overrides):
    namespace = dict(
        Any=Any,
        Mapping=Mapping,
        Optional=Optional,
        datetime=Clock,
        timezone=timezone,
        timedelta=timedelta,
        Path=Path,
        math=math,
        re=re,
        shlex=shlex,
        json=json,
        pd=pd,
        os=SimpleNamespace(environ={}),
        urlsplit=urlsplit,
        _json_no_store=lambda payload: payload,
        _account_api_is_paper_mode=lambda: True,
        _EXECUTE_STATUS_SCOPES={"all", "open", "closed"},
        _EXECUTE_LSX_SCOPES={"all", "l", "s", "e", "x"},
    )
    required = set(names) | {"_to_float", "_coerce_datetime_utc"}
    nodes = []
    for original in TREE.body:
        if isinstance(original, ast.FunctionDef) and original.name in required:
            node = deepcopy(original)
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(original, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id
            in {
                "_MONITOR_LOG_LINE_PATTERNS",
                "_ACCOUNT_OPEN_ORDER_STATUSES",
            }
            for target in original.targets
        ):
            nodes.append(deepcopy(original))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), namespace)
    namespace.update(overrides)
    return namespace


STATUS_HELPERS = (
    "_execute_order_status_bucket",
    "_execute_trailing_status_bucket",
    "_execute_status_scope_matches",
    "_execute_parse_scope",
    "_order_identity",
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("done_for_day", "DONE_FOR_DAY"),
        ("pending_replace", "PENDING_REPLACE"),
        ("pending_cancel", "PENDING_CANCEL"),
        ("canceled", "CANCELED"),
        ("cancelled", "CANCELED"),
        ("rejected", "REJECTED"),
        ("expired", "EXPIRED"),
        ("replaced", "REPLACED"),
        ("filled", "FILLED"),
        ("partially_filled", "PARTIAL"),
        ("closed", "CLOSED"),
        ("future_broker_state", "FUTURE_BROKER_STATE"),
        ("", "UNKNOWN"),
    ],
)
def test_order_and_trailing_states_do_not_invent_success(raw, expected):
    ns = helpers(*STATUS_HELPERS)
    assert ns["_execute_order_status_bucket"](raw) == expected
    assert ns["_execute_trailing_status_bucket"](raw) == expected


def test_broker_state_wins_over_historical_event_and_pending_is_not_terminal():
    ns = helpers(*STATUS_HELPERS)
    assert ns["_execute_trailing_status_bucket"]("canceled", "TRAIL_TRIGGER") == "CANCELED"
    assert ns["_execute_order_status_bucket"]("done_for_day", "BUY_FILL") == "DONE_FOR_DAY"
    assert ns["_execute_order_status_bucket"]("pending_cancel", "BUY_CANCELLED") == "PENDING_CANCEL"
    assert ns["_execute_trailing_status_bucket"]("", "TRAIL_TRIGGER") == "TRIGGERED"
    assert ns["_execute_trailing_status_bucket"]("new", "TRAIL_SUBMIT") == "ACTIVE"
    for value in ("PENDING_REPLACE", "PENDING_CANCEL", "DONE_FOR_DAY"):
        assert ns["_execute_status_scope_matches"](value, "open")
        assert not ns["_execute_status_scope_matches"](value, "closed")
    for value in ("CANCELED", "REJECTED", "EXPIRED", "REPLACED", "FILLED"):
        assert ns["_execute_status_scope_matches"](value, "closed")
        assert not ns["_execute_status_scope_matches"](value, "open")


def replacement_chain():
    # Fictional linked orders, not two proven independent stop obligations.
    return [
        dict(
            id="fictional-old",
            symbol="FICT",
            side="sell",
            type="trailing_stop",
            qty="2",
            trail_percent="3",
            status="pending_replace",
            replaced_by="fictional-new",
            submitted_at=NOW.isoformat(),
        ),
        dict(
            id="fictional-new",
            symbol="FICT",
            side="sell",
            type="trailing_stop",
            qty="2",
            trail_percent="2",
            status="new",
            replaces="fictional-old",
            submitted_at=NOW.isoformat(),
        ),
    ]


def test_account_preserves_raw_status_and_both_replacement_ids():
    ns = helpers(
        *STATUS_HELPERS,
        "api_account_open_orders",
        request=SimpleNamespace(args={}),
        _parse_positive_int=lambda *a, **kw: 50,
        _alpaca_rest_get_json=Mock(return_value=(replacement_chain(), "ok")),
        _order_price_or_stop=lambda raw: raw["trail_percent"],
        source_contracts=SimpleNamespace(number=lambda value: value),
    )
    result = ns["api_account_open_orders"]()
    assert result["ok"] and len(result["rows"]) == 2
    old, new = result["rows"]
    assert old["status"] == old["raw_status"] == "pending_replace"
    assert old["order_id"] == new["replaces"] == "fictional-old"
    assert new["order_id"] == old["replaced_by"] == "fictional-new"


@pytest.mark.parametrize("trailing", [False, True])
def test_execute_broker_and_serialized_rows_keep_replacement_identity(trailing):
    reader = "_execute_trailing_from_alpaca" if trailing else "_execute_orders_from_alpaca"
    serializer = "_execute_filter_trailing_rows" if trailing else "_execute_filter_order_rows"
    ns = helpers(
        *STATUS_HELPERS,
        reader,
        serializer,
        "_execute_side_label",
        "_execute_type_label",
        "_execute_lsx_matches",
        "_execute_query_matches",
        "_execute_iso_utc",
        _alpaca_rest_get_json=Mock(return_value=(replacement_chain(), "ok")),
        _execute_trail_text=lambda raw: "2%",
        _execute_stop_price_from_raw=lambda raw: None,
        _execute_parent_leg=lambda raw, order_id: order_id,
        _execute_limit_stop_trail=lambda raw: "2%",
        _execute_notes_from_alpaca_order=lambda *a, **kw: "",
    )
    rows, _ = ns[reader](fetch_limit=10, status_scope="open")
    output = ns[serializer](rows, status_scope="open", query="fictional-old", lsx="all", limit=10)
    assert len(output) == 2
    assert output[0]["status"] == "PENDING_REPLACE"
    assert output[0]["raw_status"] == "pending_replace"
    assert output[0]["replaced_by"] == output[1]["order_id"] == "fictional-new"
    assert output[1]["replaces"] == output[0]["order_id"] == "fictional-old"


def test_db_enrichment_cannot_overwrite_broker_status_or_chain():
    ns = helpers(
        "_execute_enrich_alpaca_orders_with_db", _execute_is_generic_order_note=lambda _: False
    )
    broker = dict(
        order_id="fictional-old",
        status="PENDING_REPLACE",
        raw_status="pending_replace",
        replaces=None,
        replaced_by="fictional-new",
        _ts=NOW,
    )
    stale_db = dict(
        order_id="fictional-old",
        status="FILLED",
        raw_status="filled",
        replaces="unrelated",
        replaced_by="unrelated",
        _ts=NOW - timedelta(days=1),
    )
    row = ns["_execute_enrich_alpaca_orders_with_db"]([broker], [stale_db])[0]
    for key in ("status", "raw_status", "replaces", "replaced_by"):
        assert row[key] == broker[key]


def test_merge_does_not_reintroduce_stale_replacement_links():
    ns = helpers("_execute_merge_order_rows", _execute_is_generic_order_note=lambda _: False)
    historical = dict(
        order_id="fictional-order", raw_status="pending_replace", replaced_by="stale-id"
    )
    current = dict(
        order_id="fictional-order", status="FILLED", raw_status="filled", replaced_by=None
    )
    row = ns["_execute_merge_order_rows"]([historical], [current])[0]
    assert row["replaced_by"] is None and row["raw_status"] == "filled"


def test_broker_summary_counts_do_not_include_session_end_or_pending_cancellation():
    ns = helpers(
        *STATUS_HELPERS,
        "_execute_summary_counts_from_alpaca",
        _alpaca_rest_get_json=lambda *a, **kw: (
            [
                dict(status=status)
                for status in (
                    "done_for_day",
                    "pending_cancel",
                    "pending_replace",
                    "canceled",
                    "rejected",
                    "filled",
                )
            ],
            "ok",
        ),
    )
    result, _ = ns["_execute_summary_counts_from_alpaca"](None)
    assert result == dict(submitted=6, filled=1, rejected=1)


@pytest.mark.parametrize("trailing", [False, True])
def test_database_fallback_preserves_recorded_status_and_lineage(trailing):
    reader = "_execute_trailing_from_db" if trailing else "_execute_orders_from_db"
    raw = replacement_chain()[0]
    records = [
        dict(
            order_id=raw["id"],
            status="pending_replace",
            event_type="TRAIL_SUBMIT",
            symbol="FICT",
            qty=2,
            event_time=NOW,
            raw=raw,
        )
    ]
    ns = helpers(
        *STATUS_HELPERS,
        reader,
        "_execute_iso_utc",
        "_execute_side_label",
        "_execute_type_label",
        _db_fetch_all=Mock(return_value=records),
        _db_fetch_all_strict=Mock(return_value=records),
        _coerce_json_mapping=lambda value: value,
        _execute_trail_text=lambda raw: "3%",
        _execute_stop_price_from_raw=lambda raw: None,
        _execute_parent_leg=lambda raw, order_id: order_id,
        _execute_limit_stop_trail=lambda raw: "3%",
        _execute_notes_from_row=lambda *a, **kw: "",
    )
    rows, _ = ns[reader](10)
    assert rows[0]["status"] == "PENDING_REPLACE" and rows[0]["raw_status"] == "pending_replace"
    assert rows[0]["order_id"] == "fictional-old" and rows[0]["replaced_by"] == "fictional-new"


USAGE_HELPERS = (
    "_percent_used",
    "_usage_metric",
    "_usage_first",
    "_pythonanywhere_file_storage_usage",
    "_pythonanywhere_postgres_usage",
    "_pythonanywhere_cpu_usage",
    "api_pythonanywhere_resources",
)


def test_storage_prefers_bytes_over_old_clamped_percentage_and_records_age():
    snapshot = dict(
        generated_at=(NOW - timedelta(days=4)).isoformat(),
        _source="fictional-snapshot",
        file_storage=dict(used_bytes=1536, limit_bytes=1024, percent=100),
        postgres_storage=dict(pg_used_bytes=0, pg_quota_bytes=1024, percent=35),
    )
    ns = helpers(
        *USAGE_HELPERS,
        _pythonanywhere_usage_snapshot=lambda: snapshot,
        _pythonanywhere_api_get_json=lambda _: None,
    )
    disk = ns["_pythonanywhere_file_storage_usage"]()
    assert disk["value"] == 150 and disk["used"] == 1536 and disk["quota"] == 1024
    assert disk["over_quota"] is disk["stale"] is True
    assert disk["status"] == "stale" and disk["age_seconds"] == 4 * 86400
    assert (
        disk["captured_at"] == snapshot["generated_at"] and disk["source"] == "fictional-snapshot"
    )
    postgres = ns["_pythonanywhere_postgres_usage"]()
    assert postgres["value"] == postgres["used"] == 0
    assert postgres["over_quota"] is False


@pytest.mark.parametrize(
    "used,quota,percent",
    [
        (None, 100, 35),
        (10, 0, 35),
        (-1, 100, 35),
        (float("nan"), 100, 35),
        (10, float("inf"), 35),
        (None, None, float("nan")),
    ],
)
def test_invalid_usage_is_unknown_not_normal_looking(used, quota, percent):
    ns = helpers(*USAGE_HELPERS)
    row = ns["_usage_metric"]("Test", used=used, quota=quota, percent=percent, captured_at=NOW)
    assert row["status"] == "unknown" and row["value"] is None
    assert row["over_quota"] is None


def test_unknown_missing_future_and_live_cpu_are_separate():
    ns = helpers(
        *USAGE_HELPERS,
        _pythonanywhere_usage_snapshot=lambda: None,
        _pythonanywhere_api_get_json=lambda _: None,
    )
    missing = ns["api_pythonanywhere_resources"]()
    assert not missing["ok"] and len(missing["resources"]) == 3
    assert all(row["value"] is None and row["status"] == "unknown" for row in missing["resources"])
    ns["os"].environ["PYTHONANYWHERE_FILE_STORAGE_PCT"] = "145.5"
    env = ns["_pythonanywhere_file_storage_usage"]()
    assert env["value"] == 145.5 and env["status"] == "unknown"
    assert env["captured_at"] is env["stale"] is None
    future = ns["_usage_metric"]("Test", used=10, quota=100, captured_at=NOW + timedelta(seconds=1))
    assert future["status"] == "unknown" and future["age_seconds"] is None
    ns["_pythonanywhere_api_get_json"] = lambda _: dict(
        daily_cpu_total_usage_seconds=0, daily_cpu_limit_seconds=100
    )
    cpu = ns["_pythonanywhere_cpu_usage"]()
    assert cpu["value"] == 0 and cpu["status"] == "fresh" and cpu["captured_at"] == NOW.isoformat()


MONITOR_HELPERS = (
    "_monitor_log_entries_from_text",
    "_parse_monitor_log_line",
    "_format_monitor_timestamp",
    "_format_utc_datetime_for_monitor",
    "_is_position_log_message",
    "_monitor_log_type",
    "_monitor_observation_metadata",
    "_fetch_pythonanywhere_monitor_log_source",
    "_strategy_monitor_log_entries",
    "_positions_logs_live",
)


def test_generic_task_backtest_content_is_never_used_as_monitor_log(tmp_path):
    fetch = Mock(side_effect=AssertionError("No provider log should be fetched"))
    scheduler = Mock(side_effect=AssertionError("Generic task logs must not be inspected"))
    ns = helpers(
        *MONITOR_HELPERS,
        BASE_DIR=tmp_path,
        _fetch_pythonanywhere_file=fetch,
        _pythonanywhere_log_sources=scheduler,
        _alpaca_trailing_logs=lambda **_: [],
    )
    # These exact keywords previously let a backtest masquerade as live closure.
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/pipeline.log").write_text(
        "2030-01-02 11:00:00 [INFO] Closed FICT @67.59 reason=TRAIL_STOP\n"
    )
    assert ns["_positions_logs_live"]() == ([], "none")
    ns["os"].environ["PYTHONANYWHERE_MONITOR_LOG_PATH"] = "/var/log/schedule-log-999.log"
    assert ns["_positions_logs_live"]() == ([], "none")
    fetch.assert_not_called()
    scheduler.assert_not_called()


def test_exact_configured_log_and_local_logs_keep_row_source_run_and_time(tmp_path):
    line = "2030-01-02 11:00:00,123 [INFO] position FICT checked run_id=fictional-run"
    ns = helpers(
        *MONITOR_HELPERS,
        BASE_DIR=tmp_path,
        _fetch_pythonanywhere_file=Mock(return_value=(line, "fictional:monitor.log")),
        _alpaca_trailing_logs=lambda **_: [],
    )
    ns["os"].environ["PYTHONANYWHERE_MONITOR_LOG_PATH"] = "/fictional/logs/monitor.log"
    rows, source = ns["_positions_logs_live"]()
    assert source == "pythonanywhere" and len(rows) == 1
    assert rows[0]["source"] == "fictional:monitor.log" and rows[0]["run_id"] == "fictional-run"
    assert rows[0]["ts_utc"] == "2030-01-02T11:00:00.123000+00:00"
    assert rows[0]["timestamp_kind"] == "log_event" and "_ts" not in rows[0]
    ns["os"].environ.clear()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/monitor.log").write_text(line)
    rows, source = ns["_positions_logs_live"]()
    assert source == "local-fallback" and rows[0]["source"] == "logs/monitor.log"


def test_monitor_receipt_is_observation_not_invented_event_time_and_keeps_pending_chain(tmp_path):
    path = tmp_path / "reports/strategy_execution/monitor/latest.json"
    path.parent.mkdir(parents=True)
    receipt = dict(
        paper_only=True,
        finished_at=(NOW - timedelta(minutes=10)).isoformat(),
        native_stops_trigger_only_regular_hours=True,
        extended_hours_native_protection=False,
        events=[
            dict(
                code="STOP_REPLACEMENT_PENDING",
                old_order_id="fictional-old",
                replacement_order_id="fictional-new",
                protection_confirmed=False,
                occurrences=2,
            )
        ],
    )
    path.write_text(json.dumps(receipt))
    ns = helpers(*MONITOR_HELPERS, BASE_DIR=tmp_path)
    row = ns["_strategy_monitor_log_entries"]()[0]
    assert row["source_kind"] == "monitor_receipt" and row["run_id"] is None
    assert row["timestamp_kind"] == "monitor_report_observation"
    assert row["ts_utc"] == row["report_observed_at"] == receipt["finished_at"]
    assert row["freshness"] == "stale" and row["stale"] is True
    assert row["protection_confirmed"] is False and row["type"] == "warning"
    assert (row["old_order_id"], row["replacement_order_id"]) == ("fictional-old", "fictional-new")
    assert row["native_stops_trigger_only_regular_hours"] is True
    assert row["extended_hours_native_protection"] is False


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"paper_only": True, "finished_at": "bad", "events": []},
        {"paper_only": False, "finished_at": NOW.isoformat(), "events": []},
    ],
)
def test_invalid_monitor_receipt_never_claims_success(tmp_path, receipt):
    path = tmp_path / "reports/strategy_execution/monitor/latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))
    ns = helpers(*MONITOR_HELPERS, BASE_DIR=tmp_path)
    assert ns["_strategy_monitor_log_entries"]() == []


def test_dedupe_does_not_erase_source_or_run_provenance(tmp_path):
    base = dict(
        timestamp="display",
        ts_utc=NOW.isoformat(),
        _ts=NOW.isoformat(),
        type="info",
        message="Same observation",
        source="monitor.log",
        source_kind="monitor_log",
    )
    ns = helpers(
        *MONITOR_HELPERS,
        BASE_DIR=tmp_path,
        _strategy_monitor_log_entries=lambda **_: [
            dict(base, run_id="run-a"),
            dict(base, run_id="run-b"),
        ],
        _fetch_pythonanywhere_monitor_log_source=lambda: None,
        _alpaca_trailing_logs=lambda **_: [
            dict(base, run_id=None, source="alpaca", source_kind="broker_observation")
        ],
    )
    rows, _ = ns["_positions_logs_live"]()
    assert len(rows) == 3 and {row["run_id"] for row in rows} == {"run-a", "run-b", None}


def test_broker_log_keeps_raw_status_and_identity(tmp_path):
    class Status(Enum):
        PENDING_REPLACE = "pending_replace"

    order = SimpleNamespace(**{**replacement_chain()[0], "status": Status.PENDING_REPLACE})
    ns = helpers(
        *MONITOR_HELPERS,
        "_alpaca_trailing_logs",
        "_order_identity",
        "_normalize_symbol",
        "_float_text",
        trading_client=SimpleNamespace(get_orders=Mock(return_value=[order])),
        GetOrdersRequest=lambda **kwargs: kwargs,
    )
    row = ns["_alpaca_trailing_logs"]()[0]
    assert row["source_kind"] == "broker_observation" and row["run_id"] is None
    assert row["order_id"] == "fictional-old" and row["raw_status"] == "pending_replace"
    assert row["replaced_by"] == "fictional-new"
    assert row["timestamp_kind"] == "broker_order_event"


def test_summary_and_stream_share_verified_schedule_and_unknown_window(monkeypatch, tmp_path):
    from dashboards import operational_status

    schedule = dict(status="UNKNOWN", in_window=None, entry_window=None)
    read = Mock(return_value=schedule)
    monkeypatch.setattr(operational_status, "execution_schedule", read)
    ns = helpers(
        "_execute_summary_payload",
        "_execute_iso_utc",
        "_execute_to_int_optional",
        "_execute_to_int",
        BASE_DIR=tmp_path,
        _pythonanywhere_schedule_tasks=lambda: [],
        _execute_summary_from_metrics_table=lambda: (
            dict(
                last_run_utc=NOW.isoformat(),
                submitted=0,
                filled=0,
                rejected=0,
                candidates=0,
                in_window=True,
            ),
            "test",
        ),
        _execute_result_pl_from_db=lambda _: None,
        source_contracts=SimpleNamespace(number=lambda value: value),
    )
    result = ns["_execute_summary_payload"]()
    assert result["execution_schedule"] == schedule and result["in_window"] is None
    assert result["submitted"] == result["filled"] == 0
    read.assert_called_once_with(tmp_path, scheduled_tasks=[])
    route = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "api_execute_summary_stream"
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "_execute_summary_payload"
        for node in ast.walk(route)
    )


def scheduled_executor_fixture():
    path = "/home/test/jbravo-research-tools/FICTIONAL_RELEASE/executor_boot.py"
    command = (
        f"printf '%s\\n' '{'a' * 64}  {path}' | /usr/bin/sha256sum --check --status && "
        f"/usr/bin/timeout --signal=TERM --kill-after=10s 40200s "
        f"/home/test/.virtualenvs/test/bin/python -I -B {path} --scheduled"
    )
    return dict(
        id=1234,
        command=command,
        enabled=True,
        interval="daily",
        hour=7,
        minute=55,
        logfile="/var/log/schedule-log-1234.log",
    )


@pytest.mark.parametrize(
    "entrypoint,name",
    [
        ("market_daily_boot.py", "Market Daily Producer"),
        ("executor_boot.py", "Execute Trades"),
        ("strategy_monitor_boot.py", "Monitor Positions"),
    ],
)
def test_current_wrapper_task_names_use_invoked_entrypoint(entrypoint, name):
    ns = helpers("_task_name_from_command")
    command = scheduled_executor_fixture()["command"].replace("executor_boot.py", entrypoint)
    assert ns["_task_name_from_command"](command) == name
    assert ns["_task_name_from_command"](f"echo '{entrypoint}'") is None


def test_task_names_never_infer_activity_from_description_or_command_text():
    ns = helpers("_task_name_from_command", "_format_task_time", "_normalize_schedule_task")
    assert ns["_task_name_from_command"]("python -m scripts.run_pipeline") == "Run Pipeline"
    assert ns["_task_name_from_command"]("python -c 'print(\"executor_boot.py\")'") is None
    row = ns["_normalize_schedule_task"](
        dict(id=99, description="Execute Trades", command="echo run_pipeline", interval="daily")
    )
    assert row["name_status"] == "unknown" and row["name"] == "Unclassified scheduled task"
    assert row["time"] == "Unknown" and row["enabled"] is None
    assert "command" not in row and "description" not in row


@pytest.mark.parametrize(
    "interval,hour,minute,expected",
    [
        ("hourly", None, None, "Unknown"),
        ("hourly", None, 61, "Unknown"),
        ("hourly", None, 0, "**:00 UTC"),
        ("hourly", None, 15, "**:15 UTC"),
        ("daily", 7, 55, "07:55 UTC"),
        ("daily", 24, 0, "Unknown"),
        ("daily", 7, True, "Unknown"),
        ("unknown", 7, 55, "Unknown"),
    ],
)
def test_task_time_is_observed_or_unknown_never_defaulted(interval, hour, minute, expected):
    ns = helpers("_format_task_time")
    assert ns["_format_task_time"](dict(interval=interval, hour=hour, minute=minute)) == expected


def test_tasks_api_includes_actual_always_on_and_unknown_rows_without_commands():
    scheduled = scheduled_executor_fixture()
    monitor = dict(
        scheduled,
        id=1235,
        command=scheduled["command"].replace("executor_boot.py", "strategy_monitor_boot.py"),
    )
    ns = helpers(
        "_task_name_from_command",
        "_format_task_time",
        "_normalize_schedule_task",
        "api_pythonanywhere_tasks",
        _pythonanywhere_schedule_tasks=lambda: [scheduled, dict(id=1236, command="echo ignored")],
        _pythonanywhere_always_on_tasks=lambda: [monitor],
    )
    result = ns["api_pythonanywhere_tasks"]()
    assert result["ok"] and len(result["tasks"]) == 3 and result["observed_at"] == NOW.isoformat()
    assert result["tasks"][0]["time"] == "07:55 UTC"
    assert result["tasks"][1]["time"] == "Unknown"
    assert result["tasks"][2]["name"] == "Monitor Positions"
    assert result["tasks"][2]["task_kind"] == "always_on"
    assert (
        result["tasks"][2]["frequency"] == "Always-on"
        and result["tasks"][2]["time"] == "Not scheduled"
    )
    assert all("command" not in row for row in result["tasks"])
    ns["_pythonanywhere_schedule_tasks"] = lambda: []
    ns["_pythonanywhere_always_on_tasks"] = lambda: []
    assert ns["api_pythonanywhere_tasks"]()["status"] == "unknown"


def test_execute_log_selector_binds_guarded_wrapper_to_provider_attempt():
    from dashboards.operational_status import scheduled_attempt

    task = scheduled_executor_fixture()
    log = (
        "2030-01-02 07:55:01 Started task\n"
        '{"status":"SPLIT_EXECUTOR_LAUNCHER_REFUSED","reason":"executor_scheduled_start_outside_bound","orders_submitted":0}\n'
        "2030-01-02 07:55:02 Completed task, return code was 1\n"
    )
    fetch = Mock(return_value=(log, "https://fictional.example/var/log/schedule-log-1234.log"))
    ns = helpers(
        "_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: [
            dict(task, id=1233, enabled=False),
            dict(
                id=1236, enabled=True, description="pre-market executor", command="echo unrelated"
            ),
            task,
        ],
        _fetch_pythonanywhere_file=fetch,
    )
    result = ns["_fetch_pythonanywhere_execute_task_log"]()
    fetch.assert_called_once_with(None, task["logfile"])
    assert result[1] == "pythonanywhere:schedule/1234/"
    attempt = scheduled_attempt(result, now=NOW)
    assert attempt["task_id"] == 1234 and attempt["status"] == "GUARDED_NO_ENTRY"
    assert attempt["orders_submitted"] == 0


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        [dict(id=1, enabled=True, command="echo executor_boot.py")],
        [scheduled_executor_fixture(), dict(scheduled_executor_fixture(), logfile="/conflicting.log")],
    ],
)
def test_execute_log_selector_rejects_missing_unverified_or_ambiguous_task(tasks):
    fetch = Mock(side_effect=AssertionError("No unverified log fetch"))
    ns = helpers(
        "_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: tasks,
        _fetch_pythonanywhere_file=fetch,
    )
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None
    fetch.assert_not_called()


def test_execute_log_selector_fetches_detail_when_provider_list_omits_logfile():
    task = scheduled_executor_fixture()
    task.pop("logfile")
    detail = Mock(return_value="/provider/custom-name.log")
    ns = helpers(
        "_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: [task],
        _task_logfile_from_detail=detail,
        _fetch_pythonanywhere_file=lambda *a: (
            "2030-01-02 07:55:01 Started task\n2030-01-02 07:55:02 Completed task, return code was 0",
            "provider source",
        ),
    )
    assert ns["_fetch_pythonanywhere_execute_task_log"]() == (
        "2030-01-02 07:55:01 Started task\n2030-01-02 07:55:02 Completed task, return code was 0",
        "pythonanywhere:schedule/1234/",
    )
    detail.assert_called_once_with("schedule", "1234")


def _attempt_log(day, *, rc=1, running=False, refusal=True):
    text = f"2030-01-{day:02d} 07:55:09 Started task\n"
    if refusal:
        text += '{"status":"SPLIT_EXECUTOR_LAUNCHER_REFUSED","reason":"executor_scheduled_start_outside_bound","orders_submitted":0}\n'
    if not running:
        text += f"2030-01-{day:02d} 07:55:14 Completed task, return code was {rc}\n"
    return text


@pytest.mark.parametrize("reverse", [False, True])
def test_attempt_selection_uses_newest_current_invocation_not_feed_or_stale_env(reverse):
    from dashboards.operational_status import scheduled_attempt

    older = dict(scheduled_executor_fixture(), id=1233, logfile="/older.log")
    current = dict(scheduled_executor_fixture(), id=1234, logfile="/current.log")
    tasks = [older, current]
    if reverse:
        tasks.reverse()
    logs = {"/older.log": _attempt_log(1), "/current.log": _attempt_log(2)}
    ns = helpers(
        "_fetch_pythonanywhere_execute_task_log",
        os=SimpleNamespace(environ={
            "PYTHONANYWHERE_EXECUTE_TASK_ID": "1233",
            "PYTHONANYWHERE_EXECUTE_TASK_LOG_PATH": "/older.log",
            "PYTHONANYWHERE_EXECUTE_TASK_LOG_URL": "https://example.invalid/older.log",
        }),
        _pythonanywhere_schedule_tasks=lambda: tasks,
        _pythonanywhere_api_get_json=lambda *a: pytest.fail("Stale environment task detail queried"),
        _fetch_pythonanywhere_file=lambda url, path: (logs[path], "provider log"),
    )
    result = ns["_fetch_pythonanywhere_execute_task_log"]()
    attempt = scheduled_attempt(result, now=NOW)
    assert attempt["task_id"] == 1234
    assert attempt["started_utc"] == "2030-01-02T07:55:09+00:00"
    assert attempt["status"] == "GUARDED_NO_ENTRY" and attempt["orders_submitted"] == 0
    assert attempt["release_attribution"] == "unverified_historical_attempt"


def test_superseded_env_task_is_not_read_even_when_its_log_would_be_newer():
    current = dict(scheduled_executor_fixture(), id=1234, logfile="/current.log")
    fetch = Mock(return_value=(_attempt_log(1), "provider log"))
    ns = helpers(
        "_fetch_pythonanywhere_execute_task_log",
        os=SimpleNamespace(environ={"PYTHONANYWHERE_EXECUTE_TASK_ID": "9999"}),
        _pythonanywhere_schedule_tasks=lambda: [current],
        _pythonanywhere_api_get_json=lambda *a: pytest.fail("Superseded task queried"),
        _fetch_pythonanywhere_file=fetch,
    )
    assert ns["_fetch_pythonanywhere_execute_task_log"]()[1] == "pythonanywhere:schedule/1234/"
    fetch.assert_called_once_with(None, "/current.log")


@pytest.mark.parametrize("bad_log", ["", "no invocation", _attempt_log(3),
    "2030-01-02 07:55:09 Started task\n2030-01-02 07:55:14 Completed task, return code was unknown"])
def test_incomplete_attempt_coverage_rejects_missing_malformed_or_future_evidence(bad_log):
    from dashboards.operational_status import latest_scheduled_log

    valid = (_attempt_log(1), "pythonanywhere:schedule/1233/")
    invalid = (bad_log, "pythonanywhere:schedule/1234/")
    assert latest_scheduled_log([invalid, valid], now=NOW) is None
    assert latest_scheduled_log([valid, invalid], now=NOW) is None
    assert latest_scheduled_log([invalid], now=NOW) is None


def test_latest_attempt_ties_unknown_but_running_invocation_supersedes_old_completion():
    from dashboards.operational_status import latest_scheduled_log, scheduled_attempt

    old = (_attempt_log(1), "pythonanywhere:schedule/1233/")
    running = (_attempt_log(2, running=True, refusal=False), "pythonanywhere:schedule/1234/")
    assert latest_scheduled_log([running, old], now=NOW) == running
    assert scheduled_attempt(running, now=NOW)["status"] == "RUNNING"
    duplicate_time = (_attempt_log(2), "pythonanywhere:schedule/1235/")
    assert latest_scheduled_log([running, duplicate_time], now=NOW) is None
    assert latest_scheduled_log([(_attempt_log(2), "unbound.log")], now=NOW) is None


def test_duplicate_identical_provider_tasks_are_read_once():
    task = scheduled_executor_fixture()
    fetch = Mock(return_value=(_attempt_log(2), "provider log"))
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: [task, dict(task)],
        _fetch_pythonanywhere_file=fetch)
    assert ns["_fetch_pythonanywhere_execute_task_log"]()[1] == "pythonanywhere:schedule/1234/"
    fetch.assert_called_once()


def test_no_current_task_does_not_revive_an_old_explicit_source():
    fetch = Mock(side_effect=AssertionError("Unverified historical source read"))
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        os=SimpleNamespace(environ={"PYTHONANYWHERE_EXECUTE_TASK_ID": "1233",
            "PYTHONANYWHERE_EXECUTE_TASK_LOG_PATH": "/older.log"}),
        _pythonanywhere_schedule_tasks=lambda: [], _fetch_pythonanywhere_file=fetch)
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None
    fetch.assert_not_called()


def test_excessive_current_task_inventory_is_unknown_without_partial_fetches():
    template = scheduled_executor_fixture()
    tasks = [dict(template, id=2000 + i, logfile=f"/task-{i}.log") for i in range(9)]
    fetch = Mock(side_effect=AssertionError("No truncated task search"))
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: tasks, _fetch_pythonanywhere_file=fetch)
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None
    fetch.assert_not_called()


def test_unavailable_candidate_log_is_not_fabricated_as_an_attempt():
    task = scheduled_executor_fixture()
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: [task],
        _fetch_pythonanywhere_file=lambda *a: None)
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None


def test_missing_detail_logfile_prevents_a_latest_attempt_assertion():
    missing = dict(scheduled_executor_fixture(), id=1233)
    missing.pop("logfile")
    current = dict(scheduled_executor_fixture(), id=1234, logfile="/current.log")
    detail = Mock(return_value=None)
    fetch = Mock(return_value=(_attempt_log(2), "provider log"))
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: [missing, current],
        _task_logfile_from_detail=detail, _fetch_pythonanywhere_file=fetch)
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None
    detail.assert_called_once_with("schedule", "1233")
    fetch.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("failure", ["missing_logfile", "unreadable", "malformed"])
def test_partial_enabled_task_coverage_stays_unknown_in_both_feed_orders(reverse, failure):
    valid = dict(scheduled_executor_fixture(), id=1233, logfile="/valid.log")
    missing = dict(scheduled_executor_fixture(), id=1234, logfile="/missing.log")
    if failure == "missing_logfile":
        missing.pop("logfile")
    tasks = [valid, missing]
    if reverse:
        tasks.reverse()
    def read(url, path):
        if path == "/valid.log":
            return _attempt_log(1), "provider log"
        return None if failure == "unreadable" else ("invalid attempt evidence", "provider log")
    ns = helpers("_fetch_pythonanywhere_execute_task_log",
        _pythonanywhere_schedule_tasks=lambda: tasks,
        _task_logfile_from_detail=lambda *a: None,
        _fetch_pythonanywhere_file=read)
    assert ns["_fetch_pythonanywhere_execute_task_log"]() is None
