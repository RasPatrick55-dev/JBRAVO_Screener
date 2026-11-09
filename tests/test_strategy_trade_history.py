"""Synthetic, offline web projection tests. No native store, credentials or DB."""

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone, tzinfo
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID
import threading

import pytest
from flask import Flask, request

from dashboards import strategy_trade_history as H

pytestmark = pytest.mark.alpaca_optional
ACCOUNT = "00000000-0000-4000-8000-000000000001"
SID = "7" * 64
NOW = datetime(2026, 11, 9, 17, tzinfo=timezone.utc)
ENTRY_TIME = "2026-10-01T10:00:00+00:00"
EXIT_TIME = "2026-10-05T10:00:00+00:00"
IDENTITY = ("synthetic-deployment", "00000000-0000-4000-8000-000000000002")
DATABASE_NAME = "synthetic-db"


def signal():
    return dict(
        account_id=ACCOUNT,
        signal_id=SID,
        symbol="SYNTH",
        state="CLOSED",
        created_at=ENTRY_TIME,
        updated_at=EXIT_TIME,
        payload=dict(
            account_id=ACCOUNT,
            proposal=dict(signal_id=SID, symbol="SYNTH", strategy_id="synthetic-pattern"),
            handoff=dict(
                paper_only=True,
                handoff_id="fixture-handoff",
                strategy_contract={"strategy_id": "synthetic-pattern", "version": 1},
                proposals=[dict(signal_id=SID, symbol="SYNTH", strategy_id="synthetic-pattern")],
            ),
            position={"qty": 40},
        ),
    )


def claim(
    role,
    sequence=0,
    *,
    qty=40,
    price="12.00",
    filled=None,
    status="filled",
    oid=None,
    operation="POST",
    target=None,
    at=EXIT_TIME,
):
    key = H.client_key(ACCOUNT, SID, role, sequence)
    side = "buy" if role == "entry" else "sell"
    order = dict(
        client_order_id=key,
        symbol="SYNTH",
        side=side,
        type="limit",
        qty=qty,
        limit_price=20 if side == "buy" else 11.00,
        time_in_force="day",
        extended_hours=True,
    )
    descriptor = (
        dict(order_id=target, order=order) if operation != "DELETE" else dict(order_id=target)
    )
    lineage = dict(
        signal_id=SID,
        handoff_id="fixture-handoff",
        strategy_contract={"strategy_id": "synthetic-pattern", "version": 1},
    )
    observation = dict(
        order,
        id=oid or f"synthetic-{role}-{sequence}",
        status=status,
        filled_qty=str(qty if filled is None else filled),
        filled_avg_price=price,
        updated_at=at,
        filled_at=at if status == "filled" else None,
    )
    return dict(
        account_id=ACCOUNT,
        signal_id=SID,
        client_order_id=key,
        role=role,
        sequence=sequence,
        operation=operation,
        request_sha256=H.digest(dict(operation=operation, request=descriptor, lineage=lineage)),
        request=descriptor,
        lineage=lineage,
        state="OBSERVED",
        observation=observation,
        claimed_at=at,
        updated_at=at,
    )


def orders():
    return [claim("entry", price="10.00", at=ENTRY_TIME), claim("exit")]


def project(s=None, records=None):
    return H.project_signal(s or signal(), orders() if records is None else records, ACCOUNT, NOW)


def rehash(record):
    record["request_sha256"] = H.digest(
        {key: record[key] for key in ("operation", "request", "lineage")}
    )


def test_closed_fictional_fixture_uses_fills_not_stale_payload_position():
    row = project()
    assert row["status"] == "complete"
    assert row["state"] == "CLOSED"
    assert row["completion_status"] == "FULL_EXIT_OBSERVED"
    assert (row["entry_qty"], row["exited_qty"], row["remaining_qty"]) == (40, 40, 0)
    assert (row["entry_avg_price"], row["exit_avg_price"], row["gross_realized_pnl"]) == (
        10.0,
        12.0,
        80.0,
    )
    assert (row["entry_time"], row["exit_time"]) == (ENTRY_TIME, EXIT_TIME)
    assert row["fees"] is row["net_realized_pnl"] is None
    assert row["fees_status"] == "UNKNOWN_UNALLOCATED"
    assert ACCOUNT not in json.dumps(row)


@pytest.mark.parametrize("state", ["OPEN", "EXIT_PENDING", "UNKNOWN", "DISPUTED"])
def test_native_state_never_fabricated(state):
    s = signal()
    s["state"] = state
    row = project(s)
    assert row["state"] == state and row["status"] == "partial"
    assert row["completion_status"] == "FULL_EXIT_OBSERVED"
    assert row["reason"] == "native_state_reconciliation_required"
    assert row["gross_realized_pnl"] is None


def test_partial_exit_and_multiple_final_exits_weighted_with_decimal():
    records = [orders()[0], claim("partial", qty=10, price="11.50")]
    s = signal()
    s["state"] = "OPEN"
    row = project(s, records)
    assert row["completion_status"] == "PARTIAL_EXIT_OBSERVED"
    assert (row["remaining_qty"], row["gross_realized_pnl"], row["exit_time"]) == (30, None, None)
    records.append(claim("exit", qty=30, price="12.00"))
    row = project(records=records)
    assert row["status"] == "complete"
    assert row["gross_realized_pnl"] == 75.0
    assert row["exit_avg_price"] == pytest.approx(475 / 40)


def test_canceled_partially_filled_orders_use_actual_filled_qty():
    records = [
        claim("entry", qty=50, filled=40, price="10.00", status="canceled", at=ENTRY_TIME),
        claim("exit", qty=50, filled=40, status="canceled"),
    ]
    records[0]["observation"]["filled_at"] = ENTRY_TIME
    records[1]["observation"]["filled_at"] = EXIT_TIME
    row = project(records=records)
    assert row["status"] == "complete" and row["entry_qty"] == 40
    assert row["gross_realized_pnl"] == 80.0


def test_cancel_receipt_is_not_an_extra_fill():
    records = orders()
    cancel = claim("cancel", operation="DELETE", target=records[1]["observation"]["id"])
    cancel["observation"] = deepcopy(records[1]["observation"])
    records.append(cancel)
    row = project(records=records)
    assert row["status"] == "complete" and row["exited_qty"] == 40


def test_newer_cancel_receipt_can_refresh_same_proved_order_not_add_it():
    records = orders()
    records[1]["observation"].update(
        filled_qty="25",
        status="partially_filled",
        filled_at=None,
        updated_at="2026-10-05T09:59:59+00:00",
    )
    cancel = claim("cancel", operation="DELETE", target=records[1]["observation"]["id"])
    cancel["observation"] = deepcopy(orders()[1]["observation"])
    records.append(cancel)
    row = project(records=records)
    assert row["status"] == "complete" and row["gross_realized_pnl"] == 80.0


def test_conflicting_or_regressed_cumulative_cancel_receipt_is_incomplete():
    records = orders()
    cancel = claim("cancel", operation="DELETE", target=records[1]["observation"]["id"])
    cancel["observation"] = deepcopy(records[1]["observation"])
    cancel["observation"].update(
        filled_qty="39", status="canceled", updated_at="2026-10-05T11:00:00Z"
    )
    row = project(records=records + [cancel])
    assert row["status"] == "incomplete" and row["reason"] == "cumulative_fill_conflict"


@pytest.mark.parametrize(
    "change",
    [
        dict(filled_avg_price="18"),
        dict(status="canceled"),
        dict(filled_at="2026-10-05T10:00:01Z"),
        dict(limit_price=22),
    ],
)
def test_newer_cancel_cannot_rewrite_same_quantity_price_clock_or_order(change):
    records = orders()
    cancel = claim("cancel", operation="DELETE", target=records[1]["observation"]["id"])
    cancel["observation"] = deepcopy(records[1]["observation"])
    cancel["observation"].update(updated_at="2026-10-05T11:00:00Z", **change)
    row = project(records=records + [cancel])
    assert row["status"] == "incomplete" and row["gross_realized_pnl"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, r: s.update(account_id="other-account"),
        lambda s, r: s["payload"].update(account_id="other-account"),
        lambda s, r: s["payload"]["proposal"].update(signal_id="8" * 64),
        lambda s, r: s["payload"]["handoff"].update(paper_only=False),
        lambda s, r: r[0].update(client_order_id="jbr2-" + "0" * 40),
        lambda s, r: r[0].update(request_sha256="0" * 64),
        lambda s, r: r[0].update(signal_id="8" * 64),
        lambda s, r: r[0]["observation"].update(symbol="OTHER"),
        lambda s, r: r[0]["observation"].update(side="sell"),
        lambda s, r: r[0]["observation"].update(client_order_id="unrelated"),
        lambda s, r: r[0]["observation"].update(filled_qty="NaN"),
        lambda s, r: r[0]["observation"].update(filled_avg_price="Infinity"),
        lambda s, r: r[0]["observation"].update(filled_qty=True),
        lambda s, r: r[1]["observation"].update(filled_qty="41"),
        lambda s, r: r[1]["observation"].update(qty="41"),
        lambda s, r: r[1]["observation"].update(limit_price=21),
        lambda s, r: r[1]["observation"].update(filled_avg_price="10.99"),
        lambda s, r: r[1]["observation"].update(filled_at=None),
        lambda s, r: r[1]["observation"].update(filled_at="2026-09-01T10:00:00Z"),
        lambda s, r: r[1]["observation"].update(updated_at="2027-01-01T14:20:25Z"),
        lambda s, r: r[1].update(state="UNKNOWN", observation=None),
    ],
)
def test_bad_evidence_fails_closed_without_exposing_payload(mutation):
    s, records = signal(), orders()
    mutation(s, records)
    row = project(s, records)
    assert row["status"] == "incomplete"
    for key in (
        "entry_qty",
        "exited_qty",
        "remaining_qty",
        "entry_avg_price",
        "exit_avg_price",
        "gross_realized_pnl",
    ):
        assert row[key] is None
    assert ACCOUNT not in json.dumps(row)


def test_missing_orders_are_incomplete_not_no_trades():
    assert project(records=[])["status"] == "incomplete"


@pytest.mark.parametrize(
    "state,status", [("ENTRY_PENDING", "new"), ("EXPIRED", "canceled"), ("EXPIRED", "expired")]
)
def test_bound_unfilled_entry_is_distinct_from_missing_evidence(state, status):
    s = signal()
    s["state"] = state
    row = project(s, [claim("entry", filled=0, status=status)])
    assert row["status"] == "partial" and row["completion_status"] == "NO_ENTRY_FILL_OBSERVED"
    assert row["entry_qty"] == 0 and row["gross_realized_pnl"] is None
    assert row["entry_time"] is row["entry_avg_price"] is None


def test_missing_fill_timestamp_does_not_fall_back_to_updated_at():
    records = orders()
    records[0]["observation"]["filled_at"] = None
    row = project(records=records)
    assert row["status"] == "incomplete" and row["entry_time"] is None


def test_replaced_zero_fill_stop_is_not_an_exit():
    records = orders()
    old = claim("stop", filled=0, status="replaced", oid="old-stop")
    new = claim(
        "stop", 1, filled=0, status="canceled", oid="new-stop", operation="PATCH", target="old-stop"
    )
    old["observation"]["replaced_by"] = "new-stop"
    new["observation"]["replaces"] = "old-stop"
    row = project(records=records + [old, new])
    assert row["status"] == "complete" and row["exited_qty"] == 40
    old["observation"]["filled_qty"] = "1"
    row = project(records=records + [old, new])
    assert row["status"] == "incomplete" and row["reason"] == "replacement_fill_ambiguity"


def test_entry_recognition_ownership_is_allowlisted_not_a_trading_permission():
    s = signal()
    s["state"] = "OPEN"
    s["payload"]["execution_owner"] = dict(
        schema="jbravo.execution-entry-recognition.v1",
        scope="ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS",
        signal_id=SID,
        approval_reference="DO-NOT-EXPOSE",
    )
    row = project(s, orders()[:1])
    assert row["ownership_scope"] == "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS"
    assert row["ownership_basis"] == "STORED_SIGNAL_PAYLOAD_NOT_REVALIDATED"
    assert "DO-NOT-EXPOSE" not in json.dumps(row)


@pytest.mark.parametrize("change", ["wrong_strategy", "missing_membership", "duplicate_membership"])
def test_proposal_must_belong_to_its_frozen_strategy_handoff(change):
    s = signal()
    if change == "wrong_strategy":
        s["payload"]["proposal"]["strategy_id"] = "different-strategy"
    elif change == "missing_membership":
        s["payload"]["handoff"]["proposals"] = []
    else:
        s["payload"]["handoff"]["proposals"] *= 2
    row = project(s)
    assert row["status"] == "incomplete" and row["reason"] == "signal_binding_mismatch"
    assert row["strategy_id"] is row["gross_realized_pnl"] is None


@pytest.mark.parametrize(
    "marker,value",
    [
        ("execution_owner", None),
        ("ownership_history", []),
        ("ownership_history", [{"scope": "obsolete"}]),
    ],
)
def test_any_ownership_marker_without_valid_owner_stays_unknown(marker, value):
    s = signal()
    s["payload"][marker] = value
    row = project(s)
    assert row["ownership_scope"] == "UNKNOWN"


@pytest.mark.parametrize("value", [None, False, {}, {"automatic_sells": False}])
def test_even_empty_manual_marker_is_not_an_ordinary_strategy_signal(value):
    s = signal()
    s["payload"]["handoff"]["manual_paper_test"] = value
    assert project(s)["ownership_scope"] == "MANUAL_PAPER_TEST"


@pytest.mark.parametrize("index", [0, 1])
def test_fill_clock_cannot_follow_its_stored_observation_clock(index):
    records = orders()
    records[index]["observation"]["filled_at"] = "2026-10-05T11:00:00Z"
    row = project(records=records)
    assert row["status"] == "incomplete" and row["reason"] == "fill_timestamp_after_observation"
    assert row["entry_time"] is row["exit_time"] is row["gross_realized_pnl"] is None


def run_snapshot(monkeypatch, *, s=None, records=None, overlap=None, client=None):
    monkeypatch.setattr(
        H,
        "read_snapshot",
        lambda *args: ([s or signal()], records or orders(), overlap or [], False),
    )
    return H.snapshot(
        Mock(),
        client or SimpleNamespace(get_account=lambda: SimpleNamespace(id=ACCOUNT)),
        IDENTITY,
        read=lambda fn, **kw: fn(),
        now=NOW,
    )


def test_envelope_and_exact_legacy_id_overlap_not_symbol(monkeypatch):
    result = run_snapshot(monkeypatch)
    assert result["schema_version"] == H.SCHEMA and result["status"] == "ready"
    assert result["broker_account_match"] == "verified" and result["accounting_complete"] is False
    assert ACCOUNT not in json.dumps(result)
    result = run_snapshot(
        monkeypatch, overlap=[dict(trade_id=1, entry_order_id="unrelated", symbol="SYNTH")]
    )
    assert len(result["rows"]) == 1
    result = run_snapshot(
        monkeypatch, overlap=[dict(trade_id=1, entry_order_id="synthetic-entry-0")]
    )
    assert result["rows"] == [] and result["legacy_overlap_count"] == 1


def test_account_or_db_unavailable_is_not_empty_success(monkeypatch):
    client = SimpleNamespace(get_account=Mock(side_effect=RuntimeError("secret-provider-error")))
    result = run_snapshot(monkeypatch, client=client)
    assert (
        result["ok"] is False and result["status"] == "unavailable" and result["has_more"] is None
    )
    assert "secret-provider-error" not in json.dumps(result)
    assert H.snapshot(Mock(), None, IDENTITY)["status"] == "unavailable"


def test_invalid_identity_cannot_hide_signal_as_legacy_duplicate(monkeypatch):
    records = orders()
    records[0]["observation"]["client_order_id"] = "unrelated"
    result = run_snapshot(
        monkeypatch,
        records=records,
        overlap=[dict(trade_id=1, entry_order_id="synthetic-entry-0")],
    )
    assert result["legacy_overlap_count"] == 0
    assert len(result["rows"]) == 1 and result["rows"][0]["status"] == "incomplete"


def test_non_object_observation_is_incomplete_not_uncaught_error(monkeypatch):
    records = orders()
    records[0]["observation"] = "invalid observation"
    result = run_snapshot(monkeypatch, records=records)
    assert result["status"] == "partial" and result["rows"][0]["status"] == "incomplete"


@pytest.mark.parametrize(
    "account_id", [None, True, 42, 1.5, "", "None", "not-an-account", " " + ACCOUNT]
)
def test_invalid_broker_account_never_queries_db_or_claims_match(monkeypatch, account_id):
    reader = Mock()
    monkeypatch.setattr(H, "read_snapshot", reader)
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=account_id))
    result = H.snapshot(Mock(), client, IDENTITY, read=lambda fn, **kw: fn(), now=NOW)
    assert result["status"] == "unavailable" and result["broker_account_match"] == "unavailable"
    reader.assert_not_called()


def test_broker_uuid_object_is_explicitly_supported(monkeypatch):
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=UUID(ACCOUNT)))
    assert run_snapshot(monkeypatch, client=client)["broker_account_match"] == "verified"


def test_cancel_receipt_cannot_hide_fill_below_sell_limit():
    records = orders()
    records[1]["observation"].update(
        filled_qty="10",
        status="partially_filled",
        filled_at=None,
        updated_at="2026-10-05T09:59:59Z",
    )
    cancel = claim("cancel", operation="DELETE", target=records[1]["observation"]["id"])
    cancel["observation"] = deepcopy(orders()[1]["observation"])
    cancel["observation"]["filled_avg_price"] = "10.99"
    assert project(records=records + [cancel])["reason"] == "broker_price_mismatch"


def test_reader_budget_wire_sentinel_and_cleanup():
    batches = [
        [
            dict(
                deployment_id=IDENTITY[0],
                database_id=IDENTITY[1],
                database_name=DATABASE_NAME,
                connected_database=DATABASE_NAME,
            )
        ],
        [signal()],
        orders(),
        [],
    ]
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)

    def fetch(cap):
        rows = batches.pop(0)
        bodies = [json.dumps(r) for r in rows]
        return [(body, len(rows), sum(len(v.encode()) for v in bodies)) for body in bodies]

    cursor.fetchmany.side_effect = fetch
    conn = Mock(autocommit=False)
    conn.get_transaction_status.return_value = 0
    conn.cursor.return_value = cursor
    signals, records, overlap, has_more = H.read_snapshot(lambda: conn, ACCOUNT, IDENTITY, 25)
    assert len(signals) == 1 and len(records) == 2 and not has_more
    sqls = [call.args[0] for call in cursor.execute.call_args_list]
    assert sqls[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert sum(sql.startswith("WITH q AS MATERIALIZED") for sql in sqls) == 4
    assert all(sql.startswith(("SET ", "WITH q AS MATERIALIZED")) for sql in sqls)
    assert all("octet_length(body)" in sql for sql in sqls if sql.startswith("WITH"))
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()
    conn.commit.assert_not_called()


@pytest.mark.parametrize("mode", ["autocommit", "busy", "oversize", "query_error"])
def test_reader_rejects_bad_context_and_always_closes(mode):
    conn = Mock(autocommit=mode == "autocommit")
    conn.get_transaction_status.return_value = 1 if mode == "busy" else 0
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    conn.cursor.return_value = cursor
    cursor.fetchmany.return_value = [(None, 1, H.MAX_BYTES + 1)]
    if mode == "query_error":
        cursor.execute.side_effect = RuntimeError("sensitive DB exception")
    with pytest.raises(Exception):
        H.read_snapshot(lambda: conn, ACCOUNT, IDENTITY, 25)
    conn.close.assert_called_once()
    conn.commit.assert_not_called()


def test_connection_timeout_does_not_mutate_config_or_log_credentials():
    config = dict(host="h", port=5432, dbname="db", user="u", password="secret", connect_timeout=60)
    db = SimpleNamespace(
        _resolve_db_config=lambda: config,
        psycopg2=SimpleNamespace(connect=Mock(return_value="conn")),
    )
    assert H.connect_current_db(db) == "conn"
    assert db.psycopg2.connect.call_args.kwargs["connect_timeout"] == 2
    assert config["connect_timeout"] == 60


def test_route_is_no_store_public_get_with_paper_guard(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "dashboards/dashboard_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "api_trades_strategy_history"
    )
    assert node.decorator_list[0].args[0].value == "/api/trades/strategy-history"
    node.decorator_list = []
    capture = Mock(return_value={"ok": True})
    monkeypatch.setattr(H, "snapshot", capture)
    namespace = dict(
        request=request,
        _parse_positive_int=lambda *a, **kw: 25,
        _dashboard_broker_context=lambda: ("paper-client", "configured_paper_broker"),
        _dashboard_read_policy_identity=lambda: IDENTITY[:2],
        db=SimpleNamespace(_PAPER_DISPOSITION_BINDING=(*IDENTITY, "synthetic-evidence-reference")),
        _json_no_store=lambda obj: (obj, "no-store"),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with Flask(__name__).test_request_context("/api/trades/strategy-history"):
        assert namespace[node.name]() == ({"ok": True}, "no-store")
    assert capture.call_args.args[1:3] == ("paper-client", IDENTITY)


def test_module_does_not_import_execution_stores_or_broker_writers():
    tree = ast.parse(Path(H.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("scripts.") for name in imports)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"commit", "submit_order", "replace_order", "cancel_order", "execute_trades"}


def identity_connection(row):
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    body = json.dumps(row)
    cursor.fetchmany.side_effect = [[(body, 1, len(body.encode()))], [], [], []]
    conn = Mock(autocommit=False)
    conn.get_transaction_status.return_value = 0
    conn.cursor.return_value = cursor
    return conn


def test_canonical_identity_pair_and_connected_database_are_separate_contracts():
    expected = ("synthetic-deployment", "00000000-0000-4000-8000-000000000002")
    conn = identity_connection(
        dict(
            deployment_id=expected[0],
            database_id=expected[1],
            database_name="synthetic-db",
            connected_database="synthetic-db",
        )
    )
    assert H.read_snapshot(lambda: conn, ACCOUNT, expected, 25) == ([], [], [], False)
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


def test_route_passes_canonical_pair_not_evidence_reference(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "dashboards/dashboard_app.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "api_trades_strategy_history"
    )
    node.decorator_list = []
    expected = ("synthetic-deployment", "00000000-0000-4000-8000-000000000002")
    pin = (*expected, "metadata-owner-fixture:sha256:" + "a" * 64)
    capture = Mock(return_value={"ok": True})
    monkeypatch.setattr(H, "snapshot", capture)
    namespace = dict(
        request=request,
        _parse_positive_int=lambda *a, **kw: 25,
        _dashboard_broker_context=lambda: ("paper-client", "configured_paper_broker"),
        _dashboard_read_policy_identity=lambda: expected,
        db=SimpleNamespace(_PAPER_DISPOSITION_BINDING=pin),
        _json_no_store=lambda value: value,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with Flask(__name__).test_request_context("/api/trades/strategy-history"):
        namespace[node.name]()
    assert capture.call_args.args[2] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_id", "other-deployment"),
        ("database_id", "00000000-0000-4000-8000-000000000003"),
        ("database_name", "wrong-database"),
        ("connected_database", "wrong-database"),
        ("database_name", None),
        ("database_name", ""),
    ],
)
def test_database_binding_mismatch_still_fails_before_history_queries(field, value):
    row = dict(
        deployment_id=IDENTITY[0],
        database_id=IDENTITY[1],
        database_name=DATABASE_NAME,
        connected_database=DATABASE_NAME,
    )
    row[field] = value
    conn = identity_connection(row)
    with pytest.raises(H.EvidenceError, match="database_identity_mismatch"):
        H.read_snapshot(lambda: conn, ACCOUNT, IDENTITY, 25)
    assert conn.cursor.return_value.fetchmany.call_count == 1
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


def test_three_field_evidence_binding_is_not_a_read_identity():
    client = SimpleNamespace(get_account=Mock())
    factory = Mock()
    result = H.snapshot(factory, client, (*IDENTITY, "synthetic-evidence-reference"))
    assert (
        result["status"] == "unavailable" and result["reason"] == "paper_read_context_unavailable"
    )
    client.get_account.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("bad_name", [None, "", 42, True, [], {}])
def test_equal_invalid_database_names_cannot_satisfy_identity(bad_name):
    conn = identity_connection(
        dict(
            deployment_id=IDENTITY[0],
            database_id=IDENTITY[1],
            database_name=bad_name,
            connected_database=bad_name,
        )
    )
    with pytest.raises(H.EvidenceError, match="database_identity_mismatch"):
        H.read_snapshot(lambda: conn, ACCOUNT, IDENTITY, 25)


def test_missing_database_names_cannot_satisfy_identity():
    conn = identity_connection(dict(deployment_id=IDENTITY[0], database_id=IDENTITY[1]))
    with pytest.raises(H.EvidenceError, match="database_identity_mismatch"):
        H.read_snapshot(lambda: conn, ACCOUNT, IDENTITY, 25)


def test_actual_route_through_snapshot_uses_evidence_reference_only_for_context(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "dashboards/dashboard_app.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "api_trades_strategy_history"
    )
    node.decorator_list = []
    pin = (*IDENTITY, "metadata-owner-fixture:sha256:" + "a" * 64)
    assert pin[2] != DATABASE_NAME
    conn = identity_connection(
        dict(
            deployment_id=IDENTITY[0],
            database_id=IDENTITY[1],
            database_name=DATABASE_NAME,
            connected_database=DATABASE_NAME,
        )
    )
    monkeypatch.setattr(H, "connect_current_db", lambda db: conn)
    client = SimpleNamespace(get_account=Mock(return_value=SimpleNamespace(id=UUID(ACCOUNT))))
    namespace = dict(
        request=request,
        _parse_positive_int=lambda *a, **kw: 25,
        _dashboard_broker_context=lambda: (client, "configured_paper_broker"),
        _dashboard_read_policy_identity=lambda: IDENTITY,
        db=SimpleNamespace(_PAPER_DISPOSITION_BINDING=pin),
        _json_no_store=lambda value: value,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with Flask(__name__).test_request_context("/api/trades/strategy-history"):
        result = namespace[node.name]()
    assert result["ok"] is True and result["status"] == "empty"
    assert result["broker_account_match"] == "verified"
    client.get_account.assert_called_once()
    assert conn.cursor.return_value.fetchmany.call_count == 4
    assert pin[2] not in json.dumps(result) and DATABASE_NAME not in json.dumps(result)


@pytest.mark.parametrize(
    "failure,expected",
    [
        (H.EvidenceError("database_identity_mismatch"), "database_identity_mismatch"),
        (RuntimeError("private SQL/DSN/account error"), "strategy_history_unavailable"),
    ],
)
def test_failure_diagnostics_are_fixed_codes_without_private_exception(
    monkeypatch, failure, expected
):
    monkeypatch.setattr(H, "read_snapshot", Mock(side_effect=failure))
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=UUID(ACCOUNT)))
    result = H.snapshot(Mock(), client, IDENTITY, read=lambda fn, **kw: fn())
    assert result["reason"] == expected and result["failure_stage"] == "database_history_read"
    assert result["database_task_entered"] is True
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("code", ["source_busy", "source_timeout"])
@pytest.mark.parametrize(
    "failed_read,stage", [(1, "broker_account_read"), (2, "database_history_read")]
)
def test_bounded_admission_failures_retain_only_fixed_codes(code, failed_read, stage):
    calls = []
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=UUID(ACCOUNT)))
    factory = Mock()

    def read(function, **kwargs):
        calls.append(kwargs["seconds"])
        if len(calls) == failed_read:
            raise RuntimeError(code)
        return function()

    result = H.snapshot(factory, client, IDENTITY, read=read)
    assert result["reason"] == code and result["failure_stage"] == stage
    assert result["database_task_entered"] is False
    assert result["ok"] is False and result["status"] == "unavailable"
    assert result["broker_account_match"] == "unavailable" and result["rows"] == []
    assert calls == ([2] if failed_read == 1 else [2, 7])
    factory.assert_not_called()


def test_second_admission_can_fail_before_first_completion_callback_releases_slot(monkeypatch):
    """Deterministically reproduce a possible admission race, not a live diagnosis."""
    from dashboards import source_contracts as S

    slots = threading.BoundedSemaphore(4)
    for _ in range(3):
        assert slots.acquire(blocking=False)
    pending = []

    class DelayedCallbackFuture:
        def __init__(self, function):
            self.function = function
            self.callback = None

        def add_done_callback(self, callback):
            self.callback = callback

        def result(self, timeout):
            # Hold the scheduling gap between result visibility and callback.
            value = self.function()
            pending.append(self)
            return value

    executor = SimpleNamespace(submit=Mock(side_effect=DelayedCallbackFuture))
    monkeypatch.setattr(S, "_slots", slots)
    monkeypatch.setattr(S, "_workers", executor)
    client = SimpleNamespace(get_account=Mock(return_value=SimpleNamespace(id=UUID(ACCOUNT))))
    factory = Mock()
    result = H.snapshot(factory, client, IDENTITY, read=S.bounded_read)
    assert result["reason"] == "source_busy" and result["failure_stage"] == "database_history_read"
    assert result["database_task_entered"] is False
    assert executor.submit.call_count == 1
    client.get_account.assert_called_once()
    factory.assert_not_called()
    for future in pending:
        future.callback(future)
    for _ in range(3):
        slots.release()
    assert all(slots.acquire(blocking=False) for _ in range(4))
    assert not slots.acquire(blocking=False)


def test_arbitrary_runtime_exception_is_not_exposed():
    def read(function, **kwargs):
        raise RuntimeError("private pool details and credentials")

    client = SimpleNamespace(get_account=Mock())
    result = H.snapshot(Mock(), client, IDENTITY, read=read)
    assert result["reason"] == "strategy_history_unavailable"
    assert result["database_task_entered"] is False
    assert "private" not in json.dumps(result)


def test_timeout_after_database_task_entry_is_distinct(monkeypatch):
    def fail(*args):
        raise RuntimeError("source_timeout")

    monkeypatch.setattr(H, "read_snapshot", fail)
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=UUID(ACCOUNT)))
    result = H.snapshot(Mock(), client, IDENTITY, read=lambda function, **kwargs: function())
    assert result["reason"] == "source_timeout"
    assert result["database_task_entered"] is True


def python310_fraction_parser(monkeypatch):
    """Compatibility shim, not a claim that these tests run on Python 3.10."""
    from dashboards import source_contracts as S

    class Python310DateTime(datetime):
        @classmethod
        def fromisoformat(cls, value):
            fraction = re.search(r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.([0-9]+)", value)
            if fraction is not None and len(fraction[1]) not in (3, 6):
                raise ValueError("Python 3.10 fractional second grammar")
            return datetime.fromisoformat(value)

    monkeypatch.setattr(H, "datetime", Python310DateTime)
    monkeypatch.setattr(S, "datetime", Python310DateTime)
    return S


@pytest.mark.parametrize("fraction", ["", ".1", ".12", ".123", ".1234", ".12345", ".123456"])
@pytest.mark.parametrize("offset", ["Z", "+00:00", "+05:30", "-04:00"])
def test_zoned_fraction_precision_survives_python310_grammar(monkeypatch, fraction, offset):
    source = f"2026-10-05T10:00:00{fraction}{offset}"
    expected = datetime.fromisoformat(source.replace("Z", "+00:00")).astimezone(timezone.utc)
    S = python310_fraction_parser(monkeypatch)
    if len(fraction) in (2, 3, 5, 6):
        assert S.timestamp(source) is None  # Reproduce the old helper's rejection.
    assert H.timestamp(source) == expected.isoformat()
    assert H.instant(source) == expected


class UnknownOffset(tzinfo):
    def utcoffset(self, value):
        return None


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        20261005,
        {},
        "2026-10-05",
        "2026-10-05T10:00:00",
        "2026-10-05T10:00:00.12345",
        "2026-10-05T10:00:00.Z",
        "2026-10-05T10:00:00.1234567Z",
        "2026-10-05T10:00:00.123456789+00:00",
        "2026-10-05T10:00:00.12345+24:00",
        "2026-10-05T10:00:00.12345+01:60",
        "2026-10-05T10:00:00.12345+0000",
        "2026-10-05T10:00:00.12345UTC",
        "2026-10-05T10:00:00.12345Z\n",
        "2026-10-05T10:00:00,12345Z",
        "2026-02-30T10:00:00.12345Z",
        "2026-10-05T10:00:60.12345Z",
        "0001-01-01T00:00:00+01:00",
        "9999-12-31T23:59:59-01:00",
        datetime(2026, 10, 5, 10),
        datetime(2026, 10, 5, 10, tzinfo=UnknownOffset()),
    ],
)
def test_timestamp_does_not_invent_zone_or_truncate_precision(value):
    assert H.timestamp(value) is None
    with pytest.raises(H.EvidenceError, match="invalid_timestamp"):
        H.instant(value)


def test_aware_datetime_and_postgresql_separator_preserve_instant():
    value = datetime(2026, 10, 5, 10, 0, 0, 12000, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    expected = "2026-10-05T04:30:00.012000+00:00"
    assert H.timestamp(value) == expected
    assert H.timestamp(value.isoformat(sep="T")) == expected
    assert H.timestamp(value.isoformat(sep=" ")) == expected


@pytest.mark.parametrize("bad_signal_time", [False, True])
def test_wire_reader_through_projection_uses_same_strict_timestamp_contract(
    monkeypatch, bad_signal_time
):
    s, records = signal(), orders()
    s["updated_at"] = "2026-10-05T10:00:00.12345" + ("" if bad_signal_time else "+00:00")
    records[0]["updated_at"] = "2026-10-01T10:00:00.1+00:00"
    records[0]["observation"].update(
        updated_at="2026-10-01T15:30:00.1+05:30", filled_at="2026-10-01T10:00:00.1Z"
    )
    records[1]["updated_at"] = "2026-10-05T10:00:00.1234+00:00"
    records[1]["observation"].update(
        updated_at="2026-10-05T06:00:00.12-04:00", filled_at="2026-10-05T10:00:00.12Z"
    )
    policy = dict(
        deployment_id=IDENTITY[0],
        database_id=IDENTITY[1],
        database_name=DATABASE_NAME,
        connected_database=DATABASE_NAME,
    )
    conn = identity_connection(policy)
    batches = [[policy], [s], records, []]
    encoded = [[json.dumps(row) for row in batch] for batch in batches]
    conn.cursor.return_value.fetchmany.side_effect = [
        [(body, len(batch), sum(len(item.encode()) for item in batch)) for body in batch]
        for batch in encoded
    ]
    python310_fraction_parser(monkeypatch)
    client = SimpleNamespace(get_account=Mock(return_value=SimpleNamespace(id=UUID(ACCOUNT))))
    result = H.snapshot(lambda: conn, client, IDENTITY, read=lambda fn, **kw: fn(), now=NOW)
    if bad_signal_time:
        assert result["status"] == "unavailable" and result["rows"] == []
        assert result["failure_stage"] == "database_history_read"
        assert conn.cursor.return_value.fetchmany.call_count == 2
    else:
        assert result["status"] == "ready" and result["broker_account_match"] == "verified"
        assert result["source_as_of"] == "2026-10-05T10:00:00.123450+00:00"
        row = result["rows"][0]
        assert row["status"] == "complete" and row["gross_realized_pnl"] == 80.0
        assert row["updated_at"] == result["source_as_of"]
        assert row["entry_time"] == "2026-10-01T10:00:00.100000+00:00"
        assert row["exit_time"] == "2026-10-05T10:00:00.120000+00:00"
        assert conn.cursor.return_value.fetchmany.call_count == 4
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()
    conn.commit.assert_not_called()
