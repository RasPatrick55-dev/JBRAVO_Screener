"""Synthetic, offline web projection tests. No native store, credentials or DB."""

import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from flask import Flask, request

from dashboards import strategy_trade_history as H

pytestmark = pytest.mark.alpaca_optional
ACCOUNT = "00000000-0000-4000-8000-000000000001"
SID = "7" * 64
NOW = datetime(2026, 11, 9, 17, tzinfo=timezone.utc)
ENTRY_TIME = "2026-10-01T10:00:00+00:00"
EXIT_TIME = "2026-10-05T10:00:00+00:00"
IDENTITY = ("synthetic-deployment", "synthetic-database", "synthetic-db")


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
                database_name=IDENTITY[2],
                connected_database=IDENTITY[2],
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
        db=SimpleNamespace(_PAPER_DISPOSITION_BINDING=IDENTITY),
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
