"""Fictional offline cohort/transport tests; no live database or broker."""

import ast
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask, jsonify

from dashboards import combined_trade_gross as C
from test_execution_gross_view import inputs, Connection
from test_public_dashboard_integration import policy, IDENTITY, NOW
import test_strategy_trade_history as S
from sql_transport_fixture import Cursor

pytestmark = pytest.mark.alpaca_optional


def strategy(sid=S.SID, state="CLOSED", symbol="SYNTH"):
    signal = S.signal()
    signal.update(
        signal_id=sid,
        state=state,
        symbol=symbol,
        created_at=(NOW - timedelta(hours=2)).isoformat(),
        updated_at=NOW.isoformat(),
    )
    proposal = dict(signal["payload"]["proposal"], signal_id=sid, symbol=symbol)
    signal["payload"]["proposal"] = proposal
    signal["payload"]["handoff"]["proposals"] = [deepcopy(proposal)]
    records = [
        S.claim("entry", price="10", at=(NOW - timedelta(hours=2)).isoformat()),
        S.claim("exit", price="12", at=(NOW - timedelta(hours=1)).isoformat()),
    ]
    for record in records:
        record["signal_id"] = sid
        record["lineage"]["signal_id"] = sid
        key = C.H.client_key(S.ACCOUNT, sid, record["role"], record["sequence"])
        record["client_order_id"] = key
        record["request"]["order"].update(client_order_id=key, symbol=symbol)
        record["observation"].update(
            client_order_id=key, symbol=symbol, id=f"fixture-{sid[:4]}-{record['role']}"
        )
        S.rehash(record)
    return signal, records


def cohort(*, signals=None, orders=None, records=None, fills=None, claims=None, p=None):
    row, fs, owner = inputs()
    signal, os = strategy()
    p = p or policy()
    records = [row] if records is None else records
    fills = fs if fills is None else fills
    claims = [owner] if claims is None else claims
    signals = [signal] if signals is None else signals
    orders = os if orders is None else orders
    execution = C.E.reconstruct(p, records, fills, claims)
    return C.project(
        p,
        records,
        execution,
        signals,
        orders,
        claims,
        IDENTITY,
        S.ACCOUNT,
        NOW,
        "a" * 64,
        legacy_fills=fills,
    )


def test_one_full_cohort_counts_list_stats_and_groups_symbols_before_ranking():
    result = cohort()
    assert result["population_complete"] and result["status"] == "ready"
    assert result["coverage"]["included"] == 2
    assert len(result["rows"]) == result["stats"][-1]["tradesCount"] == 2
    assert Decimal(result["stats"][-1]["totalPL"]) == 84
    assert result["rankings"]["all"]["winners"] == [
        dict(rank=1, symbol="SYNTH", pl="80"),
        dict(rank=2, symbol="FICT", pl="4"),
    ]
    assert result["strategy_account_binding"] == "verified"
    assert result["legacy_account_binding"] == "unverified"
    assert result["net_pnl"] is result["fees"] is None
    assert result["accounting_complete"] is result["net_eligible"] is False
    source = result["strategy_history"]
    assert source["performance_status"] == "NOT_QUALIFIED_STRATEGY_HISTORY"
    assert source["account_scope"] == "configured_paper_broker"
    assert source["rows"][0]["combined_inclusion"] == "INCLUDED"
    assert "_exact" not in json.dumps(result) and S.ACCOUNT not in json.dumps(result)
    legacy = next(row for row in result["rows"] if row["trade_id"])
    assert legacy["recorded_pnl"] == "2" and legacy["gross_pnl"] == "4"


@pytest.mark.parametrize("state", ["OPEN", "EXPIRED", "ENTRY_PENDING", "UNKNOWN", "DISPUTED"])
def test_nonclosed_signals_never_count_as_realized_or_zero_profit(state):
    signal, orders = strategy(state=state)
    if state == "EXPIRED":
        orders = orders[:1]
        orders[0]["observation"].update(filled_qty="0", status="canceled", filled_at=None)
    if state == "OPEN":
        signal["payload"]["execution_owner"] = dict(
            schema="jbravo.execution-entry-recognition.v1",
            scope="ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS",
            signal_id=signal["signal_id"],
        )
    result = cohort(signals=[signal], orders=orders)
    assert result["coverage"]["strategy_not_closed"] == 1
    assert len(result["rows"]) == result["stats"][-1]["tradesCount"] == 1
    assert result["strategy_history"]["rows"][0]["combined_inclusion"] == "NOT_CLOSED"


def test_unknown_closed_strategy_and_legacy_remain_unknown_in_every_range():
    signal, orders = strategy()
    orders[1]["observation"]["filled_at"] = None
    result = cohort(signals=[signal], orders=orders, fills=[])
    assert result["status"] == "partial"
    for metric in result["stats"]:
        assert metric["coverage"]["unknown"] == 2
        assert metric["totalPL"] is None and metric["tradesCount"] == 0


def test_legacy_exclusion_and_reserved_claim_cannot_be_reintroduced_as_strategy():
    row, fills, owner = inputs()
    signal, orders = strategy()
    orders[0]["observation"]["id"] = row["entry_order_id"]
    result = cohort(signals=[signal], orders=orders, p=policy((row["trade_id"],)))
    assert result["coverage"]["included"] == 0 and result["coverage"]["excluded"] == 2
    assert all(metric["coverage"]["excluded"] == 2 for metric in result["stats"])
    assert (
        result["strategy_history"]["rows"][0]["combined_reason"] == "LEGACY_POLICY_ORDER_RESERVED"
    )


def test_nonclosed_physical_claim_blocks_strategy_even_when_not_in_legacy_population():
    signal, orders = strategy()
    extra = dict(
        trade_id=9998,
        status="OPEN",
        entry_order_id=orders[0]["observation"]["id"],
        exit_order_id=None,
    )
    result = cohort(signals=[signal], orders=orders, claims=[inputs()[2], extra])
    assert result["coverage"]["included"] == 1 and result["coverage"]["unknown"] == 1
    assert result["strategy_history"]["rows"][0]["combined_reason"] == "LEGACY_ORDER_CLAIM_RESERVED"


def test_reserved_allocation_owner_not_in_closed_population_blocks_strategy():
    from scripts import trade_allocation as A

    signal, orders = strategy()
    overlay = A.result_read("VALIDATED", (), IDENTITY, 2, {orders[1]["observation"]["id"]: 9999})
    p = C.D.Policy(*IDENTITY, 2, (), "VALIDATED_PERFORMANCE", (), overlay._metadata_json)
    result = cohort(signals=[signal], orders=orders, p=p)
    assert result["coverage"]["unknown"] == 1
    assert result["strategy_history"]["rows"][0]["combined_inclusion"] == "UNKNOWN"


@pytest.mark.parametrize("other_account", [False, True])
def test_competing_strategy_claim_outside_selected_signals_blocks_contribution(other_account):
    signal, orders = strategy()
    foreign = deepcopy(orders[1])
    foreign.update(client_order_id="foreign-claim", signal_id="8" * 64)
    if other_account:
        foreign["account_id"] = "00000000-0000-4000-8000-000000000099"
    result = cohort(signals=[signal], orders=orders + [foreign])
    assert result["coverage"]["included"] == 1 and result["coverage"]["unknown"] == 1


def test_exact_duplicate_prefers_canonical_legacy_without_counting_it_twice():
    row, fills, owner = inputs(qty="40", symbol="SYNTH")
    signal, orders = strategy()
    # Native weighted legacy exit average is12, matching the strategy evidence.
    for record, field in zip(orders, ["entry_order_id", "exit_order_id"]):
        record["observation"]["id"] = row[field]
    result = cohort(records=[row], fills=fills, claims=[owner], signals=[signal], orders=orders)
    assert result["coverage"]["included"] == 1 and result["coverage"]["deduplicated"] == 1
    assert Decimal(result["stats"][-1]["totalPL"]) == 80
    assert len(result["rows"]) == 1
    lifecycle = result["strategy_history"]
    assert lifecycle["legacy_overlap_count"] == 0 and len(lifecycle["rows"]) == 1
    assert lifecycle["rows"][0]["combined_inclusion"] == "EXCLUDED"
    assert lifecycle["rows"][0]["combined_reason"] == "IDENTICAL_LEGACY_EXECUTION"


def test_partial_cross_source_overlap_taints_both_contributions_not_arbitrary_winner():
    row, fills, owner = inputs()
    signal, orders = strategy()
    orders[0]["observation"]["id"] = row["entry_order_id"]
    result = cohort(signals=[signal], orders=orders)
    assert result["coverage"]["included"] == 0 and result["coverage"]["unknown"] == 2
    assert result["stats"][-1]["totalPL"] is None
    lifecycle = result["strategy_history"]
    assert lifecycle["legacy_overlap_count"] == 0 and len(lifecycle["rows"]) == 1
    assert lifecycle["rows"][0]["combined_inclusion"] == "UNKNOWN"
    assert lifecycle["rows"][0]["combined_reason"] == "LEGACY_ORDER_CLAIM_RESERVED"


def test_same_symbol_distinct_orders_are_aggregated_not_deduplicated():
    signal, orders = strategy(symbol="FICT")
    result = cohort(signals=[signal], orders=orders)
    assert result["coverage"]["included"] == 2
    assert result["rankings"]["all"]["winners"] == [dict(rank=1, symbol="FICT", pl="84")]


def test_complete_cohort_not_top25_drives_rankings_and_range_counts():
    signals, orders = [], []
    for index in range(30):
        signal, os = strategy(sid=f"{index + 1:064x}", symbol=f"F{index}")
        # Distinct IDs even for signal IDs with common prefixes.
        for order in os:
            order["observation"]["id"] = f"order-{index}-{order['role']}"
        signals.append(signal)
        orders.extend(os)
    result = cohort(signals=signals, orders=orders)
    assert result["stats"][-1]["tradesCount"] == 31
    assert len(result["strategy_history"]["rows"]) == 25 and result["strategy_history"]["has_more"]
    assert len(result["rows"]) == 31 and len(result["rankings"]["all"]["winners"]) == 12
    assert Decimal(result["stats"][-1]["totalPL"]) == 2404


@pytest.mark.parametrize("overflow", ["signals", "orders", "claims"])
def test_full_population_overflow_is_not_successful_partial(overflow):
    kw = {
        overflow: (
            [strategy()[0]] * 201
            if overflow == "signals"
            else strategy()[1] * 501
            if overflow == "orders"
            else [inputs()[2]] * 1001
        )
    }
    with pytest.raises(ValueError):
        cohort(**kw)


def test_unknown_missing_excluded_dates_do_not_disappear_from_daily_uncertainty():
    row, fills, owner = inputs()
    row["exit_time"] = None
    result = cohort(records=[row], fills=fills, claims=[owner], p=policy((row["trade_id"],)))
    assert all(metric["coverage"]["excluded"] == 1 for metric in result["stats"])


def test_strategy_decimal_gross_is_not_roundtripped_through_public_float():
    signal, orders = strategy()
    orders[1]["observation"]["filled_avg_price"] = "12.123456789012"
    result = cohort(signals=[signal], orders=orders)
    expected = Decimal("40") * (Decimal("12.123456789012") - Decimal("10"))
    assert Decimal(next(r for r in result["rows"] if r["signal_id"])["gross_pnl"]) == expected
    assert Decimal(result["stats"][-1]["totalPL"]) == expected + 4


class CombinedCursor(Cursor):
    def native(self, sql, params):
        if sql == C.STRATEGY_COHORT_SQL:
            output = []
            for section, query in (
                ("visibility", C.STRATEGY_VISIBILITY_SQL),
                ("signals", C.SIGNALS_SQL),
                ("orders", C.ORDERS_SQL),
                ("claims", C.CLAIMS_SQL),
            ):
                for index, row in enumerate(self.native(query, params), 1):
                    body = json.dumps(list(row)) if section == "visibility" else row[0]
                    output.append((section, str(index), body))
            return sorted(output, key=lambda row: (row[0], int(row[1])))
        if sql == C.STRATEGY_VISIBILITY_SQL:
            return getattr(
                self.conn,
                "strategy_visibility",
                [
                    ("strategy_execution_orders_app", "r", False, False, False, False, True),
                    ("strategy_execution_signals_app", "r", False, False, False, False, True),
                ],
            )
        if sql in C.PLANS:
            field = {
                C.SIGNALS_SQL: "signals",
                C.ORDERS_SQL: "strategy_orders",
                C.CLAIMS_SQL: "claims",
            }[sql]
            rows = deepcopy(getattr(self.conn, field))
            if (
                self.conn.source_reads > 1
                and getattr(self.conn, "strategy_drift", False)
                and field == "signals"
            ):
                rows[0]["updated_at"] = (NOW + timedelta(seconds=1)).isoformat()
            return [(json.dumps(dict(row=row, version="100")),) for row in rows]
        return super().native(sql, params)


def connection():
    conn = Connection()
    signal, orders = strategy()
    conn.signals, conn.strategy_orders, conn.claims = [signal], orders, deepcopy(conn.owners)
    conn.cursor = lambda: CombinedCursor(conn)
    conn.commit = Mock()
    return conn


def wire(monkeypatch, conn):
    # The fixture's SQL catalog uses E.READ_PLANS; only the test catalog grows.
    monkeypatch.setattr(C.E, "READ_PLANS", {**C.E.READ_PLANS, **C.PLANS})
    monkeypatch.setattr(C.D, "read_policy", lambda *a, **kw: policy())
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=S.ACCOUNT))
    return C.snapshot(lambda: conn, client, IDENTITY, now=NOW, read=lambda fn, **kw: fn())


def test_real_wire_reader_through_complete_projection_and_cleanup(monkeypatch):
    conn = connection()
    result = wire(monkeypatch, conn)
    assert result["status"] == "ready" and result["coverage"]["included"] == 2
    assert conn.source_reads == 2
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()
    conn.commit.assert_not_called()
    assert all(
        not any(word in sql.upper() for word in ["INSERT ", "UPDATE ", "DELETE ", "CREATE "])
        for sql, _ in conn.calls
    )


def test_read_cohort_change_is_unavailable_and_not_retainable(monkeypatch):
    conn = connection()
    conn.strategy_drift = True
    result = wire(monkeypatch, conn)
    assert result["status"] == "unavailable" and result["reason"] == "cohort_changed"
    assert result["rows"] == [] and result["coverage"] is None and not result["population_complete"]
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.parametrize(
    "account", [None, True, 123, "invalid", "00000000-0000-4000-8000-000000000001 "]
)
def test_invalid_current_account_never_reads_database(account):
    factory = Mock()
    result = C.snapshot(
        factory,
        SimpleNamespace(get_account=lambda: SimpleNamespace(id=account)),
        IDENTITY,
        now=NOW,
        read=lambda fn, **kw: fn(),
    )
    assert result["status"] == "unavailable" and result["rows"] == []
    factory.assert_not_called()


def test_new_route_is_distinct_and_calls_one_cohort_reader(monkeypatch):
    path = Path("dashboards/dashboard_app.py")
    tree = ast.parse(path.read_text(encoding="utf8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "api_trades_combined_gross"
    )
    assert node.decorator_list[0].args[0].value == "/api/trades/combined-gross"
    existing = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "api_trades_overview"
    ]
    assert (
        len(existing) == 1 and existing[0].decorator_list[0].args[0].value == "/api/trades/overview"
    )
    node.decorator_list = []
    reader = Mock(return_value=cohort())
    monkeypatch.setattr(C, "snapshot", reader)
    ns = dict(
        _json_no_store=jsonify,
        _dashboard_broker_context=lambda: ("client", "configured_paper_broker"),
        _execution_gross_db_conn=lambda: None,
        _dashboard_read_policy_identity=lambda: IDENTITY,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    app = Flask("combined-cohort-test")
    app.add_url_rule("/api/trades/combined-gross", view_func=ns[node.name])
    assert app.test_client().get("/api/trades/combined-gross").json["coverage"]["included"] == 2
    reader.assert_called_once()


def test_real_canonical_policy_reader_and_replay_remain_in_combined_wire(monkeypatch):
    conn = connection()
    monkeypatch.setattr(C.E, "READ_PLANS", {**C.E.READ_PLANS, **C.PLANS})
    result = C.snapshot(
        lambda: conn,
        SimpleNamespace(get_account=lambda: SimpleNamespace(id=S.ACCOUNT)),
        IDENTITY,
        now=NOW,
        read=lambda fn, **kw: fn(),
    )
    assert result["status"] == "ready" and conn.policy_reads == 4
    assert len(conn.calls) <= C.T.MAX_SQL and max(conn.fetch_sizes) <= 16


@pytest.mark.parametrize("drift", ["revision_drift", "version_drift", "canonical_version_drift"])
def test_policy_or_native_version_drift_withholds_entire_cohort(monkeypatch, drift):
    conn = connection()
    setattr(conn, drift, True)
    monkeypatch.setattr(C.E, "READ_PLANS", {**C.E.READ_PLANS, **C.PLANS})
    result = C.snapshot(
        lambda: conn,
        SimpleNamespace(get_account=lambda: SimpleNamespace(id=S.ACCOUNT)),
        IDENTITY,
        now=NOW,
        read=lambda fn, **kw: fn(),
    )
    assert result["status"] == "unavailable" and result["rows"] == [] and result["stats"] == []
    assert result["coverage"] is None and result["strategy_history"] is None
    conn.close.assert_called_once()


@pytest.mark.parametrize(
    "index,field,value", [(0, 1, "v"), (0, 2, True), (0, 5, True), (1, 6, False)]
)
def test_strategy_view_rls_inheritance_or_missing_privilege_is_not_complete(
    monkeypatch, index, field, value
):
    conn = connection()
    conn.strategy_visibility = [
        ["strategy_execution_orders_app", "r", False, False, False, False, True],
        ["strategy_execution_signals_app", "r", False, False, False, False, True],
    ]
    conn.strategy_visibility[index][field] = value
    conn.strategy_visibility = [tuple(row) for row in conn.strategy_visibility]
    result = wire(monkeypatch, conn)
    assert result["status"] == "unavailable" and result["coverage"] is None


@pytest.mark.parametrize(
    "code", ["source_busy", "source_timeout", "private database error with account details"]
)
def test_only_fixed_admission_failure_codes_are_public(code):
    calls = []

    def read(fn, **kwargs):
        calls.append(kwargs["seconds"])
        if len(calls) == 2:
            raise RuntimeError(code)
        return fn()

    result = C.snapshot(
        Mock(),
        SimpleNamespace(get_account=lambda: SimpleNamespace(id=S.ACCOUNT)),
        IDENTITY,
        now=NOW,
        read=read,
    )
    assert result["reason"] == (
        code if code.startswith("source_") else "combined_gross_unavailable"
    )
    assert result["failure_stage"] == "database_cohort_read" and result["rows"] == []
    assert "private" not in json.dumps(result) and calls == [2, 10]


def test_strategy_wire_fractional_request_numbers_are_not_canonicalized_to_strings(monkeypatch):
    conn = connection()
    # Existing signatures intentionally distinguish float11.0 from integer11.
    conn.strategy_orders[1]["request"]["order"]["limit_price"] = 11.001
    conn.strategy_orders[1]["observation"]["limit_price"] = 11.001
    S.rehash(conn.strategy_orders[1])
    result = wire(monkeypatch, conn)
    assert result["coverage"]["by_source"]["strategy"]["included"] == 1


@pytest.mark.parametrize("nanosecond", ["000000000", "000000001"])
def test_rolling_day_boundary_uses_complete_exact_exit_instant(nanosecond):
    signal, orders = strategy()
    entered = (NOW - timedelta(days=2)).isoformat()
    # One ns before the rolling boundary must not be rounded into the day.
    exit_time = (
        "2001-02-02T12:00:00Z" if nanosecond == "000000000" else "2001-02-02T11:59:59.999999999Z"
    )
    orders[0]["observation"].update(filled_at=entered, updated_at=entered)
    orders[1]["observation"].update(filled_at=exit_time, updated_at=exit_time)
    result = cohort(signals=[signal], orders=orders)
    assert result["stats"][0]["tradesCount"] == (2 if nanosecond == "000000000" else 1)
    assert result["stats"][-1]["tradesCount"] == 2


def test_unknown_legacy_symbol_keeps_identity_without_exposing_raw_shape():
    row, fills, owner = inputs()
    row["symbol"] = None
    result = cohort(records=[row], fills=fills, claims=[owner])
    row = next(row for row in result["rows"] if row["trade_id"])
    assert row["symbol"] is None and row["row_id"] == "legacy:9001"
    assert row["inclusion"] == "UNKNOWN" and row["gross_pnl"] is None


def allocation_cohort(*, state="ACTIVE", conflict=False):
    """Synthetic nonempty audited-allocation projection; no native manifest claim."""
    A = C.G.A
    row, fills, owner = inputs()
    row.update(exit_order_id=None, exit_price=None, realized_pnl=None)
    owner["exit_order_id"] = None
    fills[-1]["order_id"] = fills[-1]["provider_order_id"] = "fixture-terminal-order"
    clocks = tuple(
        (
            index + 1,
            f["activity_id"],
            f["side"],
            f["order_id"],
            f["qty"],
            f["price"],
            f["transaction_time"],
        )
        for index, f in enumerate(fills)
    )
    ids = tuple(dict.fromkeys(f["order_id"] for f in fills))
    allocation = A.Allocation(
        "synthetic-allocation",
        row["trade_id"],
        "ACTIVE",
        1,
        Decimal(2),
        Decimal(2),
        Decimal(4),
        "synthetic-approval",
        (),
        ids,
        tuple(f["activity_id"] for f in fills),
        clock_fills=clocks,
        row_fingerprint=C.D.fingerprint(row),
    )
    reserved = {order: row["trade_id"] for order in ids}
    allocations = (allocation,) if state == "ACTIVE" else ()
    overlay = A.result_read(
        "VALIDATED",
        allocations,
        IDENTITY,
        2,
        reserved,
        [
            dict(
                trade_id=row["trade_id"],
                state=state,
                allocation_revision=1 if state == "ACTIVE" else 2,
            )
        ],
    )
    p = C.D.Policy(*IDENTITY, 2, (), "VALIDATED_PERFORMANCE", allocations, overlay._metadata_json)
    signal, orders = strategy()
    if conflict:
        orders[-1]["observation"]["id"] = ids[-1]
    return p, row, fills, owner, signal, orders


@pytest.mark.parametrize("state", ["ACTIVE", "REVOKED"])
def test_nonempty_allocation_preserves_active_and_revoked_order_reservations(state):
    p, row, fills, owner, signal, orders = allocation_cohort(state=state, conflict=True)
    result = cohort(
        p=p, records=[row], fills=fills, claims=[owner], signals=[signal], orders=orders
    )
    strategy_row = next(r for r in result["rows"] if r["signal_id"])
    assert strategy_row["inclusion"] == "UNKNOWN" and strategy_row["gross_pnl"] is None
    assert strategy_row["reason"] == "LEGACY_ORDER_CLAIM_RESERVED"
    assert result["coverage"]["included"] == 0
    assert result["fees"] is result["net_pnl"] is None and not result["accounting_complete"]


def test_nonempty_allocation_keeps_canonical_clock_gross_and_unknown_scalar_prices():
    p, row, fills, owner, signal, orders = allocation_cohort()
    result = cohort(
        p=p, records=[row], fills=fills, claims=[owner], signals=[signal], orders=orders
    )
    legacy = next(r for r in result["rows"] if r["trade_id"])
    assert legacy["inclusion"] == "INCLUDED" and legacy["gross_pnl"] == "4"
    assert legacy["exit_price"] is legacy["recorded_pnl"] is None
    assert legacy["exit_time"] == C.H.timestamp(fills[-1]["transaction_time"])
    assert result["stats"][-1]["totalPL"] == "84"
    assert p.allocations[0].financial_eligible is False


def test_nonempty_allocation_full_row_binding_change_fails_closed():
    p, row, fills, owner, signal, orders = allocation_cohort()
    row["updated_at"] = (NOW + timedelta(seconds=1)).isoformat()
    with pytest.raises(C.G.A.AllocationError, match="ROW_CHANGED"):
        cohort(p=p, records=[row], fills=fills, claims=[owner], signals=[signal], orders=orders)


def test_sum_and_symbol_ranking_keep_exact_widely_separated_decimal_scales():
    rows, economics = [], {}
    for index, amount in enumerate(
        [Decimal("1e70"), Decimal("-1e70"), Decimal("0.00000000000000000001")]
    ):
        row = C.common_row(
            f"legacy:{index + 1}", "LEGACY_EXECUTION", "SYNTH", "CLOSED", trade_id=index + 1
        )
        row.update(
            inclusion="INCLUDED", gross_pnl=C.decimal_text(amount), exit_time=NOW.isoformat()
        )
        rows.append(row)
        economics[row["row_id"]] = amount
    result = C.summarize(rows, economics, [], 0, 0, NOW, "a" * 64)
    assert result["stats"][-1]["totalPL"] == "0.00000000000000000001"
    assert result["rankings"]["all"]["winners"] == [
        dict(rank=1, symbol="SYNTH", pl="0.00000000000000000001")
    ]


@pytest.mark.parametrize("helper", ["_dashboard_broker_context", "_dashboard_read_policy_identity"])
def test_route_context_error_is_safe_unavailable_before_snapshot_dispatch(monkeypatch, helper):
    path = Path("dashboards/dashboard_app.py")
    tree = ast.parse(path.read_text(encoding="utf8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "api_trades_combined_gross"
    )
    node.decorator_list = []
    reader = Mock(side_effect=AssertionError("snapshot must not run"))
    monkeypatch.setattr(C, "snapshot", reader)
    ns = dict(
        _json_no_store=jsonify,
        _dashboard_broker_context=lambda: ("client", "configured_paper_broker"),
        _execution_gross_db_conn=Mock(),
        _dashboard_read_policy_identity=lambda: IDENTITY,
    )
    ns[helper] = Mock(side_effect=RuntimeError("private credentials must not escape"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    app = Flask("combined-context-error-test")
    app.add_url_rule("/api/trades/combined-gross", view_func=ns[node.name])
    response = app.test_client().get("/api/trades/combined-gross")
    assert response.status_code == 200 and response.json["status"] == "unavailable"
    assert response.json["failure_stage"] == "paper_context" and response.json["rows"] == []
    assert "private" not in response.get_data(as_text=True)
    reader.assert_not_called()


def test_original_strategy_history_wire_has_no_private_exact_fields(monkeypatch):
    signal, orders = strategy()
    expected = C.H.project_signal(signal, orders, S.ACCOUNT, NOW)
    enriched = C.H.project_signal(signal, orders, S.ACCOUNT, NOW, include_exact=True)
    assert enriched.pop("_exact")["gross"] == Decimal(80)
    assert expected == enriched and "_exact" not in expected
    monkeypatch.setattr(C.H, "read_snapshot", lambda *args: ([signal], orders, [], False))
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(id=S.ACCOUNT))
    result = C.H.snapshot(Mock(), client, IDENTITY, now=NOW, read=lambda fn, **kw: fn())
    assert result["rows"] == [expected]
    assert "_exact" not in json.dumps(result)


def synthetic_wire():
    """Reproducible fictional wire fixture shared with real-component UI tests."""
    positive, positive_orders = strategy()
    negative, negative_orders = strategy(sid="8" * 64, symbol="DECLINE")
    negative_orders[-1]["observation"].update(filled_avg_price="9", limit_price="8")
    negative_orders[-1]["request"]["order"]["limit_price"] = "8"
    S.rehash(negative_orders[-1])
    recognition, recognition_orders = strategy(sid="9" * 64, state="OPEN", symbol="RECOG")
    recognition["payload"]["execution_owner"] = dict(
        schema="jbravo.execution-entry-recognition.v1",
        scope="ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS",
        signal_id=recognition["signal_id"],
    )
    unfilled, unfilled_orders = strategy(sid="a" * 64, state="EXPIRED", symbol="UNFILLED")
    unfilled_orders[0]["observation"].update(filled_qty="0", status="canceled", filled_at=None)
    missing, missing_orders = strategy(sid="b" * 64, symbol="MISSING")
    missing_orders[-1]["observation"]["filled_at"] = None
    first, first_fills, first_owner = inputs()
    excluded, excluded_fills, excluded_owner = inputs(9002, "EXCL")
    return cohort(
        signals=[positive, negative, recognition, unfilled, missing],
        orders=positive_orders
        + negative_orders
        + recognition_orders[:1]
        + unfilled_orders[:1]
        + missing_orders,
        records=[first, excluded],
        fills=first_fills + excluded_fills,
        claims=[first_owner, excluded_owner],
        p=policy((9002,)),
    )


def test_synthetic_wire_is_exact_shared_frontend_contract():
    result = synthetic_wire()
    assert result["coverage"]["included"] == 3 and result["coverage"]["unknown"] == 1
    assert result["coverage"]["excluded"] == 1 and result["coverage"]["strategy_not_closed"] == 2
    assert result["stats"][-1]["totalPL"] == "44"
    fixture = Path("frontend/qa/fixtures/combinedGrossWire.synthetic.json")
    if fixture.exists():
        assert json.loads(fixture.read_text(encoding="utf8")) == result


@pytest.mark.parametrize("conflict", ["symbol", "cash", "date", "raw_nanosecond"])
def test_equal_gross_same_orders_do_not_hide_conflicting_identity_or_timing(conflict):
    row, fills, owner = inputs(qty="40", symbol="SYNTH")
    signal, orders = strategy()
    for record, field in zip(orders, ["entry_order_id", "exit_order_id"]):
        record["observation"]["id"] = row[field]
    if conflict == "symbol":
        row["symbol"] = "OTHER"
        for fill in fills:
            fill["symbol"] = fill["provider_symbol"] = "OTHER"
    elif conflict == "cash":
        for fill in fills:
            fill["price"] = fill["provider_price"] = str(Decimal(fill["price"]) + 1)
    elif conflict == "date":
        from datetime import datetime

        for field in ("entry_time", "exit_time"):
            row[field] = (datetime.fromisoformat(row[field]) - timedelta(days=1)).isoformat()
        for fill in fills:
            for field in ("transaction_time", "provider_transaction_time"):
                fill[field] = (datetime.fromisoformat(fill[field]) - timedelta(days=1)).isoformat()
    else:
        fills[-1]["provider_transaction_time"] = "2001-02-03T11:00:00.000000001Z"
    result = cohort(records=[row], fills=fills, claims=[owner], signals=[signal], orders=orders)
    assert result["coverage"]["included"] == 0 and result["coverage"]["unknown"] == 2
    assert result["coverage"]["deduplicated"] == 0 and result["stats"][-1]["totalPL"] is None
    assert len(result["strategy_history"]["rows"]) == 1


def test_latest25_orders_by_exact_updated_instant_not_signal_hash():
    signals, orders = [], []
    for index in range(30):
        signal, observed = strategy(sid=f"{index + 1:064x}", symbol=f"F{index}")
        signal["updated_at"] = (NOW - timedelta(seconds=index)).isoformat()
        for order in observed:
            order["observation"]["id"] = f"order-{index}-{order['role']}"
        signals.append(signal)
        orders.extend(observed)
    signals[-1]["updated_at"] = "2001-02-03T12:00:00.000000001Z"
    result = cohort(signals=signals, orders=orders)
    history = result["strategy_history"]
    assert history["has_more"] and len(history["rows"]) == 25
    assert history["rows"][0]["signal_id"] == signals[-1]["signal_id"]
    assert history["source_as_of"] == "2001-02-03T12:00:00.000000001+00:00"
    assert len(result["rows"]) == 31
