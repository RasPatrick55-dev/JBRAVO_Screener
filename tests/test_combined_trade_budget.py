"""Actual reader/transport fixture; fictional wire responses, not PostgreSQL."""

import inspect, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from dashboards import combined_trade_gross as C
import test_combined_trade_gross as F

sys.path.insert(0, str(Path(__file__).parent / "owned_exit"))
from test_owned_exit_full_consumer import Native

pytestmark = pytest.mark.alpaca_optional


class FullNative(Native):
    def __init__(self):
        super().__init__()
        self.active = False
        self.autocommit = False
        self.connection = self
        self.rollbacks = self.closes = self.batch_reads = 0
        self.signals, self.strategy_orders = F.strategy()

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_transaction_status(self):
        return 2 if self.active else 0

    def rollback(self):
        self.rollbacks += 1
        self.active = False

    def close(self):
        self.closes += 1

    def execute(self, sql, params=None):
        if sql.startswith("SET TRANSACTION"):
            self.active = True
        return super().execute(sql, params)

    def native(self, sql, params):
        if sql in (C.GUARD_PAIR_SQL, C.EXECUTION_PAIR_SQL):
            specs = (
                [("visibility", C.E.VISIBILITY_SQL, None), ("schema", C.E.SCHEMA_SQL, None)]
                if sql == C.GUARD_PAIR_SQL
                else [
                    ("owners", C.E.OWNERS_SQL, params[:3]),
                    ("fills", C.E.FILLS_SQL, (params[3],)),
                ]
            )
            return sorted(
                [
                    (
                        name,
                        str(i),
                        json.dumps(list(row)) if name in ("visibility", "schema") else row[0],
                    )
                    for name, query, args in specs
                    for i, row in enumerate(self.native(query, args), 1)
                ],
                key=lambda row: (row[0], int(row[1])),
            )
        if sql == C.STRATEGY_COHORT_SQL:
            self.batch_reads += 1
            visibility = [
                ["strategy_execution_orders_app", "r", False, False, False, False, True],
                ["strategy_execution_signals_app", "r", False, False, False, False, True],
            ]
            values = {
                "visibility": visibility,
                "signals": [dict(row=self.signals, version="100")],
                "orders": [dict(row=r, version="100") for r in self.strategy_orders],
                "claims": [
                    dict(
                        row={
                            k: r.get(k)
                            for k in ("trade_id", "status", "entry_order_id", "exit_order_id")
                        },
                        version="100",
                    )
                    for r in self.owners
                ],
            }
            return [
                (name, str(i), json.dumps(row))
                for name, rows in sorted(values.items())
                for i, row in enumerate(rows, 1)
            ]
        return super().native(sql, params)


def test_real_policy_two_owned_exit_groups_and_legacy_allocation_fit_budget():
    native = FullNative()
    with patch.object(C.E, "READ_PLANS", {**C.E.READ_PLANS, **C.PLANS}):
        cohort = C.read_cohort(lambda: native, native.identity, F.S.ACCOUNT)
    assert {a.trade_id for a in cohort[0].allocations} == {110, 1003, 1006}
    assert native.batch_reads == 2 and native.rollbacks == native.closes == 1
    assert len(native.calls) <= 128
    assert not any(
        sql.startswith(("INSERT ", "UPDATE ", "DELETE ", "ALTER ", "CREATE "))
        for sql, _ in native.calls
    )
    assert (C.T.MAX_SQL, C.T.MAX_ROWS, C.T.MAX_BYTES) == (128, 4096, 16 * 1024 * 1024)


def batch(rows):
    return C.strategy_cohort(
        SimpleNamespace(execute=lambda *args: None, fetchall=lambda: rows), F.S.ACCOUNT
    )


def valid():
    return [
        (
            "visibility",
            "1",
            json.dumps(["strategy_execution_orders_app", "r", False, False, False, False, True]),
        ),
        (
            "visibility",
            "2",
            json.dumps(["strategy_execution_signals_app", "r", False, False, False, False, True]),
        ),
    ]


@pytest.mark.parametrize(
    "row",
    [
        ("unknown", "1", "{}"),
        ("signals", "2", "{}"),
        ("signals", "1", None),
        ("signals", "x", "{}"),
        ("signals", "\u0661", "{}"),
        ("signals", "1", "not json"),
    ],
)
def test_batch_invalid_wire(row):
    with pytest.raises((ValueError, RuntimeError)):
        batch(valid() + [row])


def test_batch_keeps_strategy_numeric_types():
    signals, orders, claims = batch(
        valid()
        + [
            ("signals", "1", '{"row":{"price":1.25},"version":"100"}'),
            ("claims", "1", '{"row":{"trade_id":1},"version":"100"}'),
        ]
    )
    assert type(signals[0]["row"]["price"]) is float and signals[0]["row"]["price"] == 1.25
    assert claims[0]["row"]["trade_id"] == 1 and orders == []


@pytest.mark.parametrize("section,cap", [("signals", 200), ("orders", 1000), ("claims", 1000)])
def test_batch_overflow_not_truncated(section, cap):
    with pytest.raises(ValueError, match="population_overflow"):
        batch(valid() + [(section, str(i), "{}") for i in range(1, cap + 2)])


def test_native_semantics_preserved():
    source = inspect.getsource(C.read_cohort)
    assert (
        "first = inputs()" in source
        and "second = inputs()" in source
        and "cursor.recheck_canonical()" in source
    )
    assert "T.Cursor(raw, started + 8" in source
    assert C.STRATEGY_COHORT_SQL.count("%s") == 1
    assert all(
        q in C.STRATEGY_COHORT_SQL
        for q in (C.SIGNALS_SQL, C.ORDERS_SQL, C.CLAIMS_SQL, C.STRATEGY_VISIBILITY_SQL)
    )


def test_guarded_case_ctes_have_explicit_body_column_alias():
    for name in ("signals", "orders", "claims"):
        assert name + "(body) AS MATERIALIZED (" in C.STRATEGY_COHORT_SQL


class PairCursor:
    canonical_sql = set()

    def __init__(self, rows=()):
        self.rows, self.calls = list(rows), []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return self.rows


def test_pairing_preserves_exact_sql_and_parameter_arity():
    assert C.E.VISIBILITY_SQL in C.GUARD_PAIR_SQL and C.E.SCHEMA_SQL in C.GUARD_PAIR_SQL
    assert C.E.OWNERS_SQL in C.EXECUTION_PAIR_SQL and C.E.FILLS_SQL in C.EXECUTION_PAIR_SQL
    assert C.EXECUTION_PAIR_SQL.count("%s") == 4
    assert C.GUARD_PAIR_SQL.count("%s") == 0
    assert not {C.E.VISIBILITY_SQL, C.E.SCHEMA_SQL, C.E.OWNERS_SQL, C.E.FILLS_SQL} & {
        sql for sql, plan in C.T.plan_table().items() if plan["kind"] != "control"
    }


def test_pair_rejects_native_replay_overlap():
    cursor = PairCursor()
    cursor.canonical_sql = {C.E.FILLS_SQL}
    with pytest.raises(ValueError, match="projection_pair_native_overlap"):
        C.LegacyInputCapture(cursor)


@pytest.mark.parametrize(
    "sql,params",
    [
        ("SELECT unexpected", None),
        (C.E.SCHEMA_SQL, None),
        (C.E.FILLS_SQL, ([],)),
        (C.E.OWNERS_SQL, ([1], ["a"], ["b"])),
        (C.E.VISIBILITY_SQL, (1,)),
    ],
)
def test_pair_rejects_unexpected_sql_or_parameters(sql, params):
    cursor = PairCursor()
    with pytest.raises(ValueError):
        C.LegacyInputCapture(cursor).execute(sql, params)
    assert cursor.calls == []


def test_pair_requires_matching_fill_population_and_does_not_cache_across_instances():
    cursor = PairCursor()
    capture = C.LegacyInputCapture(cursor)
    capture.execute(C.E.OWNERS_SQL, ([1], ["a"], ["a"]))
    assert capture.fetchall() == []
    with pytest.raises(ValueError, match="projection_pair_sequence_changed"):
        capture.execute(C.E.FILLS_SQL, (["b"],))
    capture.execute(C.E.FILLS_SQL, (["a"],))
    assert capture.fetchall() == [] and capture.fills == []
    second = C.LegacyInputCapture(cursor)
    second.execute(C.E.OWNERS_SQL, ([1], ["a"], ["a"]))
    assert len(cursor.calls) == 2
    assert cursor.calls[0] == (C.EXECUTION_PAIR_SQL, ([1], ["a"], ["a"], ["a"]))


@pytest.mark.parametrize(
    "row",
    [
        ("unknown", "1", "{}"),
        ("owners", "2", "{}"),
        ("fills", "1", None),
        ("schema", "1", "{}"),
        ("visibility", "1", "[true]"),
    ],
)
def test_pair_rejects_malformed_wire(row):
    capture = C.LegacyInputCapture(PairCursor([row]))
    with pytest.raises(ValueError):
        capture.pair(C.GUARD_PAIR_SQL, None, dict(visibility=3, schema=65, owners=1000, fills=4000))


@pytest.mark.parametrize("section,cap", [("owners", 1000), ("fills", 4000)])
def test_pair_rejects_overflow(section, cap):
    capture = C.LegacyInputCapture(PairCursor([(section, str(i), "{}") for i in range(1, cap + 2)]))
    with pytest.raises(ValueError, match="population_overflow"):
        capture.pair(C.EXECUTION_PAIR_SQL, None, {section: cap})


def test_pair_requires_consumption_before_next_query():
    capture = C.LegacyInputCapture(PairCursor())
    capture.execute(C.E.VISIBILITY_SQL)
    with pytest.raises(ValueError, match="projection_pair_unconsumed"):
        capture.execute(C.E.SCHEMA_SQL)
