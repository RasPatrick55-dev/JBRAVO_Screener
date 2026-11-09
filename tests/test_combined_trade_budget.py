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
