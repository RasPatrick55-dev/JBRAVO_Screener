"""Offline fixture tests: no broker, production database or secret access."""

from copy import deepcopy
from datetime import datetime, timezone
import unittest
from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as m

A = "11111111-1111-1111-1111-111111111111"
ORDER_ID = "22222222-2222-2222-2222-222222222222"
C = "33333333-3333-3333-3333-333333333333"
S = "a" * 64
CLIENT = "jbr2-" + "b" * 40


def fixtures():
    signal = dict(
        account_id=A,
        signal_id=S,
        symbol="XYZ",
        payload=dict(
            proposal=dict(signal_id=S, symbol="XYZ"),
            handoff=dict(
                paper_only=True,
                handoff_id="handoff",
                strategy_contract={"version": "1"},
            ),
        ),
    )
    order = dict(
        account_id=A,
        signal_id=S,
        client_order_id=CLIENT,
        operation="POST",
        role="entry",
        state="OBSERVED",
        lineage=dict(
            signal_id=S, handoff_id="handoff", strategy_contract={"version": "1"}
        ),
        observation=dict(
            id=ORDER_ID,
            client_order_id=CLIENT,
            symbol="XYZ",
            side="buy",
            filled_qty="2",
            qty="2",
            status="filled",
        ),
    )
    row = dict(
        activity_id="fill1",
        activity_type="FILL",
        transaction_time=datetime(2026, 11, 5, 9, tzinfo=timezone.utc),
        symbol="XYZ",
        side="buy",
        qty="2",
        price="10.125",
        amount=None,
        order_id=ORDER_ID,
        description=None,
        raw={"id": "fill1", "qty": "2"},
    )
    proof = dict(
        checkpoint_id=C,
        source_fingerprint=cp.source_fingerprint(cp.ENDPOINT, A),
        recipe_sha256="c" * 64,
        query_axis="ALPACA_CREATED_AT",
        query_after="2026-11-04T00:00:00Z",
        query_until="2026-11-05T10:00:00Z",
        persisted_manifest=cp.manifestation([row]),
        fetched_count=1,
        manifest_sha256=cp.digest(cp.manifestation([row])),
    )
    return dict(
        account_id=A,
        signals=[signal],
        orders=[order],
        fills=[dict(row, activity_pk=7)],
        checkpoint=proof,
        expected_source=proof["source_fingerprint"],
        expected_recipe="c" * 64,
        checkpoint_rows=[row],
    )


class Tests(unittest.TestCase):
    def test_exact_link(self):
        p = m.build_plan(**fixtures())
        self.assertEqual(p["fills"][0]["price"], "10.125")
        self.assertEqual(p["coverage"][0]["status"], "MATCHED")
        self.assertEqual(p["trading_operations"], 0)

    def test_replay(self):
        self.assertEqual(m.build_plan(**fixtures()), m.build_plan(**fixtures()))

    def test_cancel_not_new_order(self):
        f = fixtures()
        o = deepcopy(f["orders"][0])
        o["operation"] = "DELETE"
        o["role"] = "cancel"
        f["orders"].append(o)
        self.assertEqual(len(m.build_plan(**f)["orders"]), 1)

    def test_missing_activity_not_manufactured(self):
        f = fixtures()
        f["fills"] = []
        p = m.build_plan(**f)
        self.assertEqual(p["fills"], [])
        self.assertEqual(p["coverage"][0]["status"], "MISSING_FILL_ACTIVITIES")

    def test_unmatched_external(self):
        f = fixtures()
        f["orders"] = []
        p = m.build_plan(**f)
        self.assertEqual(p["fills"], [])
        self.assertEqual(p["unmatched"][0]["reason"], "ORDER_NOT_STRATEGY_OWNED")

    def test_precision(self):
        self.assertEqual(m.number("0.00000001"), m.Decimal("0.00000001"))

    def test_numbers(self):
        for v in (True, 1.2, "NaN", "Infinity", "-1", None):
            with self.subTest(v=v), self.assertRaises(ValueError):
                m.number(v)

    def test_wrong_account(self):
        f = fixtures()
        f["account_id"] = ORDER_ID
        with self.assertRaisesRegex(ValueError, "ACCOUNT_SOURCE"):
            m.build_plan(**f)

    def test_missing_signal(self):
        f = fixtures()
        f["signals"] = []
        with self.assertRaisesRegex(ValueError, "SIGNAL_MISSING"):
            m.build_plan(**f)

    def test_duplicate_order(self):
        f = fixtures()
        f["orders"] *= 2
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            m.build_plan(**f)

    def test_duplicate_fill(self):
        f = fixtures()
        f["fills"] *= 2
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            m.build_plan(**f)

    def test_uncheckpointed(self):
        f = fixtures()
        f["fills"][0]["price"] = "11"
        with self.assertRaisesRegex(ValueError, "NOT_CHECKPOINTED"):
            m.build_plan(**f)

    def test_checkpoint_drift(self):
        f = fixtures()
        f["checkpoint_rows"][0]["price"] = "11"
        with self.assertRaisesRegex(ValueError, "LEDGER_MISMATCH"):
            m.build_plan(**f)

    def test_recipe(self):
        f = fixtures()
        f["expected_recipe"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE_MISMATCH"):
            m.build_plan(**f)

    def test_fill_exceeds_observation(self):
        f = fixtures()
        f["orders"][0]["observation"]["filled_qty"] = "1"
        self.assertEqual(
            m.build_plan(**f)["coverage"][0]["status"], "ORDER_SNAPSHOT_BEHIND"
        )

    def test_fill_exceeds_order_quantity(self):
        f = fixtures()
        f["orders"][0]["observation"].update(qty="1", filled_qty="1")
        with self.assertRaisesRegex(ValueError, "EXCEEDS_ORDER_QUANTITY"):
            m.build_plan(**f)

    def test_wrong_symbol(self):
        f = fixtures()
        f["orders"][0]["observation"]["symbol"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "IDENTITY"):
            m.build_plan(**f)

    def test_wrong_lineage(self):
        f = fixtures()
        f["orders"][0]["lineage"]["handoff_id"] = "other"
        with self.assertRaisesRegex(ValueError, "LINEAGE"):
            m.build_plan(**f)

    def test_unknown_order_not_invented(self):
        f = fixtures()
        f["orders"][0].update(state="UNKNOWN", observation=None)
        p = m.build_plan(**f)
        self.assertFalse(p["orders"])
        self.assertEqual(len(p["unmatched"]), 1)

    def test_sell_side_contract(self):
        f = fixtures()
        f["orders"][0]["role"] = "exit"
        with self.assertRaisesRegex(ValueError, "SIDE"):
            m.build_plan(**f)

    def test_no_accounting_claim(self):
        self.assertEqual(
            m.build_plan(**fixtures())["accounting_status"],
            "NOT_IMPLEMENTED_NO_LOT_ALLOCATION",
        )


if __name__ == "__main__":
    unittest.main()


# All provider and database ports in these tests are synthetic.
import pytest
pytestmark = pytest.mark.alpaca_optional
