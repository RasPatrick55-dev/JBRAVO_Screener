from copy import deepcopy
import unittest
from scripts import strategy_fill_linkage as m
from scripts import activity_ingestion_checkpoint as cp
from scripts.strategy_fill_sync import sync_once
from test_strategy_fill_linkage import fixtures, C


class Additional(unittest.TestCase):
    def test_same_activity_different_pk_rejected(self):
        f = fixtures()
        extra = dict(f["fills"][0], activity_pk=8)
        f["fills"].append(extra)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_IDENTITY"):
            m.build_plan(**f)

    def test_prior_checkpoint_coverage(self):
        f = fixtures()
        prior = (deepcopy(f["checkpoint"]), deepcopy(f["checkpoint_rows"]))
        f["checkpoint"].update(
            checkpoint_id="44444444-4444-4444-4444-444444444444",
            persisted_manifest=[],
            fetched_count=0,
            manifest_sha256=cp.digest([]),
        )
        f["checkpoint_rows"] = []
        f["prior_proofs"] = [prior]
        p = m.build_plan(**f)
        self.assertEqual(p["coverage"][0]["status"], "MATCHED")
        self.assertEqual(p["fills"][0]["checkpoint_id"], C)

    def test_different_partial_fill_prices(self):
        f = fixtures()
        f["fills"][0]["qty"] = "1"
        f["checkpoint_rows"][0]["qty"] = "1"
        row = dict(f["checkpoint_rows"][0], activity_id="fill2", price="10.135")
        f["checkpoint_rows"].append(row)
        f["fills"].append(dict(row, activity_pk=8))
        f["checkpoint"].update(
            persisted_manifest=cp.manifestation(f["checkpoint_rows"]),
            fetched_count=2,
            manifest_sha256=cp.digest(cp.manifestation(f["checkpoint_rows"])),
        )
        p = m.build_plan(**f)
        self.assertEqual(p["coverage"][0]["linked_qty"], "2")
        self.assertEqual([x["price"] for x in p["fills"]], ["10.125", "10.135"])

    def test_ingestion_failure_prevents_links(self):
        called = []

        def fail():
            raise ValueError("incomplete pagination")

        with self.assertRaises(ValueError):
            sync_once(
                ingest=fail,
                read_plan=lambda x: called.append("read"),
                persist=lambda p: called.append("write"),
                reconcile_uncertain=None,
            )
        self.assertEqual(called, [])

    def test_lost_ack_not_retried(self):
        p = m.build_plan(**fixtures())
        called = []

        def fail(p):
            called.append(1)
            raise m.CommitUncertain("lost")

        r = sync_once(
            ingest=lambda: C,
            read_plan=lambda c: p,
            persist=fail,
            reconcile_uncertain=lambda p: {"status": "ABSENT_UNRESOLVED"},
        )
        self.assertEqual(called, [1])
        self.assertEqual(r["status"], "COMMIT_UNRESOLVED")
        self.assertFalse(r["retry_authorized"])

    def test_lost_ack_recovered(self):
        p = m.build_plan(**fixtures())

        def fail(p):
            raise m.CommitUncertain("lost")

        r = sync_once(
            ingest=lambda: C,
            read_plan=lambda c: p,
            persist=fail,
            reconcile_uncertain=lambda p: {"status": "COMMITTED_VERIFIED"},
        )
        self.assertEqual(r["status"], "LINKED")
        self.assertFalse(r["accounting_complete"])

    def test_missing_fill_report(self):
        f = fixtures()
        f["fills"] = []
        p = m.build_plan(**f)
        r = sync_once(
            ingest=lambda: C,
            read_plan=lambda c: p,
            persist=lambda p: {"status": "COMMITTED"},
            reconcile_uncertain=None,
        )
        self.assertEqual(r["status"], "LINKED_WITH_GAPS")
        self.assertEqual(r["missing_fill_orders"], 1)

    def test_unconfirmed_writer_not_success(self):
        p = m.build_plan(**fixtures())
        with self.assertRaisesRegex(ValueError, "PERSISTENCE_NOT_CONFIRMED"):
            sync_once(
                ingest=lambda: C,
                read_plan=lambda c: p,
                persist=lambda p: {"status": "UNKNOWN"},
                reconcile_uncertain=None,
            )

    def test_stale_snapshot_is_not_missing_fills(self):
        f = fixtures()
        f["orders"][0]["observation"]["filled_qty"] = "1"
        p = m.build_plan(**f)
        r = sync_once(
            ingest=lambda: C,
            read_plan=lambda _: p,
            persist=lambda _: {"status": "COMMITTED"},
            reconcile_uncertain=None,
        )
        self.assertEqual(r["stale_order_snapshots"], 1)
        self.assertEqual(r["missing_fill_orders"], 0)
        self.assertEqual(r["status"], "LINKED_WITH_GAPS")


if __name__ == "__main__":
    unittest.main()

import pytest
pytestmark = pytest.mark.alpaca_optional
