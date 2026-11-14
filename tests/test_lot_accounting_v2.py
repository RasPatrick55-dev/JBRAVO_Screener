from copy import deepcopy
from unittest import TestCase
from scripts import strategy_lot_accounting_v2 as lot
from scripts import strategy_fill_linkage as core
from test_data_projection import cohort
from test_strategy_fill_linkage import A


def inputs():
    f = cohort()
    return lot.source_lots(core.build_plan(**f), f["fills"])


def reviewed(lots, **kwargs):
    return lot.plan(
        lots,
        account_id=A,
        history_review={
            "manifest_sha256": core.digest(lots),
            "review_id": "synthetic-fixture-only",
        },
        **kwargs,
    )


class LotPlanner(TestCase):
    def test_basic_and_no_pnl(self):
        p = reviewed(inputs())
        self.assertEqual(len(p["events"]), 1)
        self.assertEqual(p["events"][0]["quantity"], "2")
        self.assertIsNone(p["net_pnl"])
        self.assertIsNone(p["gross_pnl"])
        self.assertFalse(p["accounting_complete"])

    def test_history_not_inferred(self):
        p = lot.plan(inputs(), account_id=A)
        self.assertFalse(p["events"])
        self.assertEqual(p["held"][0]["reason"], "BOUNDED_HISTORY_REVIEW_REQUIRED")

    def test_timestamp_offsets_nanoseconds_and_precision(self):
        a = lot.timestamp("2026-11-05T23:59:59.123456789-01:00")
        b = lot.timestamp("2026-11-06T00:59:59.123456789Z")
        self.assertEqual(a["start_ns"], b["start_ns"])
        self.assertEqual(a["end_ns"] - a["start_ns"], 1)
        c = lot.timestamp("2026-11-06T00:59:59.1Z")
        self.assertEqual(c["end_ns"] - c["start_ns"], 100000000)
        for invalid in [
            "2026-11-05T01:02:03",
            "2026-11-05T01:02:03.1234567890Z",
            "2026-11-05T01:02:03+02:99",
        ]:
            with self.assertRaises(ValueError):
                lot.timestamp(invalid)

    def test_db_time_is_only_microseconds(self):
        self.assertTrue(
            all(
                x["time_source"] == "DB_MICROSECOND_ONLY"
                and x["ordering"]["fraction_digits"] == 6
                for x in inputs()
            )
        )

    def test_precision_overlap_held(self):
        ls = inputs()
        ls[0]["ordering"] = lot.timestamp("2026-11-05T09:00:00.1Z")
        ls[1]["ordering"] = lot.timestamp("2026-11-05T09:00:00.123456Z")
        p = reviewed(ls)
        self.assertFalse(p["events"])
        self.assertEqual(p["held"][0]["reason"], "ORDERING_PRECISION_OVERLAP")

    def test_recipe_does_not_double_allocate(self):
        ls = inputs()
        p = reviewed(ls)
        q = reviewed(ls, prior_events=p["events"], recipe="v2")
        self.assertEqual(q["events"], [])

    def test_oversell_and_future_buy_held(self):
        ls = inputs()
        next(x for x in ls if x["side"] == "sell")["quantity"] = "3"
        self.assertFalse(reviewed(ls)["events"])
        ls = inputs()
        next(x for x in ls if x["side"] == "sell")["ordering"] = lot.timestamp(
            "2026-11-05T08:00:00Z"
        )
        self.assertFalse(reviewed(ls)["events"])

    def test_late_entry_held(self):
        ls = inputs()
        p = reviewed(ls)
        new = deepcopy(next(x for x in ls if x["side"] == "buy"))
        new.update(
            activity_id="late",
            activity_pk=99,
            ordering=lot.timestamp("2026-11-05T08:00:00Z"),
        )
        q = reviewed(ls + [new], prior_events=p["events"])
        self.assertEqual(q["held"][0]["reason"], "LATE_EVIDENCE_REVIEW_REQUIRED")

    def test_fractional_and_partial(self):
        ls = inputs()
        for x in ls:
            x["quantity"] = "0.123456789012345678"
        self.assertEqual(reviewed(ls)["events"][0]["quantity"], "0.123456789012345678")

    def test_raw_timestamp_mismatch_rejected(self):
        f = cohort()
        f["fills"][0]["raw"]["transaction_time"] = "2026-11-05T08:00:00Z"
        with self.assertRaises(ValueError):
            lot.source_lots(core.build_plan(**f), f["fills"])

    def test_raw_nanosecond_time_without_datetime_precision_loss(self):
        f = cohort()
        f["fills"][0]["raw"]["transaction_time"] = "2026-11-05T09:00:00.000000789Z"
        f["checkpoint_rows"][0]["raw"] = f["fills"][0]["raw"]
        from scripts import activity_ingestion_checkpoint as cp

        manifest = cp.manifestation(f["checkpoint_rows"])
        f["checkpoint"].update(
            persisted_manifest=manifest, manifest_sha256=cp.digest(manifest)
        )
        ls = lot.source_lots(core.build_plan(**f), f["fills"])
        buy = next(x for x in ls if x["side"] == "buy")
        self.assertEqual(buy["ordering"]["start_ns"] % 1000, 789)
        self.assertEqual(buy["ordering"]["fraction_digits"], 9)

    def test_rounding_uncertainty_crosses_reported_boundary(self):
        a = lot.timestamp("2026-11-05T09:00:00.186204Z")
        b = lot.timestamp("2026-11-05T09:00:00.186203862Z")
        self.assertLess(a["lower_ns"], b["start_ns"])
        self.assertGreater(a["end_ns"], b["start_ns"])
        ls = inputs()
        ls[0]["ordering"] = a
        ls[1]["ordering"] = b
        self.assertEqual(
            reviewed(ls)["held"][0]["reason"], "ORDERING_PRECISION_OVERLAP"
        )

import pytest
pytestmark = pytest.mark.alpaca_optional
