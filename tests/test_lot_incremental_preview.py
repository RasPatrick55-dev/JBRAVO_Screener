from copy import deepcopy
import json
from pathlib import Path
from unittest import TestCase
from test_data_projection import cohort
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot
from scripts import strategy_lot_incremental_preview as auto


def inputs():
    f = cohort()
    linked = core.build_plan(**f)
    origins = {
        o["observation"]["id"]: {
            **o["observation"],
            "created_at": "2026-11-05T08:00:00Z",
        }
        for o in f["orders"]
    }
    return dict(
        account_id=f["account_id"],
        linked=linked,
        source_rows=f["fills"],
        canonical_checkpoints={
            x["activity_id"]: x["checkpoint_id"] for x in linked["fills"]
        },
        signals=f["signals"],
        orders=f["orders"],
        checkpoint=f["checkpoint"],
        checkpoint_rows=f["checkpoint_rows"],
        expected_recipe=f["expected_recipe"],
        origins=origins,
        prior_events=[],
    )


class AutomaticPreview(TestCase):
    def setUp(self):
        self.i = inputs()

    def run_preview(self):
        return auto.preview(**self.i)

    def reason(self):
        return self.run_preview()["held"][0]["reason"]

    def test_eligible_exact_partial_exit_lots(self):
        p = self.run_preview()
        self.assertEqual(len(p["candidates"]), 1)
        packet = p["candidates"][0]["packet"]
        self.assertEqual([e["quantity"] for e in packet["events"]], ["2"])
        self.assertFalse(p["execution_allowed"])
        self.assertFalse(p["accounting_complete"])
        self.assertIsNone(packet["net_pnl"])

    def test_replay_no_duplicate_packet(self):
        first = self.run_preview()["candidates"][0]["packet"]
        self.i["prior_events"] = first["events"]
        p = self.run_preview()
        self.assertEqual(p["candidates"], [])
        self.assertEqual(p["noops"][0]["reason"], "ALREADY_FULLY_ALLOCATED")

    def test_residual_allocation_exact(self):
        first = self.run_preview()["candidates"][0]["packet"]
        e = deepcopy(first["events"][0])
        e["quantity"] = "1"
        e["event_id"] = lot.event_id(e)
        self.i["prior_events"] = [e]
        p = self.run_preview()
        self.assertEqual(p["candidates"][0]["packet"]["events"][0]["quantity"], "1")

    def test_origin_not_inferred_from_submitted(self):
        for x in self.i["origins"].values():
            x["submitted_at"] = x.pop("created_at")
        self.assertEqual(self.reason(), "IMMUTABLE_ORDER_ORIGIN_REQUIRED")

    def test_protected(self):
        self.i["signals"][0]["payload"]["execution_owner"] = "protected"
        self.assertEqual(self.reason(), "PROTECTED_EXTERNAL_OWNERSHIP")

    def test_open_order(self):
        self.i["orders"][1]["observation"]["status"] = "partially_filled"
        self.assertEqual(self.reason(), "NONTERMINAL_OR_PARTIAL_ORDER")

    def test_unknown_claim(self):
        self.i["orders"][1]["state"] = "UNKNOWN"
        self.assertEqual(self.reason(), "UNRESOLVED_ORDER_CLAIM")

    def test_fill_gap(self):
        self.i["orders"][1]["observation"]["filled_qty"] = "1"
        self.assertEqual(self.reason(), "INCOMPLETE_ORDER_FILLS")

    def test_history_gap(self):
        self.i["checkpoint"]["query_after"] = "2026-11-05T08:30:00Z"
        self.assertEqual(self.reason(), "FULL_ORIGIN_HISTORY_NOT_COVERED")

    def test_changed_manifest(self):
        self.i["checkpoint"]["manifest_sha256"] = "0" * 64
        self.assertEqual(self.reason(), "CHECKPOINT_PROVENANCE_INCOMPLETE")

    def test_changed_origin_snapshot(self):
        next(iter(self.i["origins"].values()))["symbol"] = "OTHER"
        self.assertEqual(self.reason(), "ORIGIN_SNAPSHOT_MISMATCH")

    def test_changed_owner_lineage(self):
        self.i["orders"][0]["lineage"]["handoff_id"] = "different"
        self.assertEqual(self.reason(), "OWNERSHIP_LINEAGE_CHANGED")

    def test_correction_holds(self):
        e = self.run_preview()["candidates"][0]["packet"]["events"][0]
        self.i["prior_events"] = [
            e,
            lot.reversal(e, reason="fixture", evidence_sha256="e" * 64),
        ]
        self.assertEqual(self.reason(), "CORRECTION_REVIEW_REQUIRED")

    def test_duplicate_prior_rejected(self):
        e = self.run_preview()["candidates"][0]["packet"]["events"][0]
        self.i["prior_events"] = [e, e]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_EVENT"):
            self.run_preview()

    def test_prior_overallocation_rejected(self):
        e = self.run_preview()["candidates"][0]["packet"]["events"][0]
        e["quantity"] = "3"
        e["event_id"] = lot.event_id(e)
        self.i["prior_events"] = [e]
        with self.assertRaisesRegex(ValueError, "PRIOR_BUDGET_INVALID"):
            self.run_preview()


    def test_origin_after_actual_fill_holds(self):
        next(iter(self.i["origins"].values()))["created_at"] = "2026-11-05T09:20:00Z"
        self.assertEqual(self.reason(), "ORDER_ORIGIN_AFTER_OR_OVERLAPS_FILL")

    def test_missing_old_activity_stops_global_reconciliation(self):
        e = self.run_preview()["candidates"][0]["packet"]["events"][0]
        e["entry_activity_id"] = "unknown-prior"
        e["event_id"] = lot.event_id(e)
        self.i["prior_events"] = [e]
        with self.assertRaisesRegex(ValueError, "PRIOR_HISTORY_MISSING"):
            self.run_preview()

    def test_narrow_window_keeps_old_cohort_global_budget(self):
        old = self.run_preview()["candidates"][0]["packet"]
        previous = deepcopy(old["lots"])
        for x in previous:
            x["activity_id"] = "old-" + x["activity_id"]
            x["signal_id"] = "old-signal"
        events = deepcopy(old["events"])
        for e in events:
            for key in ("entry_activity_id", "exit_activity_id"):
                e[key] = "old-" + e[key]
            e["event_id"] = lot.event_id(e)
        self.i.update(prior_events=events, prior_lots=previous)
        self.assertEqual(len(self.run_preview()["candidates"]), 1)
        previous[0]["quantity"] = "1"
        with self.assertRaisesRegex(ValueError, "PRIOR_BUDGET_INVALID"):
            self.run_preview()


    def test_full_fetch_extra_owned_fill_holds(self):
        from scripts import activity_ingestion_checkpoint as cp
        extra = deepcopy(self.i["checkpoint_rows"][0])
        extra["activity_id"] = "late-correction"
        self.i["checkpoint_rows"].append(extra)
        manifest = cp.manifestation(self.i["checkpoint_rows"])
        self.i["checkpoint"].update(persisted_manifest=manifest,
            manifest_sha256=cp.digest(manifest), fetched_count=len(manifest))
        self.assertEqual(self.reason(), "FULL_FETCH_COHORT_SET_MISMATCH")

    def test_current_ledger_extra_owned_fill_holds(self):
        extra = deepcopy(self.i["checkpoint_rows"][0])
        extra["activity_id"] = "later-ledger-correction"
        self.i["owned_ledger_rows"] = self.i["checkpoint_rows"] + [extra]
        self.assertEqual(self.reason(), "CURRENT_LEDGER_COHORT_SET_MISMATCH")

    def test_fully_allocated_later_fill_is_not_noop(self):
        self.i['prior_events']=self.run_preview()['candidates'][0]['packet']['events']
        extra=deepcopy(self.i['source_rows'][0]); extra['activity_id']='late-after-allocation'
        self.i['owned_ledger_rows']=self.i['source_rows']+[extra]
        result=self.run_preview()
        self.assertEqual(result['noops'],[])
        self.assertEqual(result['held'][0]['reason'],'CURRENT_LEDGER_COHORT_SET_MISMATCH')

import pytest
pytestmark = pytest.mark.alpaca_optional
