from copy import deepcopy
from datetime import datetime, timezone
from unittest import TestCase
from scripts import strategy_data_projection as p
from scripts import strategy_fill_linkage as core
from scripts import activity_ingestion_checkpoint as cp
from test_strategy_fill_linkage import fixtures, A, ORDER_ID

EXIT = "44444444-4444-4444-4444-444444444444"


def cohort():
    f = fixtures()
    f["signals"][0].update(state="OPEN", revision=4)
    entry = f["orders"][0]
    entry["observation"].update(
        type="limit",
        time_in_force="day",
        extended_hours=True,
        updated_at="2026-11-05T09:00:00Z",
        replaces=None,
        replaced_by=None,
    )
    stop = deepcopy(entry)
    stop.update(client_order_id="jbr2-" + "d" * 40, role="stop")
    stop["observation"].update(
        id=EXIT,
        client_order_id=stop["client_order_id"],
        side="sell",
        type="stop",
        time_in_force="gtc",
        extended_hours=False,
        filled_qty="2",
        updated_at="2026-11-05T09:30:00Z",
    )
    f["orders"].append(stop)
    row = deepcopy(f["checkpoint_rows"][0])
    row.update(
        activity_id="exit1",
        order_id=EXIT,
        side="sell",
        price="9.125",
        transaction_time=datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc),
    )
    f["checkpoint_rows"].append(row)
    f["fills"].append(dict(row, activity_pk=8))
    manifest = cp.manifestation(f["checkpoint_rows"])
    f["checkpoint"].update(
        persisted_manifest=manifest,
        manifest_sha256=cp.digest(manifest),
        fetched_count=2,
    )
    return f


def state(f):
    return {k: f[k] for k in ("signals", "orders")}


class Projection(TestCase):
    def test_actual_fill_closure(self):
        f = cohort()
        plan = p.closure_plan(
            state(f),
            core.build_plan(**f),
            account_id=A,
        )
        self.assertEqual(plan["signals"][0]["after_state"], "CLOSED")
        self.assertEqual(plan["signals"][0]["sold_qty"], "2")
        self.assertFalse(plan["accounting_complete"])

    def test_partial_does_not_close(self):
        f = cohort()
        f["orders"][1]["observation"]["filled_qty"] = "1"
        f["fills"][1]["qty"] = "1"
        f["checkpoint_rows"][1]["qty"] = "1"
        manifest = cp.manifestation(f["checkpoint_rows"])
        f["checkpoint"].update(
            persisted_manifest=manifest, manifest_sha256=cp.digest(manifest)
        )
        plan = p.closure_plan(state(f), core.build_plan(**f), account_id=A)
        self.assertEqual(plan["signals"], [])
        self.assertEqual(plan["held"][0]["reason"], "OPEN_OR_PARTIAL_QUANTITY")

    def test_unknown_claim_blocks_closure(self):
        f = cohort()
        link = core.build_plan(**f)
        f["orders"].append(
            dict(
                f["orders"][0],
                state="UNKNOWN",
                observation=None,
                client_order_id="jbr2-" + "e" * 40,
            )
        )
        self.assertEqual(
            p.closure_plan(state(f), link, account_id=A)["held"][0]["reason"],
            "UNRESOLVED_CLAIM",
        )

    def test_live_order_blocks_closure(self):
        f = cohort()
        f["orders"][1]["observation"]["status"] = "partially_filled"
        self.assertEqual(
            p.closure_plan(state(f), core.build_plan(**f), account_id=A)["held"][0][
                "reason"
            ],
            "LIVE_ORDER_OR_FILL_GAP",
        )

    def test_snapshot_refresh_exact_ids(self):
        f = cohort()
        old = deepcopy(f["orders"][1]["observation"])
        f["orders"][1]["observation"].update(
            filled_qty="0", status="new", updated_at="2026-11-05T09:00:00Z"
        )
        plan = p.snapshot_plan(
            state(f),
            {ORDER_ID: f["orders"][0]["observation"], EXIT: old},
            account_id=A,
            observed_at="2026-11-05T10:00:00Z",
        )
        self.assertEqual(len(plan["orders"]), 1)
        self.assertEqual(plan["orders"][0]["before"]["filled_qty"], "0")
        self.assertEqual(plan["orders"][0]["after"]["filled_qty"], "2")

    def test_snapshot_identity_change_rejected(self):
        f = cohort()
        snap = {o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]}
        snap[EXIT]["symbol"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "IDENTITY_CHANGED"):
            p.snapshot_plan(f, snap, account_id=A, observed_at="2026-11-05T10:00:00Z")

    def test_replacement_relationship_change_rejected(self):
        f = cohort()
        snap = {o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]}
        snap[EXIT]["replaced_by"] = "55555555-5555-5555-5555-555555555555"
        with self.assertRaisesRegex(ValueError, "IDENTITY_CHANGED"):
            p.snapshot_plan(f, snap, account_id=A, observed_at="2026-11-05T10:00:00Z")

    def test_quantity_regression_rejected(self):
        f = cohort()
        snap = {o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]}
        snap[EXIT]["filled_qty"] = "1"
        with self.assertRaisesRegex(ValueError, "QUANTITY_REGRESSED"):
            p.snapshot_plan(f, snap, account_id=A, observed_at="2026-11-05T10:00:00Z")

    def test_unknown_cancel_blocks_closure(self):
        f = cohort()
        link = core.build_plan(**f)
        f["orders"].append(
            dict(
                f["orders"][0],
                operation="DELETE",
                role="cancel",
                state="UNKNOWN",
                observation=None,
                client_order_id="jbr2-" + "e" * 40,
            )
        )
        self.assertEqual(
            p.closure_plan(state(f), link, account_id=A)["held"][0]["reason"],
            "UNRESOLVED_CLAIM",
        )

    def test_zero_fill_not_expired_by_data_sync(self):
        f = cohort()
        f["orders"] = f["orders"][:1]
        f["orders"][0]["observation"].update(status="canceled", filled_qty="0")
        f["fills"] = []
        f["checkpoint_rows"] = []
        f["checkpoint"].update(
            persisted_manifest=[], manifest_sha256=cp.digest([]), fetched_count=0
        )
        plan = p.closure_plan(state(f), core.build_plan(**f), account_id=A)
        self.assertEqual(plan["signals"], [])
        self.assertEqual(plan["held"][0]["reason"], "NO_ACTUAL_ENTRY_FILLS")

    def test_recognition_only_original_observation_preserved(self):
        f = cohort()
        f["signals"][0]["payload"]["execution_owner"] = {
            "scope": "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS"
        }
        snaps = {
            o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]
        }
        snaps[ORDER_ID]["filled_avg_price"] = "10.125"
        plan = p.snapshot_plan(
            state(f), snaps, account_id=A, observed_at="2026-11-05T10:00:00Z"
        )
        self.assertEqual(plan["orders"], [])
        self.assertEqual(len(plan["protected_observation_differences"]), 1)
        closed = p.closure_plan(state(f), core.build_plan(**f), account_id=A)
        self.assertEqual(closed["signals"], [])
        self.assertEqual(closed["held"][0]["reason"], "PROTECTED_EXTERNAL_OWNERSHIP")

    def test_native_python310_nine_digit_broker_timestamp(self):
        stamp = "2026-11-05T09:20:13.735519768+00:00"
        whole, nanos = p.broker_time_key(stamp)
        self.assertEqual(whole, datetime(2026, 11, 5, 9, 20, 13, tzinfo=timezone.utc))
        self.assertEqual(nanos, 735519768)
        f = cohort()
        f["orders"][1]["observation"]["updated_at"] = stamp
        snaps = {
            o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]
        }
        snaps[EXIT]["updated_at"] = "2026-11-05T09:20:13.735519769Z"
        plan = p.snapshot_plan(
            state(f), snaps, account_id=A, observed_at="2026-11-05T10:00:00Z"
        )
        self.assertEqual(len(plan["orders"]), 1)

    def test_submicrosecond_regression_is_not_truncated_away(self):
        f = cohort()
        f["orders"][1]["observation"]["updated_at"] = "2026-11-05T09:20:13.735519768Z"
        snaps = {
            o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]
        }
        snaps[EXIT]["updated_at"] = "2026-11-05T09:20:13.735519767Z"
        with self.assertRaisesRegex(ValueError, "TIME_REGRESSED"):
            p.snapshot_plan(
                state(f), snaps, account_id=A, observed_at="2026-11-05T10:00:00Z"
            )

    def test_broker_time_offsets_and_supported_precision(self):
        a = p.broker_time_key("2026-11-05T09:20:13.1Z")
        b = p.broker_time_key("2026-11-05T03:20:13.100000000-06:00")
        self.assertEqual(a, b)
        self.assertEqual(p.broker_time_key("2026-11-05T09:20:13Z")[1], 0)
        self.assertEqual(p.broker_time_key("2026-11-05T09:20:13.000001Z")[1], 1000)
        for value in [
            "2026-11-05T09:20:13.1234567890Z",
            "2026-11-05T09:20:13",
            "2026-13-05T09:20:13Z",
        ]:
            with self.assertRaisesRegex(ValueError, "BROKER_TIME_INVALID"):
                p.broker_time_key(value)

    def test_protected_timestamp_regression_is_evidence_not_update(self):
        f = cohort()
        protected_signal = deepcopy(f["signals"][0])
        protected_signal["signal_id"] = "protected-signal"
        protected_signal["payload"]["execution_owner"] = {"scope": "RECOGNITION_ONLY"}
        f["signals"].append(protected_signal)
        f["orders"][0]["signal_id"] = "protected-signal"
        f["orders"][0]["observation"]["updated_at"] = "2026-11-04T15:22:26.18791746Z"
        snaps = {
            o["observation"]["id"]: deepcopy(o["observation"]) for o in f["orders"]
        }
        snaps[ORDER_ID]["updated_at"] = "2026-11-04T15:22:26.187917Z"
        snaps[EXIT]["updated_at"] = "2026-11-05T09:31:00Z"
        before = deepcopy(state(f))
        plan = p.snapshot_plan(
            state(f), snaps, account_id=A, observed_at="2026-11-05T10:00:00Z"
        )
        self.assertEqual(state(f), before)
        self.assertEqual(len(plan["orders"]), 1)
        self.assertEqual(
            plan["orders"][0]["client_order_id"], f["orders"][1]["client_order_id"]
        )
        evidence = plan["protected_observation_evidence"][0]
        self.assertTrue(evidence["timestamp_regressed"])
        self.assertEqual(evidence["before"], before["orders"][0]["observation"])
        self.assertEqual(
            evidence["broker_snapshot"]["updated_at"], snaps[ORDER_ID]["updated_at"]
        )

    def test_protected_observations_keep_all_safety_checks(self):
        for field, value, error in [
            ("updated_at", "2026-11-05T11:00:00Z", "TIME_REGRESSED"),
            ("updated_at", "invalid", "BROKER_TIME_INVALID"),
            ("symbol", "WRONG", "IDENTITY_CHANGED"),
            ("filled_qty", "1", "QUANTITY_REGRESSED"),
            ("status", "new", "TERMINAL_CHANGED"),
        ]:
            with self.subTest(field=field, value=value):
                f = cohort()
                f["signals"][0]["payload"]["execution_owner"] = {
                    "scope": "RECOGNITION_ONLY"
                }
                snaps = {
                    o["observation"]["id"]: deepcopy(o["observation"])
                    for o in f["orders"]
                }
                snaps[ORDER_ID][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    p.snapshot_plan(
                        state(f),
                        snaps,
                        account_id=A,
                        observed_at="2026-11-05T10:00:00Z",
                    )


# All provider and database ports in these tests are synthetic.
import pytest
pytestmark = pytest.mark.alpaca_optional
