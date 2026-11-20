import json
import hashlib
from unittest import TestCase
from unittest.mock import Mock
from scripts import strategy_lot_hourly_phase as phase


class Handoff(TestCase):
    def setUp(self):
        self.receipts = {name: b"{}" for name in phase.REQUIRED}
        checkpoint = {"checkpoint_id": "fixture"}
        self.receipts["linkage_result.json"] = json.dumps(
            {
                "status": "LINKED",
                "linkage": {"status": "COMMITTED", "plan_sha256": "fixture"},
                "trading_operations": 0,
                "checkpoint": checkpoint,
            }
        ).encode()
        self.c = {
            "status": "COMPLETED_EXPECTED_HOLDS",
            "native_run_id": "a" * 32,
            "slot": "20261106T06",
            "trading_operations": 0,
            "accounting_complete": False,
            "checkpoint": checkpoint,
            "native_receipts": {
                n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                for n, b in self.receipts.items()
            },
        }
        self.loader = Mock(side_effect=ValueError("READONLY_LOADER_REACHED"))

    def invoke(self):
        raw = json.dumps(self.c).encode()
        return phase.after_sync_preview(
            completion_bytes=raw,
            expected_completion_sha256=hashlib.sha256(raw).hexdigest(),
            receipt_bytes=self.receipts,
            expected_slot="20261106T06",
            load_inputs=self.loader,
        )

    def test_real_completed_handoff_reaches_loader(self):
        with self.assertRaisesRegex(ValueError, "READONLY_LOADER_REACHED"):
            self.invoke()
        self.loader.assert_called_once()

    def change_linkage(self, *, status="LINKED", commit="COMMITTED_VERIFIED"):
        name = "linkage_result.json"
        receipt = json.loads(self.receipts[name])
        receipt["status"] = status
        receipt["linkage"]["status"] = commit
        raw = json.dumps(receipt).encode()
        self.receipts[name] = raw
        self.c["native_receipts"][name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def test_verified_commit_reaches_loader_with_original_bindings(self):
        self.change_linkage()
        with self.assertRaisesRegex(ValueError, "READONLY_LOADER_REACHED"):
            self.invoke()
        self.loader.assert_called_once_with(
            run_id="a" * 32, checkpoint_id="fixture", plan_sha256="fixture"
        )

    def test_verified_commit_with_fill_gaps_does_not_load(self):
        self.change_linkage(status="LINKED_WITH_GAPS")
        with self.assertRaisesRegex(ValueError, "COMMITTED_LINKAGE_REQUIRED"):
            self.invoke()
        self.loader.assert_not_called()

    def test_unresolved_commit_does_not_load(self):
        self.change_linkage(commit="ABSENT_UNRESOLVED")
        with self.assertRaisesRegex(ValueError, "COMMITTED_LINKAGE_REQUIRED"):
            self.invoke()
        self.loader.assert_not_called()

    def test_skipped_run_does_not_load(self):
        self.c["status"] = "SKIPPED_ALREADY_COMPLETE"
        with self.assertRaisesRegex(ValueError, "NO_COMPLETED_SYNC"):
            self.invoke()
        self.loader.assert_not_called()

    def test_stale_slot_rejected(self):
        self.c["slot"] = "20261106T05"
        with self.assertRaisesRegex(ValueError, "SYNC_RUN_OR_SLOT_MISMATCH"):
            self.invoke()
        self.loader.assert_not_called()

    def test_modified_receipt_rejected(self):
        self.receipts["summary.json"] = b"changed"
        with self.assertRaisesRegex(ValueError, "SYNC_RECEIPT_PIN"):
            self.invoke()
        self.loader.assert_not_called()

    def test_missing_receipt_rejected(self):
        self.receipts.pop("summary.json")
        with self.assertRaisesRegex(ValueError, "SYNC_RECEIPT_SET"):
            self.invoke()
        self.loader.assert_not_called()

import pytest
pytestmark = pytest.mark.alpaca_optional
