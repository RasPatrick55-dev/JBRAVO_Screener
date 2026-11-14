"""Production-adapter contract tests; broker/DB are mocks only."""

from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, patch
from scripts import strategy_fill_sync_adapter as adapter
from scripts import strategy_fill_linkage as core
from scripts import activity_ingestion_checkpoint as cp
from test_strategy_fill_linkage import fixtures, A


class Adapter(TestCase):
    def setUp(self):
        self.recipe = patch.object(cp, "recipe_fingerprint", return_value="c" * 64)
        self.recipe.start()
        self.addCleanup(self.recipe.stop)
        self.f = fixtures()
        self.packet = adapter.prepare(
            account_id=A,
            source=self.f["expected_source"],
            before="2026-11-04T01:00:00Z",
            required_after="2026-11-04T00:00:00Z",
            cutoff=datetime(2026, 11, 5, 10, tzinfo=timezone.utc),
            activities=self.f["checkpoint_rows"],
            normalize=lambda x: (x, None),
        )
        self.connect = MagicMock(side_effect=lambda ro: MagicMock())

    def apply(self):
        return adapter.apply_packet(
            self.packet,
            account_id=A,
            connect=self.connect,
            policy_lock=lambda c: None,
            verify_broker=lambda p: self.f["checkpoint_rows"],
            expected_sha256=self.packet["sha256"],
        )

    def test_prepare_window_and_no_write(self):
        self.assertEqual(self.packet["after"], "2026-11-03T23:55:00+00:00")
        adapter.validate(self.packet, A)
        self.connect.assert_not_called()

    def test_edited_packet_rejected(self):
        self.packet["rows"][0]["price"] = "100"
        with self.assertRaisesRegex(ValueError, "PACKET_CHANGED"):
            self.apply()
        self.connect.assert_not_called()

    def test_forged_rehashed_manifest_rejected_by_broker(self):
        self.packet["rows"][0]["price"] = "100"
        self.packet["manifest"] = cp.manifestation(self.packet["rows"])
        self.packet["sha256"] = core.digest(
            {k: v for k, v in self.packet.items() if k != "sha256"}
        )
        with self.assertRaisesRegex(ValueError, "BROKER_EVIDENCE_CHANGED"):
            self.apply()
        self.connect.assert_not_called()

    def test_changed_account_rejected(self):
        with self.assertRaisesRegex(ValueError, "ACCOUNT_CHANGED"):
            adapter.validate(self.packet, "00000000-0000-0000-0000-000000000000")

    def test_uncertain_ingestion_missing_checkpoint_never_links(self):
        with (
            patch.object(cp, "persist", side_effect=OSError("lost")) as persist,
            patch.object(
                cp, "consume", side_effect=RuntimeError("ACT_CHECKPOINT_MISSING")
            ),
            patch.object(core, "persist_verified_plan") as linker,
        ):
            with self.assertRaisesRegex(RuntimeError, "MISSING"):
                self.apply()
            persist.assert_called_once()
            linker.assert_not_called()
            self.assertEqual(
                [x.args[0] for x in self.connect.call_args_list], [False, True]
            )

    def test_uncertain_ingestion_proven_and_link_ack_lost_recovered(self):
        plan = core.build_plan(**self.f)
        proof = {
            "checkpoint_id": self.f["checkpoint"]["checkpoint_id"],
            "manifest_sha256": cp.digest(self.packet["manifest"]),
        }
        with (
            patch.object(cp, "persist", side_effect=OSError("lost")) as persist,
            patch.object(cp, "consume", return_value=proof),
            patch.object(adapter.store, "load_plan", return_value=plan),
            patch.object(
                core, "persist_verified_plan", side_effect=core.CommitUncertain("sha")
            ) as linker,
            patch.object(
                core, "recover_receipt", return_value={"status": "COMMITTED_VERIFIED"}
            ),
        ):
            result = self.apply()
            self.assertEqual(result["linkage"]["status"], "COMMITTED_VERIFIED")
            self.assertFalse(result["accounting_complete"])
            persist.assert_called_once()
            linker.assert_called_once()
            self.assertEqual(
                [x.args[0] for x in self.connect.call_args_list],
                [False, True, False, True],
            )

    def test_wrong_approved_digest_prevents_work(self):
        with self.assertRaisesRegex(ValueError, "PACKET_NOT_APPROVED"):
            adapter.apply_packet(
                self.packet,
                account_id=A,
                connect=self.connect,
                policy_lock=None,
                verify_broker=None,
                expected_sha256="0" * 64,
            )
        self.connect.assert_not_called()

    def test_uncertain_link_absent_is_explicit(self):
        plan = core.build_plan(**self.f)
        proof = {
            "checkpoint_id": self.f["checkpoint"]["checkpoint_id"],
            "manifest_sha256": cp.digest(self.packet["manifest"]),
        }
        with (
            patch.object(cp, "persist"),
            patch.object(cp, "consume", return_value=proof),
            patch.object(adapter.store, "load_plan", return_value=plan),
            patch.object(
                core, "persist_verified_plan", side_effect=core.CommitUncertain("sha")
            ),
            patch.object(
                core, "recover_receipt", return_value={"status": "ABSENT_UNRESOLVED"}
            ),
        ):
            result = self.apply()
            self.assertEqual(result["status"], "COMMIT_UNRESOLVED")
            self.assertFalse(result["retry_authorized"])

    def test_cli_unresolved_is_nonzero(self):
        import json
        import tempfile
        import sys
        from pathlib import Path

        fake_ingest = MagicMock()
        fake_ingest.requests.get.return_value.json.return_value = {"id": A}
        fake_ingest.requests.get.return_value.status_code = 200
        fake_ingest.fetch_activities.return_value = self.f["checkpoint_rows"]
        fake_ingest._normalize_activity.side_effect = lambda x: (x, None)
        fake_ingest.DEFAULT_BASE_URL = cp.ENDPOINT
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "packet.json"
            path.write_text(json.dumps(self.packet))
            with (
                patch.dict(
                    sys.modules,
                    {
                        "scripts.db": MagicMock(),
                    "dotenv": MagicMock(),
                        "scripts.fetch_account_activities": fake_ingest,
                    },
                ),
                patch.dict("os.environ", {"APCA_API_BASE_URL": cp.ENDPOINT}),
                patch.object(
                    adapter,
                    "apply_packet",
                    return_value={"status": "COMMIT_UNRESOLVED"},
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    adapter.main(
                        [
                            "--packet",
                            str(path),
                            "--apply",
                            "--expected-sha256",
                            self.packet["sha256"],
                        ]
                    ),
                    2,
                )

    def test_unowned_fills_return_gap(self):
        plan = core.build_plan(**self.f)
        plan["unmatched"] = [{"activity_pk": 99, "reason": "ORDER_NOT_STRATEGY_OWNED"}]
        proof = {
            "checkpoint_id": self.f["checkpoint"]["checkpoint_id"],
            "manifest_sha256": cp.digest(self.packet["manifest"]),
        }
        with (
            patch.object(cp, "persist"),
            patch.object(cp, "consume", return_value=proof),
            patch.object(adapter.store, "load_plan", return_value=plan),
            patch.object(
                core, "persist_verified_plan", return_value={"status": "COMMITTED"}
            ),
        ):
            result = self.apply()
            self.assertEqual(result["status"], "LINKED_WITH_GAPS")
            self.assertEqual(result["unmatched_fills"], 1)

    def test_transport_blocks_other_hosts_and_mutation_paths(self):
        request = MagicMock()
        transport = adapter.PaperGetTransport(request)
        for url in [
            "https://api.alpaca.markets/v2/account",
            cp.ENDPOINT + "/v2/orders",
            cp.ENDPOINT + "/v2/account/activities?redirect=elsewhere",
            "https://evil.invalid/v2/account",
        ]:
            with self.assertRaisesRegex(
                ValueError, "READONLY_PAPER_TRANSPORT_REQUIRED"
            ):
                transport.get(url, headers={})
        request.assert_not_called()

    def test_transport_disables_redirects_and_rejects_302(self):
        request = MagicMock()
        request.return_value.status_code = 302
        transport = adapter.PaperGetTransport(request)
        with self.assertRaisesRegex(ValueError, "PAPER_GET_NOT_SUCCESSFUL"):
            transport.get(cp.ENDPOINT + "/v2/account", headers={})
        self.assertIs(request.call_args.kwargs["allow_redirects"], False)
        self.assertEqual(request.call_count, 1)

    def test_transport_exact_order_get(self):
        request = MagicMock()
        request.return_value.status_code = 200
        adapter.PaperGetTransport(request).get(
            cp.ENDPOINT + "/v2/orders/" + A, headers={}
        )
        self.assertEqual(request.call_count, 1)


# All provider and database ports in these tests are synthetic.
import pytest
pytestmark = pytest.mark.alpaca_optional
