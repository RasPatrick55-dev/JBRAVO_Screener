import tempfile
from pathlib import Path
from unittest import TestCase
from scripts import strategy_fill_auto_sync as a
from scripts import strategy_fill_linkage as core
from scripts import activity_ingestion_checkpoint as cp
from test_strategy_fill_linkage import A


class AutomaticConfig(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pins = {}
        for name in a.PIN_FILES:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture")
            pins[name] = a.hashlib.sha256(p.read_bytes()).hexdigest()
        self.config = dict(
            schema="jbravo.data-sync-authorization.v1",
            scope="AUTOMATIC_PAPER_DATA_INGEST_LINK_OBSERVE_CLOSE_ONLY",
            account_id=A,
            endpoint=cp.ENDPOINT,
            max_window_days=45,
            max_pages=1000,
            initial_required_after="2026-11-04T00:00:00Z",
            source_pins=pins,
            database_policy=["a", "b", "fixture", 1, "fixture"],
        )

    def test_config_pins_exact_scope(self):
        a.verify_config(self.config, core.digest(self.config), self.root)

    def test_unapproved_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "CONFIG_NOT_APPROVED"):
            a.verify_config(self.config, "0" * 64, self.root)

    def test_changed_source_rejected(self):
        (self.root / "scripts/db.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SOURCE_CHANGED"):
            a.verify_config(self.config, core.digest(self.config), self.root)

    def test_live_endpoint_rejected(self):
        self.config["endpoint"] = "https://api.alpaca.markets"
        with self.assertRaisesRegex(ValueError, "PAPER_REQUIRED"):
            a.verify_config(self.config, core.digest(self.config), self.root)

    def test_incomplete_source_scope_rejected(self):
        self.config["source_pins"].pop("scripts/db.py")
        with self.assertRaisesRegex(ValueError, "SOURCE_SCOPE"):
            a.verify_config(self.config, core.digest(self.config), self.root)

    def test_interrupted_run_blocks_future_automatic_attempt(self):
        from unittest.mock import MagicMock, patch
        import sys
        from scripts import strategy_fill_auto_sync as auto

        pending = self.root / "reports" / "strategy_fill_sync" / ("a" * 32)
        pending.mkdir(parents=True)
        config_path = self.root / "config.json"
        config_path.write_text("{}")
        fake_ingest = MagicMock()
        with (
            patch.object(
                auto,
                "__file__",
                str(self.root / "scripts" / "strategy_fill_auto_sync.py"),
            ),
            patch.object(auto, "verify_config", return_value=self.config),
            patch.dict(
                sys.modules,
                {
                    "scripts.db": MagicMock(),
                    "dotenv": MagicMock(),
                    "scripts.fetch_account_activities": fake_ingest,
                },
            ),
            patch.object(auto, "run") as run,
            patch("builtins.print"),
        ):
            self.assertEqual(
                auto.main(
                    [
                        "--config",
                        str(config_path),
                        "--expected-config-sha256",
                        "x",
                        "--run",
                    ]
                ),
                2,
            )
            run.assert_not_called()

    def test_failed_stage_latches_review_and_no_second_attempt(self):
        from unittest.mock import MagicMock, patch
        import sys
        from scripts import strategy_fill_auto_sync as auto

        config_path = self.root / "config.json"
        config_path.write_text("{}")
        with (
            patch.object(
                auto,
                "__file__",
                str(self.root / "scripts" / "strategy_fill_auto_sync.py"),
            ),
            patch.object(auto, "verify_config", return_value=self.config),
            patch.dict(
                sys.modules,
                {
                    "scripts.db": MagicMock(),
                    "dotenv": MagicMock(),
                    "scripts.fetch_account_activities": MagicMock(),
                },
            ),
            patch.object(auto, "run", side_effect=RuntimeError("uncertain")) as run,
            patch("builtins.print"),
        ):
            argv = [
                "--config",
                str(config_path),
                "--expected-config-sha256",
                "x",
                "--run",
            ]
            self.assertEqual(auto.main(argv), 2)
            self.assertTrue(
                (
                    self.root
                    / "reports"
                    / "strategy_fill_sync"
                    / "REVIEW_REQUIRED.json"
                ).is_file()
            )
            self.assertEqual(auto.main(argv), 2)
            run.assert_called_once()


# All provider and database ports in these tests are synthetic.
import pytest
pytestmark = pytest.mark.alpaca_optional
