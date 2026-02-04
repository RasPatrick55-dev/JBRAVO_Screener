import importlib
import os
import sys
import unittest


class TestMonitorPaperGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.copy()
        os.environ["APCA_API_KEY_ID"] = "test_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["APCA_DATA_API_BASE_URL"] = "https://data.alpaca.markets"
        os.environ["ALPACA_DATA_FEED"] = "iex"

        sys.modules.pop("scripts.monitor_positions", None)
        cls.monitor = importlib.import_module("scripts.monitor_positions")

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        sys.modules.pop("scripts.monitor_positions", None)

    def test_guard_blocks_live_base_url(self):
        os.environ["APCA_API_BASE_URL"] = "https://api.alpaca.markets"
        with self.assertRaises(SystemExit) as ctx:
            self.monitor.assert_paper_mode()
        self.assertEqual(ctx.exception.code, 2)

    def test_guard_allows_paper_base_url(self):
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        try:
            self.monitor.assert_paper_mode()
        except SystemExit as exc:
            self.fail(f"Unexpected SystemExit: {exc}")
