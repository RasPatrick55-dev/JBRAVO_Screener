import importlib
import os
import sys
import unittest
from unittest import mock


class TestMonitorDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.copy()
        os.environ["APCA_API_KEY_ID"] = "test_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["APCA_DATA_API_BASE_URL"] = "https://data.alpaca.markets"
        os.environ["ALPACA_DATA_FEED"] = "iex"
        os.environ["MONITOR_DISABLE_SELLS"] = "true"

        sys.modules.pop("scripts.monitor_positions", None)
        cls.monitor = importlib.import_module("scripts.monitor_positions")

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        sys.modules.pop("scripts.monitor_positions", None)

    def test_dryrun_blocks_alpaca_mutations(self):
        with mock.patch.object(
            self.monitor.trading_client, "submit_order", return_value=None
        ) as submit_mock, mock.patch.object(
            self.monitor.trading_client, "cancel_order_by_id", return_value=None
        ) as cancel_by_id_mock, mock.patch.object(
            self.monitor.trading_client, "cancel_order", return_value=None, create=True
        ) as cancel_mock, mock.patch.object(
            self.monitor.trading_client, "close_position", return_value=None
        ) as close_mock:
            self.monitor.submit_new_trailing_stop("AAPL", 1, 3.0)
            self.monitor.cancel_order_safe("OID-1", "AAPL", reason="test_cancel")
            self.monitor.broker_close_position("AAPL", {"reason": "test_close"})

            self.assertFalse(submit_mock.called)
            self.assertFalse(cancel_by_id_mock.called)
            self.assertFalse(cancel_mock.called)
            self.assertFalse(close_mock.called)
