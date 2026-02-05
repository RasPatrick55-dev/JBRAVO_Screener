import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


class TestMonitorLatestTradePrices(unittest.TestCase):
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

    def test_get_latest_trade_prices(self):
        self.monitor.MONITOR_METRICS["live_price_ok"] = 0
        with mock.patch.object(
            self.monitor.data_client,
            "get_stock_latest_trade",
            return_value={
                "AAPL": SimpleNamespace(price=123.45),
                "MSFT": SimpleNamespace(price="250.1"),
            },
        ) as latest_mock:
            prices = self.monitor.get_latest_trade_prices(["AAPL", "MSFT", "GOOG"])

        self.assertEqual(prices, {"AAPL": 123.45, "MSFT": 250.1})
        self.assertEqual(self.monitor.MONITOR_METRICS.get("live_price_ok"), 1)
        self.assertEqual(latest_mock.call_count, 1)
