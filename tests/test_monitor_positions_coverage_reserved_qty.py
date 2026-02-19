import importlib
import os
import sys
import types
import unittest
from unittest import mock

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType


class TestMonitorCoverageReservedQty(unittest.TestCase):
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

    def test_reserved_qty_counts_as_protected(self):
        position = types.SimpleNamespace(
            symbol="AAPL",
            qty=5,
            qty_available=0,
            side="long",
        )
        order = types.SimpleNamespace(
            id="OID-1",
            symbol="AAPL",
            qty=5,
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
        )

        with (
            mock.patch.object(self.monitor.trading_client, "get_orders", return_value=[order]),
            mock.patch.object(self.monitor, "_attach_long_protective_stop") as attach_long,
            mock.patch.object(self.monitor, "_attach_short_protective_stop") as attach_short,
            mock.patch.object(self.monitor, "_record_stop_missing") as stop_missing,
            mock.patch.object(self.monitor, "_persist_metrics"),
            mock.patch.object(self.monitor, "_save_monitor_state"),
        ):
            protected_count, coverage_pct, trailing_count = self.monitor.enforce_stop_coverage(
                [position]
            )

        self.assertEqual(protected_count, 1)
        self.assertAlmostEqual(coverage_pct, 1.0)
        self.assertEqual(trailing_count, 1)
        attach_long.assert_not_called()
        attach_short.assert_not_called()
        stop_missing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
