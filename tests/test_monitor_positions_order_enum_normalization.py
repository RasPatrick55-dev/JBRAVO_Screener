import importlib
import os
import sys
import types
import unittest

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType


class TestMonitorPositionOrderEnumNormalization(unittest.TestCase):
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

    def _classify(self, order):
        order_type = self.monitor.order_attr_str(order, ("order_type", "type"))
        side = self.monitor.order_attr_str(order, ("side",))
        status = self.monitor.order_attr_str(order, ("status",))
        return {
            "trailing_stop": order_type == "trailing_stop",
            "sell": side == "sell",
            "status": status,
        }

    def test_enum_trailing_stop_sell_normalization(self):
        order = types.SimpleNamespace(
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
        )
        result = self._classify(order)
        self.assertTrue(result["trailing_stop"])
        self.assertTrue(result["sell"])
        self.assertEqual(result["status"], "new")

    def test_buy_limit_not_trailing_stop(self):
        order = types.SimpleNamespace(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            status=OrderStatus.NEW,
        )
        result = self._classify(order)
        self.assertFalse(result["trailing_stop"])


if __name__ == "__main__":
    unittest.main()
