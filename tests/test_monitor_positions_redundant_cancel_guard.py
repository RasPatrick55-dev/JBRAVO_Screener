import importlib
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType


class TestMonitorRedundantCancelGuard(unittest.TestCase):
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

    def test_single_trailing_stop_no_cancel(self):
        order = types.SimpleNamespace(
            id="OID-1",
            symbol="AAPL",
            qty=5,
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )

        cancel_mock = mock.Mock(return_value=True)
        with (
            mock.patch.object(self.monitor.trading_client, "get_orders", return_value=[order]),
            mock.patch.object(
                self.monitor.trading_client,
                "get_order_by_id",
                return_value=types.SimpleNamespace(status="submitted"),
            ),
            mock.patch.object(self.monitor, "broker_cancel_order", cancel_mock),
        ):
            self.monitor.check_pending_orders()

        cancel_mock.assert_not_called()

    def test_two_trailing_stops_cancel_older(self):
        now_utc = datetime.now(timezone.utc)
        older = types.SimpleNamespace(
            id="OID-OLD",
            symbol="AAPL",
            qty=5,
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
            created_at=now_utc - timedelta(minutes=10),
        )
        newer = types.SimpleNamespace(
            id="OID-NEW",
            symbol="AAPL",
            qty=5,
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
            created_at=now_utc - timedelta(minutes=1),
        )

        cancel_mock = mock.Mock(return_value=True)
        with (
            mock.patch.object(
                self.monitor.trading_client, "get_orders", return_value=[older, newer]
            ),
            mock.patch.object(
                self.monitor.trading_client,
                "get_order_by_id",
                return_value=types.SimpleNamespace(status="submitted"),
            ),
            mock.patch.object(self.monitor, "broker_cancel_order", cancel_mock),
        ):
            self.monitor.check_pending_orders()

        cancel_mock.assert_called_once()
        called_id = cancel_mock.call_args[0][0]
        self.assertEqual(called_id, "OID-OLD")


if __name__ == "__main__":
    unittest.main()
