import importlib
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType


class TestMonitorTightenReservedQty(unittest.TestCase):
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

    def test_tighten_with_reserved_qty(self):
        position = types.SimpleNamespace(
            symbol="AAPL",
            qty=5,
            qty_available=0,
            avg_entry_price=100.0,
            current_price=110.0,
            live_price=110.0,
            created_at=datetime.now(timezone.utc),
        )
        trailing_order = types.SimpleNamespace(
            id="OID-1",
            symbol="AAPL",
            qty=5,
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            status=OrderStatus.NEW,
            trail_percent=3.0,
            hwm=110.0,
            stop_price=None,
        )

        cancel_mock = mock.Mock(return_value=True)
        submit_mock = mock.Mock(
            return_value=types.SimpleNamespace(
                id="OID-2",
                status="submitted",
                dryrun=False,
            )
        )
        parent = mock.Mock()
        parent.attach_mock(cancel_mock, "cancel")
        parent.attach_mock(submit_mock, "submit")

        original_cooldowns = dict(self.monitor.TIGHTEN_COOLDOWNS)
        self.monitor.TIGHTEN_COOLDOWNS.clear()
        try:
            with (
                mock.patch.object(
                    self.monitor.trading_client, "get_orders", return_value=[trailing_order]
                ),
                mock.patch.object(self.monitor, "broker_cancel_order", cancel_mock),
                mock.patch.object(self.monitor, "broker_submit_order", submit_mock),
                mock.patch.object(self.monitor, "log_trailing_stop_event") as log_attach,
                mock.patch.object(self.monitor, "_save_tighten_cooldowns"),
            ):
                self.monitor.manage_trailing_stop(position)
        finally:
            self.monitor.TIGHTEN_COOLDOWNS.clear()
            self.monitor.TIGHTEN_COOLDOWNS.update(original_cooldowns)

        self.assertTrue(cancel_mock.called)
        self.assertTrue(submit_mock.called)
        call_names = [call_item[0] for call_item in parent.mock_calls]
        self.assertIn("cancel", call_names)
        self.assertIn("submit", call_names)
        self.assertLess(call_names.index("cancel"), call_names.index("submit"))

        submit_args = submit_mock.call_args[0]
        request = submit_args[0]
        self.assertEqual(int(float(getattr(request, "qty", 0))), 5)
        self.assertTrue(log_attach.called)
        self.assertTrue(
            any(
                call_args[0][2] == "OID-2" and call_args[0][3] == "adjusted"
                for call_args in log_attach.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
