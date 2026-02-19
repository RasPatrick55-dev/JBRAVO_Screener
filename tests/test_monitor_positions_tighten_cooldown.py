import importlib
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType


class TestMonitorTightenCooldown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.copy()
        os.environ["APCA_API_KEY_ID"] = "test_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["APCA_DATA_API_BASE_URL"] = "https://data.alpaca.markets"
        os.environ["ALPACA_DATA_FEED"] = "iex"
        os.environ["MONITOR_TIGHTEN_COOLDOWN_MINUTES"] = "15"

        sys.modules.pop("scripts.monitor_positions", None)
        cls.monitor = importlib.import_module("scripts.monitor_positions")

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        sys.modules.pop("scripts.monitor_positions", None)

    def test_tighten_cooldown_blocks_adjust(self):
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

        now_utc = datetime.now(timezone.utc)
        original_cooldowns = dict(self.monitor.TIGHTEN_COOLDOWNS)
        self.monitor.TIGHTEN_COOLDOWNS.clear()
        self.monitor.TIGHTEN_COOLDOWNS["AAPL"] = (now_utc - timedelta(minutes=5)).isoformat()

        cancel_mock = mock.Mock(return_value=True)
        metric_mock = mock.Mock()

        try:
            with (
                mock.patch.object(
                    self.monitor.trading_client, "get_orders", return_value=[trailing_order]
                ),
                mock.patch.object(self.monitor, "cancel_order_safe", cancel_mock),
                mock.patch.object(self.monitor, "increment_metric", metric_mock),
                self.assertLogs(self.monitor.logger, level="INFO") as log_ctx,
            ):
                self.monitor.manage_trailing_stop(position)
        finally:
            self.monitor.TIGHTEN_COOLDOWNS.clear()
            self.monitor.TIGHTEN_COOLDOWNS.update(original_cooldowns)

        self.assertFalse(cancel_mock.called)
        self.assertTrue(any("STOP_TIGHTEN_COOLDOWN" in message for message in log_ctx.output))
        self.assertTrue(
            any(
                call_args[0][0] == "stop_tighten_cooldown"
                for call_args in metric_mock.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
