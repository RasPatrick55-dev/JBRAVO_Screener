import importlib
import os
import sys
import types
import unittest
from unittest import mock


class TestMonitorOrphanTrailingStops(unittest.TestCase):
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

    def _positions(self):
        return [
            types.SimpleNamespace(symbol="A", qty="1"),
            types.SimpleNamespace(symbol="B", qty="2"),
        ]

    def _orders(self):
        return [
            types.SimpleNamespace(id="trail-a", symbol="A", order_type="trailing_stop", side="sell", qty="1", trail_percent="4"),
            types.SimpleNamespace(id="trail-b", symbol="B", order_type="trailing_stop", side="sell", qty="2", trail_percent="4"),
            types.SimpleNamespace(id="trail-x", symbol="X", order_type="trailing_stop", side="sell", qty="1", trail_percent="4"),
            types.SimpleNamespace(id="trail-y", symbol="Y", order_type="trailing_stop", side="sell", qty="1", trail_percent="4"),
            types.SimpleNamespace(id="limit-x", symbol="X", order_type="limit", side="sell", qty="1"),
        ]

    def test_cancel_called_only_for_orphan_trailing_stops(self):
        os.environ["ORPHAN_TRAIL_CANCEL"] = "1"
        os.environ["ORPHAN_TRAIL_DRY_RUN"] = "0"

        cancel_mock = mock.Mock()
        with mock.patch.object(self.monitor.trading_client, "cancel_order_by_id", cancel_mock):
            cleaned = self.monitor.cleanup_orphan_trailing_stops(self._positions(), self._orders())

        self.assertEqual(cancel_mock.call_count, 2)
        canceled_ids = {call.args[0] for call in cancel_mock.call_args_list}
        self.assertEqual(canceled_ids, {"trail-x", "trail-y"})
        self.assertNotIn("trail-x", {o.id for o in cleaned})
        self.assertNotIn("trail-y", {o.id for o in cleaned})
        self.assertIn("limit-x", {o.id for o in cleaned})

    def test_dry_run_does_not_call_cancel(self):
        os.environ["ORPHAN_TRAIL_CANCEL"] = "1"
        os.environ["ORPHAN_TRAIL_DRY_RUN"] = "1"

        cancel_mock = mock.Mock()
        with mock.patch.object(self.monitor.trading_client, "cancel_order_by_id", cancel_mock):
            cleaned = self.monitor.cleanup_orphan_trailing_stops(self._positions(), self._orders())

        cancel_mock.assert_not_called()
        self.assertNotIn("trail-x", {o.id for o in cleaned})
        self.assertNotIn("trail-y", {o.id for o in cleaned})

    def test_non_trailing_orders_are_not_canceled(self):
        os.environ["ORPHAN_TRAIL_CANCEL"] = "1"
        os.environ["ORPHAN_TRAIL_DRY_RUN"] = "0"

        orders = [
            types.SimpleNamespace(id="limit-x", symbol="X", order_type="limit", side="sell", qty="1"),
            types.SimpleNamespace(id="stop-y", symbol="Y", order_type="stop", side="sell", qty="1"),
        ]

        cancel_mock = mock.Mock()
        with mock.patch.object(self.monitor.trading_client, "cancel_order_by_id", cancel_mock):
            cleaned = self.monitor.cleanup_orphan_trailing_stops(self._positions(), orders)

        cancel_mock.assert_not_called()
        self.assertEqual([o.id for o in cleaned], ["limit-x", "stop-y"])


if __name__ == "__main__":
    unittest.main()
