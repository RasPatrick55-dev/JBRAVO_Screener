import importlib
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorStopTightenEmit(unittest.TestCase):
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

    def test_stop_tighten_emits_and_increments(self):
        monitor = self.monitor
        start_breakeven = int(monitor.MONITOR_METRICS.get("breakeven_tightens", 0))
        start_stops_tightened = int(monitor.MONITOR_METRICS.get("stops_tightened", 0))

        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=102.5,
            created_at=datetime.now(timezone.utc),
        )
        trailing_order = SimpleNamespace(
            id="OID-1",
            order_type="trailing_stop",
            status="accepted",
            trail_percent=3.0,
            stop_price=99.0,
        )

        with mock.patch.dict(os.environ, {"MONITOR_ENABLE_BREAKEVEN_TIGHTEN": "true"}), \
            mock.patch.object(monitor.trading_client, "get_orders", return_value=[trailing_order]), \
            mock.patch.object(monitor, "cancel_order_safe", return_value=None), \
            mock.patch.object(monitor, "broker_submit_order", return_value=SimpleNamespace(id="DRYRUN")), \
            mock.patch.object(monitor, "_persist_metrics", return_value=None):
            with self.assertLogs(monitor.logger, level="INFO") as log_ctx:
                monitor.manage_trailing_stop(position)

        tighten_line = next(
            (
                message
                for message in log_ctx.output
                if "STOP_TIGHTEN " in message and "STOP_TIGHTEN_SKIP" not in message
            ),
            None,
        )
        self.assertIsNotNone(tighten_line, "Expected STOP_TIGHTEN log entry")
        prefix = "STOP_TIGHTEN "
        start = tighten_line.find(prefix)
        self.assertNotEqual(start, -1, "STOP_TIGHTEN payload missing")
        payload = json.loads(tighten_line[start + len(prefix):])

        required_keys = {"symbol", "from", "to", "gain_pct", "days_held", "reason"}
        self.assertTrue(required_keys.issubset(payload), "STOP_TIGHTEN missing keys")
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertIn("breakeven_lock", str(payload["reason"]))

        from_val = float(payload["from"])
        to_val = float(payload["to"])
        gain_pct = float(payload["gain_pct"])
        self.assertLess(to_val, from_val)
        self.assertAlmostEqual(gain_pct, 2.5, places=1)
        self.assertIsInstance(payload["days_held"], int)
        self.assertGreaterEqual(payload["days_held"], 0)
        self.assertEqual(
            int(monitor.MONITOR_METRICS.get("breakeven_tightens", 0)),
            start_breakeven + 1,
        )
        self.assertEqual(
            int(monitor.MONITOR_METRICS.get("stops_tightened", 0)),
            start_stops_tightened + 1,
        )

        monitor.MONITOR_METRICS["breakeven_tightens"] = start_breakeven
        monitor.MONITOR_METRICS["stops_tightened"] = start_stops_tightened
