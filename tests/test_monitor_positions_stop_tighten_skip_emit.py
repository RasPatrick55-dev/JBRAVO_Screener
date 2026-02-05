import importlib
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorStopTightenSkipEmit(unittest.TestCase):
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

    def test_stop_tighten_skip_emits(self):
        monitor = self.monitor
        start_skipped = int(monitor.MONITOR_METRICS.get("stop_tighten_skipped", 0))
        monitor.MONITOR_METRICS["stop_tighten_skipped"] = 0

        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=110.0,
            created_at=datetime.now(timezone.utc),
        )
        trailing_order = SimpleNamespace(
            id="OID-1",
            order_type="trailing_stop",
            status="accepted",
            trail_percent=3.0,
            hwm=120.0,
            stop_price=None,
        )

        with mock.patch.object(monitor.trading_client, "get_orders", return_value=[trailing_order]), \
            mock.patch.object(monitor, "calculate_days_held", return_value=3), \
            mock.patch.object(monitor, "cancel_order_safe", return_value=None) as cancel_mock, \
            mock.patch.object(monitor, "broker_submit_order", return_value=SimpleNamespace(id="DRYRUN")) as submit_mock, \
            mock.patch.object(monitor, "_persist_metrics", return_value=None):
            with self.assertLogs(monitor.logger, level="INFO") as log_ctx:
                monitor.manage_trailing_stop(position)

        self.assertFalse(cancel_mock.called)
        self.assertFalse(submit_mock.called)
        skip_line = next(
            (message for message in log_ctx.output if "STOP_TIGHTEN_SKIP" in message),
            None,
        )
        self.assertIsNotNone(skip_line, "Expected STOP_TIGHTEN_SKIP log entry")
        prefix = "STOP_TIGHTEN_SKIP "
        start = skip_line.find(prefix)
        self.assertNotEqual(start, -1, "STOP_TIGHTEN_SKIP payload missing")
        payload = json.loads(skip_line[start + len(prefix):])

        required_keys = {"symbol", "reason", "existing_stop", "proposed_stop"}
        self.assertTrue(required_keys.issubset(payload), "STOP_TIGHTEN_SKIP missing keys")
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["reason"], "would_lower_stop")

        existing_stop = float(payload["existing_stop"])
        proposed_stop = float(payload["proposed_stop"])
        self.assertLess(proposed_stop, existing_stop)
        self.assertAlmostEqual(existing_stop, 116.4, places=1)
        self.assertEqual(
            int(monitor.MONITOR_METRICS.get("stop_tighten_skipped", 0)),
            1,
        )

        monitor.MONITOR_METRICS["stop_tighten_skipped"] = start_skipped
