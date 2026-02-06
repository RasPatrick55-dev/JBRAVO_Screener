import importlib
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorExitPayloadDb(unittest.TestCase):
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

    def test_time_stop_payload_contains_exit_fields(self):
        monitor = self.monitor
        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=100.0,
            created_at=datetime.now(timezone.utc)
            - timedelta(days=monitor.MAX_HOLD_DAYS + 1),
        )

        with mock.patch.dict(os.environ, {"MONITOR_ENABLE_EXIT_INTELLIGENCE": "true"}), \
            mock.patch.object(monitor, "db_logging_enabled", return_value=True), \
            mock.patch("scripts.db.insert_order_event", return_value=True) as insert_event, \
            mock.patch.object(monitor, "get_open_positions", return_value=[position]), \
            mock.patch.object(monitor, "save_positions_csv", return_value=None), \
            mock.patch.object(monitor, "log_closed_positions", return_value=None), \
            mock.patch.object(monitor, "refresh_ml_risk_state", return_value={}), \
            mock.patch.object(monitor, "has_pending_sell_order", return_value=False), \
            mock.patch.object(monitor, "get_trailing_stop_order", return_value=None), \
            mock.patch.object(monitor, "cancel_order_safe", return_value=None), \
            mock.patch.object(
                monitor,
                "broker_submit_order",
                return_value=SimpleNamespace(id="OID-TS", status="accepted", dryrun=True),
            ), \
            mock.patch.object(monitor, "log_trade_exit", return_value=None), \
            mock.patch.object(monitor, "_persist_metrics", return_value=None), \
            mock.patch.object(monitor.os.path, "exists", return_value=False):
            monitor.process_positions_cycle()

        submit_calls = [
            call
            for call in insert_event.call_args_list
            if call.kwargs.get("event_type") == monitor.MONITOR_DB_EVENT_TYPES["sell_submit"]
        ]
        self.assertTrue(submit_calls)
        raw = submit_calls[0].kwargs.get("raw") or {}

        self.assertIn("exit_reason_code", raw)
        self.assertIn("exit_reason_detail", raw)
        self.assertIn("exit_snapshot", raw)
        self.assertIn("dryrun", raw)
        self.assertEqual(raw.get("exit_reason_code"), "TIME_STOP")
        self.assertIsInstance(raw.get("exit_snapshot"), dict)

    def test_db_event_ok_not_logged_on_insert_false(self):
        monitor = self.monitor
        start_ok = int(monitor.MONITOR_METRICS.get("db_event_ok", 0))
        start_fail = int(monitor.MONITOR_METRICS.get("db_event_fail", 0))

        with mock.patch.object(monitor, "db_logging_enabled", return_value=True), \
            mock.patch("scripts.db.insert_order_event", return_value=False), \
            self.assertLogs(monitor.logger, level="INFO") as captured:
            ok = monitor.db_log_event(
                event_type=monitor.MONITOR_DB_EVENT_TYPES["sell_submit"],
                symbol="AAPL",
                qty=1,
                order_id="OID-FAIL",
                status="submitted",
                raw={"exit_reason_code": "TIME_STOP", "exit_snapshot": {"gain_pct": 1.0}},
            )

        self.assertFalse(ok)
        output = "\n".join(captured.output)
        self.assertIn("DB_EVENT_FAIL", output)
        self.assertNotIn("DB_EVENT_OK", output)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_ok", 0)), start_ok)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_fail", 0)), start_fail + 1)

        monitor.MONITOR_METRICS["db_event_ok"] = start_ok
        monitor.MONITOR_METRICS["db_event_fail"] = start_fail
