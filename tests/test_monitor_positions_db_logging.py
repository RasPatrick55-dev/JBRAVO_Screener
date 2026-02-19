import importlib
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorDbLogging(unittest.TestCase):
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

    def test_sell_fill_logs_and_closes_trade(self):
        monitor = self.monitor
        start_ok = int(monitor.MONITOR_METRICS.get("db_event_ok", 0))
        start_fail = int(monitor.MONITOR_METRICS.get("db_event_fail", 0))

        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=101.23,
            created_at=datetime.now(timezone.utc),
        )

        with (
            mock.patch.object(monitor, "db_logging_enabled", return_value=True),
            mock.patch("scripts.db.insert_order_event", return_value=True) as insert_event,
            mock.patch("scripts.db.insert_executed_trade", return_value=True) as insert_exec,
            mock.patch("scripts.db.close_trade_on_sell_fill", return_value=True) as close_trade,
            mock.patch.object(monitor, "wait_for_order_terminal", return_value="filled"),
            mock.patch.object(
                monitor,
                "broker_submit_order",
                return_value=SimpleNamespace(id="OID-1", status="accepted", dryrun=False),
            ),
            mock.patch.object(
                monitor.trading_client,
                "get_all_positions",
                return_value=[SimpleNamespace(symbol="AAPL", qty_available=10, qty=10)],
            ),
            mock.patch.object(monitor, "log_trade_exit", return_value=None),
            mock.patch.object(monitor, "_persist_metrics", return_value=None),
        ):
            monitor.submit_sell_market_order(
                position,
                reason="Test reason",
                reason_code="monitor",
            )

        self.assertEqual(insert_event.call_count, 2)
        close_trade.assert_called_once()
        insert_exec.assert_called_once()

        event_types = [call.kwargs.get("event_type") for call in insert_event.call_args_list]
        self.assertIn(monitor.MONITOR_DB_EVENT_TYPES["sell_submit"], event_types)
        self.assertIn(monitor.MONITOR_DB_EVENT_TYPES["sell_fill"], event_types)

        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_ok", 0)), start_ok + 2)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_fail", 0)), start_fail)

        monitor.MONITOR_METRICS["db_event_ok"] = start_ok
        monitor.MONITOR_METRICS["db_event_fail"] = start_fail

    def test_trailing_adjust_logs_cancel_and_adjust(self):
        monitor = self.monitor
        start_ok = int(monitor.MONITOR_METRICS.get("db_event_ok", 0))
        start_fail = int(monitor.MONITOR_METRICS.get("db_event_fail", 0))

        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=120.0,
            created_at=datetime.now(timezone.utc),
        )
        trailing_order = SimpleNamespace(
            id="TS-1",
            order_type="trailing_stop",
            status="accepted",
            trail_percent=3.0,
            stop_price=99.0,
        )

        original_cooldowns = dict(monitor.TIGHTEN_COOLDOWNS)
        monitor.TIGHTEN_COOLDOWNS.clear()
        try:
            with (
                mock.patch.object(monitor, "db_logging_enabled", return_value=True),
                mock.patch("scripts.db.insert_order_event", return_value=True) as insert_event,
                mock.patch.object(
                    monitor.trading_client, "get_orders", return_value=[trailing_order]
                ),
                mock.patch.object(monitor, "cancel_order_safe", return_value=True),
                mock.patch.object(
                    monitor,
                    "broker_submit_order",
                    return_value=SimpleNamespace(id="TS-2", status="accepted", dryrun=False),
                ),
                mock.patch.object(monitor, "_persist_metrics", return_value=None),
                mock.patch.object(monitor, "_save_tighten_cooldowns", return_value=None),
            ):
                monitor.manage_trailing_stop(position)
        finally:
            monitor.TIGHTEN_COOLDOWNS.clear()
            monitor.TIGHTEN_COOLDOWNS.update(original_cooldowns)

        event_types = [call.kwargs.get("event_type") for call in insert_event.call_args_list]
        self.assertIn(monitor.MONITOR_DB_EVENT_TYPES["trail_cancel"], event_types)
        self.assertIn(monitor.MONITOR_DB_EVENT_TYPES["trail_adjust"], event_types)
        self.assertEqual(insert_event.call_count, 2)

        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_ok", 0)), start_ok + 2)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_fail", 0)), start_fail)

        monitor.MONITOR_METRICS["db_event_ok"] = start_ok
        monitor.MONITOR_METRICS["db_event_fail"] = start_fail

    def test_db_logging_disabled_skips_events(self):
        monitor = self.monitor
        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=101.23,
            created_at=datetime.now(timezone.utc),
        )
        with (
            mock.patch.object(monitor, "db_logging_enabled", return_value=False),
            mock.patch("scripts.db.insert_order_event", return_value=True) as insert_event,
            mock.patch("scripts.db.close_trade_on_sell_fill", return_value=True) as close_trade,
            mock.patch.object(monitor, "wait_for_order_terminal", return_value="filled"),
            mock.patch.object(
                monitor,
                "broker_submit_order",
                return_value=SimpleNamespace(id="OID-2", status="accepted", dryrun=False),
            ),
            mock.patch.object(
                monitor.trading_client,
                "get_all_positions",
                return_value=[SimpleNamespace(symbol="AAPL", qty_available=10, qty=10)],
            ),
            mock.patch.object(monitor, "log_trade_exit", return_value=None),
            mock.patch.object(monitor, "_persist_metrics", return_value=None),
        ):
            monitor.submit_sell_market_order(
                position,
                reason="Test reason",
                reason_code="monitor",
            )
        self.assertFalse(insert_event.called)
        self.assertFalse(close_trade.called)

    def test_partial_exit_does_not_close_trade(self):
        monitor = self.monitor
        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=101.23,
            created_at=datetime.now(timezone.utc),
        )
        with (
            mock.patch.object(monitor, "db_logging_enabled", return_value=True),
            mock.patch("scripts.db.insert_order_event", return_value=True) as insert_event,
            mock.patch("scripts.db.close_trade_on_sell_fill", return_value=True) as close_trade,
            mock.patch.object(monitor, "wait_for_order_terminal", return_value="filled"),
            mock.patch.object(
                monitor,
                "broker_submit_order",
                return_value=SimpleNamespace(id="OID-3", status="accepted", dryrun=False),
            ),
            mock.patch.object(
                monitor.trading_client,
                "get_all_positions",
                return_value=[SimpleNamespace(symbol="AAPL", qty_available=10, qty=10)],
            ),
            mock.patch.object(monitor, "log_trade_exit", return_value=None),
            mock.patch.object(monitor, "_persist_metrics", return_value=None),
        ):
            monitor.submit_sell_market_order(
                position,
                reason="Partial exit",
                reason_code="partial_gain",
                qty_override=5,
            )
        self.assertEqual(insert_event.call_count, 2)
        self.assertFalse(close_trade.called)

    def test_db_event_fail_metric_increments(self):
        monitor = self.monitor
        start_ok = int(monitor.MONITOR_METRICS.get("db_event_ok", 0))
        start_fail = int(monitor.MONITOR_METRICS.get("db_event_fail", 0))
        with (
            mock.patch.object(monitor, "db_logging_enabled", return_value=True),
            mock.patch("scripts.db.insert_order_event", side_effect=RuntimeError("boom")),
        ):
            ok = monitor.db_log_event(
                event_type=monitor.MONITOR_DB_EVENT_TYPES["sell_submit"],
                symbol="AAPL",
                qty=1,
                order_id="OID-9",
                status="submitted",
            )
        self.assertFalse(ok)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_ok", 0)), start_ok)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_fail", 0)), start_fail + 1)
        monitor.MONITOR_METRICS["db_event_ok"] = start_ok
        monitor.MONITOR_METRICS["db_event_fail"] = start_fail

    def test_invalid_event_type_increments_fail(self):
        monitor = self.monitor
        start_ok = int(monitor.MONITOR_METRICS.get("db_event_ok", 0))
        start_fail = int(monitor.MONITOR_METRICS.get("db_event_fail", 0))
        with mock.patch.object(monitor, "db_logging_enabled", return_value=True):
            ok = monitor.db_log_event(
                event_type="NOT_REAL",
                symbol="AAPL",
                qty=1,
                order_id="OID-X",
                status="submitted",
            )
        self.assertFalse(ok)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_ok", 0)), start_ok)
        self.assertEqual(int(monitor.MONITOR_METRICS.get("db_event_fail", 0)), start_fail + 1)
        monitor.MONITOR_METRICS["db_event_ok"] = start_ok
        monitor.MONITOR_METRICS["db_event_fail"] = start_fail
