import importlib
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorTrailingIntelligence(unittest.TestCase):
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

    def test_trail_never_loosens(self):
        monitor = self.monitor
        position = SimpleNamespace(
            symbol="AAPL",
            qty=10,
            qty_available=10,
            avg_entry_price=100.0,
            current_price=101.0,
            created_at=datetime.now(timezone.utc),
        )
        trailing_order = SimpleNamespace(
            id="TS-1",
            order_type="trailing_stop",
            status="accepted",
            trail_percent=1.0,
            stop_price=99.0,
        )
        with (
            mock.patch.object(monitor.trading_client, "get_orders", return_value=[trailing_order]),
            mock.patch.object(monitor, "broker_submit_order") as submit_mock,
            mock.patch.object(monitor, "cancel_order_safe", return_value=None),
            mock.patch.object(monitor, "_persist_metrics", return_value=None),
        ):
            monitor.manage_trailing_stop(position)
        self.assertFalse(submit_mock.called)

    def test_exit_signal_tightens_target(self):
        monitor = self.monitor
        position = SimpleNamespace(
            symbol="AAPL",
            avg_entry_price=100.0,
            current_price=103.0,
            created_at=datetime.now(timezone.utc),
        )
        indicators = {"close": 103.0}
        base_target, _ = monitor._compute_target_trail_meta(position, indicators, [])
        signals = [{"code": "SIGNAL_REVERSAL", "confidence": 0.9, "detail": "x", "priority": 1}]
        tightened, reasons = monitor._compute_target_trail_meta(position, indicators, signals)
        self.assertLessEqual(tightened, base_target)
        self.assertIn("exit_signal", reasons)

    def test_primary_signal_priority(self):
        monitor = self.monitor
        signals = [
            {"code": "PROFIT_TARGET_HIT", "confidence": 0.6, "detail": "a", "priority": 5},
            {"code": "SIGNAL_REVERSAL", "confidence": 0.7, "detail": "b", "priority": 2},
            {"code": "RISK_OFF", "confidence": 0.8, "detail": "c", "priority": 2},
        ]
        primary = monitor._select_primary_exit_signal(signals)
        self.assertEqual(primary["code"], "RISK_OFF")

    def test_exit_snapshot_keys(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=105.0,
            created_at=datetime.now(timezone.utc),
        )
        indicators = {"close": 105.0, "high": 106.0, "low": 99.0}
        signals = [{"code": "PROFIT_TARGET_HIT", "confidence": 0.6, "detail": "x", "priority": 5}]
        snapshot = monitor._build_exit_snapshot(position, indicators, signals, 2.0, 0.4)
        required = {
            "gain_pct",
            "days_held",
            "max_runup_pct",
            "max_drawdown_pct",
            "signals",
            "trail_pct",
            "risk_score",
        }
        self.assertTrue(required.issubset(set(snapshot.keys())))
