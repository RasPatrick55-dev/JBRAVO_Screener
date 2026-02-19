import importlib
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock


class TestMonitorExitSignals(unittest.TestCase):
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

    def test_signal_reversal(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=98.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        indicators = {
            "close": 98.0,
            "RSI": 48.0,
            "RSI_prev": 72.0,
            "SMA9": 95.0,
            "EMA20": 99.0,
        }
        signals = monitor.evaluate_exit_signals(position, indicators, datetime.now(timezone.utc))
        codes = {signal["code"] for signal in signals}
        self.assertIn("SIGNAL_REVERSAL", codes)

    def test_profit_target_hit(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=110.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        indicators = {"close": 110.0}
        with mock.patch.dict(os.environ, {"PROFIT_TARGET_PCT": "5"}):
            signals = monitor.evaluate_exit_signals(
                position, indicators, datetime.now(timezone.utc)
            )
        codes = {signal["code"] for signal in signals}
        self.assertIn("PROFIT_TARGET_HIT", codes)

    def test_time_stop(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=95.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=monitor.MAX_HOLD_DAYS + 2),
        )
        signals = monitor.evaluate_exit_signals(position, {}, datetime.now(timezone.utc))
        codes = {signal["code"] for signal in signals}
        self.assertIn("TIME_STOP", codes)

    def test_risk_off(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=100.0,
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        indicators = {"close": 100.0, "risk_score": 0.9}
        with mock.patch.dict(os.environ, {"RISK_OFF_SCORE_CUTOFF": "0.7"}):
            signals = monitor.evaluate_exit_signals(
                position, indicators, datetime.now(timezone.utc)
            )
        codes = {signal["code"] for signal in signals}
        self.assertIn("RISK_OFF", codes)

    def test_momentum_fade(self):
        monitor = self.monitor
        position = SimpleNamespace(
            avg_entry_price=100.0,
            current_price=99.8,
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        indicators = {
            "close": 99.8,
            "close_prev": 100.0,
            "high": 100.0,
            "high_prev": 100.0,
        }
        with mock.patch.dict(os.environ, {"MOMENTUM_STALL_PCT": "0.5"}):
            signals = monitor.evaluate_exit_signals(
                position, indicators, datetime.now(timezone.utc)
            )
        codes = {signal["code"] for signal in signals}
        self.assertIn("MOMENTUM_FADE", codes)
