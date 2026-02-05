import importlib
import os
import sys
import unittest
from unittest import mock


class TestMonitorTrailingLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.copy()
        os.environ["APCA_API_KEY_ID"] = "test_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["APCA_DATA_API_BASE_URL"] = "https://data.alpaca.markets"
        os.environ["ALPACA_DATA_FEED"] = "iex"
        os.environ.pop("MONITOR_ENABLE_BREAKEVEN_TIGHTEN", None)
        os.environ.pop("MONITOR_ENABLE_TIMEDECAY_TIGHTEN", None)

        sys.modules.pop("scripts.monitor_positions", None)
        cls.monitor = importlib.import_module("scripts.monitor_positions")

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        sys.modules.pop("scripts.monitor_positions", None)

    def test_profit_tier_default(self):
        target, reason = self.monitor.compute_target_trail_pct(
            gain_pct=1.0,
            days_held=1,
            current_trail_pct=self.monitor.TRAIL_START_PERCENT,
        )
        self.assertEqual(target, float(self.monitor.TRAIL_START_PERCENT))
        self.assertEqual(reason, "profit_tier")

    def test_breakeven_tighten(self):
        with mock.patch.dict(os.environ, {"MONITOR_ENABLE_BREAKEVEN_TIGHTEN": "true"}):
            target, reason = self.monitor.compute_target_trail_pct(
                gain_pct=2.5,
                days_held=1,
                current_trail_pct=3.0,
            )
        self.assertLessEqual(target, 2.5)
        self.assertIn("breakeven_lock", reason)

    def test_time_decay_tighten(self):
        with mock.patch.dict(os.environ, {"MONITOR_ENABLE_TIMEDECAY_TIGHTEN": "true"}):
            target, reason = self.monitor.compute_target_trail_pct(
                gain_pct=1.0,
                days_held=5,
                current_trail_pct=None,
            )
        self.assertLess(target, float(self.monitor.TRAIL_START_PERCENT))
        self.assertIn("time_decay", reason)

    def test_ratchet_only(self):
        target, reason = self.monitor.compute_target_trail_pct(
            gain_pct=1.0,
            days_held=1,
            current_trail_pct=1.5,
        )
        self.assertEqual(target, 1.5)
        self.assertEqual(reason, "profit_tier")
