import importlib
import json
import os
import sys
import unittest


class TestMonitorExitSignalsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.copy()
        os.environ["APCA_API_KEY_ID"] = "test_key"
        os.environ["APCA_API_SECRET_KEY"] = "test_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ["APCA_DATA_API_BASE_URL"] = "https://data.alpaca.markets"
        os.environ["ALPACA_DATA_FEED"] = "iex"
        os.environ["MONITOR_ENABLE_EXIT_SIGNALS_V2"] = "true"

        sys.modules.pop("scripts.monitor_positions", None)
        cls.monitor = importlib.import_module("scripts.monitor_positions")

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        sys.modules.pop("scripts.monitor_positions", None)

    def setUp(self):
        self.monitor.RSI_HIGH_MEMORY.clear()

    def _base_indicators(self) -> dict:
        return {
            "close": 100.0,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close_prev": 100.0,
            "open_prev": 100.0,
            "high_prev": 101.0,
            "low_prev": 99.0,
            "EMA20": 90.0,
            "EMA20_prev": 90.0,
            "SMA9": 95.0,
            "SMA9_prev": 95.0,
            "RSI": 60.0,
            "RSI_prev": 60.0,
            "MACD": 1.0,
            "MACD_signal": 0.5,
            "MACD_prev": 1.0,
            "MACD_signal_prev": 0.5,
            "MACD_hist": 0.5,
            "MACD_hist_prev": 0.5,
        }

    def _extract_payload(self, log_output: list[str]) -> dict:
        for message in log_output:
            prefix = "EXIT_SIGNAL_V2 "
            idx = message.find(prefix)
            if idx != -1:
                return json.loads(message[idx + len(prefix):])
        self.fail("EXIT_SIGNAL_V2 log entry not found")

    def _assert_payload(self, payload: dict) -> None:
        required_keys = {
            "symbol",
            "enabled",
            "v2_reasons",
            "baseline_reasons",
            "final_reasons",
            "price",
            "close_prev",
            "rsi_now",
            "rsi_prev",
            "sma9",
            "sma9_prev",
            "ema20",
            "ema20_prev",
            "macd_hist",
            "macd_hist_prev",
            "patterns",
            "thresholds",
        }
        self.assertTrue(required_keys.issubset(payload), "EXIT_SIGNAL_V2 payload missing keys")
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertIsInstance(payload["enabled"], bool)
        self.assertTrue(payload["enabled"])
        self.assertIsInstance(payload["v2_reasons"], list)
        self.assertIsInstance(payload["baseline_reasons"], list)
        self.assertIsInstance(payload["final_reasons"], list)
        for key in (
            "price",
            "close_prev",
            "rsi_now",
            "rsi_prev",
            "sma9",
            "sma9_prev",
            "ema20",
            "ema20_prev",
            "macd_hist",
            "macd_hist_prev",
        ):
            self.assertIsInstance(payload[key], (int, float))
        self.assertIsInstance(payload["patterns"], dict)
        self.assertIsInstance(payload["thresholds"], dict)
        self.assertIn("shooting_star", payload["patterns"])
        self.assertIn("bearish_engulfing", payload["patterns"])
        self.assertIsInstance(payload["patterns"]["shooting_star"], bool)
        self.assertIsInstance(payload["patterns"]["bearish_engulfing"], bool)
        self.assertIn("RSI_REVERSAL_DROP", payload["thresholds"])
        self.assertIn("RSI_REVERSAL_FLOOR", payload["thresholds"])

    def _run_check(self, indicators: dict) -> tuple[list[str], dict]:
        with self.assertLogs("scripts.monitor_positions", level="INFO") as log_ctx:
            reasons = self.monitor.check_sell_signal("AAPL", indicators)
        payload = self._extract_payload(log_ctx.output)
        self._assert_payload(payload)
        return reasons, payload

    def test_rsi_reversal_trigger(self):
        indicators = self._base_indicators()
        indicators.update({"RSI_prev": 75.0, "RSI": 50.0})
        reasons, payload = self._run_check(indicators)
        self.assertIn("RSI reversal", reasons)
        self.assertIn("RSI reversal", payload["v2_reasons"])

    def test_rsi_reversal_small_drop_not_trigger(self):
        indicators = self._base_indicators()
        indicators.update({"RSI_prev": 75.0, "RSI": 70.0})
        reasons, payload = self._run_check(indicators)
        self.assertNotIn("RSI reversal", reasons)
        self.assertNotIn("RSI reversal", payload["v2_reasons"])

    def test_ema20_cross_down(self):
        indicators = self._base_indicators()
        indicators.update({"close_prev": 100.0, "EMA20_prev": 90.0, "close": 85.0, "EMA20": 90.0})
        reasons, payload = self._run_check(indicators)
        self.assertIn("EMA20 cross-down", reasons)
        self.assertIn("EMA20 cross-down", payload["v2_reasons"])

    def test_sma9_ema20_cross_down(self):
        indicators = self._base_indicators()
        indicators.update({"SMA9_prev": 95.0, "EMA20_prev": 90.0, "SMA9": 85.0, "EMA20": 90.0, "close": 95.0})
        reasons, payload = self._run_check(indicators)
        self.assertIn("SMA9/EMA20 cross-down", reasons)
        self.assertIn("SMA9/EMA20 cross-down", payload["v2_reasons"])

    def test_macd_fade(self):
        indicators = self._base_indicators()
        indicators.update(
            {
                "MACD": 0.1,
                "MACD_signal": 0.11,
                "MACD_prev": 0.2,
                "MACD_signal_prev": 0.0,
                "MACD_hist_prev": 0.2,
                "MACD_hist": -0.01,
            }
        )
        reasons, payload = self._run_check(indicators)
        self.assertIn("MACD fade", reasons)
        self.assertIn("MACD fade", payload["v2_reasons"])

    def test_bearish_engulfing(self):
        indicators = self._base_indicators()
        indicators.update(
            {
                "open_prev": 100.0,
                "close_prev": 105.0,
                "open": 106.0,
                "close": 99.0,
            }
        )
        reasons, payload = self._run_check(indicators)
        self.assertIn("Bearish engulfing", reasons)
        self.assertIn("Bearish engulfing", payload["v2_reasons"])
