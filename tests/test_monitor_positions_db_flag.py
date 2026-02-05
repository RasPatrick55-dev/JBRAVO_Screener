import importlib
import os
import sys
import unittest
from unittest import mock


class TestMonitorDbFlag(unittest.TestCase):
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

    def test_db_logging_disabled_by_default(self):
        with mock.patch("scripts.db.db_enabled", return_value=True), \
            mock.patch("scripts.db.get_db_conn") as get_conn, \
            mock.patch.dict(os.environ, {"MONITOR_ENABLE_DB_LOGGING": ""}, clear=False):
            self.assertFalse(self.monitor.db_logging_enabled())
            get_conn.assert_not_called()

    def test_db_logging_enabled_when_flag_and_db(self):
        with mock.patch("scripts.db.db_enabled", return_value=True), \
            mock.patch("scripts.db.get_db_conn") as get_conn, \
            mock.patch.dict(os.environ, {"MONITOR_ENABLE_DB_LOGGING": "true"}, clear=False):
            self.assertTrue(self.monitor.db_logging_enabled())
            get_conn.assert_not_called()

    def test_db_logging_disabled_when_db_unavailable(self):
        with mock.patch("scripts.db.db_enabled", return_value=False), \
            mock.patch("scripts.db.get_db_conn") as get_conn, \
            mock.patch.dict(os.environ, {"MONITOR_ENABLE_DB_LOGGING": "true"}, clear=False):
            self.assertFalse(self.monitor.db_logging_enabled())
            get_conn.assert_not_called()
