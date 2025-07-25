import os
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import pytz

from scripts.utils import (
    cache_bars,
    get_last_trading_day_end,
    fetch_daily_bars,
    fetch_extended_hours_bars,
    get_combined_daily_bar,
    fetch_bars_with_cutoff,
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.cache_dir = 'data/test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))

    def test_last_trading_day_weekend(self):
        dt = pytz.timezone('America/New_York').localize(datetime(2024, 8, 10, 12))
        end = get_last_trading_day_end(dt)
        self.assertEqual(end.weekday(), 4)

    def test_cache_bars_handles_empty(self):
        client = MagicMock()
        client.get_stock_bars.return_value.df = pd.DataFrame()
        df = cache_bars('FAKE', client, self.cache_dir, days=10)
        self.assertIsInstance(df, pd.DataFrame)

    def test_fetch_daily_bars_iex_feed(self):
        client = MagicMock()
        mock_df = pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            },
            index=[pd.Timestamp("2024-01-02")],
        )
        client.get_stock_bars.return_value.df = mock_df
        df = fetch_daily_bars("AAPL", "2024-01-02", client)
        req = client.get_stock_bars.call_args
        self.assertTrue(client.get_stock_bars.called)
        self.assertFalse(df.empty)

    def test_fetch_bars_with_cutoff_end_time(self):
        client = MagicMock()
        client.get_stock_bars.return_value.df = pd.DataFrame(
            {"close": [1]}, index=[pd.Timestamp("2024-01-01")]
        )
        fetch_bars_with_cutoff("AAPL", datetime(2024, 1, 1), client)
        args, _ = client.get_stock_bars.call_args
        self.assertTrue(client.get_stock_bars.called)
        req = args[0]
        self.assertIsInstance(req, StockBarsRequest)
        self.assertEqual(req.timeframe, TimeFrame.Day)

    def test_get_combined_daily_bar(self):
        client = MagicMock()
        daily_df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
            },
            index=[pd.Timestamp("2024-01-01")],
        )
        extended_df = pd.DataFrame(
            {
                "open": [1.2, 1.6, 1.7, 1.8],
                "high": [1.3, 2.1, 1.8, 1.9],
                "low": [1.2, 1.3, 1.6, 1.7],
                "close": [1.25, 1.9, 1.85, 1.8],
                "volume": [10, 5, 20, 30],
            },
            index=pd.DatetimeIndex(
                [
                    "2024-01-01 08:30",
                    "2024-01-01 09:00",
                    "2024-01-01 16:10",
                    "2024-01-01 16:50",
                ],
                tz="America/New_York",
            ),
        )
        client.get_stock_bars.side_effect = [MagicMock(df=daily_df), MagicMock(df=extended_df)]

        combined = get_combined_daily_bar("AAPL", "2024-01-01", client)
        row = combined.iloc[0]
        self.assertEqual(row["volume"], 165)
        self.assertAlmostEqual(row["high"], 2.1)
        self.assertAlmostEqual(row["close"], 1.8)

if __name__ == '__main__':
    unittest.main()
