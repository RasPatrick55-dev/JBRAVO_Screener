import os
import unittest
from unittest.mock import MagicMock
from datetime import datetime
import pandas as pd
import pytz

from scripts.utils import cache_bars, get_last_trading_day_end

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

if __name__ == '__main__':
    unittest.main()
