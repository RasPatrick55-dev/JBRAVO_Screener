import unittest
from unittest.mock import MagicMock
import pandas as pd
import datetime

from utils import fetch_bars_with_cutoff

class TestFetchBarsWithCutoff(unittest.TestCase):
    def test_returns_bars_before_cutoff(self):
        client = MagicMock()
        df = pd.DataFrame(
            {
                "open": [1, 2],
                "high": [1, 2],
                "low": [1, 2],
                "close": [1, 2],
                "volume": [10, 20],
            },
            index=pd.to_datetime([
                "2024-01-01",
                "2024-01-03",
            ], utc=True),
        )
        client.get_stock_bars.return_value.df = df
        cutoff = datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)

        result = fetch_bars_with_cutoff("FAKE", client, cutoff)

        self.assertTrue((result.index <= cutoff).all())
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
