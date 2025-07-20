import os
import sqlite3
import tempfile
import unittest
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ensure_db_indicators import (
    ensure_columns,
    sync_columns_from_dataframe,
    REQUIRED_COLUMNS,
)

class TestEnsureDBIndicators(unittest.TestCase):
    def test_columns_created(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            ensure_columns(path, REQUIRED_COLUMNS)
            conn = sqlite3.connect(path)
            cur = conn.execute("PRAGMA table_info(historical_candidates);")
            cols = [row[1] for row in cur.fetchall()]
            for col in REQUIRED_COLUMNS:
                self.assertIn(col, cols)
            conn.close()
        finally:
            os.remove(path)

    def test_sync_columns_from_dataframe(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            df = pd.DataFrame(columns=["date", "symbol", "score", "dynamic_col"])
            sync_columns_from_dataframe(df, path)
            conn = sqlite3.connect(path)
            cur = conn.execute("PRAGMA table_info(historical_candidates);")
            cols = [row[1] for row in cur.fetchall()]
            for col in df.columns:
                self.assertIn(col, cols)
            conn.close()
        finally:
            os.remove(path)

if __name__ == '__main__':
    unittest.main()
