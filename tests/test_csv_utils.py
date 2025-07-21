import os
import csv
import tempfile
import unittest

from utils import write_csv_atomic


class TestWriteCsvAtomic(unittest.TestCase):
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'out.csv')
            rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
            write_csv_atomic(path, rows)
            with open(path, newline="") as f:
                reader = list(csv.reader(f))
            self.assertEqual(reader, [["a", "b"], ["1", "2"], ["3", "4"]])

    def test_empty_rows_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'empty.csv')
            write_csv_atomic(path, [], fieldnames=["x", "y"])
            with open(path, newline="") as f:
                reader = list(csv.reader(f))
            self.assertEqual(reader, [["x", "y"]])


if __name__ == '__main__':
    unittest.main()
