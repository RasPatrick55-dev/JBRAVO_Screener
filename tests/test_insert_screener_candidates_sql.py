from datetime import date

import pandas as pd

from scripts import db


class _DummyCursor:
    def __init__(self, sql_capture: list[str]):
        self.sql_capture = sql_capture
        self._fetchone = None

    def execute(self, stmt, params=None):
        self.sql_capture.append(stmt)
        if "information_schema.columns" in stmt:
            self._fetchone = (1,)
        else:
            self._fetchone = None

    def fetchone(self):
        return self._fetchone

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        return False


class _DummyConn:
    def __init__(self, sql_capture: list[str]):
        self.sql_capture = sql_capture

    def cursor(self):
        return _DummyCursor(self.sql_capture)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        return False

    def close(self):
        return None


def test_insert_screener_candidates_uses_run_date_column(monkeypatch):
    sql_capture: list[str] = []

    def fake_execute_batch(cursor, stmt, rows, page_size=200):
        sql_capture.append(stmt)

    fake_conn = _DummyConn(sql_capture)
    monkeypatch.setattr(db, "_conn_or_none", lambda: fake_conn)
    monkeypatch.setattr(db.extras, "execute_batch", fake_execute_batch)

    payload = pd.DataFrame([{"symbol": "ABC", "timestamp": "2024-01-01T00:00:00Z"}])
    db.insert_screener_candidates(date(2024, 1, 1), payload)

    assert any(
        "INSERT INTO screener_candidates" in sql and "run_date" in sql for sql in sql_capture
    )
    assert all("run date" not in sql for sql in sql_capture)
