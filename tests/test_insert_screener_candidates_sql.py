from datetime import date

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from scripts import db


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConnection:
    def __init__(self, sql_capture):
        self.sql_capture = sql_capture
        self.last_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, stmt, params=None):
        # schema check should return a truthy scalar
        if isinstance(stmt, str) and "information_schema.columns" in stmt:
            self.sql_capture.append(stmt)
            return _FakeResult(1)

        self.sql_capture.append(stmt)
        self.last_params = params
        return _FakeResult(None)


class _FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeEngine:
    def __init__(self, sql_capture):
        self.connection = _FakeConnection(sql_capture)

    def connect(self):
        return self.connection

    def begin(self):
        return _FakeBegin(self.connection)


def test_insert_screener_candidates_uses_run_date_column(monkeypatch):
    sql_capture: list[str] = []

    def fake_text(sql):
        sql_capture.append(sql)
        return sql

    fake_engine = _FakeEngine(sql_capture)
    monkeypatch.setattr(db, "_engine_or_none", lambda: fake_engine)
    monkeypatch.setattr(db, "text", fake_text)

    payload = pd.DataFrame([{"symbol": "ABC", "timestamp": "2024-01-01T00:00:00Z"}])
    db.insert_screener_candidates(date(2024, 1, 1), payload)

    assert any("INSERT INTO screener_candidates" in sql and "run_date" in sql for sql in sql_capture)
    assert all("run date" not in sql for sql in sql_capture)
