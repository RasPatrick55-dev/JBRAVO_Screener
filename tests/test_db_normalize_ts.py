import logging
from datetime import date, datetime, timezone


from scripts import db


def test_normalize_ts_parses_and_attaches_timezone():
    parsed = db.normalize_ts("2024-05-01T12:34:56", field="entry_time")
    assert parsed == datetime(2024, 5, 1, 12, 34, 56, tzinfo=timezone.utc)


def test_normalize_ts_handles_dates_and_naive_datetimes():
    naive = datetime(2024, 5, 1, 12, 0, 0)
    normalized = db.normalize_ts(naive, field="exit_time")
    assert normalized.tzinfo == timezone.utc
    as_date = db.normalize_ts(date(2024, 5, 1), field="exit_time")
    assert as_date == datetime(2024, 5, 1, tzinfo=timezone.utc)


def test_insert_executed_trade_normalizes_and_logs(monkeypatch, caplog):
    executed: list[dict] = []

    class DummyCursor:
        def __init__(self, bucket: list[dict]):
            self.bucket = bucket

        def execute(self, stmt, payload):
            self.bucket.append(payload)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConn:
        def __init__(self, bucket: list[dict]):
            self.bucket = bucket

        def cursor(self):
            return DummyCursor(self.bucket)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            return None

    monkeypatch.setattr(db, "_conn_or_none", lambda: DummyConn(executed))

    caplog.set_level(logging.INFO)
    db.insert_executed_trade(
        {
            "symbol": "xyz",
            "qty": 1,
            "entry_time": "2024-05-01T12:00:00",
            "entry_price": 1.0,
            "exit_time": "bad-ts",
            "exit_price": 2.0,
            "pnl": 1.0,
            "net_pnl": 1.0,
            "order_id": "1",
            "status": "filled",
        }
    )

    assert len(executed) == 1
    payload = executed[0]
    assert payload["symbol"] == "XYZ"
    assert isinstance(payload["entry_time"], datetime)
    assert payload["entry_time"].tzinfo == timezone.utc
    assert payload["exit_time"] is None

    warning = next(
        (record for record in caplog.records if "EXECUTED_TRADES_TS_PARSE_FAIL" in record.message),
        None,
    )
    assert warning is not None
    info = next(
        (
            record
            for record in caplog.records
            if "DB_WRITE_OK table=executed_trades" in record.message
        ),
        None,
    )
    assert info is not None
