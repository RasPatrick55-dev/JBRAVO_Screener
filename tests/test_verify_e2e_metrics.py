from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.verify_e2e as verify_e2e

pytestmark = pytest.mark.alpaca_optional


class _FakeCursor:
    def __init__(self) -> None:
        self._results = [
            (date(2026, 2, 16),),
            (datetime(2026, 2, 16, 13, 0, tzinfo=timezone.utc),),
            (0,),
            (3,),
            (3,),
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _query, _params=None):
        return None

    def fetchone(self):
        if self._results:
            return self._results.pop(0)
        return (0,)


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def close(self):
        return None


def _write_latest_csv(path):
    path.write_text(
        "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source\n",
        encoding="utf-8",
    )


def test_verify_e2e_market_closed_metrics_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(verify_e2e.db, "get_db_conn", lambda: _FakeConn())
    monkeypatch.setattr(verify_e2e, "load_env", lambda: None)

    metrics_path = tmp_path / "execute_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "skip_counts": {"MARKET_CLOSED": 1},
                "market_clock": {
                    "ny_date": "2026-02-16",
                    "is_open": False,
                    "next_open": "2026-02-17T14:30:00Z",
                    "next_close": "2026-02-17T21:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    latest_csv = tmp_path / "latest_candidates.csv"
    _write_latest_csv(latest_csv)
    execute_log = tmp_path / "execute_trades.log"

    original_read_text = Path.read_text

    def guard_read_text(self, *args, **kwargs):
        if self == execute_log:
            raise AssertionError("verify_e2e should not read execute_trades.log")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard_read_text)

    rc = verify_e2e.main(
        [
            "--execute-metrics",
            str(metrics_path),
            "--latest-csv",
            str(latest_csv),
            "--execute-log",
            str(execute_log),
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "PASS market_closed_status_not_error" in out


def test_verify_e2e_market_closed_fails_on_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(verify_e2e.db, "get_db_conn", lambda: _FakeConn())
    monkeypatch.setattr(verify_e2e, "load_env", lambda: None)

    metrics_path = tmp_path / "execute_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "status": "error",
                "skip_counts": {"MARKET_CLOSED": 1},
                "market_clock": {
                    "ny_date": "2026-02-16",
                    "is_open": False,
                    "next_open": "2026-02-17T14:30:00Z",
                    "next_close": "2026-02-17T21:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    latest_csv = tmp_path / "latest_candidates.csv"
    _write_latest_csv(latest_csv)

    rc = verify_e2e.main(
        [
            "--execute-metrics",
            str(metrics_path),
            "--latest-csv",
            str(latest_csv),
        ]
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL market_closed_status_not_error" in out
    assert "status='error'" in out
