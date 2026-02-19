from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import metrics


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *_, **__):
        return None


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


def _patch_base_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(metrics, "logfile", str(tmp_path / "logs" / "metrics.log"), raising=False)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)


def test_metrics_uses_db_and_skips_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_base_dirs(tmp_path, monkeypatch)

    df = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "spy",
                "qty": 1,
                "entry_price": 10.0,
                "exit_price": 15.0,
                "realized_pnl": 100.0,
                "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
            {
                "trade_id": "t2",
                "symbol": "qqq",
                "qty": 2,
                "entry_price": 20.0,
                "exit_price": 10.0,
                "realized_pnl": -50.0,
                "entry_time": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 4, tzinfo=timezone.utc),
            },
        ]
    )

    def _fake_read_sql(stmt, con):
        assert con is not None
        return df.copy()

    monkeypatch.setattr(metrics.db, "db_enabled", lambda: True)
    monkeypatch.setattr(metrics.db, "get_engine", lambda: _DummyEngine())
    monkeypatch.setattr(metrics.pd, "read_sql_query", _fake_read_sql)

    def _raise_csv(*_, **__):
        raise AssertionError("csv should not load")

    monkeypatch.setattr(metrics, "load_trades_log", _raise_csv)

    rc = metrics.main()
    assert rc == 0

    summary_path = tmp_path / "data" / "metrics_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert summary.iloc[0]["total_trades"] == 2
    assert summary.iloc[0]["net_pnl"] == pytest.approx(50.0)


def test_metrics_falls_back_when_db_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_base_dirs(tmp_path, monkeypatch)

    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "net_pnl": 10.0,
                "entry_time": "2024-01-01T00:00:00Z",
                "exit_time": "2024-01-02T00:00:00Z",
            }
        ]
    )
    trades.to_csv(tmp_path / "data" / "trades_log.csv", index=False)

    monkeypatch.setattr(metrics.db, "db_enabled", lambda: False)

    rc = metrics.main()
    assert rc == 0

    summary_path = tmp_path / "data" / "metrics_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert summary.iloc[0]["total_trades"] == 1
