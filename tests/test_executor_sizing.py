import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.execute_trades import ExecutionMetrics, ExecutorConfig, TradeExecutor


@pytest.mark.alpaca_optional
def test_sizing_logs_zero_qty_after_bump(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.01,
        min_order_usd=150.0,
        allow_bump_to_one=True,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "BUMP",
                "close": 300.0,
                "entry_price": 300.0,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 1000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("ZERO_QTY_AFTER_BUMP" in msg for msg in messages)
    assert metrics.skipped_reasons.get("ZERO_QTY", 0) == 1


@pytest.mark.alpaca_optional
def test_min_order_floor_sets_quantity(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.01,
        min_order_usd=500.0,
        allow_bump_to_one=False,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "FLOOR",
                "close": 95.0,
                "entry_price": 95.0,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 2000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("CALC symbol=FLOOR" in msg and "qty=5" in msg for msg in messages)
    assert any("notional=500.00" in msg for msg in messages)
    assert metrics.skipped_reasons.get("ZERO_QTY", 0) == 0
