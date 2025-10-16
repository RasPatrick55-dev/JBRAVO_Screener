import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.execute_trades import ExecutionMetrics, ExecutorConfig, TradeExecutor


@pytest.mark.alpaca_optional
def test_min_order_prevents_zero_qty(monkeypatch, caplog, tmp_path: Path) -> None:
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.02,
        min_order_usd=200.0,
        allow_bump_to_one=True,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "LIMIT",
                "close": 50.0,
                "entry_price": 50.0,
                "score": 5.0,
                "universe_count": 50,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 5000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.INFO, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("DRY_RUN_ORDER" in msg and "qty=3" in msg for msg in messages)
    assert "ZERO_QTY_AFTER_BUMP" not in "\n".join(messages)
    assert metrics.skipped_reasons.get("ZERO_QTY", 0) == 0
