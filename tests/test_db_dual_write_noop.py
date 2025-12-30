from datetime import date, datetime, timezone

import pandas as pd
import pytest

from scripts import db


@pytest.mark.alpaca_optional
def test_dual_write_noop_without_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.db_enabled() is False

    today = date(2024, 1, 1)
    candidates = pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
                "symbol": "ABC",
                "score": 1.23,
                "exchange": "NASDAQ",
                "close": 10.5,
                "volume": 1_000,
                "universe_count": 200,
                "score_breakdown": '{"factor": 1}',
                "entry_price": 10.0,
                "adv20": 2_000,
                "atrp": 0.02,
                "source": "screener",
            }
        ]
    )
    backtest_results = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "trades": 5,
                "win_rate": 60.0,
                "net_pnl": 12.5,
                "expectancy": 2.5,
                "profit_factor": 1.4,
                "max_drawdown": -0.03,
                "sharpe": 1.1,
                "sortino": 1.5,
            }
        ]
    )
    metrics_summary = {
        "total_trades": 5,
        "net_pnl": 12.5,
        "win_rate": 60.0,
        "expectancy": 2.5,
        "profit_factor": 1.4,
        "max_drawdown": -0.03,
        "sharpe": 1.1,
        "sortino": 1.5,
    }

    db.upsert_pipeline_run(today, datetime.now(timezone.utc), datetime.now(timezone.utc), 0, {"rows": 1})
    db.insert_screener_candidates(today, candidates)
    db.insert_backtest_results(today, backtest_results)
    db.upsert_metrics_daily(today, metrics_summary)
    db.insert_executed_trade(
        {
            "symbol": "ABC",
            "qty": 10,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": 10.0,
            "exit_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "net_pnl": 0.0,
            "order_id": "test-1",
            "order_status": "filled",
            "side": "buy",
        }
    )
