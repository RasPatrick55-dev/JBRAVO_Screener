from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.trade_performance import (
    compute_exit_quality_columns,
    compute_trade_excursions,
    load_trades_log,
    summarize_by_window,
)

pytestmark = pytest.mark.alpaca_optional


def test_load_trades_log_handles_missing(tmp_path):
    df = load_trades_log(tmp_path)
    assert df.empty


def test_trade_excursions_and_exit_quality_bounds():
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = start + timedelta(days=2)
    trades = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "entry_time": start,
                "exit_time": end,
                "qty": 10,
                "entry_price": 10.0,
                "exit_price": 12.0,
                "pnl": 20.0,
            }
        ]
    )

    def fake_fetch(symbol, fetch_start, fetch_end, timeframe):
        return pd.DataFrame(
            {
                "timestamp": [
                    fetch_start + timedelta(hours=1),
                    fetch_start + timedelta(hours=5),
                    fetch_end - timedelta(hours=2),
                ],
                "high": [11.0, 13.0, 12.5],
                "low": [9.5, 9.8, 10.2],
                "close": [10.8, 12.8, 12.2],
            }
        )

    excursions = compute_trade_excursions(trades, data_client=None, bar_fetcher=fake_fetch)
    enriched = compute_exit_quality_columns(excursions)
    for column in (
        "mfe_pct",
        "mae_pct",
        "missed_profit_pct",
        "exit_efficiency_pct",
        "hold_days",
    ):
        assert column in enriched.columns
    eff = enriched.loc[0, "exit_efficiency_pct"]
    assert 0 <= eff <= 100


def test_summarize_by_window_counts_recent_trades():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "XYZ",
                "entry_time": now - timedelta(days=3),
                "exit_time": now - timedelta(days=1),
                "pnl": 15.0,
                "return_pct": 5.0,
                "hold_days": 2.0,
                "exit_efficiency_pct": 80.0,
                "missed_profit_pct": 4.0,
                "mfe_pct": 6.0,
                "mae_pct": -2.0,
                "peak_price": 11.0,
                "trough_price": 9.5,
                "exit_price": 10.5,
                "entry_price": 10.0,
                "sold_too_soon": True,
            }
        ]
    )
    summary = summarize_by_window(frame)
    assert {"7D", "30D", "365D", "ALL"}.issubset(summary.keys())
    assert summary["7D"]["trades"] == 1
    assert summary["7D"]["sold_too_soon"] == 1
