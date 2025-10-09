from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scripts import screener


def _build_symbol_frame(symbol: str, base_price: float, slope: float, *, periods: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, tz="UTC", freq="B")
    close = base_price + slope * np.arange(len(dates))
    open_price = close * 0.995
    high = close * 1.01
    low = close * 0.99
    volume = np.full(len(dates), 400_000, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "exchange": "NASDAQ",
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.mark.alpaca_optional
def test_run_screener_generates_shortlist_and_backtest(tmp_path):
    periods = 260
    primary = _build_symbol_frame("AAA", base_price=40.0, slope=0.25, periods=periods)
    secondary = _build_symbol_frame("BBB", base_price=60.0, slope=0.10, periods=periods)
    universe = pd.concat([primary, secondary], ignore_index=True)

    shortlist_path = tmp_path / "shortlist.csv"
    now = datetime(2024, 12, 31, tzinfo=timezone.utc)

    (
        top_df,
        scored_df,
        stats,
        skips,
        reject_samples,
        gate_counters,
        ranker_cfg,
        timings,
    ) = screener.run_screener(
        universe,
        top_n=2,
        min_history=60,
        now=now,
        shortlist_size=5,
        shortlist_path=shortlist_path,
        backtest_top_k=2,
        backtest_lookback=30,
        dollar_vol_min=0,
    )

    assert shortlist_path.exists()
    shortlist_df = pd.read_csv(shortlist_path)
    assert set(shortlist_df.columns) == {"symbol", "coarse_score", "coarse_rank"}
    assert 0 < len(shortlist_df) <= 5
    assert shortlist_df["symbol"].str.upper().isin({"AAA", "BBB"}).all()
    assert stats["shortlist_path"] == str(shortlist_path)

    assert not scored_df.empty
    expected_backtest_cols = {
        "backtest_expectancy",
        "backtest_win_rate",
        "backtest_samples",
        "backtest_adjustment",
    }
    assert expected_backtest_cols.issubset(scored_df.columns)
    for column in expected_backtest_cols:
        assert scored_df[column].notna().any()
    assert scored_df["Score"].is_monotonic_decreasing
    assert scored_df["rank"].tolist() == list(range(1, len(scored_df) + 1))

    assert "gate_total_evaluated" in gate_counters
    assert gate_counters["gate_total_evaluated"] >= len(scored_df)

    if not top_df.empty:
        assert expected_backtest_cols.issubset(top_df.columns)
        assert top_df["Score"].is_monotonic_decreasing
        assert top_df["symbol"].str.upper().isin({"AAA", "BBB"}).all()
