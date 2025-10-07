import json
from datetime import datetime, timezone

import pandas as pd

from scripts import screener


def _sample_universe() -> pd.DataFrame:
    rows = [
        {"symbol": "EQ1", "exchange": "NASDAQ", "timestamp": "2024-01-01", "open": 10, "high": 11, "low": 9.5, "close": 10.5, "volume": 1_000},
        {"symbol": "EQ1", "exchange": "NASDAQ", "timestamp": "2024-01-02", "open": 10.5, "high": 11.2, "low": 10.1, "close": 11.0, "volume": 1_200},
        {"symbol": "EQ1", "exchange": "NASDAQ", "timestamp": "2024-01-03", "open": 11.0, "high": 11.5, "low": 10.8, "close": 11.4, "volume": 1_300},
        {"symbol": "CR1", "exchange": "CRYPTO", "timestamp": "2024-01-01", "open": 5, "high": 5.6, "low": 4.8, "close": 5.2, "volume": 500},
        {"symbol": "CR1", "exchange": "CRYPTO", "timestamp": "2024-01-02", "open": 5.1, "high": 5.5, "low": 5.0, "close": 5.3, "volume": 450},
        {"symbol": "UNK1", "exchange": "XFTXU", "timestamp": "2024-01-01", "open": 7, "high": 7.5, "low": 6.8, "close": 7.1, "volume": 600},
        {"symbol": "BLANK", "exchange": "", "timestamp": "2024-01-01", "open": 3, "high": 3.3, "low": 2.9, "close": 3.1, "volume": 300},
    ]
    return pd.DataFrame(rows)


def test_screener_skips_unknown_exchanges(tmp_path):
    df = _sample_universe()
    now = datetime(2024, 1, 10, 14, tzinfo=timezone.utc)

    top_df, scored_df, stats, skips = screener.run_screener(
        df,
        top_n=5,
        min_history=2,
        now=now,
    )

    assert stats["symbols_in"] == 4
    assert stats["candidates_out"] == 1
    assert skips["UNKNOWN_EXCHANGE"] >= 1
    assert skips["NON_EQUITY"] >= 1
    assert "EQ1" in scored_df["symbol"].tolist()

    metrics_path = screener.write_outputs(tmp_path, top_df, scored_df, stats, skips, now=now)

    top_path = tmp_path / "data" / "top_candidates.csv"
    scored_path = tmp_path / "data" / "scored_candidates.csv"
    assert top_path.exists()
    assert scored_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["rows"] == scored_df.shape[0]
    assert metrics["skips"]["UNKNOWN_EXCHANGE"] >= 1
    assert metrics["status"] == "ok"
    if not top_df.empty:
        assert int(top_df["universe_count"].iloc[0]) == stats["symbols_in"]
