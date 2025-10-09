import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts import screener


def _sample_universe() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=200, tz="UTC", freq="B")
    trend = np.linspace(50, 130, len(dates))
    trend[-2] = trend[-3] - 15
    trend[-1] = trend[-3] + 10
    eq_frame = pd.DataFrame(
        {
            "symbol": "EQ1",
            "exchange": "NASDAQ",
            "timestamp": dates,
            "open": trend - 0.5,
            "high": trend + 1.0,
            "low": trend - 1.0,
            "close": trend,
            "volume": np.full(len(dates), 1_000_000, dtype=float),
        }
    )

    crypto_dates = pd.date_range("2024-01-01", periods=3, tz="UTC", freq="B")
    crypto_frame = pd.DataFrame(
        {
            "symbol": "CR1",
            "exchange": "CRYPTO",
            "timestamp": crypto_dates,
            "open": [5.0, 5.1, 5.2],
            "high": [5.6, 5.5, 5.4],
            "low": [4.8, 4.9, 5.0],
            "close": [5.2, 5.3, 5.1],
            "volume": [500, 450, 425],
        }
    )

    unknown_frame = pd.DataFrame(
        {
            "symbol": "UNK1",
            "exchange": "XFTXU",
            "timestamp": crypto_dates,
            "open": [7.0, 7.1, 7.2],
            "high": [7.5, 7.6, 7.7],
            "low": [6.8, 6.9, 7.0],
            "close": [7.1, 7.0, 6.9],
            "volume": [600, 580, 560],
        }
    )

    blank_frame = pd.DataFrame(
        {
            "symbol": "BLANK",
            "exchange": "",
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "open": [3.0],
            "high": [3.3],
            "low": [2.9],
            "close": [3.1],
            "volume": [300],
        }
    )

    return pd.concat([eq_frame, crypto_frame, unknown_frame, blank_frame], ignore_index=True)


def test_screener_skips_unknown_exchanges(tmp_path):
    df = _sample_universe()
    now = datetime(2024, 1, 10, 14, tzinfo=timezone.utc)

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

    metrics_path = screener.write_outputs(
        tmp_path,
        top_df,
        scored_df,
        stats,
        skips,
        reject_samples,
        now=now,
        gate_counters=gate_counters,
        fetch_metrics={},
        asset_metrics={},
        ranker_cfg=ranker_cfg,
        timings=timings,
    )

    top_path = tmp_path / "data" / "top_candidates.csv"
    scored_path = tmp_path / "data" / "scored_candidates.csv"
    diag_dir = tmp_path / "data" / "diagnostics"
    diag_csv = diag_dir / f"top10_{now.date().isoformat()}.csv"
    diag_json = diag_dir / f"top10_{now.date().isoformat()}.json"
    assert top_path.exists()
    assert scored_path.exists()
    assert diag_csv.exists()
    assert diag_json.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["rows"] == scored_df.shape[0]
    assert metrics["skips"]["UNKNOWN_EXCHANGE"] >= 1
    assert metrics["status"] == "ok"
    assert "reject_samples" in metrics
    assert "gate_fail_counts" in metrics
    assert "timings" in metrics
    assert metrics.get("http") == {"429": 0, "404": 0, "empty_pages": 0}
    assert metrics.get("cache") == {"batches_hit": 0, "batches_miss": 0}
    assert metrics.get("universe_prefix_counts") == {}
    expected_gate_keys = {
        "failed_sma_stack",
        "failed_rsi",
        "failed_adx",
        "failed_aroon",
        "failed_volexp",
        "failed_gap",
        "failed_liquidity",
        "nan_data",
        "insufficient_history",
        "gate_preset",
        "gate_relax_mode",
        "gate_total_evaluated",
        "gate_total_passed",
        "gate_total_failed",
    }
    assert expected_gate_keys.issubset(metrics["gate_fail_counts"].keys())
    if not top_df.empty:
        assert int(top_df["universe_count"].iloc[0]) == stats["symbols_in"]
