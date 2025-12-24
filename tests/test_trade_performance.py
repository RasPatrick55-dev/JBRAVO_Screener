from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.trade_performance import (
    cache_refresh_summary_token,
    read_cache,
    SUMMARY_WINDOWS,
    compute_exit_quality_columns,
    compute_rebound_metrics,
    compute_trade_excursions,
    load_trades_log,
    refresh_trade_performance_cache,
    summarize_by_window,
)

pytestmark = pytest.mark.alpaca_optional


def test_load_trades_log_handles_missing(tmp_path):
    df = load_trades_log(tmp_path)
    assert df.empty


def test_cache_refresh_summary_token_includes_counts():
    token = cache_refresh_summary_token(10, 3, {"7D": {}, "ALL": {}}, 400, 0)
    assert "trades_total=10" in token
    assert "trades_enriched=3" in token
    assert "lookback_days=400" in token
    assert "windows=7D,ALL" in token
    assert token.endswith("rc=0")


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
    assert summary["7D"]["net_pnl"] == pytest.approx(15.0)
    assert summary["7D"]["win_rate"] == pytest.approx(1.0)
    assert summary["7D"]["win_rate_pct"] == pytest.approx(100.0)


def test_summary_always_includes_windows_and_numbers():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "exit_time": now,
                "return_pct": float("nan"),
                "pnl": float("nan"),
            }
        ]
    )
    summary = summarize_by_window(frame)
    for window in SUMMARY_WINDOWS:
        assert window in summary
        assert isinstance(summary[window]["net_pnl"], float)
        assert isinstance(summary[window]["win_rate"], float)
        assert summary[window]["win_rate"] == summary[window]["win_rate"]  # not NaN
        assert summary[window]["net_pnl"] == summary[window]["net_pnl"]  # not NaN


def test_summary_computes_pnl_and_win_rate_when_missing_column():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "entry_time": now - timedelta(days=1),
                "exit_time": now,
                "entry_price": 5.0,
                "exit_price": 7.0,
                "qty": 10,
            },
            {
                "symbol": "DEF",
                "entry_time": now - timedelta(days=2),
                "exit_time": now - timedelta(days=1),
                "entry_price": 10.0,
                "exit_price": 8.0,
                "qty": 5,
            },
        ]
    )
    summary = summarize_by_window(frame)
    window = summary["ALL"]
    assert window["net_pnl"] == pytest.approx((7 - 5) * 10 + (8 - 10) * 5)
    assert window["win_rate"] == pytest.approx(0.5)


def test_refresh_best_effort_when_excursions_fail(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    trades = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "qty": 10,
                "entry_price": 10.0,
                "exit_price": 11.0,
                "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 2, tzinfo=timezone.utc),
            }
        ]
    )
    trades.to_csv(data_dir / "trades_log.csv", index=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.trade_performance.compute_trade_excursions", boom)

    cache_path = data_dir / "trade_performance_cache.json"
    df, summary = refresh_trade_performance_cache(
        base_dir=tmp_path, data_client=None, force=True, cache_path=cache_path
    )
    assert not df.empty
    for window in SUMMARY_WINDOWS:
        assert window in summary
        assert summary[window]["win_rate"] >= 0.0
        assert summary[window]["net_pnl"] >= 0.0


def test_refresh_populates_exit_efficiency_with_daily_peak(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    entry_ts = now - timedelta(days=2)
    exit_ts = now - timedelta(days=1)
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "qty": 5,
                "entry_price": 10.0,
                "exit_price": 11.0,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
            }
        ]
    )
    trades.to_csv(data_dir / "trades_log.csv", index=False)

    def fake_fetch(_symbol, start, end, _timeframe):
        return pd.DataFrame(
            {
                "timestamp": [start + timedelta(hours=1), end - timedelta(hours=1)],
                "high": [12.0, 11.5],
                "low": [9.5, 10.5],
                "close": [11.0, 11.2],
            }
        )

    cache_path = data_dir / "trade_performance_cache.json"
    df, summary = refresh_trade_performance_cache(
        base_dir=tmp_path,
        data_client=None,
        lookback_days=30,
        force=True,
        cache_path=cache_path,
        bar_fetcher=fake_fetch,
    )
    expected_eff = (11.0 / 12.0) * 100
    assert pytest.approx(df.loc[0, "peak_price"], rel=1e-6) == 12.0
    assert pytest.approx(df.loc[0, "exit_efficiency_pct"], rel=1e-6) == expected_eff

    payload = read_cache(cache_path)
    assert payload is not None
    assert pytest.approx(payload["summary"]["ALL"]["avg_exit_efficiency_pct"], rel=1e-6) == expected_eff
    cached_trade = payload["trades"][0]
    assert pytest.approx(cached_trade["exit_efficiency_pct"], rel=1e-6) == expected_eff


def test_refresh_uses_full_history_and_enrichment_mask(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "qty": 10,
                "entry_price": 5.0,
                "exit_price": 6.0,
                "entry_time": now - timedelta(days=500, hours=2),
                "exit_time": now - timedelta(days=500),
            },
            {
                "symbol": "NEW",
                "qty": 8,
                "entry_price": 15.0,
                "exit_price": 16.0,
                "entry_time": now - timedelta(days=3),
                "exit_time": now - timedelta(days=1),
            },
        ]
    )
    trades.to_csv(data_dir / "trades_log.csv", index=False)

    calls: list[str] = []

    def fake_fetch(symbol, start, end, _timeframe):
        calls.append(symbol)
        return pd.DataFrame(
            {
                "timestamp": [start + timedelta(hours=1), end - timedelta(hours=1)],
                "high": [float(symbol != "OLD") + 16.0, float(symbol != "OLD") + 16.5],
                "low": [15.0, 15.5],
                "close": [15.5, 16.1],
            }
        )

    cache_path = data_dir / "trade_performance_cache.json"
    df, summary = refresh_trade_performance_cache(
        base_dir=tmp_path,
        data_client=None,
        lookback_days=30,
        force=True,
        cache_path=cache_path,
        bar_fetcher=fake_fetch,
    )
    payload = read_cache(cache_path)

    assert summary["ALL"]["trades"] == 2
    assert summary["ALL"]["net_pnl"] == pytest.approx((6 - 5) * 10 + (16 - 15) * 8)
    assert summary["30D"]["trades"] == 1
    assert int(df["needs_enrichment"].sum()) == 1
    assert payload["trades_total"] == 2
    assert payload["trades_enriched"] == 1
    assert payload["lookback_days_used"] == 30
    assert len(calls) == 1


def test_refresh_skips_enrichment_outside_lookback(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "qty": 5,
                "entry_price": 20.0,
                "exit_price": 22.0,
                "entry_time": now - timedelta(days=100),
                "exit_time": now - timedelta(days=90),
            }
        ]
    )
    trades.to_csv(data_dir / "trades_log.csv", index=False)

    def boom_fetch(*_args, **_kwargs):
        raise AssertionError("Enrichment should not run for trades outside lookback.")

    cache_path = data_dir / "trade_performance_cache.json"
    df, summary = refresh_trade_performance_cache(
        base_dir=tmp_path,
        data_client=None,
        lookback_days=7,
        force=True,
        cache_path=cache_path,
        bar_fetcher=boom_fetch,
    )
    payload = read_cache(cache_path)

    assert summary["ALL"]["trades"] == 1
    assert summary["ALL"]["net_pnl"] == pytest.approx((22 - 20) * 5)
    assert int(df["needs_enrichment"].sum()) == 0
    assert payload["trades_enriched"] == 0
    assert payload["trades_total"] == 1


def test_exit_efficiency_nan_when_peak_missing_or_clamped():
    now = datetime.now(timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "symbol": "XYZ",
                "entry_time": now - timedelta(days=1),
                "exit_time": now,
                "entry_price": 10.0,
                "exit_price": 11.0,
                "peak_price": float("nan"),
                "trough_price": 9.0,
            },
            {
                "symbol": "QRS",
                "entry_time": now - timedelta(days=2),
                "exit_time": now - timedelta(days=1),
                "entry_price": 10.0,
                "exit_price": 12.0,
                "peak_price": 9.5,
                "trough_price": 8.0,
            },
        ]
    )
    enriched = compute_exit_quality_columns(trades)
    assert pd.isna(enriched.loc[0, "exit_efficiency_pct"])
    assert enriched.loc[1, "exit_efficiency_pct"] == pytest.approx(100.0)


def test_rebound_columns_exist():
    exit_ts = datetime(2024, 1, 5, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "entry_time": exit_ts - timedelta(days=3),
                "exit_time": exit_ts,
                "entry_price": 95.0,
                "exit_price": 97.0,
                "peak_price": 100.0,
                "trough_price": 90.0,
                "trailing_pct": 3.0,
                "exit_reason": "TrailingStop",
            }
        ]
    )
    enriched = compute_exit_quality_columns(trades)
    def fake_fetch(_symbol, start, end, _timeframe):
        return pd.DataFrame(
            {
                "timestamp": [start + timedelta(days=1)],
                "high": [101.0],
                "low": [100.0],
                "close": [100.5],
                "open": [100.0],
            }
        )

    rebound_df = compute_rebound_metrics(
        enriched,
        data_client=None,
        bar_fetcher=fake_fetch,
        rebound_window_days=5,
        rebound_threshold_pct=3.0,
    )
    assert "rebound_pct" in rebound_df.columns
    assert "rebounded" in rebound_df.columns
    assert "post_exit_high" in rebound_df.columns
    assert bool(rebound_df.loc[0, "rebounded"]) is True
    assert bool(rebound_df.loc[0, "is_trailing_stop_exit"]) is True
    assert rebound_df.loc[0, "rebound_window_days"] == 5


def test_rebound_rate_numeric():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": now - timedelta(days=2),
                "exit_time": now - timedelta(days=1),
                "entry_price": 100.0,
                "exit_price": 97.0,
                "rebound_pct": 4.0,
                "rebounded": True,
                "is_trailing_stop_exit": True,
            },
            {
                "symbol": "BBB",
                "entry_time": now - timedelta(days=3),
                "exit_time": now - timedelta(days=1),
                "entry_price": 50.0,
                "exit_price": 48.5,
                "rebound_pct": 1.0,
                "rebounded": False,
                "is_trailing_stop_exit": True,
            },
        ]
    )
    summary = summarize_by_window(frame)
    metrics = summary["ALL"]
    assert isinstance(metrics["rebound_rate"], float)
    assert metrics["rebound_rate"] == pytest.approx(0.5)
    assert metrics["stop_exits"] == 2
    assert metrics["rebounds"] == 1
    assert metrics["avg_rebound_pct"] == pytest.approx(2.5)


def test_rebound_rate_zero_when_no_stop_exits():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": now - timedelta(days=2),
                "exit_time": now - timedelta(days=1),
                "entry_price": 100.0,
                "exit_price": 101.0,
                "is_trailing_stop_exit": False,
            }
        ]
    )
    summary = summarize_by_window(frame)
    metrics = summary["ALL"]
    assert metrics["stop_exits"] == 0
    assert metrics["rebound_rate"] == 0.0
    assert metrics["rebounds"] == 0


def test_summary_contains_rebound_metrics():
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": now - timedelta(days=1),
                "exit_time": now,
                "pnl": float("nan"),
                "return_pct": float("nan"),
            }
        ]
    )
    summary = summarize_by_window(frame)
    for window in SUMMARY_WINDOWS:
        metrics = summary[window]
        for key in ("stop_exits", "rebounds", "rebound_rate", "avg_rebound_pct"):
            assert key in metrics
            assert metrics[key] == metrics[key]
