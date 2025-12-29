from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import screener
from utils.features.sentiment import JsonHttpSentimentProvider, load_sentiment_cache, persist_sentiment_cache

pytestmark = pytest.mark.alpaca_optional


def test_sentiment_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "data" / "cache" / "sentiment"
    run_date = datetime(2024, 1, 2, tzinfo=timezone.utc).date()

    persist_sentiment_cache(cache_dir, run_date, {"AAPL": 0.4, "tsla": -0.2, "skip": float("nan")})
    cached = load_sentiment_cache(cache_dir, run_date)

    assert cached == {"AAPL": 0.4, "TSLA": -0.2}


def test_json_provider_fail_soft() -> None:
    class FailingSession:
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    provider = JsonHttpSentimentProvider("http://sentiment.test", session=FailingSession())
    result = provider.get_symbol_sentiment("AAPL", datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert result is None


def test_apply_sentiment_scores_injects_breakdown_and_gates(tmp_path: Path) -> None:
    run_ts = datetime(2024, 3, 1, tzinfo=timezone.utc)
    cache_dir = tmp_path / "data" / "cache" / "sentiment"
    persist_sentiment_cache(cache_dir, run_ts.date(), {"AAA": -0.5, "BBB": 0.3})

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "Score": [1.0, 2.0],
            "score_breakdown": ["{}", "{}"],
        }
    )
    settings = {"enabled": True, "weight": 1.0, "min": -0.1}

    updated, summary = screener._apply_sentiment_scores(
        frame,
        run_ts=run_ts,
        settings=settings,
        cache_dir=cache_dir,
        provider_factory=lambda *_: None,
    )

    assert summary["sentiment_gated"] == 1
    assert summary["sentiment_missing_count"] == 0
    assert summary["sentiment_avg"] == pytest.approx(0.3)
    assert updated["Score"].tolist() == pytest.approx([2.3])
    breakdown = json.loads(updated["score_breakdown"].iat[0])
    assert breakdown["sentiment"] == pytest.approx(0.3)

    candidates_df = updated.assign(
        close=10.0,
        volume=1_000_000.0,
        adv20=2_000_000.0,
        atrp=0.05,
        timestamp=pd.Timestamp(run_ts),
        entry_price=10.0,
        universe_count=2,
    )
    top_df = screener._prepare_top_frame(candidates_df, top_n=1)
    assert "sentiment" not in top_df.columns
    assert len(top_df.columns) == len(screener.TOP_CANDIDATE_COLUMNS)


def test_sentiment_missing_does_not_gate(tmp_path: Path) -> None:
    run_ts = datetime(2024, 4, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame({"symbol": ["ZZZ"], "Score": [1.5], "score_breakdown": ["{}"]})
    settings = {"enabled": True, "weight": 2.0, "min": 0.5}

    updated, summary = screener._apply_sentiment_scores(
        frame,
        run_ts=run_ts,
        settings=settings,
        cache_dir=tmp_path / "data" / "cache" / "sentiment",
        provider_factory=lambda *_: None,
    )

    assert summary["sentiment_gated"] == 0
    assert summary["sentiment_missing_count"] == 1
    assert updated.shape[0] == 1
    breakdown = json.loads(updated["score_breakdown"].iat[0])
    assert breakdown.get("sentiment") is None
    assert updated["Score"].iat[0] == pytest.approx(1.5)
