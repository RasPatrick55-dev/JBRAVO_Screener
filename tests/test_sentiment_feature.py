from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import screener
from utils.sentiment import JsonHttpSentimentClient, load_sentiment_cache, persist_sentiment_cache

pytestmark = pytest.mark.alpaca_optional


def test_sentiment_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "data" / "cache" / "sentiment"
    run_date = date(2024, 1, 2)

    persist_sentiment_cache(cache_dir, run_date, {"AAPL": 0.4, "tsla": -1.5, "skip": float("nan")})
    cached = load_sentiment_cache(cache_dir, run_date)

    assert cached == {"AAPL": 0.4, "TSLA": -1.0}


def test_json_client_enabled_and_clamps(tmp_path: Path) -> None:
    class StubResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(self.status_code)

        def json(self):
            return self._payload

    class RecordingSession:
        def __init__(self, payload: dict[str, object]):
            self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
            self.payload = payload

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append((url, params or {}, headers or {}))
            return StubResponse(self.payload)

    env = {
        "USE_SENTIMENT": "true",
        "SENTIMENT_API_URL": "http://sentiment.test/api",
        "SENTIMENT_API_KEY": "token-123",
    }
    session = RecordingSession({"sentiment": 4})
    client = JsonHttpSentimentClient(env=env, session=session)

    assert client.enabled() is True
    score = client.get_score("tsla", "2024-02-03")

    assert score == 1.0
    assert session.calls, "expected outbound request to sentiment endpoint"
    url, params, headers = session.calls[0]
    assert url == env["SENTIMENT_API_URL"]
    assert params == {"symbol": "TSLA", "date": "2024-02-03"}
    assert headers["Authorization"].endswith(env["SENTIMENT_API_KEY"])
    assert headers["X-API-Key"] == env["SENTIMENT_API_KEY"]

    disabled_client = JsonHttpSentimentClient(env={"USE_SENTIMENT": "false"})
    assert disabled_client.enabled() is False
    assert disabled_client.get_score("AAPL", "2024-02-03") is None


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
        client_factory=lambda *_: None,
    )

    assert summary["sentiment_gated"] == 1
    assert summary["sentiment_missing_count"] == 0
    assert summary["sentiment_avg"] == pytest.approx(0.3)
    assert updated["Score"].tolist() == pytest.approx([2.3])
    breakdown = json.loads(updated["score_breakdown"].iat[0])
    assert breakdown["sentiment"] == pytest.approx(0.3)
    assert breakdown["sentiment_weight"] == pytest.approx(1.0)
    assert breakdown["min_sentiment"] == pytest.approx(-0.1)

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
        client_factory=lambda *_: None,
    )

    assert summary["sentiment_gated"] == 0
    assert summary["sentiment_missing_count"] == 1
    assert updated.shape[0] == 1
    breakdown = json.loads(updated["score_breakdown"].iat[0])
    assert breakdown.get("sentiment") is None
    assert breakdown["sentiment_weight"] == pytest.approx(2.0)
    assert breakdown["min_sentiment"] == pytest.approx(0.5)
    assert updated["Score"].iat[0] == pytest.approx(1.5)


def test_screener_metrics_include_sentiment_keys(tmp_path: Path) -> None:
    base_dir = tmp_path / "enabled"
    base_dir.mkdir()
    top_df = pd.DataFrame(columns=screener.TOP_CANDIDATE_COLUMNS)
    scored_df = pd.DataFrame({"symbol": ["AAPL"], "sentiment": [0.2], "Score": [1.0]})
    stats_common = {
        "candidates_out": 0,
        "shortlist_requested": 0,
        "shortlist_candidates": 0,
        "coarse_ranked": 0,
        "shortlist_path": "",
        "backtest_target": 0,
        "backtest_evaluated": 0,
        "backtest_lookback": 0,
        "backtest_expectancy_mean": 0.0,
        "backtest_win_rate_mean": 0.0,
    }
    skip_reasons = {key: 0 for key in screener.SKIP_KEYS}

    stats_enabled = {
        **stats_common,
        "sentiment_enabled": True,
        "sentiment_missing_count": 0,
        "sentiment_avg": 0.2,
        "sentiment_gated": 0,
    }
    screener.write_outputs(base_dir, top_df, scored_df, stats_enabled, skip_reasons)
    metrics_path = base_dir / "data" / "screener_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sentiment_enabled"] is True
    assert metrics["sentiment_missing_count"] == 0
    assert metrics["sentiment_avg"] == pytest.approx(0.2)
    assert metrics["metrics_version"] == 2

    disabled_dir = tmp_path / "disabled"
    disabled_stats = {
        **stats_common,
        "sentiment_enabled": False,
        "sentiment_missing_count": 5,
        "sentiment_avg": None,
        "sentiment_gated": 0,
    }
    screener.write_outputs(disabled_dir, top_df, scored_df, disabled_stats, skip_reasons)
    disabled_metrics = json.loads((disabled_dir / "data" / "screener_metrics.json").read_text(encoding="utf-8"))
    assert disabled_metrics["sentiment_enabled"] is False
    assert disabled_metrics["sentiment_missing_count"] == 0
    assert disabled_metrics["sentiment_avg"] is None
    assert disabled_metrics["metrics_version"] == 2
