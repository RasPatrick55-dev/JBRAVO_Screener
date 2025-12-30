from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Optional

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
    assert pd.isna(updated["sentiment"].iat[0])
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
    assert isinstance(metrics["sentiment_missing_count"], int)
    assert metrics["sentiment_missing_count"] == 0
    assert metrics["sentiment_avg"] == pytest.approx(0.2)
    assert metrics["metrics_version"] == 2

    disabled_dir = tmp_path / "disabled"
    disabled_stats = {
        **stats_common,
        "sentiment_enabled": False,
        "sentiment_missing_count": 1,
        "sentiment_avg": None,
        "sentiment_gated": 0,
    }
    screener.write_outputs(disabled_dir, top_df, scored_df, disabled_stats, skip_reasons)
    disabled_metrics = json.loads((disabled_dir / "data" / "screener_metrics.json").read_text(encoding="utf-8"))
    assert disabled_metrics["sentiment_enabled"] is False
    assert isinstance(disabled_metrics["sentiment_missing_count"], int)
    assert disabled_metrics["sentiment_missing_count"] == 1
    assert disabled_metrics["sentiment_avg"] is None
    assert disabled_metrics["metrics_version"] == 2


def test_sentiment_fetch_writes_cache_and_column(tmp_path: Path) -> None:
    run_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
    cache_dir = tmp_path / "data" / "cache" / "sentiment"
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "Score": [1, 2, 3, 4, 5],
            "score_breakdown": ["{}"] * 5,
        }
    )

    class Provider:
        def __init__(self) -> None:
            self.errors = 0

        def get_symbol_sentiment(self, symbol: str, asof_utc: datetime) -> Optional[float]:
            values = {"AAA": 0.1, "BBB": 0.2, "CCC": 0.3}
            return values.get(symbol)

    provider = Provider()
    settings = {"enabled": True, "weight": 0.0, "min": -1.0, "api_url": "http://example.test/api"}
    updated, summary = screener._apply_sentiment_scores(
        frame,
        run_ts=run_ts,
        settings=settings,
        cache_dir=cache_dir,
        client_factory=lambda *_: provider,
    )

    cache_file = cache_dir / f"{run_ts.date().isoformat()}.json"
    assert cache_file.exists()
    cached_payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached_payload == {"AAA": 0.1, "BBB": 0.2, "CCC": 0.3}
    assert "sentiment" in updated.columns
    assert summary["sentiment_missing_count"] == 2

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
    stats_enabled = {
        **stats_common,
        "sentiment_enabled": True,
        "sentiment_missing_count": summary["sentiment_missing_count"],
        "sentiment_avg": summary["sentiment_avg"],
        "sentiment_gated": summary.get("sentiment_gated", 0),
    }
    skip_reasons = {key: 0 for key in screener.SKIP_KEYS}
    top_df = pd.DataFrame(columns=screener.TOP_CANDIDATE_COLUMNS)

    screener.write_outputs(tmp_path, top_df, updated, stats_enabled, skip_reasons)
    saved = pd.read_csv(tmp_path / "data" / "scored_candidates.csv")
    assert "sentiment" in saved.columns


def test_sentiment_targets_use_symbol_column(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run_ts = datetime(2024, 7, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "symbol": ["aaa", "BBB", None, "aaa"],
            "Score": [1.0, 2.0, 3.0, 4.0],
            "score_breakdown": ["{}"] * 4,
        }
    )

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_score(self, symbol: str, date_utc: str) -> Optional[float]:
            self.calls.append(symbol)
            return {"AAA": 0.5, "BBB": -0.25}.get(symbol)

    client = RecordingClient()
    settings = {"enabled": True, "weight": 0.0, "min": -1.0, "api_url": "http://example.test/api"}

    with caplog.at_level("INFO"):
        updated, summary = screener._apply_sentiment_scores(
            frame,
            run_ts=run_ts,
            settings=settings,
            cache_dir=tmp_path / "data" / "cache" / "sentiment",
            client_factory=lambda *_: client,
        )

    assert client.calls == ["AAA", "BBB"]
    assert "[WARN] SENTIMENT_FETCH skipped reason=empty_target" not in caplog.text
    assert summary["sentiment_missing_count"] == 1
    assert pd.isna(updated.loc[2, "sentiment"])
    assert updated.loc[0:1, "sentiment"].tolist() == pytest.approx([0.5, -0.25], nan_ok=True)
    assert updated.loc[3, "sentiment"] == pytest.approx(0.5)


def test_sentiment_stage_entry_log_and_fetch_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run_ts = datetime(2024, 8, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame({"symbol": ["AAA"], "Score": [1.0], "score_breakdown": ["{}"]})
    settings = {"enabled": True, "weight": 0.0, "min": -1.0, "api_url": "http://example.test/api"}

    with caplog.at_level("INFO"):
        screener._apply_sentiment_scores(
            frame,
            run_ts=run_ts,
            settings=settings,
            cache_dir=tmp_path / "data" / "cache" / "sentiment",
            client_factory=lambda *_: None,
        )

    assert "[INFO] SENTIMENT_STAGE enter df_rows=1 cols_has_symbol=True" in caplog.text
    assert "[INFO] SENTIMENT_FETCH start target=1 unique=1" in caplog.text
    assert "[INFO] SENTIMENT_FETCH done target=1" in caplog.text


def test_sentiment_runs_when_candidates_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run_ts = datetime(2024, 9, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "symbol": [f"SYM{i}" for i in range(10)],
            "Score": list(range(10)),
            "score_breakdown": ["{}"] * 10,
        }
    )
    settings = {"enabled": True, "weight": 0.0, "min": -1.0, "api_url": "http://example.test/api"}

    class NullClient:
        def __init__(self) -> None:
            self.errors = 0

        def get_score(self, symbol: str, date_utc: str) -> Optional[float]:
            return None

    with caplog.at_level("INFO"):
        updated, summary = screener._apply_sentiment_scores(
            frame,
            run_ts=run_ts,
            settings=settings,
            cache_dir=tmp_path / "data" / "cache" / "sentiment",
            client_factory=lambda *_: NullClient(),
        )

    cache_file = tmp_path / "data" / "cache" / "sentiment" / f"{run_ts.date().isoformat()}.json"
    assert cache_file.exists()
    cache_payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cache_payload == {}
    assert cache_file.stat().st_size > 0
    assert "[INFO] SENTIMENT_STAGE enter df_rows=10 cols_has_symbol=True" in caplog.text
    assert "[INFO] SENTIMENT_FETCH start target=10 unique=10" in caplog.text
    assert "[INFO] SENTIMENT_FETCH done target=10" in caplog.text
    assert summary["sentiment_missing_count"] == len(frame)
    assert summary["sentiment_enabled"] is True

    stats_common = {
        "candidates_out": 0,
        "shortlist_requested": 10,
        "shortlist_candidates": len(frame),
        "coarse_ranked": len(frame),
        "shortlist_path": "",
        "backtest_target": 0,
        "backtest_evaluated": 0,
        "backtest_lookback": 0,
        "backtest_expectancy_mean": 0.0,
        "backtest_win_rate_mean": 0.0,
    }
    stats_enabled = {
        **stats_common,
        "sentiment_enabled": True,
        "sentiment_missing_count": summary["sentiment_missing_count"],
        "sentiment_avg": summary["sentiment_avg"],
        "sentiment_gated": summary.get("sentiment_gated", 0),
    }
    skip_reasons = {key: 0 for key in screener.SKIP_KEYS}
    top_df = pd.DataFrame(columns=screener.TOP_CANDIDATE_COLUMNS)
    screener.write_outputs(
        tmp_path,
        top_df,
        updated.iloc[0:0],
        stats_enabled,
        skip_reasons,
        now=run_ts,
    )
    metrics_path = tmp_path / "data" / "screener_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sentiment_enabled"] is True
    assert metrics["sentiment_missing_count"] == len(frame)
    assert metrics["sentiment_avg"] is None
