from pathlib import Path

import pandas as pd

from scripts.fallback_candidates import ensure_min_candidates


def test_fallback_from_scored(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "t",
                "symbol": "AAA",
                "score": 2.0,
                "exchange": "NASDAQ",
                "close": 10,
                "volume": 1_000_000,
                "universe_count": 1,
                "score_breakdown": "x",
                "adv20": 2_500_000,
            },
            {
                "timestamp": "t",
                "symbol": "BBB",
                "score": 1.0,
                "exchange": "NYSE",
                "close": 12,
                "volume": 1_000_000,
                "universe_count": 1,
                "score_breakdown": "y",
                "adv20": 2_100_000,
            },
        ]
    ).to_csv(data / "scored_candidates.csv", index=False)
    rows, reason = ensure_min_candidates(tmp_path, 1)
    assert rows >= 1
    assert reason in ("scored_candidates", "already_populated")
    out = pd.read_csv(data / "latest_candidates.csv")
    assert {
        "timestamp",
        "symbol",
        "score",
        "exchange",
        "close",
        "volume",
        "universe_count",
        "score_breakdown",
    }.issubset(out.columns)
