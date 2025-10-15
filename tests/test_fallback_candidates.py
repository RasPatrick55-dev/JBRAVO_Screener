from pathlib import Path

import pandas as pd

from scripts.fallback_candidates import CANONICAL_COLUMNS, ensure_min_candidates


def test_fallback_from_scored(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    pd.DataFrame(columns=["symbol", "score"]).to_csv(data / "top_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "score": 2.0,
                "exchange": "NASDAQ",
                "close": 10,
                "volume": 1_000_000,
                "universe count": 1,
                "score breakdown": "x",
            },
            {
                "symbol": "BBB",
                "score": 4.0,
                "exchange": "NYSE",
                "close": 12,
                "volume": 1_000_000,
                "universe count": 1,
                "score breakdown": "y",
            },
            {
                "symbol": "CCC",
                "score": 3.5,
                "exchange": "AMEX",
                "close": 11,
                "volume": 900_000,
                "universe count": 1,
                "score breakdown": "z",
            },
        ]
    ).to_csv(data / "scored_candidates.csv", index=False)
    rows, reason = ensure_min_candidates(
        tmp_path,
        1,
        canonicalize=True,
        prefer="top_then_scored",
    )
    assert rows >= 1
    assert reason.startswith("scored")
    out = pd.read_csv(data / "latest_candidates.csv")
    assert out.columns.tolist() == list(CANONICAL_COLUMNS)
    assert out.iloc[0]["symbol"] == "BBB"
