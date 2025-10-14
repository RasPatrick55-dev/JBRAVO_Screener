import pandas as pd
import pytest

from scripts.fallback_candidates import normalize_candidate_df


pytestmark = pytest.mark.alpaca_optional


def test_normalize_candidate_df_handles_synonyms_and_entry_price():
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "AAPL",
                "score": "3.2",
                "exchange": "NASDAQ",
                "close": "150.5",
                "volume": "1000000",
                "universe count": "10",
                "score breakdown": "{}",
            },
            {
                "timestamp": "2024-01-01T00:05:00Z",
                "symbol": "MSFT",
                "score": "2.1",
                "exchange": "NASDAQ",
                "close": "320.10",
                "volume": "2000000",
                "Universe Count": "15",
                "Score Breakdown": "{}",
            },
        ]
    )

    normalized = normalize_candidate_df(frame)

    assert "score_breakdown" in normalized.columns
    assert "universe_count" in normalized.columns
    assert "entry_price" in normalized.columns
    assert not normalized["entry_price"].isna().any()
    # Scores should be sorted descending
    assert list(normalized["symbol"]) == ["AAPL", "MSFT"]
