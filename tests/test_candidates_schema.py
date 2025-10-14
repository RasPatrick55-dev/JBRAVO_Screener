import pandas as pd
import pytest

from scripts.fallback_candidates import normalize_candidate_df


pytestmark = pytest.mark.alpaca_optional


def test_normalize_candidate_df_handles_synonyms_and_entry_price():
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "score": "3.2",
                "exchange": "NASDAQ",
                "close": "150.5",
                "volume": "1000000",
                "universe count": "10",
                "score breakdown": "{}",
            },
            {
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

    normalized = normalize_candidate_df(frame, now_ts="2024-01-01T00:00:00Z")

    assert "score_breakdown" in normalized.columns
    assert "universe_count" in normalized.columns
    assert "entry_price" in normalized.columns
    assert (normalized["entry_price"] == normalized["close"]).all()
    assert len(normalized) >= 1
    assert normalized.iloc[0]["symbol"] == "AAPL"
