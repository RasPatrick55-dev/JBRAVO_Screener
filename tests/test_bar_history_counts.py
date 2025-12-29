import pandas as pd

from scripts.screener import summarize_bar_history_counts, BARS_REQUIRED_FOR_INDICATORS


def test_summarize_bar_history_counts_respects_threshold_and_any_count():
    history = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "C", "C", "C", "D"],
            "n": [BARS_REQUIRED_FOR_INDICATORS, 0, BARS_REQUIRED_FOR_INDICATORS + 10, 100, 100, 100, 1],
        }
    )

    summary = summarize_bar_history_counts(history, required_bars=BARS_REQUIRED_FOR_INDICATORS)

    assert summary["any"] == 4
    assert summary["required"] == 3
