import pandas as pd
import pytest

from scripts.screener import _prepare_predictions_frame


pytestmark = pytest.mark.alpaca_optional


def test_prepare_predictions_frame_orders_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "Score": [1.2, 0.5],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
        }
    )

    prepared = _prepare_predictions_frame(df)
    assert list(prepared.columns[:4]) == ["timestamp", "symbol", "Score", "rank"]
    assert prepared["gates_passed"].dtype == bool
    assert prepared.loc[0, "rank"] == 1
