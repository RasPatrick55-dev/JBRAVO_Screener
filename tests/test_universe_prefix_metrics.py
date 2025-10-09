import pandas as pd
import pytest

from scripts.screener import _write_universe_prefix_metrics


@pytest.mark.alpaca_optional
def test_universe_prefix_counts_updates_metrics():
    metrics: dict[str, object] = {}
    df = pd.DataFrame({"symbol": ["AAPL", "AMZN", "MSFT", "META", "GOOG"]})

    prefix_counts = _write_universe_prefix_metrics(df, metrics)

    assert set(prefix_counts.keys()) >= {"A", "M", "G"}
    assert metrics["universe_prefix_counts"] == prefix_counts
