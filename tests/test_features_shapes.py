import numpy as np
import pandas as pd
import pytest

from scripts.features import ALL_FEATURE_COLUMNS, compute_all_features


pytestmark = pytest.mark.alpaca_optional


def _make_synthetic_bars(rows: int = 220) -> pd.DataFrame:
    symbols = ["AAA", "BBB"]
    rng = np.random.default_rng(42)
    frames = []
    for symbol in symbols:
        idx = pd.date_range("2023-01-01", periods=rows, freq="B", tz="UTC")
        base = np.linspace(10, 50, num=rows)
        noise = np.sin(np.linspace(0, np.pi * 4, num=rows))
        close = base + noise
        open_ = close + rng.normal(0, 0.5, size=rows)
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        volume = np.linspace(100_000, 250_000, num=rows) * (1 + 0.1 * noise)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": idx,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_feature_columns_and_no_nan_zscores():
    bars = _make_synthetic_bars()
    features = compute_all_features(bars, cfg={"gates": {"min_history": 50}})
    assert not features.empty

    missing = [col for col in ALL_FEATURE_COLUMNS if col not in features.columns]
    assert missing == []

    z_columns = [col for col in features.columns if col.endswith("_z")]
    assert z_columns, "expected at least one z-score column"
    z_block = features[z_columns]
    assert not z_block.isna().any().any()

    assert features.groupby("symbol").size().min() > 0
