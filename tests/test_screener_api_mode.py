import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import screener

os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "secret")


def _build_mock_bars() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=200, tz="UTC", freq="B")
    trend = np.linspace(50, 130, len(dates))
    trend[-20:] = np.linspace(110, 130, 20)
    good_close = trend.copy()
    good_close[-2] = good_close[-3] - 15
    good_close[-1] = good_close[-3] + 10
    other_close = np.linspace(30, 60, len(dates))

    def make_frame(symbol: str, exchange: str, close_values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": dates,
                "open": close_values - 0.5,
                "high": close_values + 1.0,
                "low": close_values - 1.0,
                "close": close_values,
                "volume": np.full(len(dates), 1_000_000, dtype=float),
            }
        )

    frames = [
        make_frame("GOOD", "NASDAQ", good_close),
        make_frame("UNKNOWN1", "PINK", other_close),
        make_frame("CRYPTO1", "CRYPTO", other_close),
    ]
    return pd.concat(frames, ignore_index=True)


def test_screener_api_mode_creates_outputs(tmp_path, monkeypatch):
    # Ensure credentials are visible to autouse fixtures and the screener under test
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

    mock_bars = _build_mock_bars()

    monkeypatch.setattr(screener, "_create_trading_client", lambda: object())
    monkeypatch.setattr(screener, "_create_data_client", lambda: object())
    monkeypatch.setattr(
        screener,
        "fetch_active_equity_symbols",
        lambda *args, **kwargs: (
            ["GOOD", "UNKNOWN1", "CRYPTO1"],
            {
                "GOOD": {"exchange": "NASDAQ", "asset_class": "US_EQUITY", "tradable": True},
                "UNKNOWN1": {"exchange": "PINK", "asset_class": "OTC", "tradable": True},
                "CRYPTO1": {"exchange": "CRYPTO", "asset_class": "CRYPTO", "tradable": True},
            },
            {"assets_total": 3, "assets_tradable_equities": 1, "assets_after_filters": 1},
        ),
    )
    monkeypatch.setattr(
        screener,
        "_fetch_daily_bars",
        lambda *args, **kwargs: (
            mock_bars.copy(),
            {
                "batches_total": 0,
                "batches_paged": 0,
                "pages_total": 0,
                "bars_rows_total": int(mock_bars.shape[0]),
                "symbols_with_bars": 1,
                "symbols_no_bars": [],
                "fallback_batches": 0,
                "insufficient_history": 0,
            },
            {},
        ),
    )

    exit_code = screener.main([], output_dir=tmp_path)
    assert exit_code == 0

    data_dir = Path(tmp_path) / "data"
    top_path = data_dir / "top_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"

    assert top_path.exists()
    top_df = pd.read_csv(top_path)
    assert not top_df.empty
    assert set(top_df["symbol"]) == {"GOOD"}

    metrics = json.loads(metrics_path.read_text())
    assert metrics["symbols_in"] > 0
    assert metrics["skips"]["UNKNOWN_EXCHANGE"] == 1
    assert metrics["skips"]["NON_EQUITY"] == 1
