from pathlib import Path

import pandas as pd
import pytest

from scripts import execute_trades


pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def reset_logger():
    yield
    for handler in list(execute_trades.LOGGER.handlers):
        execute_trades.LOGGER.removeHandler(handler)
        handler.close()


def test_load_candidates_success(tmp_path: Path):
    csv_path = tmp_path / "candidates.csv"
    df = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "close": [10.5],
            "score": [1.2],
            "universe_count": [500],
            "score_breakdown": ["alpha"],
            "adv20": [3_000_000],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = execute_trades.load_candidates(csv_path)
    assert not loaded.empty
    for column in df.columns:
        assert column in loaded.columns


def test_load_candidates_missing_required_columns(tmp_path: Path):
    csv_path = tmp_path / "candidates.csv"
    pd.DataFrame({"symbol": ["ABC"], "close": [10.0]}).to_csv(csv_path, index=False)

    with pytest.raises(execute_trades.CandidateLoadError):
        execute_trades.load_candidates(csv_path)


def test_apply_guards_filters_by_price_and_adv(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(execute_trades, "_fetch_latest_daily_bars", lambda symbols: {})
    monkeypatch.setattr(execute_trades, "_fetch_latest_close_from_alpaca", lambda symbol: None)
    df = pd.DataFrame(
        {
            "symbol": ["LOW", "HIGH", "LIQ"],
            "close": [0.5, 1500.0, 10.0],
            "entry_price": [0.5, 1500.0, 10.0],
            "score": [1, 1, 1],
            "universe_count": [500, 500, 500],
            "score_breakdown": ["", "", ""],
            "adv20": [3_000_000, 3_000_000, 500_000],
        }
    )
    metrics = execute_trades.ExecutionMetrics()
    config = execute_trades.ExecutorConfig(min_price=1.0, max_price=1000.0, min_adv20=1_000_000)

    filtered = execute_trades.apply_guards(df, config, metrics)

    assert filtered.empty
    assert metrics.skipped_by_reason["PRICE_LT_MIN"] == 1
    assert metrics.skipped_by_reason["PRICE_GT_MAX"] == 1
    assert metrics.skipped_by_reason["ADV20_LT_MIN"] == 1


def test_run_executor_returns_zero_when_all_filtered(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "close": [5.0],
            "score": [0.5],
            "universe_count": [100],
            "score_breakdown": [""],
            "adv20": [100_000],
        }
    ).to_csv(csv_path, index=False)

    config = execute_trades.ExecutorConfig(
        source=csv_path,
        min_adv20=200_000,
        log_json=True,
    )

    monkeypatch.setenv("APCA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(execute_trades, "_fetch_latest_daily_bars", lambda symbols: {})
    monkeypatch.setattr(execute_trades, "_fetch_latest_close_from_alpaca", lambda symbol: None)

    exit_code = execute_trades.run_executor(config)
    assert exit_code == 0
