import pytest
from utils.screener_metrics import ensure_canonical_metrics


pytestmark = pytest.mark.alpaca_optional


def test_legacy_fetch_counts_are_mapped_to_attempted() -> None:
    payload = {
        "symbols_in": 57,
        "symbols_with_any_bars": 57,
        "symbols_with_required_bars": 57,
        "symbols_with_bars_fetch": 419,
        "rows": 2,
    }

    canonical = ensure_canonical_metrics(payload)

    assert canonical["symbols_attempted_fetch"] == 419
    assert canonical["symbols_with_any_bars"] == 57
    assert canonical["with_bars"] == 57
    assert canonical["metrics_version"] == 2


def test_required_bars_and_with_bars_are_canonicalised() -> None:
    payload = {
        "symbols_in": 10,
        "symbols_with_bars": 8,
        "rows": 1,
        "required_bars": 250,
    }

    canonical = ensure_canonical_metrics(payload)

    assert canonical["symbols_with_required_bars"] == 8
    assert canonical["with_bars_required"] == 8
    assert canonical["with_bars"] == 8
    assert canonical["required_bars"] == 250
    assert canonical["metrics_version"] == 2
