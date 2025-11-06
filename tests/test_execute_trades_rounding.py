import pandas as pd
import pytest

pytestmark = pytest.mark.alpaca_optional

from scripts.execute_trades import (
    _canonicalize_candidate_header,
    _enforce_order_price_ticks,
    round_to_tick,
)


class _DummyRequest:
    def __init__(self, limit_price: float, stop_price: float) -> None:
        self.limit_price = limit_price
        self.stop_price = stop_price


def test_round_to_tick_respects_us_equity_ticks() -> None:
    assert round_to_tick(1.2345) == pytest.approx(1.23)
    assert round_to_tick(0.123456) == pytest.approx(0.1234)


def test_enforce_order_price_ticks_rounds_all_price_fields() -> None:
    request = _DummyRequest(limit_price=1.2345, stop_price=0.456789)
    _enforce_order_price_ticks(request)
    assert request.limit_price == pytest.approx(1.23)
    assert request.stop_price == pytest.approx(0.4567)


def test_canonicalize_candidate_header_uppercases_symbol() -> None:
    df = pd.DataFrame(
        {
            "Symbol": [" abc "],
            "Price": [12.3456],
            "ScoreBreakdown": [""],
        }
    )
    result = _canonicalize_candidate_header(df, base_dir=None)
    assert list(result["symbol"]) == ["ABC"]
    # Score breakdown should default to a JSON object string and timestamp should be populated.
    assert result.loc[result.index[0], "score_breakdown"] == "{}"
    assert pd.notna(result.loc[result.index[0], "timestamp"])
