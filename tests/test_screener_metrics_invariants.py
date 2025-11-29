import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.alpaca_optional


FIXTURE_DIR = Path(__file__).parent / "data"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _assert_screener_metrics(metrics: dict) -> None:
    assert metrics["symbols_in"] >= 0
    assert metrics["symbols_with_bars"] >= 0
    assert metrics["symbols_with_bars"] <= metrics["symbols_in"]
    assert metrics["rows"] >= 0


def test_screener_metrics_fixture_ok() -> None:
    metrics = _load("screener_metrics_ok.json")
    _assert_screener_metrics(metrics)


def test_screener_metrics_fixture_violation_raises() -> None:
    metrics = _load("screener_metrics_bad.json")
    with pytest.raises(AssertionError):
        _assert_screener_metrics(metrics)
