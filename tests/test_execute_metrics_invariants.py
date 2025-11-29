import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.alpaca_optional


FIXTURE_DIR = Path(__file__).parent / "data"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _assert_execute_metrics(metrics: dict) -> None:
    assert metrics["orders_submitted"] >= 0
    assert metrics["orders_filled"] >= 0
    skips = metrics.get("skips") or {}
    assert isinstance(skips, dict)
    for value in skips.values():
        assert value >= 0
    assert metrics["orders_submitted"] + sum(skips.values()) >= 0


def test_execute_metrics_fixture_ok() -> None:
    metrics = _load("execute_metrics_ok.json")
    _assert_execute_metrics(metrics)


def test_execute_metrics_invalid_skips_raise() -> None:
    metrics = _load("execute_metrics_ok.json")
    metrics["skips"]["TIME_WINDOW"] = -1
    with pytest.raises(AssertionError):
        _assert_execute_metrics(metrics)
