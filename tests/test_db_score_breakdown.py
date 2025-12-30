import json
import logging

import pytest

from scripts.db import normalize_score_breakdown

pytestmark = pytest.mark.alpaca_optional


def test_normalize_score_breakdown_accepts_dict():
    payload = {"factor": 1, "reasons": ["alpha", "beta"]}

    serialized = normalize_score_breakdown(payload, symbol="abc")

    assert json.loads(serialized) == payload


def test_normalize_score_breakdown_raw_string_wrapped():
    serialized = normalize_score_breakdown("not-json", symbol="abc")

    assert json.loads(serialized) == {"raw": "not-json"}


def test_normalize_score_breakdown_logs_and_returns_none_on_failure(caplog):
    with caplog.at_level(logging.WARNING):
        serialized = normalize_score_breakdown({"bad": {1, 2}}, symbol="abc")

    assert serialized is None
    assert any("SCORE_BREAKDOWN_JSON_FAIL" in rec.message for rec in caplog.records)
