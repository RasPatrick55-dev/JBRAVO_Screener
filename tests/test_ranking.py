import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.ranking import apply_gates, score_universe


pytestmark = pytest.mark.alpaca_optional


def _sample_features() -> pd.DataFrame:
    rows = [
        {
            "symbol": "AAA",
            "timestamp": pd.Timestamp("2024-01-05", tz="UTC"),
            "TS": 1.2,
            "MS": 0.8,
            "BP": 0.15,
            "PT": 0.2,
            "MH": 0.1,
            "RSI": 60.0,
            "ADX": 30.0,
            "AROON": 70.0,
            "VCP": -0.5,
            "VOLexp": 1.5,
            "GAPpen": 0.02,
            "LIQpen": 0.000001,
            "SMA9": 11.0,
            "EMA20": 10.0,
            "SMA50": 9.5,
            "SMA100": 9.0,
            "history": 40,
        },
        {
            "symbol": "BBB",
            "timestamp": pd.Timestamp("2024-01-05", tz="UTC"),
            "TS": 0.4,
            "MS": -0.2,
            "BP": -0.1,
            "PT": 0.05,
            "MH": -0.1,
            "RSI": 44.0,
            "ADX": 15.0,
            "AROON": 30.0,
            "VCP": 0.2,
            "VOLexp": 0.7,
            "GAPpen": 0.01,
            "LIQpen": 0.00002,
            "SMA9": 10.5,
            "EMA20": 10.2,
            "SMA50": 10.0,
            "SMA100": 9.8,
            "history": 35,
        },
        {
            "symbol": "CCC",
            "timestamp": pd.Timestamp("2024-01-05", tz="UTC"),
            "TS": 0.9,
            "MS": 0.6,
            "BP": 0.05,
            "PT": 0.12,
            "MH": 0.05,
            "RSI": 58.0,
            "ADX": 25.0,
            "AROON": 65.0,
            "VCP": -0.2,
            "VOLexp": 1.1,
            "GAPpen": 0.03,
            "LIQpen": 0.00003,
            "SMA9": 12.0,
            "EMA20": 11.8,
            "SMA50": 11.5,
            "SMA100": 11.6,
            "history": 10,
        },
    ]
    return pd.DataFrame(rows)


def _load_ranker_cfg() -> dict:
    cfg_path = Path("config/ranker.yml")
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        with cfg_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def test_score_shapes():
    cfg = _load_ranker_cfg()
    ranked = score_universe(_sample_features(), cfg)
    assert "Score" in ranked.columns
    assert "score_breakdown" in ranked.columns
    assert ranked["Score"].is_monotonic_decreasing
    parsed = json.loads(ranked.loc[0, "score_breakdown"])
    assert isinstance(parsed, dict)
    assert "trend" in parsed


def test_apply_gates_counts():
    cfg = _load_ranker_cfg()
    ranked = score_universe(_sample_features(), cfg)
    candidates, fail_counts, rejects = apply_gates(ranked, cfg)
    total_failures = sum(fail_counts.values())
    assert total_failures + len(candidates) == len(ranked)
    assert len(rejects) <= len(ranked)
