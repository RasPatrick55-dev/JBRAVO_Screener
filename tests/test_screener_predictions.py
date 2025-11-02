from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.screener import _prepare_predictions_frame


pytestmark = pytest.mark.alpaca_optional


def test_prepare_predictions_frame_orders_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "Score": [1.2, 0.5],
            "gates_passed": [True, False],
            "close": [10.0, 20.0],
            "ADV20": [2_000_000, 1_500_000],
            "ATR14": [0.5, 1.0],
            "TS": [1.0, -0.5],
            "MS": [0.2, -0.1],
            "BP": [0.3, -0.2],
            "PT": [0.1, -0.1],
            "RSI": [60, 55],
            "MH": [0.5, -0.2],
            "ADX": [25, 18],
            "AROON": [70, 40],
            "VCP": [0.1, -0.05],
            "VOLexp": [1.2, 0.8],
            "GAPpen": [0.01, 0.02],
            "LIQpen": [0.0, 0.1],
            "trend_score": [0.8, -0.4],
            "momentum_score": [0.6, -0.3],
            "volume_score": [0.5, -0.2],
            "volatility_score": [0.4, -0.1],
            "risk_score": [0.3, -0.05],
            "trend_contribution": [0.24, -0.12],
            "momentum_contribution": [0.15, -0.075],
            "volume_contribution": [0.10, -0.04],
            "volatility_contribution": [0.06, -0.015],
            "risk_contribution": [0.03, -0.005],
            "REL_VOLUME": [2.0, 0.8],
            "OBV_DELTA": [1_000, -500],
            "BB_BANDWIDTH": [0.05, 0.12],
            "ATR_pct": [0.05, 0.08],
            "score_breakdown": ['{"TS": 1.0, "MS": 0.5}', '{"TS": -1.0, "MS": -0.5}'],
        }
    )

    prepared = _prepare_predictions_frame(
        df,
        run_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        gate_counters={"gate_preset": "standard"},
        ranker_cfg={"version": "2024.07"},
        limit=1,
    )

    expected_columns = [
        "run_date",
        "symbol",
        "rank",
        "score",
        "passed_gates",
        "ranker_version",
        "gate_preset",
        "adv20",
        "price_close",
        "atr14",
        "ts",
        "ms",
        "bp",
        "pt",
        "rsi",
        "mh",
        "adx",
        "aroon",
        "vcp",
        "volexp",
        "gap_pen",
        "liq_pen",
        "score_breakdown_json",
        "trend_score",
        "momentum_score",
        "volume_score",
        "volatility_score",
        "risk_score",
        "trend_contribution",
        "momentum_contribution",
        "volume_contribution",
        "volatility_contribution",
        "risk_contribution",
        "rel_volume",
        "obv_delta",
        "bb_bandwidth",
        "atr_pct",
        "components_json",
        "relax_gates",
    ]
    assert list(prepared.columns) == expected_columns
    assert prepared.loc[0, "run_date"] == "2024-01-02"
    assert prepared.loc[0, "rank"] == 1
    assert bool(prepared.loc[0, "passed_gates"]) is True
    assert prepared.loc[0, "score"] == pytest.approx(1.2)
    assert prepared.loc[0, "adv20"] == pytest.approx(2_000_000)
    assert prepared.loc[0, "components_json"] == prepared.loc[0, "score_breakdown_json"]
