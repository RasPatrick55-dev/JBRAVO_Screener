from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.eval_ranker import (
    EvaluationConfig,
    average_precision,
    compute_decile_lifts,
    label_predictions,
    roc_auc_score,
)


pytestmark = pytest.mark.alpaca_optional


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "as_of": [date(2024, 1, 1), date(2024, 1, 1)],
            "Score": [0.9, 0.2],
            "score_breakdown": ['{"trend": 1.0}', '{"trend": -1.0}'],
            "gates_passed": [True, False],
        }
    )


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "BBB",
            ],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
            ],
            "open": [100, 101, 102, 50, 49, 48],
            "high": [101, 106, 103, 51, 48, 47],
            "low": [99, 100, 101, 49, 45, 44],
            "close": [100, 105, 102, 50, 46, 45],
        }
    )


def test_label_predictions_hit_vs_drawdown():
    cfg = EvaluationConfig(days=5, label_horizon=3, hit_threshold=0.04)
    labelled = label_predictions(_sample_predictions(), _sample_prices(), cfg)
    assert labelled["label"].tolist() == [1, 0]


def test_auc_and_average_precision():
    y_true = np.array([0, 1, 0, 1], dtype=float)
    scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    assert roc_auc_score(y_true, scores) == 1.0
    assert average_precision(y_true, scores) == 1.0


def test_decile_lifts_shape():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    scores = np.linspace(1, 0, len(y_true))
    lifts = compute_decile_lifts(y_true, scores, deciles=5)
    assert list(lifts.keys()) == ["1", "2", "3", "4", "5"]
    assert lifts["1"]["count"] >= 1
