import numpy as np
import pandas as pd
import pytest

from scripts.features import robust_z


pytestmark = pytest.mark.alpaca_optional


def test_robust_z_clips_to_range():
    series = pd.Series(np.linspace(-10, 10, num=41))
    clipped = robust_z(series, clip=2.5)
    assert np.isfinite(clipped).all()
    assert clipped.max() <= 2.5 + 1e-9
    assert clipped.min() >= -2.5 - 1e-9


def test_robust_z_symmetry():
    data = pd.Series([-5.0, -1.0, 0.0, 1.0, 5.0])
    z_scores = robust_z(data)
    assert np.isclose(z_scores.iloc[0], -z_scores.iloc[-1])
    assert z_scores.iloc[2] == 0
