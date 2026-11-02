"""Deterministic public/synthetic OHLCV fixtures; no provider or configuration."""
import numpy as np
import pandas as pd
from unittest.mock import patch

from scripts.swing_patterns import STRATEGIES
from scripts import swing_patterns, pattern_strategy

ACTUAL_PREPARE_MANY = pattern_strategy.prepare_many


def reference_input(bars):
    """Unmodified pre-optimization validator from current main 67a63f02."""
    columns = ['open', 'high', 'low', 'close', 'volume']
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.hasnans or
        not bars.index.is_monotonic_increasing or bars.index.normalize().has_duplicates or
        not set(columns) <= set(bars.columns) or len(bars) < 31):
        return None
    frame = bars.iloc[-141:][columns].apply(pd.to_numeric, errors='coerce')
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values <= 0).any() or
        (frame.high < frame[['open', 'close']].max(axis=1)).any() or
        (frame.low > frame[['open', 'close']].min(axis=1)).any()):
        return None
    current = frame.iloc[-1]
    ratio = float(current.volume/frame.volume.iloc[-21:-1].mean())
    if current.close <= current.open or ratio < 1.5:
        return None
    return frame.iloc[:-1], current, ratio


def baseline_prepare_many(*args, **kwargs):
    with patch.object(swing_patterns, '_input', reference_input):
        return ACTUAL_PREPARE_MANY(*args, **kwargs)


class SyntheticDataset:
    market_wide = True
    provenance = {'dataset_sha256': '0'*64}

    def __init__(self, frames, spec):
        self.spec = spec
        self.raw = {name: item[0] for name, item in frames.items()}
        self.features = {name: item[1] for name, item in frames.items()}

    def prepare(self, symbol, strategy, fold):
        return strategy.prepare(self.raw[symbol], self.features[symbol], fold)

    def member(self, symbol):
        return {'actions': []}


def shared_fixture(folder, count=4, bars=600):
    frames = {'SYN%03d'%i: item for i, item in enumerate(fixtures(count, bars))}
    spec = dict(capture_manifest=str(folder/'capture/input/manifest.json'),
                actions_capture={'synthetic': True}, universe={'symbols': list(frames)},
                folds=[next(iter(frames.values()))[2]], portfolio={'max_hold_days': 7})
    return SyntheticDataset(frames, spec), spec


def pattern_frame(strategy_id, bars=600):
    # Same positive geometries as the repository's test_swing_patterns fixtures.
    if strategy_id == STRATEGIES[0]:
        tail = np.interp(np.arange(31), [0, 4, 8, 12, 16, 20, 24, 28, 29, 30],
                         [109, 95, 109, 98, 109, 102, 109, 106, 108, 113])
    elif strategy_id == STRATEGIES[1]:
        tail = np.interp(np.arange(81), [0, 20, 40, 60, 79, 80], [130, 100, 115, 101, 114, 118])
    elif strategy_id == STRATEGIES[2]:
        cup = 110 - 20 * np.sin(np.linspace(0, np.pi, 61))
        tail = np.r_[np.linspace(90, 109, 20), cup, np.linspace(109, 106, 10), 114.]
    else:
        tail = np.interp(np.arange(81), [0, 20, 30, 40, 50, 60, 79, 80],
                         [130, 100, 112, 90, 112, 101, 111, 116])
    close = np.r_[np.linspace(70, 90, bars - len(tail)), tail]
    frame = pd.DataFrame(dict(open=close-.2, high=close+1, low=close-1,
                              close=close, volume=1000.),
                         index=pd.bdate_range('2023-01-03', periods=bars))
    frame.iloc[-20:, frame.columns.get_loc('volume')] = 800.
    frame.iloc[-1, frame.columns.get_loc('volume')] = 3000.
    frame['volume'] = frame.volume.astype('int64')
    return frame


def market_frame(seed=773, bars=600):
    rng = np.random.default_rng(seed)
    close = 70 * np.exp(np.cumsum(rng.normal(.0005, .018, bars)))
    opening = close * np.exp(rng.normal(0, .009, bars))
    high = np.maximum(opening, close) * (1 + rng.uniform(.001, .025, bars))
    low = np.minimum(opening, close) * (1 - rng.uniform(.001, .025, bars))
    return pd.DataFrame(dict(open=opening, high=high, low=low, close=close,
                              volume=np.maximum(1, np.rint(rng.lognormal(7, .7, bars))).astype('int64')),
                         index=pd.bdate_range('2023-01-03', periods=bars))


def fixtures(count=12, bars=600):
    frames = [pattern_frame(sid, bars) for sid in STRATEGIES]
    frames.extend(market_frame(773+i, bars) for i in range(max(0, count-4)))
    result = []
    for frame in frames[:count]:
        raw = frame.copy(deep=True)
        raw[['open', 'high', 'low', 'close']] *= 2
        fold = dict(warmup_start=str(frame.index[0].date()),
                    test_start=str(frame.index[270].date()),
                    test_end=str(frame.index[-1].date()))
        result.append((raw, frame, fold))
    return result
