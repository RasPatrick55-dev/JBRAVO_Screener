"""Bull-flag v1 adapter; existing Pullback/Reclaim predicates/kernel stay intact."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from scripts.chart_patterns import detect_bull_flag, validated_exit_plan
from scripts.research_strategies import BASELINE, prepare_signals

BULL_FLAG = 'bull_flag_breakout_v1'


def prepare(raw, split, fold, *, latest_only=False):
    frame = prepare_signals(raw, split, BASELINE, fold)
    frame['score'] = np.nan
    prior_volume = split.volume.shift(1).rolling(20).mean()
    candidate = ((split.close > split.open) & (split.volume >= prior_volume * 1.5)
                 & (split.close > split.high.shift(1).rolling(3).max() * 1.001)
                 & (split.close <= split.high.shift(1).rolling(10).max() * 1.05))
    plans = {}
    for day in frame.index[frame.entry_ready & candidate.reindex(frame.index).fillna(False)]:
        plan = None
        if not latest_only or day == frame.index[-1]:
            history = split.loc[:day].tail(21)
            found = detect_bull_flag(history)
            if found is not None:
                plan = asdict(found)
                # Dataset validation requires a constant, known split factor.
                factor = float(raw.loc[day, 'close'] / split.loc[day, 'close'])
                for key in ('entry', 'resistance', 'invalidation', 'target'):
                    plan[key] *= factor
                plan['pattern'] = BULL_FLAG
                score = 100 * found.metrics['pole_gain'] + min(found.metrics['volume_ratio'], 10)
                frame.loc[day, 'score'] = score
        if plan is not None:
            plans[day] = plan
    frame['entry_details'] = [dict(signal=day in plans,
        failures=[] if day in plans else ['NO_CONFIRMED_BULL_FLAG'], pattern_exit=plans.get(day)) for day in frame.index]
    return frame


def plan_from_row(symbol, row):
    details = row.get('entry_details', {})
    plan = details.get('pattern_exit') if isinstance(details, dict) else None
    if not isinstance(plan, dict):
        return None
    result = dict(plan, symbol=symbol)
    # chart_patterns registers the detector's short name; the strategy catalog
    # carries the distinct full versioned identity.
    checked = validated_exit_plan(dict(result, pattern='bull_flag'), symbol)
    if checked is None:
        return None
    return dict(checked, pattern=BULL_FLAG)


class BullFlagMixin:
    """Add fixed structural exits around the unchanged portfolio kernel."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_plans = {}
        self.enable_partial_exit = False

    def _open_position(self, symbol, row, day, signal_row):
        plan = plan_from_row(symbol, signal_row)
        if plan is None or not plan['invalidation'] < float(row['open']) < plan['target']:
            return
        super()._open_position(symbol, row, day, signal_row)
        if symbol in self.positions:
            self.pattern_plans[symbol] = plan

    def _protective_stop(self, position):
        existing, reason = super()._protective_stop(position)
        plan = self.pattern_plans.get(position.symbol)
        if plan and (existing is None or plan['invalidation'] > existing):
            return plan['invalidation'], 'PATTERN_INVALIDATION'
        return existing, reason

    def _before_session(self, day):
        super()._before_session(day)
        for symbol in list(self.positions):
            frame = self.data[symbol]
            if day not in frame.index:
                continue
            opening = float(frame.loc[day, 'open'])
            stop, _ = self._protective_stop(self.positions[symbol])
            plan = self.pattern_plans.get(symbol)
            if plan and opening >= plan['target'] and (stop is None or opening > stop):
                self._close_position(symbol, opening, day, 'PATTERN_TARGET')
                self.pending_exits.pop(symbol, None)

    def _update_completed_bar(self, symbol, row, day):
        plan = self.pattern_plans.get(symbol)
        # The kernel has already applied prior stops before this method. A bar
        # touching both stop and target therefore resolves conservatively.
        if plan and float(row['high']) >= plan['target']:
            self._close_position(symbol, plan['target'], day, 'PATTERN_TARGET')
            self.pending_exits.pop(symbol, None)
            return
        super()._update_completed_bar(symbol, row, day)


def research_portfolio_class():
    from scripts.research_simulation import ResearchPortfolio

    class BullFlagPortfolio(BullFlagMixin, ResearchPortfolio):
        def __init__(self, data, **kwargs):
            # Signal preparation supplies bull-flag-only scores. Reuse the
            # baseline consumer, without changing the versioned P/R consumer.
            super().__init__(data, method=BASELINE, **kwargs)

    return BullFlagPortfolio


def scanner_portfolio_class(base):
    class BullFlagScanner(BullFlagMixin, base):
        pass
    return BullFlagScanner
