"""Isolated policy experiments over the existing causal portfolio/cash kernel."""
from __future__ import annotations

import math

import pandas as pd

from scripts.pattern_strategy import PATTERNS
from scripts.research_simulation import ResearchPortfolio
from scripts.research_strategies import BASELINE

EXIT_POLICIES = {
    'existing': 'Existing ATR/trailing/indicator/time rules; pattern structural stop and measured target; no pattern partials',
    'structural_target': 'Fixed structural stop; pattern measured target or Pullback 2R target; seven calendar days',
    'atr_stop': 'Fixed 1.5 signal-ATR stop from entry; no target; seven calendar days',
    'fixed_2r': 'Fixed structural stop and entry plus twice initial risk; seven calendar days',
    'close_trailing': 'Structural floor and 3% trailing completed-close stop effective next session; seven calendar days',
    'time_3d': 'Fixed structural stop; no target; exit queued after three calendar days',
}


class StudyPortfolio(ResearchPortfolio):
    """The supplied frozen screen controls entries; research never changes runtime rules."""
    def __init__(self, data, *, exit_policy='existing', **options):
        if exit_policy not in EXIT_POLICIES:
            raise ValueError('Unknown frozen exit policy')
        super().__init__(data, method=BASELINE, **options)
        self.exit_policy = exit_policy
        self.active = {}
        self.attempts = []

    def _entry_candidates(self, day, rows):
        return [(symbol, row['score'], row) for symbol, row in rows.items()
                if row.get('research_proposal') is not None and pd.notna(row['score']) and
                symbol not in self.positions and symbol not in self.pending_entries]

    def _open_position(self, symbol, row, day, signal_row):
        proposal = signal_row['research_proposal']
        entry = float(row['open'])
        pattern = proposal.get('pattern_exit')
        stop = float(proposal['structural_stop'])
        target = None
        attempt = dict(signal_id=proposal['signal_id'], symbol=symbol,
                       session=day.strftime('%Y-%m-%d'), status='UNFILLED')
        self.attempts.append(attempt)
        # Every exit trial keeps the original pattern's entry eligibility.
        # A different exit may not introduce buys that the baseline refuses.
        if pattern and not pattern['invalidation'] < entry < pattern['target']:
            attempt['status'] = 'OPEN_OUTSIDE_PATTERN_LEVELS'
            return
        if self.exit_policy == 'existing':
            target = pattern['target'] if pattern else None
        else:
            if self.exit_policy == 'atr_stop':
                stop = entry - 1.5 * proposal['signal_atr14_raw']
            if not 0 < stop < entry:
                attempt['status'] = 'OPEN_OUTSIDE_STOP'
                return
            if self.exit_policy == 'fixed_2r':
                target = entry + 2 * (entry-stop)
            if self.exit_policy == 'structural_target':
                target = pattern['target'] if pattern else entry + 2 * (entry-stop)
                if entry >= target:
                    attempt['status'] = 'OPEN_OUTSIDE_TARGET'
                    return
        count = len(self.fills)
        super()._open_position(symbol, row, day, signal_row)
        if len(self.fills) == count:
            attempt['status'] = 'CASH_OR_WHOLE_SHARE_BUDGET'
            return
        self.active[symbol] = dict(proposal=proposal, stop=stop, target=target)
        if self.exit_policy != 'existing':
            pos = self.positions[symbol]
            pos.atr_stop = stop
            pos.trailing_stop = entry * .97 if self.exit_policy == 'close_trailing' else None
        attempt['status'] = 'MODELED_FILL'
        initial_stop,_ = self._protective_stop(self.positions[symbol])
        self.fills[-1].update(signal_id=proposal['signal_id'], strategy_id=proposal['strategy_id'],initial_stop=initial_stop)

    def _record_exit(self, position, quantity, price, day, reason):
        proposal = self.active[position.symbol]['proposal']
        super()._record_exit(position, quantity, price, day, reason)
        self.fills[-1].update(signal_id=proposal['signal_id'], strategy_id=proposal['strategy_id'])

    def _protective_stop(self, pos):
        state = self.active.get(pos.symbol)
        if not state:
            return super()._protective_stop(pos)
        if self.exit_policy != 'existing':
            return max(state['stop'], pos.trailing_stop or 0), 'RESEARCH_POLICY_STOP'
        existing, reason = super()._protective_stop(pos)
        pattern = state['proposal'].get('pattern_exit')
        if pattern and (existing is None or pattern['invalidation'] > existing):
            return pattern['invalidation'], 'PATTERN_INVALIDATION'
        return existing, reason

    def _before_session(self, day):
        super()._before_session(day)
        for symbol in list(self.positions):
            if day not in self.data[symbol].index:
                continue
            opening = float(self.data[symbol].loc[day, 'open'])
            stop, _ = self._protective_stop(self.positions[symbol])
            target = self.active[symbol]['target']
            if target is not None and opening >= target and (stop is None or opening > stop):
                self._close_position(symbol, opening, day, 'PATTERN_TARGET' if self.exit_policy=='existing' else 'RESEARCH_TARGET')
                self.pending_exits.pop(symbol, None)

    def _update_completed_bar(self, symbol, row, day):
        target = self.active[symbol]['target']
        if target is not None and float(row['high']) >= target:
            # The shared kernel already applied prior stops, including entry stops.
            self._close_position(symbol, target, day, 'PATTERN_TARGET' if self.exit_policy=='existing' else 'RESEARCH_TARGET')
            self.pending_exits.pop(symbol, None)
            return
        if self.exit_policy == 'existing':
            before = self.enable_partial_exit
            if self.active[symbol]['proposal']['strategy_id'] in PATTERNS:
                self.enable_partial_exit = False
            try:
                super()._update_completed_bar(symbol, row, day)
            finally:
                self.enable_partial_exit = before
            return
        pos = self.positions[symbol]
        pos.max_price = max(pos.max_price, float(row['high']))
        pos.highest_close = max(pos.highest_close, float(row['close']))
        if self.exit_policy == 'close_trailing':
            pos.trailing_stop = max(pos.trailing_stop or 0, float(row['close']) * .97)
        limit = 3 if self.exit_policy == 'time_3d' else self.max_hold_days
        if (day-pos.entry_time).days >= limit:
            self.pending_exits[symbol] = ['RESEARCH_TIME_EXIT']


def frames_for_book(prepared, observations, *, family='ALL_STRATEGIES', only_signal=None):
    """Copy exact signal rows into the common kernel; no fourth-ranked backfill."""
    data = {}
    for item in observations:
        if only_signal is not None:
            use = item['signal_id'] == only_signal
        elif family == 'ALL_STRATEGIES':
            use = item['selected'] and item['is_exit_owner']
        else:
            use = item['strategy_id'] == family and item['family_rank'] <= 3
        if not use:
            continue
        symbol, sid = item['symbol'], item['strategy_id']
        frame = prepared[(sid, symbol)]
        if symbol not in data:
            data[symbol] = frame.copy(deep=True)
            data[symbol]['score'] = math.nan
            data[symbol]['research_proposal'] = pd.Series([None]*len(frame), index=frame.index, dtype=object)
        day = pd.Timestamp(item['signal_session'])
        data[symbol].loc[day] = frame.loc[day]
        rank = item['global_rank'] if family=='ALL_STRATEGIES' else item['family_rank']
        # The screen has already used the production common ranking key. Equal
        # rank never occurs within one portfolio, so the kernel's sort is stable.
        data[symbol].at[day, 'score'] = -float(rank)
        data[symbol].at[day, 'research_proposal'] = item['proposal']
    if only_signal is not None:
        chosen = next(r for r in observations if r['signal_id']==only_signal)
        symbol = chosen['symbol']
        data[symbol] = data[symbol].loc[chosen['signal_session']:].copy()
    return data
