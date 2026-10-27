"""Memory-bounded preparation for the unchanged portfolio/strategy kernels.

Only rows that cannot affect an entry, held position or its next-bar exit are
omitted from the portfolio input. All symbols still receive a current decision.
"""
import time

import numpy as np
import pandas as pd

from scripts.research_simulation import availability_clock


def prepare_market(dataset, spec, strategy, fold, *, deadline=None, stop_file=None):
    data, latest, dividends, observations = {}, {}, [], 0
    for symbol in spec['universe']['symbols']:
        if (deadline is not None and time.monotonic() >= deadline) or (stop_file and stop_file.exists()):
            raise RuntimeError('Market preparation stopped by deadline or stop file')
        try:
            frame = dataset.prepare(symbol, strategy, fold)
        except ValueError as error:
            # The strategy kernel can refuse finite but unready indicator history
            # (for example a newly listed or flat-price security). Never swallow
            # an unexpected kernel/data failure as a no-buy decision.
            if str(error) != 'Insufficient fresh warmup at evaluation boundary':
                raise
            raw=dataset.raw[symbol].loc[fold['warmup_start']:fold['test_end']]
            split=dataset.features[symbol].loc[fold['warmup_start']:fold['test_end']]
            from scripts.indicators import compute_indicators
            from scripts.research_strategies import ready
            featured = compute_indicators(split)
            records = featured.to_dict('records')
            first = next((i for i in range(249, len(records)) if
                          featured.index[i].strftime('%Y-%m-%d') >= fold['test_start'] and ready(records, i, 250)), None)
            if first is None:
                latest[symbol] = dict(symbol=symbol, session=fold['test_end'],
                    reason='WARMUP_OR_FEATURE_READINESS', consumed=False, selected=False, score=None,
                    pullback_start=None, available_at_assumed=availability_clock(pd.Timestamp(fold['test_end'])),
                    conditions=dict(signal=False, failures=['READINESS']))
                continue
            frame = strategy.prepare(raw, split, dict(fold, test_start=featured.index[first].strftime('%Y-%m-%d')))
        observations += len(frame)
        row = frame.iloc[-1]
        latest[symbol] = dict(symbol=symbol, session=fold['test_end'],
            reason='NO_QUALIFYING_SETUP' if bool(row['entry_ready']) else 'WARMUP_OR_FEATURE_READINESS',
            consumed=False, selected=False, score=float(row['score']) if pd.notna(row['score']) else None,
            pullback_start=row['entry_details'].get('pullback_start'),
            available_at_assumed=availability_clock(pd.Timestamp(fold['test_end'])), conditions=row['entry_details'])
        finite = np.flatnonzero(np.isfinite(frame['score'].to_numpy(dtype=float)))
        if not len(finite):
            continue
        # The existing kernel exits after max_hold_days (calendar days), then
        # fills at the next supplied session. A larger session-count margin
        # preserves every possible position/exit row plus the previous MACD row.
        keep = np.zeros(len(frame), dtype=bool)
        for i in finite:
            keep[max(0, i-1):min(len(frame), i+spec['portfolio']['max_hold_days']+6)] = True
        data[symbol] = frame.iloc[np.flatnonzero(keep)].copy()
        for category, event in dataset.member(symbol)['actions']:
            if category == 'cash_dividends' and fold['warmup_start'] <= event['ex_date'] <= fold['test_end']:
                dividends.append(event)
    return data, latest, dividends, observations


def merge_latest(result, missing, strategy, episode_id):
    from scripts.strategy_scanner import signal_id
    actual = {r['symbol']: r for r in result['latest_observations']}
    for symbol, row in missing.items():
        if symbol not in actual:
            row.update(strategy_id=strategy.id, strategy_version=strategy.version,
                signal_id=signal_id(strategy, episode_id, symbol, row['session']))
            actual[symbol] = row
    result['latest_observations'] = [actual[s] for s in sorted(actual)]
    result['observation_storage'] = 'SPARSE_PORTFOLIO_ROWS_PLUS_COMPLETE_CURRENT_DECISIONS'
    return result
