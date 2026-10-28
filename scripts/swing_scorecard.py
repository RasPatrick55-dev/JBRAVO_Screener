"""Closed-signal research statistics; never counts partial exits as winning trades."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd


def require(value, message):
    if not value:
        raise ValueError(message)


def closed_signals(fills, frames=None):
    """Reconcile complete quantities first; retain open/partially exited episodes."""
    groups = defaultdict(list)
    for fill in fills:
        require(fill['side'] in {'BUY', 'SELL'}, 'Unknown fill side')
        require(type(fill['quantity']) is int and fill['quantity'] > 0, 'Invalid fill quantity')
        require(all(math.isfinite(float(fill[k])) for k in ('raw_price', 'fee', 'slippage')) and
                fill['raw_price'] > 0 and fill['fee'] >= 0 and fill['slippage'] >= -1e-8,
                'Invalid fill price/cost')
        groups[fill['signal_id']].append(fill)
    result = []
    for sid, rows in groups.items():
        buys = [r for r in rows if r['side'] == 'BUY']
        sells = [r for r in rows if r['side'] == 'SELL']
        require(len(buys) == 1 and rows[0]['side'] == 'BUY', 'One originating entry required')
        entry = buys[0]
        require(all(r['symbol'] == entry['symbol'] and r['strategy_id'] == entry['strategy_id'] and
                    r['session'] >= entry['session'] for r in rows), 'Mixed fill lineage')
        sold = sum(r['quantity'] for r in sells)
        require(sold <= entry['quantity'], 'Exit quantity exceeds entry')
        net = sum(float(r['net_pnl']) for r in sells)
        require(math.isfinite(net), 'Nonfinite net P&L')
        # SELL net_pnl already allocates entry costs. Never subtract them twice.
        gross = sum((r['raw_price'] - entry['raw_price']) * r['quantity'] for r in sells)
        costs = sum(r['fee'] + r['slippage'] for r in rows)
        closed = sold == entry['quantity']
        if closed:
            require(math.isclose(gross - costs, net, rel_tol=1e-9, abs_tol=1e-6),
                    'Closed signal costs do not reconcile')
        last = max(r['session'] for r in rows)
        item = dict(signal_id=sid, strategy_id=entry['strategy_id'], symbol=entry['symbol'],
            signal_session=entry['signal_session'], entry_session=entry['session'], exit_session=last if closed else None,
            quantity=entry['quantity'], quantity_sold=sold, state='CLOSED' if closed else 'OPEN',
            net_pnl=net if closed else None, realized_pnl_to_date=net, costs=costs,
            holding_calendar_days=(date.fromisoformat(last) - date.fromisoformat(entry['session'])).days if closed else None,
            mfe_observed_lower_bound_pct=None, mae_observed_lower_bound_pct=None)
        stop=entry.get('initial_stop')
        risk=entry['quantity']*(entry['raw_price']-stop) if stop is not None and 0<stop<entry['raw_price'] else None
        item.update(initial_risk_cash=risk,net_r_multiple=net/risk if closed and risk else None)
        if closed and frames is not None:
            frame = frames[entry['symbol']]
            inside = frame.loc[(frame.index > entry['session']) & (frame.index < last)]
            known = [entry['raw_price'], *[r['raw_price'] for r in sells]]
            high = max(known + inside.high.tolist())
            low = min(known + inside.low.tolist())
            item.update(mfe_observed_lower_bound_pct=100 * (high / entry['raw_price'] - 1),
                        mae_observed_lower_bound_pct=100 * (1 - low / entry['raw_price']))
        result.append(item)
    return result


def cluster_interval(trades, *, samples=400, seed=271828):
    """Resample signal weeks, not individual correlated same-session trades."""
    groups = defaultdict(list)
    for trade in trades:
        day = date.fromisoformat(trade['signal_session'])
        groups[(day - timedelta(days=day.weekday())).isoformat()].append(trade['net_pnl'])
    if len(groups) < 2:
        return dict(low=None, high=None, signal_weeks=len(groups), method='WEEK_CLUSTER_BOOTSTRAP', samples=0)
    totals = np.array([[sum(v), len(v)] for v in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = totals[rng.integers(0, len(totals), len(totals))].sum(axis=0)
        estimates.append(chosen[0] / chosen[1])
    low, high = np.quantile(estimates, [.025, .975])
    return dict(low=float(low), high=float(high), signal_weeks=len(groups),
                method='WEEK_CLUSTER_BOOTSTRAP', samples=samples)


def statistics(episodes, *, bootstrap_samples=400):
    trades = [t for t in episodes if t['state'] == 'CLOSED']
    net = [t['net_pnl'] for t in trades]
    wins = [v for v in net if v > 0]
    losses = [v for v in net if v < 0]
    count = len(trades)
    interval = cluster_interval(trades, samples=bootstrap_samples)
    return dict(closed_signals=count, open_signals=sum(t['state'] == 'OPEN' for t in episodes),
        unfilled_signals=sum(t['state'] == 'UNFILLED' for t in episodes),
        wins=len(wins), losses=len(losses), breakeven=count-len(wins)-len(losses),
        win_rate_pct=100 * len(wins) / count if count else None,
        net_pnl=sum(net), expectancy_net_per_signal=sum(net) / count if count else None,
        average_win=sum(wins) / len(wins) if wins else None,
        average_loss=sum(losses) / len(losses) if losses else None,
        payoff_ratio=(sum(wins)/len(wins))/abs(sum(losses)/len(losses)) if wins and losses else None,
        profit_factor=sum(wins)/abs(sum(losses)) if losses else None,
        profit_factor_status='DEFINED' if losses else 'UNDEFINED_NO_LOSSES' if count else 'NO_CLOSED_SIGNALS',
        costs_closed_signals=sum(t['costs'] for t in trades),
        expectancy_net_r=float(np.mean([t['net_r_multiple'] for t in trades])) if trades and all(t.get('net_r_multiple') is not None for t in trades) else None,
        average_holding_calendar_days=sum(t['holding_calendar_days'] for t in trades)/count if count else None,
        expectancy_interval_95=interval,
        sample_status='NO_CLOSED_SIGNALS' if not count else 'OBSERVED_SAMPLE_NOT_QUALIFICATION')


def portfolio_scorecard(engine, *, bootstrap_samples=400):
    episodes = closed_signals(engine.fills, engine.data)
    values = np.array([engine.initial_cash] + [v for _, v in engine.equity_curve])
    require(len(values) > 1 and np.isfinite(values).all() and (values > 0).all(), 'Invalid NAV series')
    exposure = [r['marked_positions']/r['equity'] for r in engine.reconciliations]
    return dict(**statistics(episodes, bootstrap_samples=bootstrap_samples),
        initial_cash=engine.initial_cash, terminal_equity=float(values[-1]),
        total_return_pct=100 * (values[-1]/values[0]-1),
        max_drawdown_pct=100 * float(np.min(values/np.maximum.accumulate(values)-1)),
        average_close_exposure_pct=100 * float(np.mean(exposure)),
        entry_attempts=len(engine.attempts), modeled_entry_fills=sum(r['side']=='BUY' for r in engine.fills),
        inapplicable_exit_policy_attempts=sum(r['status'] in {'OPEN_OUTSIDE_STOP','OPEN_OUTSIDE_TARGET'} for r in engine.attempts),
        fill_rate_pct=100 * sum(r['side']=='BUY' for r in engine.fills)/len(engine.attempts) if engine.attempts else None,
        gross_turnover_initial_cash=sum(f['quantity']*f['raw_price'] for f in engine.fills)/engine.initial_cash,
        dividend_income=engine.dividend_income, terminal_open_positions=len(engine.positions),
        terminal_canceled_entries=engine.terminal_entries,
        per_strategy={sid: statistics([t for t in episodes if t['strategy_id']==sid], bootstrap_samples=bootstrap_samples)
                      for sid in sorted({r['strategy_id'] for r in engine.fills})},
        episodes=episodes, metric_basis='DAILY_OHLC_RESEARCH_PORTFOLIO', research_qualified=False)


def ranking_scorecard(observations, outcomes, *, bootstrap_samples=400):
    """Standalone counterfactual trades are NEVER added into portfolio NAV/P&L."""
    rows = [dict(row, outcome=outcomes[row['signal_id']]) for row in observations]
    selected = [r['outcome'] for r in rows if r['selected'] and r['is_exit_owner']]
    rejected = [r['outcome'] for r in rows if not r['selected'] and r['is_exit_owner']]
    first, other = statistics(selected, bootstrap_samples=bootstrap_samples), statistics(rejected, bootstrap_samples=bootstrap_samples)
    pairs = []
    for day in sorted({r['signal_session'] for r in rows}):
        day_rows = [r for r in rows if r['signal_session']==day and r['is_exit_owner']]
        if any(r['outcome']['state']!='CLOSED' for r in day_rows):
            continue
        # Stock outcomes are counted once, using the same exit owner as rank_setups.
        chosen = [r['outcome']['net_return_pct'] for r in day_rows if r['selected'] and r['is_exit_owner']]
        rest = [r['outcome']['net_return_pct'] for r in day_rows if not r['selected'] and r['is_exit_owner']]
        if chosen and rest:
            pairs.append(dict(signal_session=day, selected_mean_return_pct=float(np.mean(chosen)),
                              rejected_mean_return_pct=float(np.mean(rest)), difference_pct=float(np.mean(chosen)-np.mean(rest))))
    return dict(selected=first, rejected=other, paired_session_comparisons=pairs,
        paired_sessions=len(pairs), mean_paired_return_difference_pct=float(np.mean([p['difference_pct'] for p in pairs])) if pairs else None,
        paired_sessions_require_all_owner_outcomes_closed=True,
        basis='INDEPENDENT_FIXED_CASH_COUNTERFACTUALS_NOT_A_TRADABLE_PORTFOLIO',
        stock_level_groups_deduplicated=True,
        per_strategy={sid:statistics([r['outcome'] for r in rows if r['strategy_id']==sid],bootstrap_samples=bootstrap_samples)
                      for sid in sorted({r['strategy_id'] for r in rows})},research_qualified=False)


def benchmark_comparison(engine, raw, dividends=()):
    """Price references with explicit exposure differences, not executable returns."""
    days=pd.DatetimeIndex([d for d,_ in engine.equity_curve])
    bars=raw.reindex(days)
    require(not bars[['open','close']].isna().any().any(),'Missing benchmark session')
    returns=bars.close.pct_change()
    returns.iloc[0]=bars.close.iloc[0]/bars.open.iloc[0]-1
    for event in dividends:
        day=pd.Timestamp(event['ex_date'])
        if day in days and day!=days[0]:
            i=days.get_loc(day)
            returns.loc[day]+=float(event['rate'])/float(bars.close.iloc[i-1])
    # The preceding close's exposure is known before this session's return.
    weights=np.array([0.]+[r['marked_positions']/r['equity'] for r in engine.reconciliations[:-1]])
    passive=float(np.prod(1+returns.to_numpy())-1)*100
    matched=float(np.prod(1+weights*returns.to_numpy())-1)*100
    return dict(cash_return_pct=0.,full_exposure_spy_reference_return_pct=passive,
        lagged_exposure_spy_reference_return_pct=matched,
        portfolio_return_minus_lagged_spy_pct=100*(engine.equity_curve[-1][1]/engine.initial_cash-1)-matched,
        basis='DIAGNOSTIC_RAW_PRICE_AND_EX_DATE_DIVIDEND_REFERENCES',
        assumed_costs=0,assumed_cash_interest=0,tradable_portfolio_parity=False,
        note='Full SPY exposure differs from the strategy. Lagged exposure is a diagnostic, not matched beta/risk or achievable execution.')
