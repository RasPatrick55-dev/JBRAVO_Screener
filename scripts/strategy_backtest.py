"""Candidate backtest preview over the unchanged versioned strategy kernel.

Pure computation: no credentials, providers, database, deployment or orders.
Replay the declared historical cohort before attributing closed signals to the
current candidates. Retrospectively trading only today's winners would change
portfolio competition, sizing and strategy state.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import time


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'invalid_backtest_number')
    return float(value)


def closed_signal(trace, *, as_of):
    """One completed position is one trade, including all its partial exits."""
    require(isinstance(trace['strategy_version'],str) and trace['strategy_version'], 'missing_backtest_strategy_version')
    fills=trace['fills'];bought=sold=0;buys=[];sells=[];previous=trace['signal_session']
    for fill in fills:
        require(fill['signal_id']==trace['signal_id'] and fill['symbol']==trace['symbol'] and
            fill['strategy_id']==trace['strategy_id'] and fill['strategy_version']==trace['strategy_version'],
            'backtest_fill_lineage_mismatch')
        require(type(fill['quantity']) is int and fill['quantity']>0,'invalid_backtest_quantity')
        require(date.fromisoformat(fill['session']).isoformat()==fill['session'] and
            trace['signal_session']<fill['session']<=as_of and previous<=fill['session'],
            'backtest_future_or_same_signal_day_fill')
        previous=fill['session']
        require(number(fill['raw_price'])>0 and number(fill['fee'])>=0 and number(fill['slippage'])>=0,
                'invalid_backtest_price_or_cost')
        number(fill['cash_delta'])
        if fill['side']=='BUY':
            require(not buys and not sells,'unsupported_backtest_multiple_entries')
            bought+=fill['quantity'];buys.append(fill)
        else:
            require(fill['side']=='SELL' and buys and fill['session']>=buys[0]['session'],
                    'backtest_exit_without_entry')
            number(fill['net_pnl']);sold+=fill['quantity'];sells.append(fill)
            require(sold<=bought,'backtest_oversold_position')
    require(trace['bought_quantity']==bought and trace['sold_quantity']==sold,'backtest_lifecycle_quantity_mismatch')
    expected='CLOSED' if bought and bought==sold else 'OPEN' if bought else 'UNFILLED_PROPOSAL'
    require(trace['status']==expected,'backtest_lifecycle_state_mismatch')
    if expected!='CLOSED':
        return None
    net=math.fsum(number(f['net_pnl']) for f in sells)
    cash_net=math.fsum(number(f['cash_delta']) for f in fills)
    require(math.isclose(net,cash_net,rel_tol=1e-12,abs_tol=1e-7),'backtest_closed_cash_reconciliation_failed')
    basis=-number(buys[0]['cash_delta'])
    require(basis>0,'invalid_backtest_entry_basis')
    return dict(signal_id=trace['signal_id'],symbol=trace['symbol'],
        signal_session=trace['signal_session'],entry_session=buys[0]['session'],exit_session=sells[-1]['session'],
        quantity=bought,exit_fills=len(sells),partial_exits=max(0,len(sells)-1),
        entry_price_raw=number(buys[0]['raw_price']),net_pnl=net,
        trade_return_pct=net/basis*100,fees=math.fsum(number(f['fee']) for f in fills),
        slippage=math.fsum(number(f['slippage']) for f in fills),
        hold_calendar_days=(date.fromisoformat(sells[-1]['session'])-date.fromisoformat(buys[0]['session'])).days)


def symbol_metrics(proposal, replay, *, as_of):
    symbol=proposal['symbol']
    traces=[t for t in replay['lifecycle'] if t['symbol']==symbol]
    completed=[];open_prior=unfilled_prior=current_unfilled=0
    seen=set()
    for trace in traces:
        require(trace['strategy_id']==proposal['strategy_id'] and
            trace['strategy_version']==proposal['strategy_version'],'mixed_backtest_strategy_history')
        require(trace['signal_id'] not in seen and trace['signal_session']<=as_of,'duplicate_or_future_backtest_signal')
        seen.add(trace['signal_id'])
        closed=closed_signal(trace,as_of=as_of)
        if trace['signal_id']==proposal['signal_id']:
            require(trace['signal_session']==as_of and trace['status']=='UNFILLED_PROPOSAL',
                    'current_proposal_has_invented_historical_outcome')
            current_unfilled+=1
        elif trace['signal_session']<as_of:
            if closed is not None:completed.append(closed)
            elif trace['status']=='OPEN':open_prior+=1
            else:unfilled_prior+=1
    require(current_unfilled==1,'current_signal_missing_from_backtest_replay')
    completed.sort(key=lambda t:(t['exit_session'],t['signal_id']))
    nets=[t['net_pnl'] for t in completed];count=len(nets)
    gains=math.fsum(v for v in nets if v>0);losses=-math.fsum(v for v in nets if v<0)
    wins=sum(v>0 for v in nets);lost=sum(v<0 for v in nets)
    cumulative=peak=0.0;drawdown=0.0
    for net in nets:
        cumulative+=net;peak=max(peak,cumulative);drawdown=min(drawdown,cumulative-peak)
    observations=[o for o in replay['observations'] if o['symbol']==symbol and o['session']<as_of]
    return dict(symbol=symbol,strategy_id=proposal['strategy_id'],strategy_version=proposal['strategy_version'],
        current_signal_id=proposal['signal_id'],current_signal_session=as_of,
        prior_qualifying_setups=sum(bool(o['consumed']) for o in observations),
        closed_signals=count,winning_signals=wins,losing_signals=lost,breakeven_signals=count-wins-lost,
        prior_open_signals=open_prior,prior_unfilled_proposals=unfilled_prior,
        current_proposal_state='UNFILLED_FUTURE_OUTCOME_UNKNOWN',
        sample_status='HAS_COMPLETED_SIGNALS' if count else 'NO_COMPLETED_SIGNALS',
        win_rate_pct=wins/count*100 if count else None,
        closed_trade_net_pnl=math.fsum(nets),expectancy_net_per_closed_signal=math.fsum(nets)/count if count else None,
        profit_factor=gains/losses if losses else None,
        profit_factor_status='DEFINED' if losses else 'NO_CLOSED_SIGNALS' if not count else 'UNDEFINED_NO_LOSSES',
        max_closed_signal_drawdown_cash=drawdown,
        average_hold_calendar_days=math.fsum(t['hold_calendar_days'] for t in completed)/count if count else None,
        fees_closed_signals=math.fsum(t['fees'] for t in completed),
        slippage_closed_signals=math.fsum(t['slippage'] for t in completed),
        trade_pnl_excludes_cash_dividends=True,completed_signal_ids=[t['signal_id'] for t in completed],
        closed_signal_ledger_sha256=digest(completed),
        disposition='REPORT_ONLY_NO_PERFORMANCE_FILTER')


def report_from_replay(batch, replay, *, primary_run_id, dataset, specification):
    """Build reporting inputs for Metrics; no performance acceptance thresholds."""
    proposals=batch['proposals'];as_of=batch['session']['signal_session']
    require(isinstance(primary_run_id,str) and re.fullmatch('[0-9a-f]{32}',primary_run_id),
            'invalid_backtest_primary_identity')
    require(batch['schema']=='jbravo.strategy-primary-batch.v1' and
        type(batch['candidate_count']) is int and batch['candidate_count']==len(proposals) and
        batch['order_submission_authorized'] is False,'invalid_backtest_candidate_batch')
    require(len({p['symbol'] for p in proposals})==len(proposals) and
        len({p['signal_id'] for p in proposals})==len(proposals),'duplicate_backtest_candidate')
    for proposal in proposals:
        require(proposal['symbol'] in specification['universe']['symbols'] and
            proposal['signal_session']==as_of and proposal['side']=='BUY' and
            proposal['paper_only'] is True and proposal['order_submission_authorized'] is False,
            'backtest_candidate_outside_declared_scope')
    require(dataset.provenance['feed']=='sip' and dataset.provenance['currency']=='USD' and
        dataset.provenance['execution_prices']=='raw' and dataset.provenance['feature_prices']=='split',
        'backtest_data_convention_mismatch')
    result=dict(schema='jbravo.candidate-strategy-backtest.v1',mode='REPORT_ONLY',
        primary_run_id=primary_run_id,signal_session=as_of,screener_batch_sha256=digest(batch),
        archive_id=batch['archive_id'],archive_sha256=batch['archive_sha256'],scan_run_id=batch['scan_run_id'],
        input_candidates=len(proposals),output_candidates=len(proposals),
        candidate_signal_ids=[p['signal_id'] for p in proposals],
        strategy_contract=batch['strategy_contract'],
        parameters=dict(universe=specification['universe'],portfolio=specification['portfolio'],
            cost_cases=specification['cost_cases'],folds=specification['folds'],effective_test_end=as_of),
        data=dataset.provenance,results=[],replay_performed=bool(proposals),
        status='COMPLETED_CANDIDATE_BACKTEST' if proposals else 'SKIPPED_NO_CANDIDATES',
        research_qualified=False,order_submission_authorized=False,broker_operations=0,
        limitations=['Retrospectively selected fixed cohort; no untouched validation or promotion claim',
            'Per-symbol results attribute the shared historical cohort portfolio, not an independent single-stock account',
            'Current buy proposals have no known future fills or outcomes',
            'Closed-signal statistics group partial exits; cash dividends are separate from trade P&L',
            'No minimum-sample or performance rejection rules have been approved for this stage'])
    if proposals:
        require(replay is not None and replay['as_of']==as_of and
            canonical(replay['top_candidates'])==canonical(proposals),'backtest_replay_changed_current_signals')
        result['strategy']=replay['strategy']
        result['results']=[symbol_metrics(p,replay,as_of=as_of) for p in proposals]
        result['cohort_portfolio_metrics']=replay['metrics']
    result['result_id']=digest(result)
    require(len(canonical(result))<=24*1024,'candidate_backtest_report_oversize')
    return result


def run_candidate_backtest(batch, *, primary_run_id, dataset, specification):
    """Run the exact Screener kernel across its original cohort, capped by study budget."""
    from scripts.research_testing import validate_spec, json_default
    from scripts.strategy_library import get_strategy
    from scripts.strategy_scanner import evaluate_strategy
    validate_spec(specification)
    as_of=batch['session']['signal_session']
    folds=[f for f in specification['folds'] if f['test_start']<=as_of<=f['test_end']]
    require(len(folds)==1 and len(specification['strategies'])==1 and len(specification['cost_cases'])==1,
            'backtest_requires_one_explicit_strategy_episode_and_cost_case')
    contract=batch.get('strategy_contract')
    require(isinstance(contract,dict) and set(contract)=={'strategy_id','strategy_version','definition_sha256'},
            'missing_backtest_strategy_contract')
    strategy=get_strategy(contract['strategy_id'])
    definition=strategy.describe()
    require(contract['strategy_version']==strategy.version and contract['definition_sha256']==definition['definition_sha256'],
            'backtest_strategy_version_or_definition_mismatch')
    require(specification['strategies']==[strategy.research_method], 'backtest_specification_strategy_mismatch')
    require(batch['candidate_count']==len(batch['proposals']) and all(
        p['strategy_id']==strategy.id and p['strategy_version']==strategy.version for p in batch['proposals']),
        'backtest_candidate_strategy_mismatch')
    require(strategy.paper_scan_enabled,'backtest_strategy_not_enabled')
    own_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source=strategy.describe()
    replay=None
    if batch['candidate_count']:
        replay=evaluate_strategy(dataset,specification,strategy,folds[0],specification['cost_cases'][0],as_of,
            deadline=time.monotonic()+specification['max_seconds'])
        replay=json.loads(json.dumps(replay,default=json_default,allow_nan=False))
    result=report_from_replay(batch,replay,primary_run_id=primary_run_id,dataset=dataset,specification=specification)
    require(source==strategy.describe() and own_source==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'backtest_source_changed_during_replay')
    result['backtest_source_sha256']=own_source
    result['strategy_definition']=source
    result.pop('result_id')
    result['result_id']=digest(result)
    return result
