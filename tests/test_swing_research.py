"""Research accounting, neutral cohorts, frozen trials and causal exit experiments."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import swing_research as lab
from scripts.research_dataset import canonical_hash,digest_bytes
from scripts.swing_scorecard import closed_signals,statistics,ranking_scorecard,cluster_interval
from scripts.swing_research_portfolio import StudyPortfolio,frames_for_book,EXIT_POLICIES
from scripts import swing_execution_research as execution

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**kw):raise AssertionError('No research network')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def frame(count=12):
    from test_research_testing import simple_frame
    result=simple_frame(count)
    result['research_proposal']=pd.Series([None]*count,index=result.index,dtype=object)
    return result


def proposal(sid=lab.CHALLENGER,symbol='AAA',quality=8,session='2024-01-02'):
    from test_setup_selection import proposal as make
    p=make(sid,symbol,quality)
    p.update(signal_session=session,next_session='2024-01-03',structural_stop=90.)
    p['setup_ranking']['signal_session']=session
    return p


def portfolio(data,policy='existing',**kw):
    options=dict(initial_cash=10000.,alloc_pct=.1,max_positions=4,top_n=3,trail_pct=None,
        atr_multiple=0,max_hold_days=7,enable_macd_exit=False,enable_partial_exit=False,
        enable_candlestick_exit=False,enable_ema_exit=False,enable_rsi_divergence=False,
        enable_trailing_exit=False)
    options.update(kw)
    return StudyPortfolio(data,exit_policy=policy,**options)


def test_partial_exits_only_become_one_closed_signal_and_costs_once():
    common=dict(symbol='AAA',signal_id='a',strategy_id='test',signal_session='2024-01-02',fee=1.,slippage=0.)
    buy=dict(common,side='BUY',quantity=10,raw_price=100.,session='2024-01-03')
    first=dict(common,side='SELL',quantity=4,raw_price=110.,net_pnl=38.6,session='2024-01-04')
    second=dict(common,side='SELL',quantity=6,raw_price=90.,net_pnl=-61.6,session='2024-01-05')
    assert statistics(closed_signals([buy,first]))['closed_signals']==0
    result=statistics(closed_signals([buy,first,second]))
    assert result['closed_signals']==1 and result['net_pnl']==pytest.approx(-23)
    assert result['win_rate_pct']==0 and result['costs_closed_signals']==3
    bad=deepcopy(second);bad['quantity']=7
    with pytest.raises(ValueError,match='exceeds'):closed_signals([buy,first,bad])
    bad=deepcopy(second);bad['net_pnl']=0
    with pytest.raises(ValueError,match='reconcile'):closed_signals([buy,first,bad])
    assert statistics([])['profit_factor'] is None
    assert statistics([])['expectancy_net_per_signal'] is None


@pytest.mark.parametrize('policy',list(EXIT_POLICIES))
def test_policy_entries_next_open_cash_reconciliation_and_no_terminal_liquidation(policy):
    data=frame();data.at[data.index[0],'research_proposal']=proposal()
    engine=portfolio({'AAA':data},policy,trade_cost=1,slippage=.001)
    engine.run()
    assert engine.fills[0]['session']=='2024-01-03'
    assert all(r['equity']==pytest.approx(r['cash']+r['marked_positions']+r['dividend_receivable']) for r in engine.reconciliations)
    assert engine.fills[-1]['side']=='SELL'
    from scripts.swing_scorecard import portfolio_scorecard
    result=portfolio_scorecard(engine)
    assert result['closed_signals']==1 and result['expectancy_net_per_signal']<0


@pytest.mark.parametrize('kind,expected,reason',[
    ('target',120.,'RESEARCH_TARGET'),('both',90.,'RESEARCH_POLICY_STOP'),('gap',80.,'RESEARCH_POLICY_STOP')])
def test_fixed_r_stop_precedes_target_and_gaps_fill_at_open(kind,expected,reason):
    data=frame(4);data.at[data.index[0],'research_proposal']=proposal()
    if kind=='gap':data.loc[data.index[2],['open','high','low','close']]=[80.,130.,79.,100.]
    else:data.loc[data.index[2],['high','low']]=[130.,89. if kind=='both' else 99.]
    engine=portfolio({'AAA':data},'fixed_2r');engine.run()
    assert engine.fills[-1]['raw_price']==expected and engine.fills[-1]['reason']==reason


def test_completed_close_trail_only_applies_to_following_session():
    data=frame(4);data.at[data.index[0],'research_proposal']=proposal()
    data.loc[data.index[1],['open','high','low','close']]=[100.,120.,99.,115.]
    data.loc[data.index[2],['open','high','low','close']]=[114.,116.,110.,112.]
    engine=portfolio({'AAA':data},'close_trailing');engine.run()
    assert engine.fills[-1]['session']=='2024-01-04'
    assert engine.fills[-1]['raw_price']==pytest.approx(115*.97)


@pytest.mark.parametrize('policy',list(EXIT_POLICIES))
def test_exit_experiments_cannot_expand_original_pattern_entry_eligibility(policy):
    data=frame(3);p=proposal(lab.PATTERNS[0])
    p['pattern_exit']=dict(invalidation=90.,target=110.)
    data.at[data.index[0],'research_proposal']=p
    data.loc[data.index[1],['open','high','low','close']]=[120.,121.,119.,120.]
    engine=portfolio({'AAA':data},policy);engine.run()
    assert engine.fills==[] and engine.attempts[0]['status']=='OPEN_OUTSIDE_PATTERN_LEVELS'


def test_actual_six_strategy_book_uses_complete_cohort_and_shared_ranking(tmp_path):
    from test_research_testing import make_capture
    from test_swing_patterns import pattern_frame
    from scripts.swing_patterns import STRATEGIES
    from scripts.research_dataset import load_dataset
    spec,folder,bind,_=make_capture(tmp_path)
    source=pattern_frame(STRATEGIES[0])
    for kind in ('raw','split'):
        path=folder/f'alpaca_{kind}_daily.csv';bars=pd.read_csv(path)
        for symbol in ('AAA','SPY'):
            for to,fr in (('o','open'),('h','high'),('l','low'),('c','close'),('v','volume')):
                bars.loc[bars.symbol==symbol,to]=source[fr].to_numpy()
        bars.to_csv(path,index=False)
    bind()
    prepared,book=lab.build_book(load_dataset(spec),spec,spec['folds'][0],lambda:None)
    assert len(prepared)==6
    assert any(r['strategy_id']==STRATEGIES[0] for r in book)
    assert all('setup_ranking' in r['proposal'] and r['proposal']['structural_stop']>0 for r in book)
    assert sum(r['is_exit_owner'] and r['selected'] for r in book)>=1


def synthetic_book(monkeypatch,count=5):
    dates=frame().index
    class Strategy:
        version='1'
        def __init__(self,sid):self.id=sid
        def describe(self):return dict(definition_sha256='a'*64)
        def prepare(self,raw,split,fold):return raw
    monkeypatch.setattr(lab,'select_strategies',lambda ids:[Strategy(lab.CHALLENGER)])
    raw={}
    for i in range(count):
        data=frame();data['setup_score']=i;data['setup_volume_ratio']=2.
        data['entry_details']=[dict(pullback_start='2024-01-01') for _ in dates]
        raw['S'+str(i)]=data
    dataset=SimpleNamespace(raw=raw,features=raw,calendar=pd.DataFrame(index=dates.strftime('%Y-%m-%d')),
        dividends=(),provenance=dict(dataset_sha256='a'*64,limitations=['Synthetic test data']))
    spec=dict(universe=dict(symbols=list(raw)),portfolio=dict(initial_cash=10000.,alloc_pct=.05,max_positions=4,
        top_n=3,trail_pct=.03,atr_multiple=1.,max_hold_days=7))
    return dataset,spec


def test_ranked_cohort_retains_rejects_without_fourth_rank_backfill(monkeypatch):
    dataset,spec=synthetic_book(monkeypatch)
    prepared,book=lab.build_book(dataset,spec,dict(id='f',test_start='2024-01-02'),lambda:None)
    assert len(book)==5 and sum(r['selected'] for r in book)==3
    assert {r['symbol'] for r in book if r['selected']}=={'S2','S3','S4'}
    data=frames_for_book(prepared,book)
    assert set(data)=={'S2','S3','S4'}
    engine=portfolio(data,max_positions=1);engine.run()
    assert engine.fills[0]['symbol']=='S4'
    rejected=next(r for r in book if not r['selected'])
    counter=frames_for_book(prepared,book,only_signal=rejected['signal_id'])
    assert set(counter)=={rejected['symbol']}
    with pytest.raises(ValueError,match='budget'):lab.build_book(dataset,spec,dict(id='f',test_start='2024-01-02'),lambda:None,max_signals=4)


def test_ranking_scorecard_deduplicates_pattern_matches_and_reports_paired_returns():
    rows=[];outcomes={}
    for sid,symbol,selected,owner,pnl in [('a','AAA',True,True,10),('b','AAA',True,False,10),('c','CCC',False,True,-5)]:
        rows.append(dict(signal_id=sid,strategy_id=sid,symbol=symbol,signal_session='2024-01-02',selected=selected,is_exit_owner=owner))
        outcomes[sid]=dict(signal_id=sid,signal_session='2024-01-02',state='CLOSED',net_pnl=pnl,costs=0,
            holding_calendar_days=2,net_return_pct=pnl)
    result=ranking_scorecard(rows,outcomes)
    assert result['selected']['closed_signals']==1
    assert result['mean_paired_return_difference_pct']==15
    assert result['basis'].startswith('INDEPENDENT')


def test_week_clusters_do_not_count_same_day_trades_as_independent():
    rows=[dict(signal_session='2024-01-02',net_pnl=i) for i in range(100)]
    assert cluster_interval(rows)['low'] is None
    assert cluster_interval(rows)['signal_weeks']==1


@pytest.mark.parametrize('sid',[*lab.PATTERNS])
def test_existing_policy_reuses_pattern_adapter_fill_behavior(sid):
    from test_swing_patterns import pattern_frame
    from test_bull_flag_companion import flag_frame
    from scripts.bull_flag_strategy import BULL_FLAG,research_portfolio_class as bull_class
    from scripts.pattern_strategy import research_portfolio_class as pattern_class
    from scripts.strategy_library import get_strategy
    raw=flag_frame() if sid==BULL_FLAG else pattern_frame(sid)
    original_end=raw.index[-1]
    nextbar=raw.tail(1).copy();nextbar.index=pd.DatetimeIndex([original_end+pd.offsets.BDay()])
    price=float(raw.close.iloc[-1])+1
    nextbar[['open','high','low','close']]=[price,price+1,price-.5,price+.5]
    following=nextbar.copy();following.index=pd.DatetimeIndex([nextbar.index[-1]+pd.offsets.BDay()])
    following[['open','high','low','close']]=[price+1,price*2,price+.5,price*1.5]
    raw=pd.concat([raw,nextbar,following])
    fold=dict(warmup_start=str(raw.index[0].date()),test_start=str(raw.index[270].date()),test_end=str(raw.index[-1].date()))
    prepared=get_strategy(sid).prepare(raw,raw,fold)
    signal=prepared.loc[original_end]
    assert pd.notna(signal['score'])
    p=proposal(sid);p.update(signal_session=str(original_end.date()),pattern_exit=signal.entry_details['pattern_exit'],
        structural_stop=signal.entry_details['pattern_exit']['invalidation'])
    data=prepared.copy();data['score']=np.nan;data.at[original_end,'score']=1.
    data['research_proposal']=pd.Series([None]*len(data),index=data.index,dtype=object)
    data.at[original_end,'research_proposal']=p
    options=dict(initial_cash=10000,alloc_pct=.1,top_n=1,max_positions=1,trail_pct=.03,atr_multiple=1.,max_hold_days=7)
    actual=StudyPortfolio({'AAA':data},**options);actual.run()
    expected=(bull_class() if sid==BULL_FLAG else pattern_class())({'AAA':data},**options);expected.run()
    assert [{k:v for k,v in f.items() if k not in {'signal_id','strategy_id','initial_stop'}} for f in actual.fills]==expected.fills


def test_rank_comparison_does_not_hide_censored_rejected_outcomes():
    rows=[dict(signal_id=sid,strategy_id='x',signal_session='2024-01-02',selected=chosen,is_exit_owner=True)
          for sid,chosen in [('a',True),('b',False),('c',False)]]
    outcomes={sid:dict(state='CLOSED',net_pnl=pnl,costs=0,holding_calendar_days=1,signal_session='2024-01-02',net_return_pct=pnl)
              for sid,pnl in [('a',10),('b',-5)]}
    outcomes['c']=dict(state='OPEN',signal_session='2024-01-02',net_pnl=None)
    result=ranking_scorecard(rows,outcomes)
    assert result['paired_sessions']==0 and result['rejected']['open_signals']==1


def test_unchanged_patterns_and_no_new_runtime_import_or_production_config_change():
    import ast
    from pathlib import Path
    for name in ('swing_research.py','swing_scorecard.py','swing_research_portfolio.py','swing_execution_research.py'):
        ast.parse((lab.ROOT/'scripts'/name).read_text(encoding='utf-8'),feature_version=(3,10))
    assert lab.STRATEGIES==tuple(sorted((lab.CHALLENGER,*lab.PATTERNS)))


def plan_case(tmp_path):
    from test_research_testing import make_capture
    spec,folder,bind,_=make_capture(tmp_path)
    plan=lab.prepare_plan(spec,capture_manifest=spec['capture_manifest'])
    plan['data_kind']='SYNTHETIC_TEST_FIXTURE'
    plan['exit_policies']=['existing','fixed_2r']
    plan['budgets']['bootstrap_samples']=100
    return plan


def test_frozen_plan_refuses_mixed_axes_unsafe_paths_and_fake_holdout_promotion(tmp_path):
    plan=plan_case(tmp_path)
    assert lab.validate_plan(plan)==28
    for mutate in (lambda p:p.update(axis='entries'),lambda p:p['study']['folds'][0].update(id='../escape'),
                   lambda p:p['study'].update(strategies=[lab.CHALLENGER]),lambda p:p['budgets'].update(max_trials=1)):
        bad=deepcopy(plan);mutate(bad)
        with pytest.raises(ValueError):lab.validate_plan(bad)
    from scripts.swing_scorecard import portfolio_scorecard
    data=frame();data.at[data.index[0],'research_proposal']=proposal()
    engine=portfolio({'AAA':data});engine.run()
    evidence=lab.evidence_gate(portfolio_scorecard(engine),dict(role='HOLDOUT',exposure='PREVIOUSLY_EXPOSED'),plan['evidence_floors'])
    assert evidence['research_qualified'] is False and not evidence['checks']['declared_unseen_evaluation']


def test_full_runner_writes_all_trials_ledger_unknown_intraday_and_preserves_failed_attempts(tmp_path):
    plan=plan_case(tmp_path/'input')
    plan.update(axis='entries',exit_policies=['existing'])
    registry=tmp_path/'registry';output=tmp_path/'output'
    result=lab.run_lab(plan,output,registry)
    assert result['completed_trials']==14 and result['strategy_priority'] is False
    assert result['entry_comparisons'][0]['status']=='UNAVAILABLE_NO_CAPTURED_INTRADAY_QUOTES'
    assert result['prospective_confirmation']=='NOT_ESTABLISHED_BY_RETROSPECTIVE_STUDY'
    assert 'SYNTHETIC_TEST_FIXTURE' in (output/'report.html').read_text(encoding='utf-8')
    manifest=json.loads((output/'manifest.json').read_bytes())
    for pin in manifest['bindings']:
        body=(output/pin['path']).read_bytes()
        assert len(body)==pin['bytes'] and digest_bytes(body)==pin['sha256']
    assert lab.registry_summary(registry)['attempted_experiments']==1
    with pytest.raises(FileExistsError):lab.run_lab(plan,tmp_path/'replay',registry)
    stopped=deepcopy(plan);stopped['budgets']['max_seconds']=1
    stop=tmp_path/'STOP';stop.touch()
    with pytest.raises(RuntimeError):lab.run_lab(stopped,tmp_path/'stopped',registry,stop_file=stop)
    history=lab.registry_summary(registry)
    assert history['attempted_experiments']==2 and {r['status'] for r in history['experiments']}=={'FAILED','COMPLETED'}
    assert (tmp_path/'stopped/failure.json').exists()
