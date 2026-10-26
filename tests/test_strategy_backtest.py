from copy import deepcopy
from datetime import datetime,timezone
from types import SimpleNamespace
import json
import socket
import pytest

from scripts import strategy_backtest as backtest
from scripts import strategy_pipeline as pipeline
from scripts import strategy_library, strategy_scanner
from test_strategy_pipeline import archive

pytestmark=pytest.mark.alpaca_optional

def inputs(positive=False):
    cfg,session,value=archive(positive=positive)
    batch=pipeline.selection_batch(value,session)
    spec=dict(schema='jbravo.strategy-study.v1',purpose='exploratory_research',capture_manifest='unused',
        actions_capture=dict(path='unused',sha256='f'*64),
        universe=dict(symbols=cfg['symbols'],benchmark=cfg['benchmark'],excluded={}),
        folds=[dict(id=cfg['episode_id'],warmup_start=cfg['warmup_start'],test_start=cfg['episode_start'],test_end=session['signal_session'])],
        strategies=[cfg['strategy_id']],cost_cases=[cfg['cost_case']],portfolio=cfg['portfolio'],max_runs=2,max_seconds=60,
        research_qualified=False,execution_authorized=False)
    data=SimpleNamespace(provenance=dict(feed='sip',currency='USD',execution_prices='raw',feature_prices='split',dataset_sha256='c'*64))
    return batch,spec,data

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('Backtest core cannot use a network')
    monkeypatch.setattr(socket.socket,'connect',fail)
    monkeypatch.setattr(socket,'create_connection',fail)

def test_empty_batch_still_resolves_exact_strategy_without_fake_simulation(monkeypatch):
    batch,spec,data=inputs()
    monkeypatch.setattr(strategy_scanner,'evaluate_strategy',lambda *a,**k:pytest.fail('empty batch cannot simulate'))
    result=backtest.run_candidate_backtest(batch,primary_run_id='1'*32,dataset=data,specification=spec)
    assert result['status']=='SKIPPED_NO_CANDIDATES' and result['results']==[]
    assert result['strategy_definition']['id']==batch['strategy_contract']['strategy_id']
    assert result['strategy_definition']['version']=='2' and not result['replay_performed']
    assert result['candidate_signal_ids']==[] and result['broker_operations']==0
    assert json.loads(json.dumps(result))==result

@pytest.mark.parametrize('change',['missing','unknown','version','definition','spec','candidate'])
def test_strategy_dispatch_never_defaults_or_silently_substitutes(monkeypatch,change):
    batch,spec,data=inputs(True)
    if change=='missing':batch.pop('strategy_contract')
    if change=='unknown':batch['strategy_contract']['strategy_id']='unknown'
    if change=='version':batch['strategy_contract']['strategy_version']='1'
    if change=='definition':batch['strategy_contract']['definition_sha256']='0'*64
    if change=='spec':spec['strategies']=['legacy_simulator_core_v1']
    if change=='candidate':batch['proposals'][0]['strategy_id']='legacy_simulator_core_v1'
    monkeypatch.setattr(strategy_scanner,'evaluate_strategy',lambda *a,**k:pytest.fail('identity rejection must precede replay'))
    with pytest.raises(ValueError):backtest.run_candidate_backtest(batch,primary_run_id='1'*32,dataset=data,specification=spec)

def trace(partial=True):
    identity=dict(signal_id='b'*64,symbol='SRRK',strategy_id='trend_pullback_reclaim_qualifying_close_v2',strategy_version='2')
    fills=[dict(identity,side='BUY',session='2026-01-05',quantity=10,raw_price=10.,fee=1.,slippage=0.,cash_delta=-101.)]
    if partial:
        fills.extend([dict(identity,side='SELL',session='2026-01-06',quantity=5,raw_price=12.,fee=1.,slippage=0.,cash_delta=59.,net_pnl=8.5),
            dict(identity,side='SELL',session='2026-01-07',quantity=5,raw_price=8.,fee=1.,slippage=0.,cash_delta=39.,net_pnl=-11.5)])
    else:
        fills.append(dict(identity,side='SELL',session='2026-01-07',quantity=10,raw_price=11.,fee=1.,slippage=0.,cash_delta=109.,net_pnl=8.))
    return dict(identity,signal_session='2026-01-02',status='CLOSED',bought_quantity=10,sold_quantity=10,fills=fills)

def test_partial_exits_are_one_closed_signal_with_all_entry_costs():
    actual=backtest.closed_signal(trace(),as_of='2026-01-08')
    assert actual['net_pnl']==-3 and actual['exit_fills']==2 and actual['partial_exits']==1
    assert actual['fees']==3 and actual['hold_calendar_days']==2

@pytest.mark.parametrize('change',['future','same_day','lineage','cash','oversold','state'])
def test_bad_fill_sequences_and_invented_results_fail(change):
    value=trace()
    if change=='future':value['fills'][-1]['session']='2026-01-09'
    if change=='same_day':value['fills'][0]['session']='2026-01-02'
    if change=='lineage':value['fills'][1]['strategy_id']='wrong'
    if change=='cash':value['fills'][-1]['net_pnl']=999.
    if change=='oversold':value['fills'][-1]['quantity']=6
    if change=='state':value['status']='OPEN'
    with pytest.raises(ValueError):backtest.closed_signal(value,as_of='2026-01-08')

def test_current_signal_is_unknown_and_no_history_is_not_a_zero_win_rate():
    batch,spec,data=inputs(True);proposal=batch['proposals'][0]
    current=dict(signal_id=proposal['signal_id'],symbol=proposal['symbol'],strategy_id=proposal['strategy_id'],
        strategy_version='2',signal_session=proposal['signal_session'],status='UNFILLED_PROPOSAL',
        bought_quantity=0,sold_quantity=0,fills=[])
    replay=dict(as_of=proposal['signal_session'],top_candidates=batch['proposals'],
        strategy=strategy_library.get_strategy(proposal['strategy_id']).describe(),metrics={},lifecycle=[current],observations=[])
    result=backtest.report_from_replay(batch,replay,primary_run_id='1'*32,dataset=data,specification=spec)
    row=result['results'][0]
    assert row['closed_signals']==0 and row['win_rate_pct'] is None and row['expectancy_net_per_closed_signal'] is None
    assert row['current_signal_id']==proposal['signal_id'] and row['current_proposal_state']=='UNFILLED_FUTURE_OUTCOME_UNKNOWN'
    assert result['output_candidates']==result['input_candidates']==1
    assert json.loads(json.dumps(result))==result

def test_full_cohort_and_exact_registered_strategy_reach_the_engine(monkeypatch):
    batch,spec,data=inputs(True);proposal=batch['proposals'][0];calls=[]
    current=dict(signal_id=proposal['signal_id'],symbol=proposal['symbol'],strategy_id=proposal['strategy_id'],
        strategy_version='2',signal_session=proposal['signal_session'],status='UNFILLED_PROPOSAL',bought_quantity=0,sold_quantity=0,fills=[])
    def engine(dataset,study,strategy,fold,cost,asof,**kwargs):
        calls.append((study,strategy,asof))
        return dict(as_of=asof,top_candidates=batch['proposals'],strategy=strategy.describe(),metrics={},lifecycle=[current],observations=[])
    monkeypatch.setattr(strategy_scanner,'evaluate_strategy',engine)
    result=backtest.run_candidate_backtest(batch,primary_run_id='1'*32,dataset=data,specification=spec)
    assert len(calls)==1 and calls[0][1].id==batch['strategy_contract']['strategy_id']
    assert calls[0][0]['universe']['symbols']==spec['universe']['symbols'] and len(spec['universe']['symbols'])==10
    assert result['strategy_contract']['definition_sha256']==calls[0][1].describe()['definition_sha256']
    assert json.loads(json.dumps(result))==result


def test_engine_returning_another_strategy_cannot_publish_a_plausible_candidate(monkeypatch):
    batch,spec,data=inputs(True)
    def wrong_engine(*args,**kwargs):
        return dict(as_of=batch['session']['signal_session'],top_candidates=batch['proposals'],
            strategy=dict(id='legacy_simulator_core_v1',version='2',definition_sha256=batch['strategy_contract']['definition_sha256']))
    monkeypatch.setattr(strategy_scanner,'evaluate_strategy',wrong_engine)
    with pytest.raises(ValueError,match='engine_strategy_mismatch'):
        backtest.run_candidate_backtest(batch,primary_run_id='1'*32,dataset=data,specification=spec)
