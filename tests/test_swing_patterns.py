"""Causal geometry, actual full-cohort handoffs and durable execution for four patterns."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import hashlib,json,socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import bull_flag_companion as companion
from scripts import pattern_strategy as adapter
from scripts.swing_patterns import STRATEGIES, DETECTORS
from scripts.strategy_library import get_strategy,select_strategies
from scripts import strategy_execution as rules

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**kw):raise AssertionError('No network in pattern tests')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def pattern_frame(strategy_id):
    if strategy_id==STRATEGIES[0]:
        tail=np.interp(np.arange(31),[0,4,8,12,16,20,24,28,29,30],
                       [109,95,109,98,109,102,109,106,108,113])
    elif strategy_id==STRATEGIES[1]:
        tail=np.interp(np.arange(81),[0,20,40,60,79,80],[130,100,115,101,114,118])
    elif strategy_id==STRATEGIES[2]:
        cup=110-20*np.sin(np.linspace(0,np.pi,61))
        tail=np.r_[np.linspace(90,109,20),cup,np.linspace(109,106,10),114.]
    else:
        tail=np.interp(np.arange(81),[0,20,30,40,50,60,79,80],[130,100,112,90,112,101,111,116])
    close=np.r_[np.linspace(70,90,360-len(tail)),tail]
    frame=pd.DataFrame(dict(open=close-.2,high=close+1,low=close-1,close=close,volume=1000.),
                       index=pd.bdate_range('2023-01-03',periods=360))
    frame.iloc[-20:,frame.columns.get_loc('volume')]=800.
    frame.iloc[-1,frame.columns.get_loc('volume')]=3000.
    return frame


@pytest.mark.parametrize('strategy_id',STRATEGIES)
def test_confirmed_geometry_volume_and_causal_raw_conversion(strategy_id):
    frame=pattern_frame(strategy_id)
    detector=DETECTORS[strategy_id]
    signal=detector(frame)
    assert signal is not None and signal.pattern==strategy_id
    assert signal.invalidation<signal.resistance<signal.entry<signal.target
    assert signal.signal_time==frame.index[-1].isoformat()
    for key,value in (('volume',500.),('open',signal.entry+1),('close',signal.resistance)):
        changed=frame.copy();changed.loc[changed.index[-1],key]=value
        assert detector(changed) is None
    fold=dict(warmup_start=str(frame.index[0].date()),test_start=str(frame.index[270].date()),
              test_end=str(frame.index[-1].date()))
    raw=frame.copy();raw[['open','high','low','close']]*=2
    prepared=adapter.prepare(raw,frame,fold,strategy_id)
    assert prepared.score.notna().sum()==1
    plan=prepared.iloc[-1].entry_details['pattern_exit']
    assert plan['entry']==signal.entry*2 and plan['target']==signal.target*2
    assert adapter.plan_from_row('AAA',prepared.iloc[-1])==dict(plan,symbol='AAA')
    extra=frame.tail(3).copy();extra.index=pd.bdate_range(frame.index[-1]+pd.offsets.BDay(),periods=3)
    future=pd.concat([frame,extra]);raw_future=future.copy();raw_future[['open','high','low','close']]*=2
    later=adapter.prepare(raw_future,future,dict(fold,test_end=str(future.index[-1].date())),strategy_id)
    pd.testing.assert_frame_equal(prepared,later.loc[prepared.index])


@pytest.mark.parametrize('strategy_id',STRATEGIES)
@pytest.mark.parametrize('kind',['nan','duplicate','unsorted','invalid_ohlc'])
def test_malformed_pattern_inputs_never_signal(strategy_id,kind):
    frame=pattern_frame(strategy_id)
    if kind=='nan':frame.iloc[-5,0]=np.nan
    if kind=='duplicate':frame.index=list(frame.index[:-1])+[frame.index[-2]]
    if kind=='unsorted':frame=frame.iloc[::-1]
    if kind=='invalid_ohlc':frame.iloc[-5,frame.columns.get_loc('high')]=0
    assert DETECTORS[strategy_id](frame) is None


@pytest.mark.parametrize('strategy_id',STRATEGIES)
def test_scanner_backtest_metrics_and_separate_exact_db_handoff(tmp_path,strategy_id):
    from test_research_testing import make_capture
    from scripts.strategy_scanner import run_scan
    from scripts.research_dataset import load_dataset
    from scripts.strategy_pipeline import selection_batch
    from scripts.strategy_backtest import run_candidate_backtest,digest
    from scripts.strategy_metrics import build_metrics
    from scripts.research_simulation import availability_clock
    spec,folder,bind,_=make_capture(tmp_path)
    frame=pattern_frame(strategy_id)
    for adjustment in ('raw','split'):
        path=folder/f'alpaca_{adjustment}_daily.csv';saved=pd.read_csv(path)
        for symbol in ('AAA','SPY'):
            indices=saved.index[saved.symbol==symbol]
            for target,source in (('o','open'),('h','high'),('l','low'),('c','close'),('v','volume')):
                saved.loc[indices,target]=frame[source].to_numpy()
        saved.to_csv(path,index=False)
    last=spec['folds'][0]['test_end'];upcoming=str((pd.Timestamp(last)+pd.offsets.BDay()).date())
    path=folder/'calendar.json';calendar=json.loads(path.read_bytes())
    calendar.append(dict(date=upcoming,open='09:30:00',close='16:00:00',timezone='America/New_York'))
    path.write_text(json.dumps(calendar));bind();spec.update(strategies=[strategy_id],max_runs=2)
    now=datetime.fromisoformat(availability_clock(pd.Timestamp(last)))+timedelta(minutes=1)
    summary=run_scan(spec,as_of=last,output=tmp_path/'scan',strategy_ids=[strategy_id],now=now,cost_case_id='base')
    assert len(summary['top_candidates'])==1
    session=dict(signal_session=last,next_session=upcoming,signal_available_at=availability_clock(pd.Timestamp(last)),
        next_open_at=availability_clock(pd.Timestamp(upcoming),opening=True),execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN')
    value=dict(run_id=companion.archive_id(last,strategy_id),scan_run_id=summary['run_id'],summary=summary,
               handoff=dict(proposals=summary['top_candidates']))
    batch=selection_batch(value,session)
    backtest=run_candidate_backtest(batch,primary_run_id='1'*32,dataset=load_dataset(spec),specification=spec)
    metrics=build_metrics(batch,backtest,primary_run_id='1'*32)
    assert backtest['candidate_signal_ids']==[summary['top_candidates'][0]['signal_id']]
    parent=dict(primary_run_id='1'*32,primary_report_sha256='b'*64,session=session,source_bindings={},
        execution_policy=dict(entry_open_at=availability_clock(pd.Timestamp(upcoming),opening=True)))
    record=dict(schema='jbravo.companion-primary.v1',status='COMPLETED_COMPANION_PRIMARY',
        run_id=companion.primary_id('1'*32,strategy_id),primary_run_id='1'*32,parent_report_sha256='b'*64,
        source_bindings={},batch=batch,backtest=backtest,metrics=metrics,session=session,
        finished_at=now.isoformat(),specification_sha256='c'*64,research_qualified=False,
        execution_authorized=False,order_submission_authorized=False)
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,sql,params):self.key=params[0]
        def fetchone(self):return ('c'*64,digest(record),record) if self.key==record['run_id'] else None
    class Connection:
        def cursor(self):return Cursor()
    handoff=companion.read_handoff(Connection(),parent,now=now,strategy_id=strategy_id)
    assert handoff['strategy_contract']['strategy_id']==strategy_id
    assert handoff['proposals'][0]['pattern_exit']['pattern']==strategy_id
    assert companion.read_handoff(Connection(),parent,now=now) is None
    record['parent_report_sha256']='f'*64
    with pytest.raises(ValueError,match='lineage_changed'):
        companion.read_handoff(Connection(),parent,now=now,strategy_id=strategy_id)


@pytest.mark.parametrize('strategy_id',STRATEGIES)
def test_pattern_enters_and_uses_existing_durable_target_and_duplicate_protection(strategy_id):
    from test_strategy_execution_engine import setup,reference,enter
    from test_bull_flag_companion import bull_handoff
    engine,store,broker,parent=setup()
    handoff=bull_handoff(parent)
    handoff['strategy_contract']['strategy_id']=strategy_id
    p=handoff['proposals'][0];p.update(strategy_id=strategy_id,signal_close_raw=110.)
    p['pattern_exit']['pattern']=strategy_id
    handoff.pop('handoff_id');handoff['handoff_id']=rules.digest(handoff)
    ref=reference();ref.update(symbol='BULL',price=110.);ref.pop('reference_id');ref['reference_id']=rules.digest(ref)
    engine.enter(handoff,{'BULL':ref},now=broker.now,clock=broker.clock(),account=broker.account(),positions={},open_orders=[])
    assert len(broker.posts)==1
    broker.fill('entry',broker.posts[0]['qty'],price=110.);engine.tick(now=broker.now)
    broker.bid=121.;engine.tick(now=broker.now)
    assert any(e['code']=='PATTERN_TARGET' for e in engine.events)
    assert broker.cancels and broker.posts[-1]['side']=='sell' and broker.posts[-1]['type']=='limit'
    before=len(broker.posts);engine.tick(now=broker.now);assert len(broker.posts)==before


def test_existing_rule_files_and_default_strategy_stay_unchanged():
    assert select_strategies()[0].id=='trend_pullback_reclaim_qualifying_close_v2'
    root=Path(__file__).parents[1]
    expected={'bull_flag_strategy.py':'29e05116f8614b4e03d3eb6d626053136b26392dd9484ec69cd48472b716933f',
              'chart_patterns.py':'546cb721be567ae8aa400a55a6cd33eb265be313961b96132cd13926107657b8'}
    for name,pin in expected.items():
        assert hashlib.sha256((root/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    for sid,pin in (
        ('trend_pullback_reclaim_qualifying_close_v2','bf576123697684b933b0f6772ab0121e8546fc9aaa1daa4084dbabcc27de0207'),
        (adapter.BULL_FLAG,'6b39f5fe3d82347d6b42d0112512efdafed35481eb9b786b2429dba7b8ecfffb')):
        definition=get_strategy(sid).describe();definition.pop('research_qualified');definition.pop('definition_sha256')
        definition['source_sha256']={n:hashlib.sha256((root/'scripts'/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for n in definition['source_sha256'] if n!='strategy_library.py'}
        assert hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':')).encode()).hexdigest()==pin
    ids=(adapter.BULL_FLAG,*STRATEGIES)
    assert len({companion.archive_id('2026-10-27',s) for s in ids})==5
    assert len({companion.primary_id('a'*32,s) for s in ids})==5
    assert adapter.enabled_ids()==STRATEGIES


@pytest.mark.parametrize('strategy_id',STRATEGIES)
@pytest.mark.parametrize('outcome',['target','both','gap','entry_outside'])
def test_pattern_replay_uses_prior_stops_before_targets(strategy_id,outcome):
    frame=pattern_frame(strategy_id);signal=DETECTORS[strategy_id](frame)
    entry=(signal.entry+signal.target)/2
    nextbar=frame.tail(1).copy();nextbar.index=pd.DatetimeIndex([frame.index[-1]+pd.offsets.BDay()])
    if outcome=='entry_outside':entry=signal.target+2
    nextbar[['open','high','low','close']]=[entry,entry+.1,entry-.1,entry]
    nextbar['volume']=1000.
    after=nextbar.copy();after.index=pd.DatetimeIndex([nextbar.index[-1]+pd.offsets.BDay()])
    after[['open','high','low','close']]=[entry,signal.target+2,entry-.1,signal.target]
    if outcome=='entry_outside':after[['open','high','low','close']]=[entry,entry+.1,entry-.1,entry]
    if outcome=='both':after['low']=signal.invalidation-2
    if outcome=='gap':after[['open','high','low','close']]=[signal.invalidation-2,signal.invalidation-1,signal.invalidation-3,signal.invalidation-2]
    raw=pd.concat([frame,nextbar,after]);fold=dict(warmup_start=str(raw.index[0].date()),
        test_start=str(raw.index[270].date()),test_end=str(raw.index[-1].date()))
    prepared=adapter.prepare(raw,raw,fold,strategy_id)
    engine=adapter.research_portfolio_class()({'AAA':prepared},initial_cash=10000.,alloc_pct=.1,
        top_n=1,max_positions=1,trail_pct=None,atr_multiple=0,max_hold_days=7,
        enable_ema_exit=False,enable_macd_exit=False,enable_candlestick_exit=False)
    engine.run();sells=[f for f in engine.fills if f['side']=='SELL']
    if outcome=='entry_outside':assert engine.fills==[]
    else:
        assert len(sells)==1
        expected=signal.target if outcome=='target' else signal.invalidation if outcome=='both' else signal.invalidation-2
        assert sells[0]['raw_price']==pytest.approx(expected)
        assert sells[0]['reason']==('PATTERN_TARGET' if outcome=='target' else 'PATTERN_INVALIDATION')


def test_structurally_invalid_formations_do_not_qualify():
    descending=pattern_frame(STRATEGIES[0])
    for offset,value in ((-27,105.),(-19,100.),(-11,95.),(-3,90.)):
        descending.iloc[offset,descending.columns.get_indexer(['open','high','low','close'])]=[value-.2,value+1,value-1,value]
    assert DETECTORS[STRATEGIES[0]](descending) is None
    no_trend=pattern_frame(STRATEGIES[1]);no_trend.iloc[-81:-61,no_trend.columns.get_indexer(['open','high','low','close'])]=[102.,103.,101.5,102.]
    assert DETECTORS[STRATEGIES[1]](no_trend) is None
    sharp_v=pattern_frame(STRATEGIES[2]);values=np.r_[np.linspace(110,90,31),np.linspace(90,110,31)[1:]]
    sharp_v.iloc[-72:-11,sharp_v.columns.get_indexer(['open','high','low','close'])]=np.c_[values-.2,values+1,values-1,values]
    assert DETECTORS[STRATEGIES[2]](sharp_v) is None
    shallow_head=pattern_frame(STRATEGIES[3]);values=np.interp(np.arange(81),[0,20,30,40,50,60,79,80],[130,100,112,99,112,101,111,116])
    shallow_head.iloc[-81:,shallow_head.columns.get_indexer(['open','high','low','close'])]=np.c_[values-.2,values+1,values-1,values]
    assert DETECTORS[STRATEGIES[3]](shallow_head) is None


def test_shared_daily_budget_survives_multiple_pattern_workers():
    from test_strategy_execution_engine import setup,enter,reference
    from test_bull_flag_companion import bull_handoff
    from scripts.strategy_execution_runtime import available_entry_budget
    engine,store,broker,parent=setup();enter(engine,broker,parent)
    for i,strategy_id in enumerate(STRATEGIES):
        handoff=bull_handoff(parent,symbol='PAT'+str(i))
        handoff['strategy_contract']['strategy_id']=strategy_id
        proposal=handoff['proposals'][0]
        proposal.update(strategy_id=strategy_id,signal_close_raw=110.,signal_id=str(i+1)*64)
        proposal['pattern_exit']['pattern']=strategy_id
        handoff.pop('handoff_id');handoff['handoff_id']=rules.digest(handoff)
        ref=reference();ref.update(symbol=proposal['symbol'],price=110.);ref.pop('reference_id');ref['reference_id']=rules.digest(ref)
        engine.enter(handoff,{proposal['symbol']:ref},now=broker.now,clock=broker.clock(),account=broker.account(),
            positions={},open_orders=[],max_new_entries=available_entry_budget(store,parent['session']['next_session']))
    assert len(broker.posts)==3
    assert available_entry_budget(store,parent['session']['next_session'])==0


@pytest.mark.parametrize('strategy_id',STRATEGIES)
def test_existing_held_or_pending_symbol_cannot_get_a_second_strategy_owner(strategy_id):
    from test_strategy_execution_engine import setup,enter
    from test_bull_flag_companion import bull_handoff
    engine,_,broker,parent=setup();enter(engine,broker,parent)
    duplicate=bull_handoff(parent,symbol=parent['proposals'][0]['symbol'])
    duplicate['strategy_contract']['strategy_id']=strategy_id
    duplicate['proposals'][0]['strategy_id']=strategy_id
    duplicate['proposals'][0]['pattern_exit']['pattern']=strategy_id
    engine.enter(duplicate,{},now=broker.now,clock=broker.clock(),account=broker.account(),positions={},open_orders=[])
    assert len(broker.posts)==1 and any(e['code']=='SYMBOL_ALREADY_HELD_OR_PENDING' for e in engine.events)


def test_disabled_invalid_and_missing_extra_policy_leave_bull_flag_policy_available(tmp_path):
    assert adapter.configuration(tmp_path) is None
    folder=tmp_path/'config';folder.mkdir()
    source=Path(__file__).parents[1]/'config'
    (folder/'strategy_companions.json').write_bytes((source/'strategy_companions.json').read_bytes())
    value=json.loads((source/'chart_pattern_companions.json').read_bytes())
    path=folder/'chart_pattern_companions.json';path.write_text(json.dumps(dict(value,enabled=False)))
    assert adapter.enabled_ids(tmp_path)==()
    assert companion.policy_for(adapter.BULL_FLAG,tmp_path) is not None
    path.write_text(json.dumps(dict(value,max_positions_total=10)))
    with pytest.raises(ValueError,match='configuration_changed'):adapter.configuration(tmp_path)
    assert companion.policy_for(adapter.BULL_FLAG,tmp_path) is not None


def test_one_unavailable_handoff_does_not_hide_other_patterns(monkeypatch):
    from scripts import strategy_execution_runtime as runtime
    calls=[]
    def read(env,parent,*,strategy_id,**kwargs):
        calls.append(strategy_id)
        if strategy_id==STRATEGIES[1]:raise ValueError('unavailable')
        return dict(strategy_contract=dict(strategy_id=strategy_id))
    monkeypatch.setattr(runtime,'companion_handoff',read)
    result=runtime.additional_handoffs({}, {})
    assert calls==list(STRATEGIES)
    assert [r['strategy_contract']['strategy_id'] for r in result]==[STRATEGIES[0],STRATEGIES[2],STRATEGIES[3]]
