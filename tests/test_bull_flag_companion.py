"""Offline current-pipeline proof; no broker/provider/database connections."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from scripts import bull_flag_companion as companion
from scripts.bull_flag_strategy import BULL_FLAG, prepare, research_portfolio_class
from scripts.strategy_library import get_strategy, select_strategies
from scripts.research_strategies import CHALLENGER
from scripts import strategy_execution as rules

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No network in companion tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def flag_frame():
    close = np.r_[np.linspace(70, 95, 300), [100,104,108,112,116,115,114,113,112,113,117]]
    frame = pd.DataFrame(dict(open=close*.995, high=close+1, low=close-1, close=close,
                              volume=np.full(len(close),1000.)),
                         index=pd.bdate_range('2025-01-02',periods=len(close)))
    frame.iloc[-11:,frame.columns.get_loc('open')] = [99,102,106,110,114,115.5,114.5,113.5,112.5,112,114]
    frame.iloc[-11:,frame.columns.get_loc('low')] = [98,101,105,109,113,113,112,111,110,111,113]
    frame.iloc[-11:,frame.columns.get_loc('volume')] = [2000]*5+[600]*5+[3000]
    return frame


def fold(frame):
    return dict(id='bull-fixture',warmup_start=str(frame.index[0].date()),
                test_start=str(frame.index[270].date()),test_end=str(frame.index[-1].date()))


def test_original_signal_rule_files_are_preserved_with_all_strategies_enabled():
    # Definition excludes the catalog itself. All rule/kernel bytes remain the
    # same as the base commit; catalog extensions must not rename P/R signals.
    assert len(select_strategies())==6 and CHALLENGER in [s.id for s in select_strategies()]
    expected_sources = {
        'research_strategies.py':'2929fa5caa9b81e7176a2ee7587e25e721940795bb73c26bc38ea736004412fe',
        'indicators.py':'9feece6f7dd98dbaf9866595dd969ab993e12773a6051fd9339c9c6e051b7f70',
        'research_simulation.py':'4779ed62ab0d49f9a0432a1787b243b0835d5bc8af0fd3e384b9b57092d19b6c',
        'backtest_core.py':'348f9d53ea2ca0c506ae49aecc8b37ec7e1ae9a1c31ef253e81947a92d621fe5'}
    for name,expected in expected_sources.items():
        actual = (Path(__file__).parents[1]/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(actual).hexdigest() == expected
    assert get_strategy(BULL_FLAG).id != CHALLENGER


def test_bull_flag_features_are_causal_and_convert_raw_exit_levels():
    frame = flag_frame()
    result = prepare(frame, frame, fold(frame))
    assert result.score.notna().sum() == 1
    plan = result.iloc[-1].entry_details['pattern_exit']
    assert plan['pattern'] == BULL_FLAG and plan['target'] == 133.
    assert plan['invalidation'] == pytest.approx(109.89)
    raw = frame.copy()
    raw[['open','high','low','close']] *= 2
    scaled = prepare(raw, frame, fold(frame)).iloc[-1].entry_details['pattern_exit']
    assert scaled['target'] == 266. and scaled['invalidation'] == pytest.approx(219.78)
    future = frame.tail(2).copy()
    future.index = pd.bdate_range(frame.index[-1]+pd.offsets.BDay(), periods=2)
    expanded = pd.concat([frame, future])
    later = prepare(expanded, expanded, fold(expanded))
    pd.testing.assert_series_equal(result.score, later.loc[result.index,'score'])
    assert later.loc[result.index[-1],'entry_details'] == result.iloc[-1].entry_details


@pytest.mark.parametrize('kind',['target','both','gap','invalid_entry'])
def test_shared_kernel_executes_pattern_exits_conservatively(kind):
    frame = flag_frame()
    entry = frame.tail(1).copy()
    entry.index = pd.DatetimeIndex([frame.index[-1]+pd.offsets.BDay()])
    entry[['open','high','low','close']] = [118.,119.,117.,118.]
    following = entry.copy()
    following.index = pd.DatetimeIndex([entry.index[-1]+pd.offsets.BDay()])
    if kind == 'target':following[['open','high','low','close']] = [119.,137.,118.,136.]
    if kind == 'both':following[['open','high','low','close']] = [119.,137.,108.,120.]
    if kind == 'gap':following[['open','high','low','close']] = [105.,108.,104.,106.]
    if kind == 'invalid_entry':entry[['open','high','low','close']] = [140.,141.,139.,140.]
    raw = pd.concat([frame,entry,following])
    data = prepare(raw,raw,fold(raw))
    engine = research_portfolio_class()({'AAA': data},initial_cash=10000.,alloc_pct=.1,
        top_n=1,max_positions=1,trail_pct=None,atr_multiple=0,max_hold_days=7,
        enable_ema_exit=False,enable_macd_exit=False,enable_candlestick_exit=False)
    engine.run()
    sells = [f for f in engine.fills if f['side']=='SELL']
    if kind == 'invalid_entry':assert engine.fills == []
    else:
        assert len(sells) == 1 and sells[0]['reason'] != 'PARTIAL_5PCT'
        assert sells[0]['raw_price'] == pytest.approx({'target':133.,'both':109.89,'gap':105.}[kind])
        assert sells[0]['reason'] == ('PATTERN_TARGET' if kind=='target' else 'PATTERN_INVALIDATION')


def test_positive_scanner_backtest_metrics_and_database_handoff(tmp_path):
    from test_research_testing import make_capture
    from scripts.strategy_scanner import run_scan
    from scripts.research_dataset import load_dataset, canonical_hash
    from scripts.strategy_backtest import run_candidate_backtest, digest
    from scripts.strategy_metrics import build_metrics
    from scripts.strategy_pipeline import selection_batch
    from scripts.research_simulation import availability_clock
    spec, folder, bind, _ = make_capture(tmp_path)
    for adjustment in ('raw','split'):
        path = folder/f'alpaca_{adjustment}_daily.csv'
        csv = pd.read_csv(path)
        for symbol in ('AAA','SPY'):
            indices = csv.index[csv.symbol==symbol]
            custom = flag_frame().tail(11)
            csv.loc[indices,'v'] = 1000.
            for column,source in (('o','open'),('h','high'),('l','low'),('c','close'),('v','volume')):
                csv.loc[indices[-11:],column] = custom[source].to_numpy()
        csv.to_csv(path,index=False)
    calendar_file = folder/'calendar.json'
    calendar = json.loads(calendar_file.read_text())
    last = spec['folds'][0]['test_end']
    upcoming = str((pd.Timestamp(last)+pd.offsets.BDay()).date())
    calendar.append(dict(date=upcoming,open='09:30:00',close='16:00:00',timezone='America/New_York'))
    calendar_file.write_text(json.dumps(calendar))
    bind()
    spec.update(strategies=[BULL_FLAG],max_runs=2)
    now = datetime.fromisoformat(availability_clock(pd.Timestamp(last)))+timedelta(minutes=1)
    summary = run_scan(spec,as_of=last,output=tmp_path/'scan',strategy_ids=[BULL_FLAG],now=now,cost_case_id='base')
    assert len(summary['top_candidates']) == 1
    session = dict(signal_session=last,next_session=upcoming,
        signal_available_at=availability_clock(pd.Timestamp(last)),
        next_open_at=availability_clock(pd.Timestamp(upcoming),opening=True),
        execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN')
    value = dict(run_id='a'*64,scan_run_id=summary['run_id'],summary=summary,
                 handoff=dict(proposals=summary['top_candidates']))
    batch = selection_batch(value,session)
    dataset = load_dataset(spec)
    backtest = run_candidate_backtest(batch,primary_run_id='1'*32,dataset=dataset,specification=spec)
    metrics = build_metrics(batch,backtest,primary_run_id='1'*32)
    assert backtest['candidate_signal_ids'] == [summary['top_candidates'][0]['signal_id']]
    assert metrics['rows'][0]['sample_status'] == 'NO_COMPLETED_SIGNALS'
    parent = dict(primary_run_id='1'*32,primary_report_sha256='b'*64,session=session,
        source_bindings={},portfolio=rules.PORTFOLIO,paper_only=True,order_submission_authorized=False,
        inspection_mode=False,execution_policy=dict(entry_open_at=availability_clock(pd.Timestamp(upcoming),opening=True)))
    record = dict(schema='jbravo.companion-primary.v1',status='COMPLETED_COMPANION_PRIMARY',
        run_id=companion.primary_id('1'*32),primary_run_id='1'*32,parent_report_sha256='b'*64,
        source_bindings={},batch=batch,backtest=backtest,metrics=metrics,session=session,
        finished_at=now.isoformat(),specification_sha256='c'*64,research_qualified=False,
        execution_authorized=False,order_submission_authorized=False)
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,parameters):assert parameters == (record['run_id'],)
        def fetchone(self):return ('c'*64,digest(record),record)
    class Connection:
        def cursor(self):return Cursor()
    handoff = companion.read_handoff(Connection(),parent,now=now)
    assert handoff['strategy_contract']['strategy_id'] == BULL_FLAG
    assert handoff['proposals'][0]['pattern_exit']['symbol'] == 'AAA'
    assert handoff['handoff_id'] == rules.digest({k:v for k,v in handoff.items() if k!='handoff_id'})
    for key,changed in (('entry',118.),('signal_time','2025-01-01T00:00:00')):
        original = record['batch']['proposals'][0]['pattern_exit'][key]
        record['batch']['proposals'][0]['pattern_exit'][key] = changed
        with pytest.raises((ValueError,KeyError,TypeError)):
            companion.read_handoff(Connection(),parent,now=now)
        record['batch']['proposals'][0]['pattern_exit'][key] = original
    for key in ('parent_report_sha256','source_bindings','session'):
        original = deepcopy(record[key])
        record[key] = {} if key!='parent_report_sha256' else 'f'*64
        if record[key] == original:record[key]={'changed':True}
        with pytest.raises((ValueError,KeyError,TypeError)):
            companion.read_handoff(Connection(),parent,now=now)
        record[key] = original


def bull_handoff(parent, symbol='BULL'):
    handoff = deepcopy(parent)
    proposal = handoff['proposals'][0]
    proposal.update(strategy_id=BULL_FLAG,strategy_version='1',symbol=symbol,signal_id='b'*64,
        pattern_exit=dict(pattern=BULL_FLAG,version=1,status='confirmed',symbol=symbol,
            signal_time=proposal['signal_session']+'T00:00:00',entry=110.,resistance=109.,invalidation=105.,target=120.))
    handoff['strategy_contract'] = dict(strategy_id=BULL_FLAG,strategy_version='1',definition_sha256='f'*64)
    handoff.pop('handoff_id');handoff['handoff_id']=rules.digest(handoff)
    return handoff


def test_shared_account_prevents_duplicate_symbols_across_strategies():
    from test_strategy_execution_engine import setup,enter,reference
    from scripts.strategy_execution_runtime import available_entry_budget
    engine,store,broker,parent = setup()
    enter(engine,broker,parent)
    assert len(broker.posts)==1
    duplicate = bull_handoff(parent,parent['proposals'][0]['symbol'])
    engine.enter(duplicate,{duplicate['proposals'][0]['symbol']:reference()},now=broker.now,
        clock=broker.clock(),account=broker.account(),positions={},open_orders=[])
    assert len(broker.posts)==1
    assert available_entry_budget(store,parent['session']['next_session'])==2
    assert any(e['code']=='SYMBOL_ALREADY_HELD_OR_PENDING' for e in engine.events)


def test_bull_flag_uses_durable_stop_target_and_cancellation_recovery():
    from test_strategy_execution_engine import setup,reference
    engine,store,broker,parent = setup()
    handoff = bull_handoff(parent)
    ref = reference();ref.update(symbol='BULL',price=110.);ref.pop('reference_id');ref['reference_id']=rules.digest(ref)
    engine.enter(handoff,{'BULL':ref},now=broker.now,clock=broker.clock(),account=broker.account(),positions={},open_orders=[])
    assert len(broker.posts)==1
    broker.fill('entry',broker.posts[0]['qty'],price=110.)
    engine.tick(now=broker.now)
    assert store.signals()[0]['payload']['proposal']['pattern_exit']['target']==120.
    assert len(broker.posts)==2 and broker.posts[-1]['type']=='stop'
    broker.bid=121.
    engine.tick(now=broker.now)
    assert any(e['code']=='PATTERN_TARGET' for e in engine.events)
    assert broker.cancels
    assert broker.posts[-1]['side']=='sell' and broker.posts[-1]['type']=='limit'
    before=len(broker.posts)
    engine.tick(now=broker.now)
    assert len(broker.posts)==before


def test_no_remaining_daily_budget_cannot_submit_companion():
    from test_strategy_execution_engine import setup,reference
    engine,_,broker,parent = setup()
    handoff=bull_handoff(parent)
    engine.enter(handoff,{},now=broker.now,clock=broker.clock(),account=broker.account(),
                 positions={},open_orders=[],max_new_entries=0)
    assert broker.posts==[] and any(e['code']=='SHARED_DAILY_ENTRY_CAP' for e in engine.events)


def test_companion_archive_is_distinct_and_cannot_replace_primary():
    from test_strategy_pipeline import archive, row
    from scripts import strategy_pipeline as pipeline
    from scripts.research_dataset import canonical_hash
    cfg,session,value=archive(positive=True)
    cfg=dict(cfg,strategy_id=BULL_FLAG)
    value.update(strategy_id=BULL_FLAG,run_id=companion.archive_id(session['signal_session']),
                 configuration_sha256=canonical_hash(cfg))
    description=get_strategy(BULL_FLAG).describe()
    entry=value['summary']['strategies'][0]
    entry['strategy']=description
    proposal=value['handoff']['proposals'][0]
    proposal.update(strategy_id=BULL_FLAG,strategy_version='1')
    proposal['signal_id']=canonical_hash(dict(strategy_id=BULL_FLAG,version='1',
        episode_id=entry['episode_id'],symbol=proposal['symbol'],signal_session=session['signal_session']))
    assert pipeline.validate_archive(row(value),cfg,session,strategy_id=BULL_FLAG)==value
    with pytest.raises(ValueError,match='scope_mismatch'):
        pipeline.validate_archive(row(value),cfg,session)


def test_failed_companion_preserves_parent_and_never_replays_claim(tmp_path,monkeypatch):
    from scripts import strategy_scanner
    def fail(*args,**kwargs):raise ValueError('offline failure')
    monkeypatch.setattr(strategy_scanner,'run_scan',fail)
    parent=dict(strategies=[CHALLENGER],max_seconds=60,folds=[dict(test_end='2026-10-26')])
    before=deepcopy(parent)
    published=[]
    kwargs=dict(now=datetime.now(timezone.utc),archive=lambda *args:published.append(args),total_deadline=1e20)
    result=companion.produce(tmp_path,parent,dict(episode_id='test',cost_case=dict(id='base')),**kwargs)
    assert result['status']=='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED'
    assert parent==before and published==[]
    marker=tmp_path/'companions'/BULL_FLAG/'failure.json'
    evidence=marker.read_bytes()
    assert companion.produce(tmp_path,parent,{},**kwargs)['status']=='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED'
    assert marker.read_bytes()==evidence
