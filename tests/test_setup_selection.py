"""Cross-strategy selection is common-score ranked, complete and order independent."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import itertools,socket

import pandas as pd
import pytest

from scripts import setup_selection as selection
from scripts import bull_flag_companion as companion
from scripts.research_strategies import CHALLENGER
from scripts.pattern_strategy import PATTERNS
from scripts.strategy_backtest import digest

pytestmark=pytest.mark.alpaca_optional
STRATEGIES=(CHALLENGER,*PATTERNS)
NOW=datetime(2026,10,26,7,50,tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('No network in setup selection tests')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def proposal(sid,symbol,quality,raw_score=1000,volume=2):
    return dict(strategy_id=sid,strategy_version='2' if sid==CHALLENGER else '1',symbol=symbol,
        signal_id=digest([sid,symbol]),signal_session='2026-10-23',next_session='2026-10-26',
        next_open_at='2026-10-26T13:30:00+00:00',
        signal_close_raw=100.,signal_atr14_raw=2.,score=raw_score,
        setup_ranking=dict(model=selection.MODEL,model_sha256=selection.model_sha256(),signal_session='2026-10-23',
            score=quality,relative_volume=volume,feature_basis='COMPLETED_SPLIT_BARS',signal_volume_raw=1000.))


@pytest.mark.parametrize('winner',STRATEGIES)
def test_any_strategy_can_win_regardless_of_input_order_or_detector_score(winner):
    candidates=[proposal(sid,'S'+str(i),10 if sid==winner else 1,raw_score=-100 if sid==winner else 10000)
                for i,sid in enumerate(STRATEGIES)]
    for order in (candidates,list(reversed(candidates)),candidates[2:]+candidates[:2]):
        result=selection.rank_setups(order)
        assert result['selected_setups'][0]['proposal']['strategy_id']==winner
        assert result['strategy_priority'] is False


def test_no_family_quota_and_no_raw_score_comparison():
    rows=[proposal(PATTERNS[-1],'WIN'+str(i),10-i,raw_score=0) for i in range(3)]
    rows.extend(proposal(sid,'LOW'+str(i),0,raw_score=1000000) for i,sid in enumerate(STRATEGIES[:-1]))
    assert [r['proposal']['symbol'] for r in selection.rank_setups(rows)['selected_setups']]==['WIN0','WIN1','WIN2']


def test_duplicate_symbol_merges_matches_and_ties_do_not_use_family_order():
    rows=[proposal(sid,'SAME',8,raw_score=i*100) for i,sid in enumerate(STRATEGIES)]
    expected=selection.rank_setups(rows)
    assert len(expected['selected_setups'])==1
    assert expected['selected_setups'][0]['matching_strategies']==sorted(STRATEGIES)
    for order in itertools.permutations(rows[:3]):
        assert selection.rank_setups([*order,*rows[3:]])==expected
    changed=deepcopy(rows);changed[0]['setup_ranking']['relative_volume']=3
    with pytest.raises(ValueError,match='mixed_symbol_evidence'):selection.rank_setups(changed)


@pytest.mark.parametrize('change',['missing','model','nan','session','duplicate'])
def test_unproven_or_mixed_scores_never_fall_back_to_detector_score(change):
    rows=[proposal(CHALLENGER,'AAA',8)]
    if change=='missing':rows[0].pop('setup_ranking')
    if change=='model':rows[0]['setup_ranking']['model_sha256']='f'*64
    if change=='nan':rows[0]['setup_ranking']['score']=float('nan')
    if change=='session':rows[0]['setup_ranking']['signal_session']='2026-10-22'
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):selection.rank_setups(rows)


def test_real_scanner_finalists_use_common_score_before_local_top_three(monkeypatch):
    from test_strategy_current_handoff import case
    from scripts.strategy_scanner import evaluate_strategy
    data,spec,strategy,_=case(monkeypatch,symbols=('BBB','CCC','DDD','EEE'))
    spec['portfolio']['top_n']=3
    for symbol,score in (('AAA',0),('BBB',1),('CCC',2),('DDD',9),('EEE',10)):
        data.raw[symbol]['setup_score']=score
        data.raw[symbol]['setup_volume_ratio']=2.
    actual=evaluate_strategy(data,spec,strategy,spec['folds'][0],spec['cost_cases'][0],'2024-01-05')
    assert [p['symbol'] for p in actual['top_candidates']]==['EEE','DDD','CCC']
    assert [p['setup_ranking']['score'] for p in actual['top_candidates']]==[10,9,2]


def verified_fixture(monkeypatch):
    from test_strategy_execution_engine import setup
    _,_,_,base=setup()
    p=proposal(CHALLENGER,'LOW',1);b=proposal(PATTERNS[0],'BEST',10)
    base['proposals']=[p]
    bull=deepcopy(base);bull['proposals']=[b]
    bull['strategy_contract']=dict(strategy_id=PATTERNS[0],strategy_version='1',definition_sha256='f'*64)
    bull['primary_report_sha256']='c'*64
    handoffs={CHALLENGER:base,PATTERNS[0]:bull}
    monkeypatch.setattr(selection,'required_strategies',lambda *a:sorted(handoffs))
    monkeypatch.setattr(companion,'read_handoff',lambda conn,parent,**kw:deepcopy(handoffs.get(kw['strategy_id'])))
    ranked=selection.rank_setups([p,b]);references={CHALLENGER:base['primary_report_sha256'],PATTERNS[0]:'c'*64}
    spec=dict(policy=selection.POLICY,model_sha256=selection.model_sha256(),required_strategies=sorted(handoffs),
              strategy_result_sha256=references)
    record=dict(schema='jbravo.unified-setup-selection.v1',status='COMPLETED_UNIFIED_SETUP_SELECTION',
        run_id=selection.selection_id(base['primary_run_id']),primary_run_id=base['primary_run_id'],
        parent_report_sha256=base['primary_report_sha256'],source_bindings=base['source_bindings'],session=base['session'],
        specification=spec,selection=ranked,finished_at=(NOW-timedelta(seconds=1)).isoformat(),paper_only=True,
        research_qualified=False,order_submission_authorized=False)
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,sql,params):self.id=params[0]
        def fetchone(self):
            if self.id==record['run_id']:return (digest(spec),digest(record),record)
            return ('d'*64,'c'*64,{})
    class Connection:
        def cursor(self):return Cursor()
    return base,record,handoffs,Connection()


def test_exact_database_selection_and_queue_retain_original_strategy_handoffs(monkeypatch):
    base,record,handoffs,conn=verified_fixture(monkeypatch)
    verified=selection.read_selection(conn,base,now=NOW)
    queue=selection.execution_queue(verified)
    assert [r['proposal']['symbol'] for r in queue]==['BEST','LOW']
    assert queue[0]['handoff']['primary_report_sha256']==handoffs[PATTERNS[0]]['primary_report_sha256']
    assert queue[0]['handoff']['unified_selection']['result_sha256']==digest(record)
    record['selection']['selected_setups'].reverse()
    with pytest.raises(ValueError,match='ranking_changed'):selection.read_selection(conn,base,now=NOW)


def test_submission_guard_requires_current_global_selection_before_any_order(monkeypatch):
    from scripts import strategy_execution_runtime as runtime,strategy_execution as rules
    base,_,_,conn=verified_fixture(monkeypatch)
    verified=selection.read_selection(conn,base,now=NOW)
    first=selection.execution_queue(verified)[0]
    signal=dict(signal_id=first['proposal']['signal_id'],payload=dict(handoff=first['handoff'],account_id='paper'))
    opening=rules.utc(first['handoff']['execution_policy']['entry_open_at'])
    class Clock(datetime):
        @classmethod
        def now(cls,*a,**kw):return opening+timedelta(seconds=2)
    monkeypatch.setattr(runtime,'datetime',Clock)
    monkeypatch.setenv('APCA_API_BASE_URL','https://paper-api.alpaca.markets')
    monkeypatch.setenv('JBR_EXEC_PAPER','true')
    monkeypatch.setattr(runtime,'source_guard',lambda *a,**kw:True)
    monkeypatch.setattr(runtime,'readonly_handoff',lambda *a,**kw:base)
    monkeypatch.setattr(runtime,'unified_handoff',lambda *a,**kw:verified)
    monkeypatch.setattr(runtime,'monitor_ready',lambda *a,**kw:None)
    guard=runtime.guard_for({})
    assert guard('entry',signal) is True
    verified['record']['selection']['selected_setups']=[]
    with pytest.raises(ValueError,match='not_in_current_unified_selection'):guard('entry',signal)


def test_missing_enabled_strategy_blocks_global_selection_instead_of_favoring_available_family(monkeypatch):
    base,record,_,conn=verified_fixture(monkeypatch)
    monkeypatch.setattr(companion,'read_handoff',lambda *a,**kw:None)
    with pytest.raises(ValueError,match='required_strategy_unavailable'):selection.read_selection(conn,base,now=NOW)


def test_common_score_is_split_factor_invariant():
    from test_swing_patterns import pattern_frame
    from scripts.strategy_library import get_strategy
    frame=pattern_frame(PATTERNS[1]);fold=dict(warmup_start=str(frame.index[0].date()),
        test_start=str(frame.index[270].date()),test_end=str(frame.index[-1].date()))
    raw=frame.copy();raw[['open','high','low','close']]*=3
    first=get_strategy(PATTERNS[1]).prepare(frame,frame,fold)
    second=get_strategy(PATTERNS[1]).prepare(raw,frame,fold)
    pd.testing.assert_series_equal(first.setup_score,second.setup_score)
    pd.testing.assert_series_equal(first.setup_volume_ratio,second.setup_volume_ratio)


def test_same_bars_receive_same_quality_for_every_strategy():
    from test_swing_patterns import pattern_frame
    from scripts.strategy_library import select_strategies
    frame=pattern_frame(PATTERNS[1]);fold=dict(warmup_start=str(frame.index[0].date()),
        test_start=str(frame.index[270].date()),test_end=str(frame.index[-1].date()))
    results=[s.prepare(frame,frame,fold) for s in select_strategies()]
    for result in results[1:]:
        pd.testing.assert_series_equal(results[0].setup_score,result.setup_score)
        pd.testing.assert_series_equal(results[0].setup_volume_ratio,result.setup_volume_ratio)


def test_finalist_union_preserves_true_global_top_three_without_family_quotas():
    import random
    for seed in range(10):
        rng=random.Random(seed);all_rows=[];finalists=[]
        scores={s:rng.randrange(12) for s in range(30)}
        for sid in STRATEGIES:
            family=[proposal(sid,'S'+str(s),scores[s],raw_score=rng.randrange(1000))
                    for s in rng.sample(range(30),15)]
            all_rows.extend(family);finalists.extend(sorted(family,key=selection.key)[:3])
        assert selection.rank_setups(finalists)['selected_setups']==selection.rank_setups(all_rows)['selected_setups']


def test_durable_engine_can_submit_only_the_global_selected_signal(monkeypatch):
    from test_strategy_execution_engine import setup,reference
    from scripts import strategy_execution as rules
    engine,_,broker,base=setup()
    p=deepcopy(base['proposals'][0]);other=deepcopy(p)
    other.update(symbol='OTHER',signal_id='9'*64)
    base['proposals']=[p,other]
    base.pop('handoff_id');base['handoff_id']=rules.digest(base)
    ref=reference();ref.update(symbol='OTHER');ref.pop('reference_id');ref['reference_id']=rules.digest(ref)
    engine.enter(base,{'OTHER':ref},now=broker.now,clock=broker.clock(),account=broker.account(),
        positions={},open_orders=[],selected_signal_ids=[other['signal_id']])
    assert len(broker.posts)==1 and broker.posts[0]['symbol']=='OTHER'


def test_scanner_dashboard_projects_global_winner_instead_of_primary_family(monkeypatch):
    from scripts import strategy_dashboard as dashboard
    base,record,handoffs,conn=verified_fixture(monkeypatch)
    verified=selection.read_selection(conn,base,now=NOW)
    verified['primary']=dict(candidate_run_ts_utc=NOW.isoformat())
    rows=selection.dashboard_rows(verified)
    monkeypatch.setattr(dashboard,'snapshot',lambda:dict(ok=True,status='COMPLETE',rows=rows,
        run_ts_utc=NOW.isoformat(),_unified_selection=record['selection'],_unified_reports={}))
    result=dashboard.api_payload('candidates',{})
    assert [r['symbol'] for r in result['rows']]==['BEST','LOW']
    assert result['rows'][0]['strategy_id']==PATTERNS[0]
    assert result['rows'][0]['final_score']==10
    assert result['rows'][0]['matching_strategies']==[PATTERNS[0]]


def test_dashboard_cannot_fall_back_to_primary_candidates_when_global_selection_is_missing(monkeypatch):
    from scripts import strategy_dashboard as dashboard,db,strategy_pipeline as pipeline,strategy_pipeline_store,strategy_metrics
    from test_strategy_pipeline import report
    primary=report();stamp=datetime.fromisoformat(primary['started_at'])
    batch=(datetime.fromisoformat(primary['session']['signal_session']).date(),stamp,0,primary['batch'])
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):pass
        def fetchone(self):return (primary['run_id'],)
    class Connection:
        def set_session(self,**kw):pass
        def cursor(self):return Cursor()
        def rollback(self):pass
        def close(self):pass
    monkeypatch.setattr(db,'get_db_conn',lambda:Connection())
    monkeypatch.setattr(pipeline,'latest_batch',lambda *a,**kw:batch)
    monkeypatch.setattr(strategy_pipeline_store,'candidate_frame',lambda *a,**kw:(pd.DataFrame(),stamp))
    seen=[]
    def missing(*a,**kw):
        seen.append(True)
        raise ValueError('unified_selection_not_published')
    monkeypatch.setattr(selection,'dashboard_selection',missing)
    result=dashboard.snapshot()
    assert seen and result['ok'] is False and result['rows']==[]
