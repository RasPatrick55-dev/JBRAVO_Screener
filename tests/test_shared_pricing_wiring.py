from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from tests.test_isolated_pricing_integration import fixture, Database
from tests.test_pricing_adapter_prototype import contract, PUBLISHED, load, NATIVE
from tests.test_actual_web_binding import WEB
from shared_pricing_contract import SharedPricingContract
from shared_pricing_wiring import SharedResearchStore, wire_shared

pytestmark = pytest.mark.alpaca_optional

@pytest.fixture
def wiring(fixture, monkeypatch):
    f=fixture; f.primary['batch']={'proposals':[]}; db=Database(); statuses=[]; checks=[]
    root=Path(__file__).resolve().parents[1]
    names=['shared_pricing_contract.py','shared_pricing_wiring.py',
           'isolated_pricing_integration.py','pricing_adapter_prototype.py']
    pins={n:dict(bytes=len((root/n).read_bytes()),sha256=sha256((root/n).read_bytes()).hexdigest()) for n in names}
    runtime=load('shared_native_runtime',NATIVE/'strategy_execution_runtime.py')
    import shared_pricing_wiring as module
    monkeypatch.setattr(module,'capture_prices',lambda *a,**kw:deepcopy(f.prices))
    w=wire_shared(root=root,pins=pins,rules=f.rules,selection=f.selection,runtime=runtime,
        connect=db.connect,research_connect=lambda **kw:db.connect(write=False),market_get=None,
        check_sources=lambda p:checks.append(deepcopy(p)),report_status=statuses.append,
        base_dir='unused',challenger='test',clock=lambda:datetime.fromisoformat(PUBLISHED))
    return f,w,db,statuses,checks,runtime


def publish(wiring):
    f,w,*_=wiring
    return w.publish(f.primary,inputs=lambda p:(f.handoffs,f.references,f.selected))


def test_shared_publication_single_row_receipt_and_entry_read(wiring):
    f,w,db,statuses,checks,runtime=wiring
    status=publish(wiring)
    assert status['status']=='COMPLETED_UNIFIED_SETUP_SELECTION'
    assert len(db.rows)==1 and db.connections[0].commits==1
    assert status['selected_signal_ids']==['GOOD_signal']
    assert status['ranked_signal_ids']==['GOOD_signal','NONE_signal']
    assert status['excluded_count']==1 and len(checks)==2
    parent=f.selection.parent_context(f.primary)
    v=w.entry_runtime.unified_handoff({},parent)
    q=w.entry_runtime.pipeline_execution_queue(v)
    assert q[0]['handoff']['pipeline_limit_order']['limit_price']=='6.73'
    assert status['result_sha256']==v['result_sha256']
    assert all(c.closed for c in db.connections)
    assert w.entry_runtime.guard_for.__code__ is runtime.guard_for.__code__
    assert w.entry_runtime.guard_for.__globals__['pipeline_execution_queue'] is w.execution_queue
    assert runtime.guard_for.__globals__['pipeline_execution_queue'] is runtime.pipeline_execution_queue


@pytest.mark.parametrize('failure',['insert','readback','duplicate'])
def test_publication_failure_never_claims_complete_or_retries(wiring,failure):
    f,w,db,statuses,checks,runtime=wiring
    if failure=='insert':db.fail_insert=0
    if failure=='readback':db.fail_read=True
    if failure=='duplicate':assert publish(wiring)['status']=='COMPLETED_UNIFIED_SETUP_SELECTION'
    result=publish(wiring)
    assert result['status']=='UNIFIED_SETUP_SELECTION_UNAVAILABLE'
    assert result['failure_code']=='SHARED_PRICING_PUBLICATION_FAILED'
    assert len(db.rows)==(0 if failure=='insert' else 1)
    assert sum(c.commits for c in db.connections)==(0 if failure=='insert' else 1)
    assert all(c.closed for c in db.connections)


def test_source_drift_before_commit_blocks_write(wiring,monkeypatch):
    f,w,db,*_=wiring
    import shared_pricing_wiring as module
    original=module.verify_adapter_files;calls=[]
    def verify(*args):
        calls.append(1)
        if len(calls)>1:raise ValueError('drift')
        return original(*args)
    monkeypatch.setattr(module,'verify_adapter_files',verify)
    assert publish(wiring)['status']=='UNIFIED_SETUP_SELECTION_UNAVAILABLE'
    assert not db.connections and not db.rows


def test_retained_web_display_reads_shared_row_and_control_receipt(wiring,monkeypatch):
    f,w,db,statuses,checks,runtime=wiring
    from scripts import setup_selection as native,strategy_pipeline as pipeline,pipeline_postflight as health
    from scripts import bull_flag_companion as companion,research_strategies
    f.primary.update(candidate_run_ts_utc=PUBLISHED,backtest={},metrics={})
    status=publish(wiring);raw=json.dumps(f.primary)
    web=load('shared_web_display',WEB)
    monkeypatch.setattr(web,'primary_receipt',lambda *a,**k:(f.primary,raw))
    monkeypatch.setattr(native,'read_selection',w.read_selection)
    monkeypatch.setattr(native,'parent_context',f.selection.parent_context)
    monkeypatch.setattr(health,'_read',lambda path,*a:json.dumps(status) if 'unified_setups' in str(path) else raw)
    monkeypatch.setattr(research_strategies,'CHALLENGER','test')
    batch=(datetime(2026,11,9).date(),datetime.fromisoformat(PUBLISHED),None,f.primary['batch'])
    v=web.display_selection(db.connect(write=False),batch,'primary',now=datetime(2026,11,10,8,tzinfo=timezone.utc))
    assert v['result_sha256']==status['result_sha256']
    assert v['record']['selection']['selected_signal_ids']==['GOOD_signal']
    assert v['record']['ranking_record']['selection']['selected_signal_ids']==['GOOD_signal','NONE_signal']



def test_native_entry_guard_rechecks_shared_ticket_before_submission(wiring,monkeypatch):
    f,w,db,statuses,checks,runtime=wiring
    f.handoffs['test'].update(strategy_contract={'strategy_id':'test'},
        primary_report_sha256='report',candidate_values=[],batch_sha256='batch')
    publish(wiring)
    parent=f.selection.parent_context(f.primary)
    verified=w.entry_runtime.unified_handoff({},parent)
    setup=w.execution_queue(verified)[0]
    signal={'signal_id':'GOOD_signal','payload':{'handoff':setup['handoff'],'account_id':'synthetic'}}
    namespace=w.entry_runtime.guard_for.__globals__
    monkeypatch.setitem(namespace,'source_guard',lambda *a,**kw:None)
    monkeypatch.setitem(namespace,'expected_paper_account',lambda env:'synthetic')
    monkeypatch.setitem(namespace,'readonly_handoff',lambda *a,**kw:parent)
    monitor_checks=[]
    monkeypatch.setitem(namespace,'monitor_ready',lambda *a,**kw:monitor_checks.append(True))
    class Clock:
        @staticmethod
        def now(tz):return datetime(2026,11,10,9,0,10,tzinfo=timezone.utc)
    monkeypatch.setitem(namespace,'datetime',Clock)
    monkeypatch.setenv('APCA_API_BASE_URL','https://paper-api.alpaca.markets')
    monkeypatch.setenv('JBR_EXEC_PAPER','true')
    guard=w.entry_runtime.guard_for({})
    assert guard('entry',signal) is True
    assert monitor_checks==[True]
    original_model=signal['payload']['handoff']['execution_policy']['model']
    signal['payload']['handoff']['execution_policy']['model']=f.rules.EXECUTION_MODEL
    with pytest.raises(ValueError,match='execution_pipeline_model_changed_before_post'):
        guard('entry',signal)
    signal['payload']['handoff']['execution_policy']['model']=original_model
    def reject_monitor(*a,**kw):raise ValueError('monitor_not_ready')
    monkeypatch.setitem(namespace,'monitor_ready',reject_monitor)
    with pytest.raises(ValueError,match='monitor_not_ready'):
        guard('entry',signal)
    monkeypatch.setitem(namespace,'monitor_ready',lambda *a,**kw:None)
    signal['payload']['handoff']['pipeline_limit_order']['limit_price']='99'
    with pytest.raises(ValueError,match='execution_pipeline_limit_changed_before_post'):
        guard('entry',signal)


def test_existing_native_pipeline_queue_accepts_reconstructed_shared_tickets(wiring,monkeypatch):
    f,w,db,*_=wiring
    publish(wiring)
    v=w.entry_runtime.unified_handoff({},f.selection.parent_context(f.primary))
    from scripts import setup_selection as selected,research_strategies
    monkeypatch.setattr(selected,'execution_queue',w.native_execution_queue)
    monkeypatch.setattr(research_strategies,'CHALLENGER','test')
    native=wiring[-1]
    actual=native.pipeline_execution_queue(v)
    assert actual==w.execution_queue(v)
    assert actual[0]['handoff']['pipeline_limit_order']['limit_price']=='6.73'


def test_legacy_reader_keeps_native_validation_and_missing_ticket_rejection(wiring,monkeypatch):
    f,w,db,*_=wiring
    # An original ranking-only V1 row is still readable, but cannot become an executable price.
    ranking,pricing=f.adapter.prepare_pair(f.primary,selection=f.selected,handoffs=f.handoffs,
        references=f.references,price_inputs=f.prices,published_at=PUBLISHED)
    db.rows[ranking['run_id']]=(f.rules.digest(ranking['specification']),f.rules.digest(ranking),ranking)
    from scripts import bull_flag_companion as companion,setup_selection as selected,research_strategies
    monkeypatch.setattr(companion,'_row',lambda conn,run:db.rows.get(run))
    monkeypatch.setattr(selected,'execution_queue',w.native_execution_queue)
    monkeypatch.setattr(research_strategies,'CHALLENGER','test')
    v=w.entry_runtime.unified_handoff({},f.selection.parent_context(f.primary))
    assert v['record']==ranking
    with pytest.raises(ValueError,match='execution_pipeline_limit_price_required'):
        w.execution_queue(v)
    ranking['selection']['selected_signal_ids']=[]
    with pytest.raises(ValueError):w.entry_runtime.unified_handoff({},f.selection.parent_context(f.primary))


@pytest.fixture
def pipeline_guard_case(wiring,monkeypatch):
    from scripts.strategy_order_lifecycle import marker_for
    f,w,db,statuses,checks,runtime=wiring
    f.handoffs['test'].update(strategy_contract={'strategy_id':'test'},
        primary_report_sha256='report',candidate_values=[],batch_sha256='batch',
        session={'signal_session':'2026-11-09','next_session':'2026-11-10',
                 'next_open_at':'2026-11-10T14:30:00+00:00'})
    f.handoffs['test']['execution_policy']['cancel_at']='2026-11-10T09:35:00+00:00'
    publish(wiring)
    parent=f.selection.parent_context(f.primary)
    verified=w.entry_runtime.unified_handoff({},parent)
    handoff=w.execution_queue(verified)[0]['handoff']
    signal={'signal_id':'GOOD_signal','payload':{'handoff':handoff,'account_id':'synthetic'}}
    signal['payload']['entry_lifecycle']=marker_for(handoff,
        {'date':'2026-11-10','open':'09:30:00','close':'16:00:00'})
    namespace=w.entry_runtime.guard_for.__globals__
    source_checks=[]
    monkeypatch.setitem(namespace,'source_guard',lambda *a,**kw:source_checks.append(True))
    monkeypatch.setitem(namespace,'expected_paper_account',lambda env:'synthetic')
    monkeypatch.setitem(namespace,'readonly_handoff',lambda *a,**kw:parent)
    def forbidden_monitor(*a,**kw):raise AssertionError('monitor report must not be read')
    monkeypatch.setitem(namespace,'monitor_ready',forbidden_monitor)
    class Clock:
        value=datetime(2026,11,10,9,0,10,tzinfo=timezone.utc)
        @staticmethod
        def now(tz):return Clock.value
    monkeypatch.setitem(namespace,'datetime',Clock)
    monkeypatch.setenv('APCA_API_BASE_URL','https://paper-api.alpaca.markets')
    monkeypatch.setenv('JBR_EXEC_PAPER','true')
    return w.entry_runtime.guard_for({},pipeline_entry_only=True),signal,namespace,Clock,source_checks


def test_entry_only_pipeline_guard_has_no_monitor_dependency(pipeline_guard_case):
    guard,signal,namespace,clock,source_checks=pipeline_guard_case
    assert guard('entry',signal) is True
    assert source_checks==[True]


@pytest.mark.parametrize('change',['account','ticket','model','missing_marker','changed_marker',
                                    'expired_window','pipeline_selection','source','legacy'])
def test_entry_only_pipeline_guard_retains_safety_checks(pipeline_guard_case,monkeypatch,change):
    guard,signal,namespace,clock,source_checks=pipeline_guard_case
    handoff=signal['payload']['handoff']
    if change=='account':signal['payload']['account_id']='different'
    elif change=='ticket':handoff['pipeline_limit_order']['limit_price']='99'
    elif change=='model':handoff['execution_policy']['model']='NEXT_PREMARKET_0400_FROZEN_LIMIT_V1'
    elif change=='missing_marker':signal['payload'].pop('entry_lifecycle')
    elif change=='changed_marker':signal['payload']['entry_lifecycle']['cancel_at']='2026-11-11T21:00:00+00:00'
    elif change=='expired_window':clock.value=datetime(2026,11,10,9,1,tzinfo=timezone.utc)
    elif change=='pipeline_selection':monkeypatch.setitem(namespace,'pipeline_execution_queue',lambda v:[])
    elif change=='source':
        def reject(*a,**kw):raise ValueError('source_changed')
        monkeypatch.setitem(namespace,'source_guard',reject)
    elif change=='legacy':
        handoff['execution_policy']['model']='NEXT_PREMARKET_0400_FROZEN_LIMIT_V1'
        monkeypatch.setitem(namespace,'unified_handoff',lambda *a,**kw:{'record':{}})
    with pytest.raises(ValueError):guard('entry',signal)
