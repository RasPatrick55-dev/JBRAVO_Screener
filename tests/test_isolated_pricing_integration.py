"""Offline contract integration; fake transaction port, native ranking reader."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from types import SimpleNamespace
import pytest
from tests.test_pricing_adapter_prototype import contract,ref,SESSION,PUBLISHED,load,NATIVE
from isolated_pricing_integration import IsolatedPricing,PairedResearchStore,entry_facade,primary_main,finish_handoff_adapter

pytestmark=pytest.mark.alpaca_optional
BIND={'adapter.py':{'bytes':1,'sha256':'a'*64}}


@pytest.fixture
def fixture(contract,monkeypatch):
    c,rules,selection=contract
    primary=dict(run_id='primary',source_bindings={'native.py':{'bytes':1,'sha256':'b'*64}},session=SESSION)
    parent=selection.parent_context if hasattr(selection,'parent_context') else None
    selected=dict(selected_setups=[dict(rank=i+1,proposal=dict(symbol=s,signal_id=s+'_signal',strategy_id='test',**SESSION)) for i,s in enumerate(('GOOD','NONE'))],selected_signal_ids=['GOOD_signal','NONE_signal'])
    handoffs={'test':dict(primary_run_id='primary',pipeline_limit_placement_v2=True,inspection_mode=False,execution_policy=dict(entry_open_at='2026-11-10T09:00:00+00:00'))}
    references={'test':'result-pin'}
    selection.model_sha256=lambda:'model-pin'
    adapter=IsolatedPricing(rules=rules,selection=selection,contract=c,adapter_binding=BIND,clock=lambda:datetime.fromisoformat(PUBLISHED))
    prices={'GOOD':ref(contract,'GOOD'),'NONE':ref(contract,'NONE',regular=False)}
    ranking,pricing=adapter.prepare_pair(primary,selection=selected,handoffs=handoffs,references=references,price_inputs=prices,published_at=PUBLISHED)
    rows={x['run_id']:(rules.digest(x['specification']),rules.digest(x),deepcopy(x)) for x in (ranking,pricing)}
    from scripts import bull_flag_companion
    monkeypatch.setattr(bull_flag_companion,'_row',lambda conn,run:rows.get(run))
    selection.inputs=lambda *a,**k:(deepcopy(handoffs),deepcopy(references),deepcopy(selected))
    parent=dict(primary_run_id='primary',primary_report_sha256=rules.digest(primary),source_bindings=primary['source_bindings'],session=SESSION)
    def read():return adapter.read_verified(None,parent,native_read=selection.read_selection,row_read=lambda conn,run:rows.get(run),base_dir='unused',now=datetime(2026,11,10,8,tzinfo=timezone.utc))
    return SimpleNamespace(adapter=adapter,rules=rules,selection=selection,primary=primary,selected=selected,handoffs=handoffs,references=references,prices=prices,ranking=ranking,pricing=pricing,rows=rows,read=read)


def test_real_native_reader_entry_queue_and_dashboard_share_verified_partition(fixture):
    f=fixture;original=deepcopy(f.ranking);v=f.read()
    assert v['record']==original and 'execution_orders' not in original
    q=f.adapter.execution_queue(v,native_queue=f.selection.execution_queue,challenger='test')
    assert [r['proposal']['signal_id'] for r in q]==['GOOD_signal']
    h=q[0]['handoff'];assert h['pipeline_limit_order']['limit_price']=='6.73'
    assert h['source_bindings'] if 'source_bindings' in h else True
    assert h['unified_selection']['pricing_result_sha256']==f.rules.digest(f.pricing)
    assert h['unified_selection']['selected_signal_ids']==['GOOD_signal','NONE_signal']
    assert h['handoff_id']==f.rules.digest({k:x for k,x in h.items() if k!='handoff_id'})
    d=f.adapter.dashboard(v)
    assert d['selected_count']==2 and d['executable_count']==1 and d['excluded_count']==1
    assert d['entry_ready'] is False and d['order_submission_authorized'] is False
    assert f.ranking==original


@pytest.mark.parametrize('kind',['missing','wrong_hash','wrong_adapter','wrong_parent','wrong_price','future'])
def test_reader_rejects_missing_or_tampered_artifacts_even_if_rehashed(fixture,kind):
    f=fixture;k=f.pricing['run_id'];spec,result,p=f.rows[k]
    if kind=='missing':del f.rows[k]
    elif kind=='wrong_hash':f.rows[k]=(spec,'0'*64,p)
    else:
        p=deepcopy(p)
        if kind=='wrong_adapter':p['specification']['adapter_binding']={'different':BIND['adapter.py']}
        if kind=='wrong_parent':p['review']['parent_selection_sha256']='0'*64
        if kind=='wrong_price':p['review']['pricing']['execution_orders']['GOOD_signal']['limit_price']='88.88'
        if kind=='future':p['review']['published_at']='2027-01-01T03:30:00+00:00'
        f.rows[k]=(f.rules.digest(p['specification']),f.rules.digest(p),p)
    with pytest.raises(ValueError):f.read()


def test_native_reconstruction_rejects_rank_tampering(fixture):
    f=fixture;k=f.ranking['run_id'];_,_,r=f.rows[k];r=deepcopy(r);r['selection']['selected_setups'].reverse()
    f.rows[k]=(f.rules.digest(r['specification']),f.rules.digest(r),r)
    with pytest.raises(ValueError,match='ranking_changed'):f.read()


def test_queue_guard_and_dashboard_reject_postverification_mutation(fixture):
    f=fixture;v=f.read();v['isolated_pricing']['review']['pricing']['execution_orders']['GOOD_signal']['limit_price']='88.88'
    with pytest.raises(ValueError):f.adapter.execution_queue(v,native_queue=f.selection.execution_queue,challenger='test')
    with pytest.raises(ValueError):f.adapter.dashboard(v)


def test_primary_failure_never_reports_complete_and_reveals_no_raw_error(fixture):
    f=fixture;statuses=[];writes=[]
    def failed(*a):raise ValueError('secret-never-export')
    status=f.adapter.publish_primary(f.primary,inputs=lambda p:(f.handoffs,f.references,f.selected),capture=failed,
        check_sources=lambda p:None,publish_pair=lambda p:writes.append(p),report_status=statuses.append)
    assert status['status']=='UNIFIED_SETUP_SELECTION_UNAVAILABLE' and not writes
    assert 'secret' not in str(statuses)


def test_primary_success_only_after_verified_pair_publication(fixture):
    f=fixture;events=[]
    status=f.adapter.publish_primary(f.primary,inputs=lambda p:(f.handoffs,f.references,f.selected),capture=lambda *a:f.prices,
        check_sources=lambda p:events.append('source_check'),publish_pair=lambda p:events.append('publish_readback'),report_status=lambda p:events.append('status'))
    assert status['status']=='COMPLETED_UNIFIED_SETUP_SELECTION'
    assert events==['source_check','publish_readback','status']


class Database:
    def __init__(self,fail_insert=None,fail_read=False):self.rows={};self.connections=[];self.fail_insert=fail_insert;self.fail_read=fail_read
    def connect(self,*,write):
        db=self
        class Conn:
            def __init__(self):self.staged={};self.commits=0;self.rollbacks=0;self.closed=False;self.result=None
            def cursor(self):return self
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def execute(self,sql,args=None):
                if sql.startswith('INSERT'):
                    if len(self.staged)==db.fail_insert:raise RuntimeError('insert failure')
                    run,spec,digest,body=args
                    if run in db.rows or run in self.staged:raise RuntimeError('duplicate')
                    self.staged[run]=(spec,digest,json.loads(body))
                elif sql.startswith('SELECT'):self.result=None if db.fail_read else db.rows.get(args[0])
            def fetchone(self):return self.result
            def commit(self):self.commits+=1;db.rows.update(self.staged);self.staged={}
            def rollback(self):self.rollbacks+=1;self.staged={}
            def close(self):self.closed=True
        conn=Conn();self.connections.append(conn);return conn


def test_two_records_atomic_and_independently_read_back(fixture):
    f=fixture;db=Database();s=PairedResearchStore(db.connect,f.rules.digest)
    s.publish((f.ranking,f.pricing))
    assert len(db.rows)==2 and len(db.connections)==2
    assert db.connections[0].commits==1 and all(c.closed for c in db.connections)
    with pytest.raises(RuntimeError,match='duplicate'):s.publish((f.ranking,f.pricing))
    assert len(db.rows)==2


def test_second_insert_failure_rolls_back_first(fixture):
    f=fixture;db=Database(fail_insert=1);s=PairedResearchStore(db.connect,f.rules.digest)
    with pytest.raises(RuntimeError):s.publish((f.ranking,f.pricing))
    assert db.rows=={} and db.connections[0].commits==0 and db.connections[0].rollbacks==1


def test_ambiguous_postcommit_readback_fails_without_replay(fixture):
    f=fixture;db=Database(fail_read=True);s=PairedResearchStore(db.connect,f.rules.digest)
    with pytest.raises(ValueError,match='readback_failed'):s.publish((f.ranking,f.pricing))
    assert len(db.rows)==2 and len(db.connections)==2 and db.connections[0].commits==1


def test_entry_facade_clones_native_guard_code_without_global_replacement(fixture):
    native=load('integration_runtime',NATIVE/'strategy_execution_runtime.py')
    original=dict(native.__dict__);read=lambda *a,**k:fixture.read();queue=lambda v:fixture.adapter.execution_queue(v,native_queue=fixture.selection.execution_queue,challenger='test')
    facade=entry_facade(native,read_verified=read,queue=queue)
    assert facade.guard_for.__code__ is native.guard_for.__code__
    assert facade.guard_for.__globals__['unified_handoff'] is read
    assert native.guard_for.__globals__['unified_handoff'] is native.unified_handoff
    assert facade.source_guard is native.source_guard and facade.store_connection is native.store_connection
    assert all(native.__dict__[k] is v for k,v in original.items())


def test_primary_entrypoint_preserves_native_outer_completion_contract():
    events=[]
    def finish_strategy_handoff():events.append('old')
    namespace={'finish_strategy_handoff':finish_strategy_handoff}
    exec('def pipeline_main(argv): return finish_strategy_handoff()',namespace)
    pipeline_main=namespace['pipeline_main']
    def outer(argv,*,pipeline_main):
        events.append('native_primary');return pipeline_main(['--handoff-invocation','id'])
    native=SimpleNamespace(main=pipeline_main,enabled=lambda:True)
    run=primary_main(SimpleNamespace(main=outer),native,finish_handoff=lambda:events.append('new') or 0)
    assert run([])==0 and events==['native_primary','new']
    assert native.main.__globals__['finish_strategy_handoff'] is finish_strategy_handoff
    with pytest.raises(ValueError):run(['--manual-provisional-test'])


def test_companion_failure_blocks_isolated_pricing_publication():
    events=[]
    finish=finish_handoff_adapter(SimpleNamespace(required_strategies=lambda p:['primary','other']),prepare_primary=lambda *a,**k:{'status':'FAILED'},
        publish=lambda *a,**k:events.append('publish'),status=lambda *a:events.append('status'),digest=lambda v:'digest',challenger='primary')
    assert finish({'run_id':'x'},env={},base_dir='none')['status']=='UNIFIED_SETUP_SELECTION_UNAVAILABLE'
    assert events==['status']



def test_entry_runner_wiring_uses_facade_without_constructing_broker_or_engine():
    from isolated_pricing_integration import install_entry_runner
    calls=[];runtime=SimpleNamespace(ROOT='root');facade=object();engine=object()
    entry=SimpleNamespace(run_entry_only=lambda config,**kw:calls.append(kw) or 0)
    lifecycle=SimpleNamespace(monitor_engine=lambda e:calls.append(e),marker_for=object())
    proof=install_entry_runner(runtime,facade=facade,NativeEngine=engine,entry=entry,
        monitor=SimpleNamespace(calendar_session=object()),lifecycle=lifecycle,preflight=SimpleNamespace(broker_entry_preflight=object()))
    assert calls==[engine] and proof['entry_only']
    assert runtime._run_entry('config',progress={})==0
    assert calls[-1]['runtime'] is facade
    with pytest.raises(ValueError,match='duplicate_install'):
        install_entry_runner(runtime,facade=facade,NativeEngine=engine,entry=entry,
            monitor=None,lifecycle=lifecycle,preflight=None)


def test_dashboard_adapter_preserves_native_api_code_and_never_promotes_blocked_readiness(fixture):
    from isolated_pricing_integration import dashboard_adapter
    f=fixture;native=load('integration_dashboard',NATIVE/'strategy_dashboard.py')
    native.display_selection=lambda *a,**k:dict(f.read(),primary=f.primary)
    native.handoff_readiness=lambda *a,**k:dict(entry_ready=False,entry_block_reason='expired',entry_block_reasons=['expired'],order_submission_authorized=False)
    parent=lambda p:dict(primary_run_id=p['run_id'],primary_report_sha256=f.rules.digest(p),source_bindings=p['source_bindings'],session=p['session'])
    wrapped=dashboard_adapter(native,f.adapter,row_read=lambda conn,run:f.rows.get(run),parent_context=parent)
    v=wrapped.display_selection(None,None,'primary',now=datetime(2026,11,10,8,tzinfo=timezone.utc))
    result=wrapped.handoff_readiness(v,[])
    assert result['entry_ready'] is False and result['entry_block_reason']=='expired' and result['excluded_count']==1
    assert wrapped.snapshot.__code__ is native.snapshot.__code__
    assert wrapped.api_payload.__code__ is native.api_payload.__code__
    assert native.api_payload.__globals__['snapshot'] is native.snapshot


def test_capture_is_bounded_sip_and_never_uses_raw_trade_price(contract):
    from isolated_pricing_integration import capture_prices
    from tests.test_pricing_adapter_prototype import bar,OBSERVED
    c,rules,_=contract;calls=[]
    replies=[{'bars':{},'next_page_token':None},{'bars':{'X':[bar('2026-11-09T20:59:00Z',6.7)]},'next_page_token':None},
        {'trades':{'X':[{'t':'2026-11-09T22:00:00Z','p':99,'s':1,'c':['I','T']}]},'next_page_token':None}]
    def get(path,params):calls.append((path,params));return replies.pop(0)
    result=capture_prices({'selected_setups':[{'proposal':{'symbol':'X'}}]},{'session':SESSION},rules=rules,contract=c,get=get,now=lambda:datetime.fromisoformat(OBSERVED))
    assert result['X']['price']=='6.7' and result['X']['reference_kind']=='REGULAR_SESSION_FALLBACK'
    assert len(calls)==3 and all(p['feed']=='sip' for _,p in calls)
    assert calls[1][1]['start'].endswith('15:45:00-05:00')


@pytest.mark.parametrize('response',[{'bars':{}},{'bars':{},'next_page_token':'partial'},{'bars':{'OTHER':[]},'next_page_token':None}])
def test_incomplete_afterhours_never_triggers_fallback(contract,response):
    from isolated_pricing_integration import capture_prices
    from tests.test_pricing_adapter_prototype import OBSERVED
    c,rules,_=contract;calls=[]
    with pytest.raises(ValueError):capture_prices({'selected_setups':[{'proposal':{'symbol':'X'}}]},{'session':SESSION},rules=rules,contract=c,
        get=lambda *a:calls.append(a) or response,now=lambda:datetime.fromisoformat(OBSERVED))
    assert len(calls)==1


def test_empty_selection_makes_no_data_request(contract):
    from isolated_pricing_integration import capture_prices
    c,rules,_=contract
    assert capture_prices({'selected_setups':[]},{'session':SESSION},rules=rules,contract=c,
        get=lambda *a:pytest.fail('unexpected GET'),now=lambda:None)=={}


def test_recomputed_supplement_cannot_shift_publication_time(fixture):
    f=fixture;p=deepcopy(f.pricing)
    p['review']=f.adapter.contract.prepare_review(f.ranking,f.prices,published_at='2026-11-10T03:31:00+00:00')
    f.rows[p['run_id']]=(f.rules.digest(p['specification']),f.rules.digest(p),p)
    with pytest.raises(ValueError,match='publication_time_changed'):f.read()


def test_supplemental_pins_verified_from_independent_files(tmp_path):
    from hashlib import sha256
    from isolated_pricing_integration import verify_adapter_files
    source=tmp_path/'adapter.py';source.write_bytes(b'original')
    pins={'adapter.py':{'bytes':8,'sha256':sha256(b'original').hexdigest()}}
    assert verify_adapter_files(tmp_path,pins)==pins
    source.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='source_changed'):verify_adapter_files(tmp_path,pins)


def test_supplemental_pin_path_cannot_escape_root(tmp_path):
    from isolated_pricing_integration import verify_adapter_files
    root=tmp_path/'root';root.mkdir();(tmp_path/'outside').write_bytes(b'x')
    with pytest.raises(ValueError,match='path_invalid'):verify_adapter_files(root,{'../outside':{}})

@pytest.mark.parametrize('http,reason',[(403,'RAW_TRADES_HTTP_ERROR'),(None,'RAW_TRADES_TRANSPORT_ERROR')])
def test_raw_trade_failure_keeps_typed_diagnostic_and_eligible_fallback(contract,http,reason):
    from isolated_pricing_integration import capture_prices
    from tests.test_pricing_adapter_prototype import bar,OBSERVED
    from scripts.strategy_execution_broker import BrokerError
    c,rules,_=contract;replies=[{'bars':{},'next_page_token':None},{'bars':{'X':[bar('2026-11-09T20:59:00Z',6.7)]},'next_page_token':None}]
    def get(path,params):
        if replies:return replies.pop(0)
        raise BrokerError('private detail',http_status=http)
    result=capture_prices({'selected_setups':[{'proposal':{'symbol':'X'}}]},{'session':SESSION},rules=rules,contract=c,get=get,
        now=lambda:datetime.fromisoformat(OBSERVED),broker_error_type=BrokerError)
    assert result['X']['price']=='6.7'
    encoded=json.dumps(result)
    assert reason in encoded and 'ACTIVITY_UNKNOWN' in encoded and 'private detail' not in encoded
