from copy import deepcopy
from datetime import datetime,timezone
import json
import pytest
from tests.test_isolated_pricing_integration import fixture
from tests.test_pricing_adapter_prototype import contract,ref,PUBLISHED
from tests.test_actual_web_binding import WEB
from tests.test_pricing_adapter_prototype import load
from shared_pricing_contract import SharedPricingContract

pytestmark=pytest.mark.alpaca_optional

@pytest.fixture
def shared(fixture):
 f=fixture;c=SharedPricingContract(f.adapter)
 r=c.prepare(f.primary,selection=f.selected,handoffs=f.handoffs,references=f.references,price_inputs=f.prices,published_at=PUBLISHED)
 parent=dict(primary_run_id=f.primary['run_id'],primary_report_sha256=f.rules.digest(f.primary),source_bindings=f.primary['source_bindings'],session=f.primary['session'])
 def read(record=r):
  row=(f.rules.digest(record['specification']),f.rules.digest(record),record)
  return c.read_selection(None,parent,row_read=lambda *_:row,now=datetime(2026,11,10,8,tzinfo=timezone.utc),base_dir='unused')
 return f,c,r,read

def test_shared_queue_preserves_ranking_and_matches_execution_projection(shared):
 f,c,r,read=shared;v=read();q=c.execution_queue(v,challenger='test')
 assert r['ranking_record']['selection']==f.selected
 assert r['selection']['selected_signal_ids']==['GOOD_signal']
 assert [s['proposal']['signal_id'] for s in q]==['GOOD_signal']
 assert q[0]['handoff']['unified_selection']['result_sha256']==f.rules.digest(r)
 assert q[0]['handoff']['unified_selection']['selected_signal_ids']==['GOOD_signal']
 audit=c.audit(v)
 assert audit['dispositions']['NONE_signal']['status']=='EXCLUDED_NO_ELIGIBLE_REFERENCE'
 assert len(audit['ranked_selection']['selected_setups'])==2

@pytest.mark.parametrize('attack',['selection','ranking','prices','source','session','adapter','authority'])
def test_rehashed_tampering_refused(shared,attack):
 f,c,r,read=shared;r=deepcopy(r)
 if attack=='selection':r['selection']['selected_signal_ids'].append('NONE_signal')
 elif attack=='ranking':r['ranking_record']['selection']['selected_setups'].reverse()
 elif attack=='prices':r['pricing_record']['review']['pricing']['execution_orders']['GOOD_signal']['limit_price']='99'
 elif attack=='source':r['ranking_record']['source_bindings']={}
 elif attack=='session':r['ranking_record']['session']['signal_session']='2026-11-08'
 elif attack=='adapter':r['pricing_record']['specification']['adapter_binding']={}
 elif attack=='authority':r['order_submission_authorized']=True
 with pytest.raises(ValueError):read(r)

def test_existing_web_readiness_and_shared_queue_agree_all_unpriceable(shared,monkeypatch):
 f,c,r,read=shared
 f.prices['GOOD']=ref((f.adapter.contract,f.rules,f.selection),'GOOD',regular=False)
 r=c.prepare(f.primary,selection=f.selected,handoffs=f.handoffs,references=f.references,price_inputs=f.prices,published_at=PUBLISHED)
 v=read(r);assert c.execution_queue(v,challenger='test')==[]
 web=load('unchanged_web_shared_contract',WEB)
 from scripts import strategy_pipeline as pipeline,strategy_execution as execution,pipeline_postflight as health
 monkeypatch.setattr(health,'_read',lambda *a:json.dumps(f.primary))
 monkeypatch.setattr(pipeline,'check_report',lambda *a,**k:None)
 monkeypatch.setattr(execution,'prepare_handoff',lambda *a,**k:None)
 v['primary']=f.primary
 status=web.handoff_readiness(v,[],now=datetime(2026,11,10,8,tzinfo=timezone.utc))
 assert status['entry_ready'] is False
 assert 'unified_selection_no_buy_signals' in status['entry_block_reasons']
 assert len(c.audit(v)['dispositions'])==2

def test_verified_shared_record_mutation_refused(shared):
 f,c,r,read=shared;v=read();v['record']['selection']['selected_signal_ids']=[]
 with pytest.raises(ValueError):c.execution_queue(v,challenger='test')


def test_dashboard_retains_exclusions_and_same_ticket_prices(shared):
 f,c,r,read=shared
 for setup in f.selected['selected_setups']:
  setup['matching_strategies']=['test']
  setup['proposal'].update(signal_close_raw=6.5,setup_ranking={'score':80,'signal_volume_raw':100})
 r=c.prepare(f.primary,selection=f.selected,handoffs=f.handoffs,references=f.references,price_inputs=f.prices,published_at=PUBLISHED)
 v=read(r);v['primary']=dict(f.primary,candidate_run_ts_utc=PUBLISHED)
 rows=c.dashboard_rows(v);queue=c.execution_queue(v,challenger='test')
 assert [row['symbol'] for row in rows]==['GOOD','NONE']
 assert rows[0]['execution_eligible'] is True
 assert rows[0]['execution_limit_price']==queue[0]['handoff']['pipeline_limit_order']['limit_price']=='6.73'
 assert rows[1]['execution_eligible'] is False
 assert rows[1]['execution_limit_price'] is None
 assert rows[1]['pricing_disposition']['status']=='EXCLUDED_NO_ELIGIBLE_REFERENCE'
 assert all(row['entry_price']==6.5 for row in rows) # signal evidence is not relabeled
 assert v['record']==r


def test_dashboard_rejects_post_verification_mutation(shared):
 f,c,r,read=shared;v=read();v['record']['selection']['selected_signal_ids']=[]
 with pytest.raises(ValueError,match='shared_verified_record_changed'):c.dashboard_rows(v)


def test_reports_preserve_excluded_ranking_through_existing_web_projection(shared,monkeypatch):
 f,c,r,read=shared
 for setup in f.selected['selected_setups']:
  setup['proposal'].update(strategy_version='1',setup_ranking={'score':80})
 r=c.prepare(f.primary,selection=f.selected,handoffs=f.handoffs,references=f.references,price_inputs=f.prices,published_at=PUBLISHED)
 verified=read(r)
 reports={'test':{'metrics':{'result_id':'metrics','rows':[
  {'signal_id':s+'_signal','symbol':s,'trades':0} for s in ('GOOD','NONE')]},
  'backtest':{'result_id':'backtest'}}}
 rendered=c.dashboard_reports('metrics',{},verified['record']['selection'],reports,{})
 assert [row['symbol'] for row in rendered['rows']]==['GOOD','NONE']
 assert verified['record']['selection']['selected_signal_ids']==['GOOD_signal']
