"""Offline recognition regression tests with the real runtime/rules helpers."""
import copy
from types import SimpleNamespace
import pytest
from scripts import strategy_execution as rules
from scripts import strategy_execution_runtime as runtime
from scripts import strategy_entry_preflight as preflight
from scripts.strategy_execution_engine import Engine
pytestmark = pytest.mark.alpaca_optional
D=rules.digest
ns={'validate_entry_recognition':runtime.validate_entry_recognition}
context={'schema':'jbravo.manual-paper-pipeline-entry.v1','max_quantity':1,'paper_orders_authorized':True,'signals':{'PTCT':'signal'}}
original={'account_id':'paper','proposal':{'symbol':'PTCT'},'position':{'symbol':'PTCT','qty':1,'entry_price':'50'},'handoff':{'manual_paper_test':{'context':context,'automatic_sells':False,'context_sha256':D(context)},'execution_policy':{'model':'MANUAL_PAPER_PIPELINE_ENTRY_TEST_V1'}}}
orders=[{'state':'OBSERVED','role':'entry','operation':'POST','client_order_id':'client','observation':{'status':'filled','side':'buy','symbol':'PTCT','id':'order','qty':1,'filled_qty':1,'filled_avg_price':'50'}}]
bindings={'synthetic.py':{'bytes':1,'sha256':'a'*64}}
owner={'schema':'jbravo.execution-entry-recognition.v1','scope':'ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS','model':rules.PIPELINE_LIMIT_MODEL,'signal_id':'signal','original_revision':4,'original_payload_sha256':D(original),'original_orders_sha256':D(orders),'release_source_bindings_sha256':D(bindings),'operation_id':'adopt','entry_order_id':'order','entry_client_order_id':'client'}
payload=dict(original,execution_owner=owner,ownership_history=[owner])
base={'signal':{'signal_id':'signal','state':'OPEN','revision':5,'payload':payload},'orders':orders,'positions':{'PTCT':{'side':'long','qty':1,'avg_entry_price':'50'}},'open_orders':[],'account_id':'paper','current_source_bindings':bindings}
checks=[]
def run(label,edit=None,expected=None):
 x=copy.deepcopy(base)
 if edit:edit(x)
 before=copy.deepcopy(x)
 try:ns['validate_entry_recognition'](**x);actual='PASS'
 except ValueError as e:actual=str(e)
 assert actual==(expected or 'PASS'),(label,actual,expected)
 assert x==before
 checks.append({'test':label,'result':actual,'inputs_unchanged':True})
def compatibility(x,broken=None):
 h=x['signal']['payload']['ownership_history'];old=D(x['current_source_bindings']);x['current_source_bindings']['next']={'bytes':2}
 event={'schema':'jbravo.execution-entry-recognition-release-compat.v1','operation_id':'compat','scope':'ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS','signal_id':'signal','previous_revision':5,'previous_history_sha256':D(h),'previous_release_source_bindings_sha256':old,'release_source_bindings_sha256':D(x['current_source_bindings']),'recorded_at':'2026-11-15T18:00:00+00:00'}
 if broken=='history':event['previous_history_sha256']='broken'
 if broken=='duplicate':event['operation_id']='adopt'
 if broken=='previous':event['previous_release_source_bindings_sha256']='broken'
 h.append(event);x['signal']['revision']=6

def test_recognition_contract_matrix():
    run('valid adopted one-share position')
    run('release drift',lambda x:x['current_source_bindings'].update({'other':{}}),'strategy_recognition_release_changed')
    run('account mismatch',lambda x:x.update(account_id='other'),'strategy_recognition_original_context_invalid')
    run('revision drift',lambda x:x['signal'].update(revision=6),'strategy_recognition_audit_changed')
    run('original payload drift',lambda x:x['signal']['payload']['position'].update(entry_price='51'),'strategy_recognition_audit_changed')
    run('order audit drift',lambda x:x['orders'][0].update(client_order_id='other'),'strategy_recognition_audit_changed')
    run('broker quantity drift',lambda x:x['positions']['PTCT'].update(qty=2),'strategy_recognition_broker_position_changed')
    run('broker price drift',lambda x:x['positions']['PTCT'].update(avg_entry_price='51'),'strategy_recognition_broker_position_changed')
    run('PTCT open order present',lambda x:x['open_orders'].append({'symbol':'PTCT'}),'strategy_recognition_broker_position_changed')
    run('unrelated open order permitted',lambda x:x['open_orders'].append({'symbol':'SPY'}))
    run('scope mismatch',lambda x:x['signal']['payload']['execution_owner'].update(scope='OTHER'),'strategy_recognition_scope_invalid')
    run('valid append-only compatibility history',compatibility)
    run('history digest drift',lambda x:compatibility(x,'history'),'strategy_recognition_compatibility_audit_changed')
    run('duplicate operation',lambda x:compatibility(x,'duplicate'),'strategy_recognition_compatibility_audit_changed')
    run('previous release drift',lambda x:compatibility(x,'previous'),'strategy_recognition_compatibility_audit_changed')

@pytest.mark.parametrize('bad_time', ['2026-11-15T18:00:00', 'not-a-time'])
def test_compatibility_requires_unambiguous_timestamp(bad_time):
    x=copy.deepcopy(base);compatibility(x)
    x['signal']['payload']['ownership_history'][-1]['recorded_at']=bad_time
    with pytest.raises(ValueError, match='execution_ambiguous_timestamp'):
        runtime.validate_entry_recognition(**x)

@pytest.mark.parametrize('bad_qty', [True, False, 'NaN', 'Infinity', -1])
def test_real_numeric_helper_rejects_invalid_broker_quantity(bad_qty):
    x=copy.deepcopy(base);x['positions']['PTCT']['qty']=bad_qty
    with pytest.raises(ValueError, match='execution_invalid_number'):
        runtime.validate_entry_recognition(**x)

def test_full_readonly_preflight_uses_restored_validator():
    x=copy.deepcopy(base)
    order=x['orders'][0]['observation']
    order.update(client_order_id='client',type='limit',time_in_force='day',extended_hours=True,limit_price='50')
    x['orders'][0]['request']={'order':dict(symbol='PTCT',side='buy',type='limit',qty=1,time_in_force='day',extended_hours=True,limit_price='50')}
    x['signal']['payload']['execution_owner']['original_orders_sha256']=D(x['orders'])
    reads=[]
    class Store:
        account_id='paper'
        def signals(self,**kwargs):reads.append('signals');return [copy.deepcopy(x['signal'])]
        def orders(self,sid):reads.append('orders');return copy.deepcopy(x['orders'])
    class Broker:
        paper=True;account_id='paper'
        def account(self):reads.append('account');return {'id':'paper'}
        def positions(self):reads.append('positions');return copy.deepcopy(x['positions'])
        def open_orders(self):reads.append('open_orders');return []
        def find_by_client_order_id(self,client):reads.append('find_order');assert client=='client';return copy.deepcopy(order)
        def submit_order(self,*a,**k):raise AssertionError('no order action allowed')
        cancel_order=submit_order
        replace_order=submit_order
    facade=SimpleNamespace(source_guard=lambda *a,**k:None,
                           validate_entry_recognition=runtime.validate_entry_recognition)
    before=copy.deepcopy(x)
    result=preflight.broker_entry_preflight(facade,Engine,Store(),Broker(),base_dir=None,
                handoff={'source_bindings':x['current_source_bindings']},proposed_symbols=['NEW'])
    assert result['status']=='pass' and result['mutations']==0
    assert result['recognized_existing_positions'][0]['symbol']=='PTCT'
    assert result['scope']=='FRESH_BROKER_REGULAR_HOURS_STOPS_ONLY'
    assert x==before and reads.count('positions')==2
