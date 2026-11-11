from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
from datetime import datetime
from copy import deepcopy
import pytest
from tests.test_shared_pricing_wiring import wiring
from tests.test_isolated_pricing_integration import fixture, Database
from tests.test_pricing_adapter_prototype import contract, load, NATIVE, PUBLISHED
from shared_native_integration import install_shared_native, HELPERS, ENDPOINTS
pytestmark=pytest.mark.alpaca_optional

@pytest.fixture
def installed(fixture,tmp_path,monkeypatch):
 f=fixture;db=Database();events=[]
 source=Path(__file__).resolve().parents[1]
 for name in HELPERS:(tmp_path/name).write_bytes((source/name).read_bytes())
 (tmp_path/'native.py').write_bytes(b'native fixture')
 def pin(path):
  raw=path.read_bytes();return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())
 pins={name:pin(tmp_path/name) for name in HELPERS}
 f.primary.update(batch={'proposals':[]},source_bindings={**pins,'native.py':pin(tmp_path/'native.py')})
 runtime=load('shared_bootstrap_runtime',NATIVE/'strategy_execution_runtime.py')
 before=dict(runtime.__dict__)
 def connect(**kw):events.append(('connect',kw));return db.connect(write=kw.get('write',False))
 class ForbiddenBroker:
  def __init__(self,*a,**kw):raise AssertionError('network forbidden')
 native=SimpleNamespace(setup_selection=f.selection,strategy_pipeline=SimpleNamespace(EXTRA_SOURCES=('native.py',)),
  strategy_execution=f.rules,strategy_execution_runtime=runtime,
  research_store=SimpleNamespace(connect=connect),research_strategies=SimpleNamespace(CHALLENGER='test'),
  strategy_execution_broker=SimpleNamespace(PaperBroker=ForbiddenBroker,BrokerError=RuntimeError),
  bull_flag_companion=SimpleNamespace(_status=lambda *a:events.append(('status',a[0]))))
 import shared_pricing_wiring
 monkeypatch.setattr(shared_pricing_wiring,'capture_prices',lambda *a,**kw:deepcopy(f.prices))
 original={n:getattr(f.selection,n) for n in ENDPOINTS}
 bound=install_shared_native(native,root=tmp_path,adapter_pins=pins,clock=lambda:datetime.fromisoformat(PUBLISHED))
 return f,db,events,native,bound,before,original,tmp_path,pins


def test_install_preserves_native_guard_and_ownership_no_io(installed):
 f,db,events,native,bound,before,original,root,pins=installed
 assert not events and not db.rows
 assert all(native.strategy_execution_runtime.__dict__[k] is v for k,v in before.items())
 assert all(getattr(bound.original,k) is v for k,v in original.items())
 assert set(HELPERS)<=set(native.strategy_pipeline.EXTRA_SOURCES)
 assert bound.proof['ownership_functions_changed'] is False
 with pytest.raises(ValueError,match='duplicate_install'):
  install_shared_native(native,root=root,adapter_pins=pins,clock=lambda:None)


def test_installed_native_publication_and_reader_share_record(installed,monkeypatch):
 f,db,events,native,bound,before,original,root,pins=installed
 status=native.setup_selection.publish(f.primary,env={'synthetic':'only'},base_dir=root)
 assert status['status']=='COMPLETED_UNIFIED_SETUP_SELECTION'
 assert len(db.rows)==1
 c=db.connect(write=False)
 v=native.setup_selection.read_selection(c,bound.original.parent_context(f.primary),now=datetime.fromisoformat(PUBLISHED),base_dir=root)
 assert v['result_sha256']==status['result_sha256']
 assert native.setup_selection.execution_queue(v)[0]['handoff']['pipeline_limit_order']['limit_price']=='6.73'
 assert len(v['record']['ranking_record']['selection']['selected_setups'])==2


def test_unpinned_primary_fails_before_any_connection(installed):
 f,db,events,native,bound,before,original,root,pins=installed
 f.primary['source_bindings'].pop('shared_native_integration.py')
 with pytest.raises(ValueError,match='publication_closure_missing'):
  native.setup_selection.publish(f.primary,env={},base_dir=root)
 assert not events and not db.rows


def test_late_helper_drift_refused(installed):
 f,db,events,native,bound,before,original,root,pins=installed
 (root/'shared_pricing_contract.py').write_bytes(b'changed')
 with pytest.raises(ValueError,match='isolated_adapter_source_changed'):
  native.setup_selection.publish(f.primary,env={},base_dir=root)
 assert not db.rows

