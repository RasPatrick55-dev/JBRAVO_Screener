import argparse,ast,copy,hashlib,importlib.util,json,logging,os,re,sys,tempfile,types,unittest,uuid
from pathlib import Path
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from typing import Any,Dict,Iterable,List,Optional,Tuple,Mapping
from unittest import mock
R=Path(__file__).resolve().parents[1];C=R/'scripts'
AUDIT={'network':0,'process':0,'secret':0,'denied_checkout':0}
def audit(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):AUDIT['network']+=1;raise AssertionError('NETWORK_FORBIDDEN')
 if event in ('subprocess.Popen','os.system'):AUDIT['process']+=1;raise AssertionError('PROCESS_FORBIDDEN')
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=str(args[0]).lower().replace('\\','/')
  if p.endswith('/.env') or '/.config/jbravo/' in p:AUDIT['secret']+=1;raise AssertionError('SECRET_FORBIDDEN')
  if '/jbravogit/' in p or '/jbravo-worktrees/' in p:AUDIT['denied_checkout']+=1;raise AssertionError('DENIED_CHECKOUT')
if __name__=='__main__':sys.addaudithook(audit)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
I=load('offline_activity_checkpoint',C/'activity_ingestion_checkpoint.py')
scripts=types.ModuleType('scripts');scripts.activity_ingestion_checkpoint=I;scripts.__path__=[]
with mock.patch.dict(sys.modules,{'scripts':scripts}):S=load('offline_account_sync',C/'account_sync.py')
U=datetime(2026,10,24,12,tzinfo=timezone.utc);B=U-timedelta(days=1);A=U-timedelta(days=14,seconds=300);REQUIRED=U-timedelta(days=14)
W='a'*32;SOURCE=I.source_fingerprint(I.ENDPOINT,'00000000-0000-0000-0000-000000000001')
def row(key='one',amount=None):
 return dict(activity_id=key,activity_type='FILL',transaction_time=B,symbol='TEST',side='buy',qty=Decimal('2'),price=Decimal('10.5'),amount=amount,order_id='00000000-0000-0000-0000-000000000002',description='SYNTHETIC',raw=json.dumps({'id':key,'activity_type':'FILL','qty':'2','price':'10.5'}))
class Database:
 def __init__(self,before=B):
  self.autocommit=False;self.committed={'cursor':before,'ledger':{},'checkpoints':{}};self.active=False;self.sql=[];self.fail_prefix=None;self.bad_readback=False;self.bad_ledger=False;self.commit_error=None;self.duplicate_race=False;self.rollback_count=0;self.commit_count=0
 @property
 def data(self):return self.pending if self.active else self.committed
 def __enter__(self):
  if self.active:raise AssertionError('NESTED_TRANSACTION')
  self.pending=copy.deepcopy(self.committed);self.active=True;return self
 def __exit__(self,kind,*args):
  if kind:self.rollback_count+=1
  elif self.commit_error=='before':self.active=False;raise RuntimeError('synthetic commit failed')
  else:
   self.committed=self.pending;self.commit_count+=1
   if self.commit_error=='after':self.active=False;raise RuntimeError('synthetic commit acknowledgement lost')
  self.active=False
 def cursor(self):return Cursor(self)
class Cursor:
 def __init__(self,db):self.db=db;self.rows=[]
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def fetchall(self):return self.rows
 def execute(self,sql,args=()):
  d=self.db;d.sql.append((sql,args));self.rows=[]
  if d.fail_prefix and sql.startswith(d.fail_prefix):raise RuntimeError('synthetic database fault')
  st=d.data
  if sql.startswith('SET LOCAL') or sql.startswith('SELECT pg_catalog.pg_advisory'):return
  if sql.startswith('SELECT source_fingerprint FROM'):
   q=sorted(st['checkpoints'].values(),key=lambda x:x['until']);self.rows=[(q[-1]['source'],)] if q else [];return
  if sql.startswith('SELECT last_after FROM'):
   self.rows=[(st['cursor'],)] if st['cursor'] is not None else [];return
  if sql.startswith('SELECT activity_id,activity_type,'):
   keys=list(zip(args[::2],args[1::2]));self.rows=[tuple(st['ledger'][k][n] for n in I.FIELDS) for k in keys if k in st['ledger']]
   if d.bad_ledger and st['ledger']:self.rows=[]
   return
  if sql.startswith('INSERT INTO public.alpaca_activities'):
   for offset in range(0,len(args),len(I.FIELDS)):
    v=dict(zip(I.FIELDS,args[offset:offset+len(I.FIELDS)]));v['raw']=json.loads(v['raw']);key=(v['activity_id'],v['activity_type'])
    if d.duplicate_race and key not in st['ledger']:st['ledger'][key]=v;d.duplicate_race=False
    if key not in st['ledger']:st['ledger'][key]=v;self.rows.append(key)
   return
  if sql.startswith('INSERT INTO public.activity_ingestion_checkpoints'):
   key,source,recipe,axis,after,until,count,manifest,sha,observed,before,after_cursor=args
   if key in st['checkpoints']:raise RuntimeError('synthetic duplicate checkpoint')
   st['checkpoints'][key]=dict(source=source,recipe=recipe,axis=axis,after=after,until=until,count=count,manifest=json.loads(manifest),sha=sha,cursor_after=after_cursor);return
  if sql.startswith('INSERT INTO public.reconcile_state'):
   until,now,before=args
   if st['cursor']==before:st['cursor']=until;self.rows=[(until,)]
   return
  if sql.startswith('SELECT source_fingerprint,query_after'):
   v=st['checkpoints'].get(args[0]);self.rows=[(v['source'],v['after'],v['until'],v['count']+(1 if d.bad_readback else 0),v['manifest'],v['sha'],v['cursor_after'])] if v else [];return
  if sql.startswith('SELECT source_fingerprint,query_axis'):
   v=st['checkpoints'].get(args[0]);self.rows=[(v['source'],v['axis'],v['after'],v['until'],v['count'],v['manifest'],v['sha'],v['cursor_after'],v['recipe'])] if v else [];return
  raise AssertionError('UNRECOGNIZED_SQL:'+sql)
def persist(db,rows=None,**overrides):
 values=dict(workflow_id=W,source=SOURCE,after=A,until=U,before=B,rows=[row()] if rows is None else rows,policy_lock=lambda c:None);values.update(overrides);return I.persist(db,**values)
def consume(db,**overrides):
 values=dict(workflow_id=W,source=SOURCE,required_after=REQUIRED,required_until=U,now=U+timedelta(seconds=30));values.update(overrides);return I.consume(db,**values)
def contract(proof):
 return '\n'.join(('[INFO] RECONCILE_START open_trades=0','[INFO] RECONCILE_OPERATIONAL_COMPLETE scope=ACTIVE_RECONCILIATION policy_revision=2 excluded_count=1 recovery_pending_case_ids=[]','[INFO] RECONCILE_INGESTION_CHECKPOINT '+I.encoded(proof),'[WARN] RECONCILE_WATERMARK_HELD reason=durable_ingestion_proved_accounting_held','[INFO] RECONCILE_END closed=0 decorated=0 operational_complete=true recovery_complete=false accounting_complete=false activity_ingestion_complete=true watermark_advanced=false'))
class Atomic(unittest.TestCase):
 def test_later_required_cutoff_rejects_earlier_checkpoint(self):
  d=Database();persist(d)
  with self.assertRaisesRegex(RuntimeError,'COVERAGE_INSUFFICIENT'):consume(d,required_until=U+timedelta(seconds=1))
 def test_earlier_required_cutoff_rejects_different_checkpoint(self):
  d=Database();persist(d)
  with self.assertRaisesRegex(RuntimeError,'COVERAGE_INSUFFICIENT'):consume(d,required_until=U-timedelta(seconds=1))
 def test_future_workflow_cutoff_refuses(self):
  with self.assertRaisesRegex(RuntimeError,'CUTOFF_FUTURE'):I.bound_cutoff(W,U+timedelta(seconds=1),U)
 def test_missing_workflow_cutoff_refuses(self):
  with self.assertRaisesRegex(RuntimeError,'CUTOFF_INVALID'):I.bound_cutoff(W,None,U)
 def test_naive_workflow_cutoff_refuses(self):
  with self.assertRaisesRegex(RuntimeError,'CUTOFF_INVALID'):I.bound_cutoff(W,'2026-10-24T12:00:00',U)
 def test_commit_is_atomic_and_consumable(self):
  d=Database();r=persist(d);self.assertEqual(r['inserted'],1);self.assertEqual(d.commit_count,1);self.assertEqual(d.committed['cursor'],U);self.assertEqual(consume(d)['rows'],1)
 def test_ledger_insert_fault_rolls_back_all(self):
  d=Database();d.fail_prefix='INSERT INTO public.alpaca_activities'
  with self.assertRaises(RuntimeError):persist(d)
  self.assertEqual(d.committed,Database().committed);self.assertEqual(d.rollback_count,1)
 def test_checkpoint_insert_fault_rolls_back_ledger_and_cursor(self):
  d=Database();d.fail_prefix='INSERT INTO public.activity_ingestion_checkpoints'
  with self.assertRaises(RuntimeError):persist(d)
  self.assertEqual(d.committed,Database().committed)
 def test_cursor_fault_rolls_back_checkpoint_and_ledger(self):
  d=Database();d.fail_prefix='INSERT INTO public.reconcile_state'
  with self.assertRaises(RuntimeError):persist(d)
  self.assertEqual(d.committed,Database().committed)
 def test_readback_failure_rolls_back_everything(self):
  d=Database();d.bad_readback=True
  with self.assertRaisesRegex(RuntimeError,'CHECKPOINT_READBACK'):persist(d)
  self.assertEqual(d.committed,Database().committed)
 def test_ledger_readback_failure_rolls_back(self):
  d=Database();d.bad_ledger=True
  with self.assertRaisesRegex(RuntimeError,'LEDGER_READBACK'):persist(d)
  self.assertEqual(d.committed,Database().committed)
 def test_duplicate_original_is_not_inserted_or_changed(self):
  d=Database();v=row();v['raw']=json.loads(v['raw']);d.committed['ledger'][('one','FILL')]=v;original=copy.deepcopy(v)
  result=persist(d);self.assertEqual(result['inserted'],0);self.assertEqual(result['matched_existing'],1);self.assertEqual(d.committed['ledger'][('one','FILL')],original)
  self.assertFalse(any(q.startswith('INSERT INTO public.alpaca_activities') for q,_ in d.sql))
 def test_existing_conflict_refuses_without_overwrite(self):
  d=Database();v=row();v['price']=Decimal('11');v['raw']=json.loads(v['raw']);d.committed['ledger'][('one','FILL')]=v
  with self.assertRaisesRegex(RuntimeError,'LEDGER_CONFLICT'):persist(d)
  self.assertEqual(d.committed['cursor'],B)
 def test_returning_count_handles_concurrent_duplicate(self):
  d=Database();d.duplicate_race=True;r=persist(d);self.assertEqual(r['inserted'],0);self.assertEqual(r['matched_existing'],1)
 def test_stale_cursor_cas_refuses(self):
  d=Database()
  with self.assertRaisesRegex(RuntimeError,'CURSOR_CHANGED'):persist(d,before=B-timedelta(seconds=1))
  self.assertFalse(d.committed['ledger'])
 def test_duplicate_checkpoint_restart_never_replays(self):
  d=Database();persist(d);before=copy.deepcopy(d.committed)
  with self.assertRaises(RuntimeError):persist(d,before=U)
  self.assertEqual(d.committed,before)
 def test_commit_ack_loss_does_not_return_success(self):
  d=Database();d.commit_error='after'
  with self.assertRaisesRegex(RuntimeError,'acknowledgement'):persist(d)
  self.assertEqual(len(d.committed['checkpoints']),1);self.assertEqual(consume(d)['rows'],1)
 def test_restart_new_workflow_matches_existing_without_repair(self):
  d=Database();persist(d);r=persist(d,before=U,workflow_id='b'*32,until=U+timedelta(seconds=1));self.assertEqual(r['inserted'],0);self.assertEqual(len(d.committed['checkpoints']),2)
 def test_wrong_source_refuses_consume(self):
  d=Database();persist(d)
  with self.assertRaisesRegex(RuntimeError,'SOURCE_MISMATCH'):consume(d,source='0'*64)
 def test_account_switch_refuses_cursor_advance(self):
  d=Database();persist(d)
  with self.assertRaisesRegex(RuntimeError,'SOURCE_CHANGED'):persist(d,source='0'*64,before=U,workflow_id='b'*32)
 def test_missing_checkpoint_refuses(self):
  with self.assertRaisesRegex(RuntimeError,'MISSING'):consume(Database())
 def test_inadequate_interval_refuses(self):
  d=Database();persist(d)
  with self.assertRaisesRegex(RuntimeError,'COVERAGE_INSUFFICIENT'):consume(d,required_after=A-timedelta(seconds=1))
 def test_ledger_change_invalidates_committed_proof(self):
  d=Database();persist(d);d.committed['ledger'][('one','FILL')]['price']=Decimal('99')
  with self.assertRaisesRegex(RuntimeError,'LEDGER_CHANGED'):consume(d)
 def test_cursor_rewind_invalidates_proof(self):
  d=Database();persist(d);d.committed['cursor']=B
  with self.assertRaisesRegex(RuntimeError,'CURSOR_REWOUND_OR_FUTURE'):consume(d)
 def test_numeric_scale_is_semantically_equal(self):
  a=row();b=row();b['qty']=Decimal('2.000000');b['price']=Decimal('10.500000');self.assertEqual(I.manifestation([a]),I.manifestation([b]))
 def test_empty_interval_response_persists_zero_row_proof(self):
  d=Database();persist(d,rows=[]);self.assertEqual(consume(d)['rows'],0)
 def test_duplicate_input_identity_refuses_before_write(self):
  d=Database()
  with self.assertRaisesRegex(RuntimeError,'DUPLICATE_IDENTITY'):persist(d,rows=[row(),row()])
  self.assertEqual(d.commit_count,0)
 def test_autocommit_refuses(self):
  d=Database();d.autocommit=True
  with self.assertRaisesRegex(RuntimeError,'TRANSACTION_REQUIRED'):persist(d)
 def test_five_digit_fraction_preserves_time(self):
  self.assertEqual(I.utc('2026-10-23T12:00:00.12345+00:00'),I.utc('2026-10-23T12:00:00.123450+00:00'))
 def test_naive_timestamp_refuses(self):
  with self.assertRaises(RuntimeError):I.utc('2026-10-23T12:00:00')
class Consumers(unittest.TestCase):
 def setUp(self):self.db=Database();persist(self.db);self.proof=consume(self.db)
 def test_new_contract_preserves_financial_holds(self):
  v=S._scoped_reconciliation_result(contract(self.proof),expected_cutoff=U.isoformat(),expected_workflow_id=W);self.assertTrue(v['activity_ingestion_complete'])
  for k in ('accounting_complete','recovery_complete','watermark_advanced','snapshot_eligible'):self.assertIs(v[k],False)
 def test_wrong_workflow_rejected(self):self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof),expected_cutoff=U.isoformat(),expected_workflow_id='b'*32))
 def test_missing_expected_workflow_rejected(self):self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof)))
 def test_wrong_expected_cutoff_rejected(self):
  self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof),expected_workflow_id=W,expected_cutoff=(U+timedelta(seconds=1)).isoformat()))
 def test_missing_expected_cutoff_rejected(self):
  self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof),expected_workflow_id=W))
 def test_duplicate_proof_same_line_rejected(self):
  text=contract(self.proof).replace('RECONCILE_INGESTION_CHECKPOINT','RECONCILE_INGESTION_CHECKPOINT RECONCILE_INGESTION_CHECKPOINT');self.assertIsNone(S._scoped_reconciliation_result(text,expected_cutoff=U.isoformat(),expected_workflow_id=W))
 def test_duplicate_json_key_rejected(self):
  text=contract(self.proof).replace('"schema_version":1','"schema_version":1,"schema_version":1');self.assertIsNone(S._scoped_reconciliation_result(text,expected_cutoff=U.isoformat(),expected_workflow_id=W))
 def test_accounting_flag_cannot_be_promoted(self):self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof).replace('accounting_complete=false','accounting_complete=true'),expected_cutoff=U.isoformat(),expected_workflow_id=W))
 def test_watermark_flag_cannot_be_promoted(self):self.assertIsNone(S._scoped_reconciliation_result(contract(self.proof).replace('watermark_advanced=false','watermark_advanced=true'),expected_cutoff=U.isoformat(),expected_workflow_id=W))
 def test_proof_before_operational_marker_rejected(self):
  lines=contract(self.proof).splitlines();lines[1],lines[2]=lines[2],lines[1];self.assertIsNone(S._scoped_reconciliation_result('\n'.join(lines),expected_cutoff=U.isoformat(),expected_workflow_id=W))
 def test_bool_row_count_rejected(self):
  p=dict(self.proof,rows=True)
  with self.assertRaises(RuntimeError):I.validate_proof(p)
 def test_executor_strict_helper_accepts_only_bound_workflow(self):
  tree=ast.parse((C/'execute_trades.py').read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_ingestion_completion_valid','_operational_reconciliation_complete')]
  ns={'Mapping':Mapping,'os':os,'ingestion_checkpoint':I,'datetime':datetime,'timezone':timezone};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-executor-contract','exec'),ns)
  state=dict(scope='ACTIVE_RECONCILIATION',operational_complete=True,recovery_complete=False,accounting_complete=False,activity_ingestion_complete=True,watermark_advanced=False,reason='DURABLE_INGESTION_PROVED_ACCOUNTING_HELD',ingestion_checkpoint=self.proof,trade_policy={'scope':'ACTIVE_RECONCILIATION','policy_revision':2})
  executor=types.SimpleNamespace(metrics=types.SimpleNamespace(reconciliation=state))
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True):self.assertTrue(ns['_operational_reconciliation_complete'](executor,True))
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:'b'*32,I.ENV_CUTOFF:U.isoformat()},clear=True):self.assertFalse(ns['_operational_reconciliation_complete'](executor,True))
  state['accounting_complete']=True
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True):self.assertFalse(ns['_operational_reconciliation_complete'](executor,True))
class Clock(datetime):
 @classmethod
 def now(cls,tz=None):return U
class Integrated(unittest.TestCase):
 def order_reader(self):
  tree=ast.parse((C/'execute_trades.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='alpaca_list_orders_http')
  request=mock.Mock(return_value=types.SimpleNamespace(status_code=200,json=lambda:[]))
  ns={**globals(),'ingestion_checkpoint':I,'load_env':lambda:None,'_alpaca_url':lambda path:I.ENDPOINT+path,'alpaca_http_get':request}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-order-reader','exec'),ns);return ns['alpaca_list_orders_http'],request
 def test_order_reader_binds_exclusive_submission_upper(self):
  f,r=self.order_reader();f(after_iso=A.isoformat(),until_iso=U.isoformat())
  self.assertEqual(r.call_args.kwargs['params']['until'],U.isoformat());self.assertEqual(r.call_args.kwargs['params']['after'],A.isoformat())
 def test_unscoped_order_call_preserves_prior_no_upper(self):
  f,r=self.order_reader();f(after_iso=A.isoformat());self.assertNotIn('until',r.call_args.kwargs['params'])
 def test_invalid_order_upper_refuses_before_request(self):
  f,r=self.order_reader()
  with self.assertRaises(RuntimeError):f(after_iso=A.isoformat(),until_iso='2026-10-24T12:00:00')
  self.assertEqual(r.call_count,0)
 def test_order_filled_after_cutoff_refuses_instead_of_claiming_snapshot(self):
  tree=ast.parse((C/'execute_trades.py').read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TradeExecutor');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reconcile_closed_trades')
  nodes=[n for n in method.body if isinstance(n,ast.FunctionDef) and n.name in ('_fetch_orders','_best_order_timestamp')]
  order=dict(id='synthetic',symbol='TEST',side='sell',status='filled',filled_at=(U+timedelta(seconds=1)).isoformat(),created_at=B.isoformat(),updated_at=(U+timedelta(seconds=1)).isoformat(),filled_avg_price='10',filled_qty='1')
  request=mock.Mock(return_value=[order]);fail=mock.Mock();cache={'orders':None,'error':False}
  ns={**globals(),'Mapping':Mapping,'ingestion_checkpoint':I,'LOGGER':logging.getLogger('cutoff-test'),'now_utc':U,'orders_cache':cache,'alpaca_list_orders_http':request,'lookback_days':14,'limit':500,'fetch_after_iso':A.isoformat(),'watermark_enabled':True,'workflow_id':W,'record_failure':fail,'db':types.SimpleNamespace(normalize_ts=lambda value,**kwargs:None if value is None else I.utc(value))}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-reconciliation-order-guard','exec'),ns)
  self.assertEqual(ns['_fetch_orders'](),[]);self.assertTrue(cache['error']);self.assertEqual(request.call_args.kwargs['until_iso'],U.isoformat());self.assertTrue(fail.called)
 def test_ingestor_uses_workflow_start_not_later_child_clock(self):
  d=Database();n,args,http=self.ingestor(d);cutoff=U-timedelta(seconds=15)
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:cutoff.isoformat()},clear=True):n['_run_incremental'](args,I.ENDPOINT,d,B.isoformat())
  self.assertEqual(n['fetch_activities'].call_args.args[2],cutoff.isoformat());self.assertEqual(consume(d,required_until=cutoff)['query_until'],cutoff.isoformat())
 def paginator(self,pages):
  tree=ast.parse((C/'fetch_account_activities.py').read_bytes())
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_normalize_page_size','fetch_activities')]
  request=mock.Mock(side_effect=pages);ns={**globals(),'MAX_PAGE_SIZE':100,'logger':logging.getLogger('pagination-test'),'_request_activity_page':request}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-paginator','exec'),ns);return ns['fetch_activities'],request
 def test_documented_headerless_full_page_continues_to_terminal_empty(self):
  f,r=self.paginator([([{'id':'a'},{'id':'b'}],None),([],None)])
  self.assertEqual(len(f(I.ENDPOINT,'lower','upper',2,3)),2);self.assertEqual(r.call_args.args[1]['page_token'],'b');self.assertEqual(r.call_args.args[1]['until'],'upper')
 def test_short_page_still_requires_terminal_empty(self):
  f,r=self.paginator([([{'id':'a'}],None),([],None)]);self.assertEqual(len(f(I.ENDPOINT,'lower','upper',100,3)),1);self.assertEqual(r.call_count,2)
 def test_matching_header_id_is_accepted(self):
  f,_=self.paginator([([{'id':'a'}],'a'),([],None)]);self.assertEqual(len(f(I.ENDPOINT,'lower','upper',100,3)),1)
 def test_conflicting_header_id_refuses(self):
  f,_=self.paginator([([{'id':'a'}],'other')])
  with self.assertRaisesRegex(RuntimeError,'PAGINATION_INCOMPLETE'):f(I.ENDPOINT,'lower','upper',100,3)
 def test_repeated_activity_refuses(self):
  f,_=self.paginator([([{'id':'a'}],None),([{'id':'a'}],None)])
  with self.assertRaisesRegex(RuntimeError,'PAYLOAD_INVALID'):f(I.ENDPOINT,'lower','upper',100,3)
 def test_empty_page_with_continuation_refuses(self):
  f,_=self.paginator([([],'more')])
  with self.assertRaisesRegex(RuntimeError,'PAGINATION_INCOMPLETE'):f(I.ENDPOINT,'lower','upper',100,3)
 def test_page_cap_refuses_instead_of_certifying_partial_window(self):
  f,_=self.paginator([([{'id':'a'}],None)])
  with self.assertRaisesRegex(RuntimeError,'PAGINATION_INCOMPLETE'):f(I.ENDPOINT,'lower','upper',100,1)
 def test_missing_native_activity_identity_refuses(self):
  f,_=self.paginator([([{'description':'synthetic'}],None)])
  with self.assertRaisesRegex(RuntimeError,'PAYLOAD_INVALID'):f(I.ENDPOINT,'lower','upper',100,3)
 def test_main_finishes_cursor_read_before_strict_policy_transaction(self):
  # A real psycopg SELECT starts a transaction. Policy reads require idle,
  # so exercising only _run_incremental would miss this entrypoint defect.
  class Connection:
   reading=False;closed=False;read_finishes=0
   def __enter__(self):return self
   def __exit__(self,*args):self.reading=False;self.read_finishes+=1
   def close(self):self.closed=True
  connection=Connection();calls=[]
  def read(c,strict=False):c.reading=True;return B.isoformat()
  def run(args,endpoint,c,watermark):
   self.assertFalse(c.reading,'cursor read transaction must finish before strict policy read');calls.append(watermark)
  tree=ast.parse((C/'fetch_account_activities.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  ns={**globals(),'_configure_logging':lambda x:None,'load_dotenv':lambda:None,'validate_env':lambda:None,'db':types.SimpleNamespace(get_db_conn=lambda:connection),'load_watermark':read,'_run_incremental':run,'DEFAULT_BASE_URL':I.ENDPOINT,'DEFAULT_LOOKBACK_DAYS':30,'DEFAULT_PAGE_SIZE':100,'DEFAULT_CHUNK_DAYS':30,'DEFAULT_MAX_PAGES':2000,'logger':logging.getLogger('entrypoint-test')}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-ingestor-main','exec'),ns)
  with mock.patch.dict(os.environ,{'APCA_API_BASE_URL':I.ENDPOINT},clear=True):result=ns['main']([])
  self.assertEqual(result,0);self.assertEqual(calls,[B.isoformat()]);self.assertEqual(connection.read_finishes,1);self.assertTrue(connection.closed)
 def ingestor(self,db,activities=None):
  path=C/'fetch_account_activities.py';tree=ast.parse(path.read_bytes())
  names={'_parse_decimal','_parse_timestamp','_normalize_activity','compute_since','_run_incremental'}
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
  response=types.SimpleNamespace(ok=True,json=lambda:{'id':'00000000-0000-0000-0000-000000000001'})
  requests=types.SimpleNamespace(get=mock.Mock(return_value=response))
  lock=lambda c:None
  ns={**globals(),'datetime':Clock,'ingestion_checkpoint':I,'date_parser':types.SimpleNamespace(isoparse=lambda x:datetime.fromisoformat(x.replace('Z','+00:00'))),'logger':logging.getLogger('atomic-ingestor-test'),'_build_headers':lambda:{'SYNTHETIC':'NO_CREDENTIAL'},'requests':requests,'db':types.SimpleNamespace(get_reconcile_state=lambda c,strict:{'last_after':REQUIRED},_policy_write_lock=lock)}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
  event={'id':'one','activity_type':'FILL','transaction_time':B.isoformat(),'symbol':'TEST','side':'buy','qty':'2','price':'10.5','order_id':'00000000-0000-0000-0000-000000000002','description':'SYNTHETIC'}
  ns['fetch_activities']=mock.Mock(return_value=[event] if activities is None else activities)
  args=argparse.Namespace(since_ts=None,lookback_days=30,max_pages=3,page_size=100)
  return ns,args,requests
 def test_actual_ingestor_commits_source_window_and_created_cursor(self):
  d=Database();n,args,http=self.ingestor(d)
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True):n['_run_incremental'](args,I.ENDPOINT,d,B.isoformat())
  self.assertEqual(n['fetch_activities'].call_args.args[1:3],(A.isoformat(),U.isoformat()));self.assertEqual(d.committed['cursor'],U);self.assertEqual(consume(d)['rows'],1);self.assertEqual(http.get.call_count,1)
 def test_actual_ingestor_uncertain_commit_never_logs_completion(self):
  d=Database();d.commit_error='after';n,args,http=self.ingestor(d)
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True),self.assertLogs('atomic-ingestor-test',level='INFO') as logs:
   with self.assertRaisesRegex(RuntimeError,'PERSIST_FAILED_OR_UNCERTAIN'):n['_run_incremental'](args,I.ENDPOINT,d,B.isoformat())
  self.assertFalse(any('ACT_COVERAGE_COMPLETE' in x or 'ACT_CHECKPOINT_COMMITTED' in x for x in logs.output));self.assertEqual(len(d.committed['checkpoints']),1)
 def test_actual_ingestor_refuses_live_endpoint_before_request(self):
  d=Database();n,args,http=self.ingestor(d)
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True),self.assertRaisesRegex(RuntimeError,'PAPER_SOURCE_REQUIRED'):n['_run_incremental'](args,'https://api.alpaca.markets',d,B.isoformat())
  self.assertEqual(http.get.call_count,0);self.assertFalse(d.committed['ledger'])
 def test_since_override_cannot_hide_required_active_window(self):
  d=Database();n,args,http=self.ingestor(d);args.since_ts=U.isoformat()
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True):n['_run_incremental'](args,I.ENDPOINT,d,B.isoformat())
  self.assertEqual(n['fetch_activities'].call_args.args[1],A.isoformat())
 def test_empty_ingestor_response_is_durable_bounded_zero(self):
  d=Database();n,args,http=self.ingestor(d,activities=[])
  with mock.patch.dict(os.environ,{I.ENV_WORKFLOW:W,I.ENV_CUTOFF:U.isoformat()},clear=True):n['_run_incremental'](args,I.ENDPOINT,d,B.isoformat())
  self.assertEqual(consume(d)['rows'],0);self.assertEqual(d.committed['cursor'],U)
 def stage_call(self,workflow_id=W,environ=None):
  db=Database();persist(db);proof=consume(db);output=contract(proof)
  primary=load('offline_primary_paper',C/'primary_paper_pipeline.py')
  stage=S.Stage(S.STAGES[1].name,S.STAGES[1].module,S.STAGES[1].args,S.STAGES[1].required,S.STAGES[1].failures,workflow_id,U.isoformat())
  seen=[]
  class Process:
   returncode=0
   def poll(self):return 0
   def wait(self,timeout=None):return 0
  def popen(command,stdout,**options):stdout.write(output.encode());stdout.flush();seen.append(options);return Process()
  env={'APCA_API_BASE_URL':I.ENDPOINT} if environ is None else environ
  with mock.patch.dict(sys.modules,{'scripts':scripts,'scripts.primary_paper_pipeline':primary}),mock.patch.dict(os.environ,env,clear=True),mock.patch.object(S.subprocess,'Popen',side_effect=popen):result=S.run_stage(stage,300,123)
  return result,seen
 def test_real_stage_parser_is_degraded_with_valid_interval_and_financial_holds(self):
  result,seen=self.stage_call();self.assertEqual(result['status'],'degraded');self.assertEqual(result['completion_contract'],'ACTIVE_RECONCILIATION_INGESTION_V2');self.assertFalse(result['reconciliation']['snapshot_eligible']);self.assertEqual(seen[0]['env'][I.ENV_WORKFLOW],W);self.assertEqual(seen[0]['env']['JBR_EXEC_PAPER'],'true')
 def test_real_stage_parser_wrong_workflow_is_failure(self):
  result,_=self.stage_call('b'*32);self.assertEqual(result['status'],'failed')
 def test_explicit_false_paper_flag_stops_before_child(self):
  with self.assertRaisesRegex(ValueError,'context_conflict'):self.stage_call(environ={'APCA_API_BASE_URL':I.ENDPOINT,'JBR_EXEC_PAPER':'false'})
 def test_workflow_binds_both_children_and_keeps_snapshot_omitted(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);src=root/'sources';(src/'scripts').mkdir(parents=True)
   for name in ('account_sync.py','db.py','activity_ingestion_checkpoint.py','fetch_account_activities.py','execute_trades.py','fetch_account_snapshot.py'):(src/'scripts'/name).write_text('SYNTHETIC_NO_RUNTIME')
   seen=[]
   def runner(stage,timeout,fd):
    seen.append((stage.name,stage.workflow_id));held=stage.name=='reconciliation'
    return {'status':'degraded' if held else 'ok','exit_code':0,'reason':'RECONCILE_RECOVERY_HELD' if held else None,'elapsed_seconds':1,'output_bytes':1,'failure_markers':[],'missing_completion_markers':[],'qualifications':['RECONCILE_RECOVERY_HELD'] if held else []}
   with mock.patch.object(S,'ROOT',src):report=S.run_workflow(root/'state',runner=runner)
   self.assertEqual([x[0] for x in seen],['activities','reconciliation']);self.assertTrue(all(x[1]==report['run_id'] for x in seen));self.assertEqual(report['status'],'degraded');self.assertEqual(report['stages'][2]['status'],'not_run')
if __name__=='__main__':
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Atomic,Consumers,Integrated)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 out={'status':'PASS' if result.wasSuccessful() else 'FAIL','methods':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'audit':AUDIT,'qualification':'Actual isolated candidate code; synthetic DBAPI transaction/SQL and source-consumer tests. No native PostgreSQL/concurrency/driver semantics or host/provider calls are claimed.'}
 print(json.dumps(out));raise SystemExit(0 if result.wasSuccessful() else 1)
