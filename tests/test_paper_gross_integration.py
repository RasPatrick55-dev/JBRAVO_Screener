"""Synthetic integration: real pure code, source-extracted consumers, fake DB/UI."""
import argparse,ast,hashlib,json,logging,shutil,tempfile,unittest
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any,Dict,Iterable,Optional
from unittest import mock
from scripts import gross_paper_accounting as G,trade_disposition as D
from types import ModuleType

# Shared pure fixtures are a fixed sibling source, independent of pytest's
# import mode, sys.path aliases and bytecode-cache availability.
_fixture_path = Path(__file__).with_name('test_gross_paper_accounting.py')
F = ModuleType('gross_paper_test_fixtures')
F.__file__ = str(_fixture_path)
exec(compile(_fixture_path.read_bytes(), str(_fixture_path), 'exec'), F.__dict__)

ROOT=Path(__file__).resolve().parents[1]
NOW=datetime(2026,10,23,14,tzinfo=timezone.utc)

def extract(path,names,namespace):
 tree=ast.parse(path.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'scoped_integration_AST','exec'),namespace)
 return namespace

class GrossArtifactTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name)
  for name in G.SOURCE_PATHS:
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
  self.policy=F.policy((F.allocation(),));self.rows=[F.row()]
 def current(self):return F.build(self.policy,self.rows),G.source_bindings(self.root)
 def publish(self):return G.publish_report(self.root,self.current,now=NOW)
 def test_roundtrip_unknown_net_exact_gross_and_distinct_namespace(self):
  before=deepcopy(self.rows);result=self.publish();self.assertEqual(self.rows,before)
  self.assertEqual(result['validation_status'],'VALIDATED_PAPER_GROSS')
  self.assertEqual(result['report']['gross_pnl'],'-63.38');self.assertIsNone(result['report']['net_pnl'])
  self.assertIsNone(result['report']['fees']);self.assertFalse(result['report']['accounting_complete'])
  self.assertFalse(result['report']['net_eligible']);self.assertFalse(result['report']['recovery_complete'])
  self.assertEqual(G.read_report(self.root,self.current,now=NOW),result)
  self.assertFalse((self.root/'data/trade_performance_cache.json').exists())
  self.assertFalse((self.root/'data/metrics_summary.csv').exists())
 def test_policy_row_allocation_and_source_changes_reject_stale(self):
  self.publish();old=deepcopy(self.rows);old_policy=self.policy
  cases=[('row',lambda:self.rows[0].update(realized_pnl='unrelated-original-change')),
   ('policy',lambda:setattr(self,'policy',F.policy((F.allocation(),),revision=5))),
   ('allocation',lambda:setattr(self,'policy',F.policy((F.replace(F.allocation(),revision=2),))))]
  for label,change in cases:
   with self.subTest(case=label):
    self.rows=deepcopy(old);self.policy=old_policy;change()
    with self.assertRaises(D.TradePolicyError):G.read_report(self.root,self.current,now=NOW)
  self.rows=old;self.policy=old_policy
  p=self.root/'scripts/metrics.py';p.write_bytes(p.read_bytes()+b'\n# source-revision-change\n')
  with self.assertRaisesRegex(D.TradePolicyError,'SOURCE_CHANGED'):G.read_report(self.root,self.current,now=NOW)
 def test_revoke_missing_price_row_failure_and_no_cached_fallback(self):
  self.publish();self.policy=F.policy(revision=5)
  with self.assertRaisesRegex(D.TradePolicyError,'NUMBER_INVALID'):G.read_report(self.root,self.current,now=NOW)
  self.policy=F.policy((F.allocation(),));self.rows=[]
  with self.assertRaisesRegex(D.TradePolicyError,'ALLOCATION_ROW_MISSING'):G.read_report(self.root,self.current,now=NOW)
 def test_generation_age_is_distinct_from_fresh_current_validation(self):
  self.publish();later=NOW+timedelta(days=2)
  result=G.read_report(self.root,self.current,now=later)
  self.assertEqual(result['generated_at'],NOW.isoformat());self.assertEqual(result['validated_at'],later.isoformat())
  self.rows[0]['realized_pnl']='new-fact'
  with self.assertRaises(D.TradePolicyError):G.read_report(self.root,self.current,now=later)
 def test_tampered_net_fee_values_schema_and_future_stamp_fail(self):
  snapshot,sources=self.current();valid=G.report_document(snapshot,sources,now=NOW)
  for kind in ('net','fees','accounting','gross','namespace','boolschema','future'):
   with self.subTest(case=kind):
    doc=deepcopy(valid)
    if kind in ('net','fees'):doc['report']['net_pnl' if kind=='net' else 'fees']='0'
    elif kind=='accounting':doc['report']['accounting_complete']=True
    elif kind=='gross':doc['report']['rows'][0]['gross_pnl']='999'
    elif kind=='namespace':doc['namespace']='NET'
    elif kind=='boolschema':doc['schema_version']=True
    else:doc['generated_at']=(NOW+timedelta(seconds=1)).isoformat()
    doc['binding_fingerprint']=D.fingerprint({k:v for k,v in doc.items() if k!='binding_fingerprint'})
    with self.assertRaises(D.TradePolicyError):G.validate_report(doc,snapshot,sources,now=NOW)
 def test_atomic_prewrite_drift_preserves_prior_and_cleans_owned_temp(self):
  self.publish();target=self.root/G.REPORT_PATH;before=target.read_bytes();count=0
  def changing():
   nonlocal count
   count+=1
   if count==2:self.rows[0]['realized_pnl']='changed'
   return self.current()
  with self.assertRaises(D.TradePolicyError):G.publish_report(self.root,changing,now=NOW)
  self.assertEqual(target.read_bytes(),before);self.assertEqual(list(target.parent.glob('.paper_gross_*')),[])
 def test_postwrite_drift_never_reports_success_or_reads_stale(self):
  count=0
  def changing():
   nonlocal count
   count+=1
   if count==3:self.rows[0]['realized_pnl']='changed-after-replace'
   return self.current()
  with self.assertRaises(D.TradePolicyError):G.publish_report(self.root,changing,now=NOW)
  with self.assertRaises(D.TradePolicyError):G.read_report(self.root,self.current,now=NOW)
 def test_file_source_caps_and_outside_namespace_fail(self):
  self.publish();target=self.root/G.REPORT_PATH
  target.write_bytes(b' '*(G.MAX_REPORT_BYTES+1))
  with self.assertRaisesRegex(D.TradePolicyError,'FILE_CAP'):G.read_report(self.root,self.current,now=NOW)
  source=self.root/G.SOURCE_PATHS[0];source.write_bytes(b'x'*(1024*1024+1))
  with self.assertRaisesRegex(D.TradePolicyError,'FILE_CAP'):G.source_bindings(self.root)
  with self.assertRaisesRegex(D.TradePolicyError,'PATH_INVALID'):G._regular(ROOT/'scripts/db.py',self.root)
 def test_read_and_publication_policy_failures_have_no_fallback(self):
  self.publish();before=(self.root/G.REPORT_PATH).read_bytes()
  def refused():D.fail('UNAVAILABLE')
  for operation in (G.read_report,G.publish_report):
   with self.subTest(operation=operation.__name__),self.assertRaises(D.TradePolicyError):operation(self.root,refused,now=NOW)
  self.assertEqual((self.root/G.REPORT_PATH).read_bytes(),before)

class GrossDatabaseIntegrationTests(unittest.TestCase):
 setUp=GrossArtifactTests.setUp
 current=GrossArtifactTests.current
 def test_actual_db_read_write_sidecar_stays_readonly_and_lock_held(self):
  sql=[];active=[];test=self
  class Cursor:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def execute(self,q):sql.append(q)
   def fetchall(self):test.assertEqual(active,['locked']);return [(D.canonical(x),) for x in test.rows]
  conn=SimpleNamespace(cursor=Cursor)
  @contextmanager
  def maybe_conn(engine):self.assertIs(engine,conn);yield conn
  @contextmanager
  def policy_read(c,**kwargs):
   self.assertEqual(kwargs,dict(scope='VALIDATED_PERFORMANCE',expected_revision=4));active.append('locked')
   try:yield self.policy
   finally:active.pop()
  ns=dict(__file__=str(self.root/'scripts/db.py'),disposition=D,_paper_disposition_identity=lambda:F.IDENTITY,
   _maybe_conn=maybe_conn,_policy_read_transaction=policy_read)
  extract(ROOT/'scripts/db.py',('_paper_gross_artifact','publish_paper_gross_report','validated_paper_gross_report'),ns)
  pub=ns['publish_paper_gross_report'](conn,expected_revision=4)
  got=ns['validated_paper_gross_report'](conn,expected_revision=4)
  self.assertEqual(pub['report'],got['report']);self.assertEqual(active,[])
  self.assertEqual(sql.count("SET LOCAL statement_timeout = '30s'"),2)
  selects=[q for q in sql if q.startswith('SELECT ')];self.assertEqual(len(selects),4)
  self.assertEqual(len(sql),6)
  for q in selects:
   self.assertTrue(q.startswith('SELECT '));self.assertLess(q.index('NOT EXISTS'),q.index('LIMIT 10001'))
  with self.assertRaises(D.TradePolicyError):ns['_paper_gross_artifact']('arbitrary',conn)

class ConsumerIntegrationTests(unittest.TestCase):
 def test_metrics_explicit_mode_publishes_only_gross_and_retains_exit2(self):
  db=SimpleNamespace(publish_paper_gross_report=mock.Mock(),disposition=D)
  ns=dict(argparse=argparse,Optional=Optional,Iterable=Iterable,logger=mock.Mock(),db=db)
  extract(ROOT/'scripts/metrics.py',('parse_args','_run_metrics','main'),ns)
  self.assertFalse(ns['parse_args']([]).paper_gross_report_only)
  self.assertEqual(ns['main'](['--paper-gross-report-only']),2);db.publish_paper_gross_report.assert_called_once_with()
  db.publish_paper_gross_report.reset_mock()
  self.assertEqual(ns['main'](['--paper-gross-report-only','--run-date','2026-10-23']),2)
  db.publish_paper_gross_report.assert_not_called()
  db.publish_paper_gross_report.side_effect=OSError('synthetic error must not be logged')
  self.assertEqual(ns['main'](['--paper-gross-report-only']),2)
  self.assertFalse(any('synthetic error' in str(x) for x in ns['logger'].mock_calls))
 def test_dashboard_reader_missing_or_invalid_is_explicit_unknown_not_zero(self):
  db=SimpleNamespace(validated_paper_gross_report=mock.Mock(side_effect=D.TradePolicyError('HELD')))
  ns=dict(Dict=Dict,Any=Any,db=db,logger=mock.Mock())
  extract(ROOT/'dashboards/data_io.py',('paper_gross_summary_snapshot',),ns)
  result=ns['paper_gross_summary_snapshot']()
  self.assertEqual(result['validation_status'],'HELD_PAPER_GROSS')
  self.assertIsNone(result['gross_pnl']);self.assertIsNone(result['net_pnl']);self.assertIsNone(result['total_trades'])
  self.assertFalse(result['accounting_complete']);self.assertFalse(result['recovery_complete'])
 def test_dashboard_gross_panel_never_calls_legacy_net_normalizers(self):
  def component(name):
   return lambda *args,**kwargs:dict(component=name,args=args,props=kwargs)
  ui=SimpleNamespace(**{n:component(n) for n in ('Div','H4','Alert','DataTable')})
  report=F.build(F.policy((F.allocation(),)),[F.row()])
  result={'validation_status':'VALIDATED_PAPER_GROSS',**report,
   'generated_at':NOW.isoformat(),'validated_at':NOW.isoformat()}
  read=mock.Mock(return_value=result)
  ns=dict(html=ui,dbc=ui,dash_table=ui,paper_gross_summary_snapshot=read,
   _render_legacy_net_trade_performance_panel=mock.Mock(side_effect=AssertionError('legacy net forbidden')))
  extract(ROOT/'dashboards/dashboard_app.py',('render_paper_gross_performance_panel','render_trade_performance_panel'),ns)
  rendered=ns['render_trade_performance_panel']();text=json.dumps(rendered)
  self.assertIn('Paper gross performance',text);self.assertIn('before fees',text);self.assertIn('UNKNOWN',text)
  self.assertIn('-63.38',text);self.assertNotIn('Net P&L by window',text)
  ns['_render_legacy_net_trade_performance_panel'].assert_not_called()
  read.return_value={'validation_status':'HELD_PAPER_GROSS'}
  self.assertIn('unavailable',json.dumps(ns['render_trade_performance_panel']()))
 def test_legacy_net_cache_rejects_gross_namespace_before_numeric_normalization(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'gross.json';p.write_text(json.dumps({'namespace':'PAPER_GROSS_ONLY','net_pnl':None}),encoding='utf-8')
   db=SimpleNamespace(disposition=D,validated_performance_artifact=mock.Mock(side_effect=AssertionError('must reject first')))
   ns=dict(Path=Path,Any=Any,CACHE_PATH=p,db=db)
   extract(ROOT/'scripts/trade_performance.py',('read_cache',),ns)
   with self.assertRaisesRegex(D.TradePolicyError,'GROSS_NOT_NET_CACHE'):ns['read_cache'](p)
   db.validated_performance_artifact.assert_not_called()
 def test_ml_financial_holds_are_not_promoted_by_gross_report(self):
  p=F.policy((F.allocation(),))
  with self.assertRaisesRegex(D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):D.require_financial_eligible(p)
  with self.assertRaises(D.TradePolicyError):D.require_complete_accounting([F.row()])

if __name__=='__main__':unittest.main()
