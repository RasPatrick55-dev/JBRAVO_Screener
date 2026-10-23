"""Synthetic gross-only contract tests; no actual DB/provider/environment reads."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json,os
from pathlib import Path
import unittest
from unittest import mock
from scripts import gross_paper_accounting as G, trade_disposition as D, trade_allocation as A

ROOT=Path(__file__).resolve().parents[1]
IDENTITY=('synthetic-paper','00000000-0000-0000-0000-000000000001')

def row(trade_id=110, *, incomplete=True, symbol='TEST'):
 return dict(trade_id=trade_id,symbol=symbol,status='CLOSED',qty=30,entry_price=65,
  exit_price=None if incomplete else '62.5',realized_pnl=None,exit_order_id=None,
  entry_time='2026-04-16T15:00:00Z',exit_time='2026-04-17T15:00:00Z')

def allocation():
 return A.Allocation('synthetic-allocation',110,'ACTIVE',1,Decimal(30),Decimal(30),
  Decimal('-63.38'),'synthetic-approval',('synthetic-evidence',),('exit-one','exit-two'),
  ('entry-fill-one','entry-fill-two','exit-fill-one','exit-fill-two','exit-fill-three'))

def policy(allocations=(), *, revision=4, excluded=()):
 owners={oid:a.trade_id for a in allocations for oid in a.order_ids}
 overlay=A.result_read('VALIDATED',allocations,IDENTITY,revision,owners)
 active=tuple(D.ActiveCase('synthetic-exclusion-'+str(i),i,'synthetic-exclusion-approval',
  'TEST_GENERATED',('VALIDATED_PERFORMANCE',)) for i in excluded)
 return D.Policy(*IDENTITY,revision,active,'VALIDATED_PERFORMANCE',allocations,overlay._metadata_json)

def build(p,rows,**kw):
 return G.build_snapshot(p,rows,expected_identity=kw.get('expected_identity',IDENTITY),
  expected_revision=kw.get('expected_revision',p.revision))

class GrossProjectionTests(unittest.TestCase):
 def test_exact_multi_fill_gross_preserves_unknown_net_and_original(self):
  rows=[row()];before=deepcopy(rows);p=policy((allocation(),));r=build(p,rows)
  self.assertEqual(rows,before);self.assertEqual(r['gross_pnl'],'-63.38')
  self.assertEqual(r['pnl_basis'],G.BASIS);self.assertEqual(r['total_trades'],1)
  self.assertEqual(r['rows'][0]['lineage']['order_ids'],['exit-one','exit-two'])
  self.assertEqual(r['rows'][0]['original_row_fingerprint'],D.fingerprint(rows[0]))
  self.assertNotIn('exit_price',r['rows'][0]);self.assertIsNone(rows[0]['exit_price'])
  for obj in (r,*r['rows']):
   self.assertIsNone(obj['fees']);self.assertIsNone(obj['net_pnl'])
   self.assertEqual(obj['fees_status'],'UNKNOWN');self.assertEqual(obj['net_status'],'UNKNOWN')
   self.assertFalse(obj['net_eligible'])
  for key in ('accounting_complete','activity_ingestion_complete','recovery_complete'):
   self.assertFalse(r[key]);self.assertFalse(r['evidence'][key])

 def test_gross_view_does_not_promote_financial_or_writable_eligibility(self):
  p=policy((allocation(),));build(p,[row()])
  with self.assertRaisesRegex(D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):D.require_financial_eligible(p)
  with self.assertRaisesRegex(D.TradePolicyError,'TRADE_EXCLUDED'):D.require_writable(p,110)
  with self.assertRaises(D.TradePolicyError):D.require_order_owner(p,111,'exit-one')
  self.assertFalse(p.allocations[0].financial_eligible)

 def test_revocation_restores_original_missing_price_hold(self):
  build(policy((allocation(),)),[row()])
  with self.assertRaisesRegex(D.TradePolicyError,'GROSS_NUMBER_INVALID'):build(policy(revision=5),[row()])

 def test_exact_ids_not_symbol_or_order_pattern_filters(self):
  rows=[row(),row(111,incomplete=False,symbol='TEST')]
  r=build(policy((allocation(),)),rows)
  self.assertEqual([x['trade_id'] for x in r['rows']],[110,111])
  self.assertEqual(r['rows'][1]['gross_pnl'],'-75')
  with self.assertRaisesRegex(D.TradePolicyError,'ROW_IDENTITY_COLLISION'):build(policy(excluded=(63,)),[row(63,incomplete=False)])
  with self.assertRaises(D.TradePolicyError):build(policy(),[row(111,incomplete=False),row(111,incomplete=False)])

 def test_precise_recorded_price_derivation_never_relabels_recorded_net(self):
  x=row(112,incomplete=False);x.update(qty='0.00000001',entry_price='1.00000001',exit_price='1.00000002',realized_pnl='99')
  r=build(policy(),[x]);self.assertEqual(Decimal(r['gross_pnl']),Decimal('0.0000000000000001'))
  self.assertEqual(x['realized_pnl'],'99');self.assertIsNone(r['net_pnl'])
  self.assertEqual(r['rows'][0]['lineage'],{'kind':'RECORDED_PRICE_GROSS','fill_provenance_proved':False})
  x.update(qty=-2,entry_price=100,exit_price=90)
  self.assertEqual(build(policy(),[x])['gross_pnl'],'20')

 def test_decimal_derivation_is_independent_of_ambient_precision(self):
  from decimal import localcontext
  x=row(112,incomplete=False)
  x.update(qty='12345678901234567890123456789',entry_price='1.00000000000000000000000000001',
           exit_price='1.00000000000000000000000000002')
  with localcontext() as context:
   context.prec=6
   self.assertEqual(build(policy(),[x])['gross_pnl'],'0.12345678901234567890123456789')

 def test_reviewer_wide_exponent_counterexample_and_cancellation(self):
  from decimal import localcontext, Inexact, Rounded
  big=row(112,incomplete=False)|dict(qty='1e1000',entry_price='1',exit_price='1'+'0'*999+'e1000')
  tiny=row(113,incomplete=False)|dict(qty='1e-1000',entry_price='1e-1000',exit_price='2e-1000')
  # Exact string construction is independent of the candidate and Decimal
  # arithmetic: 10^2999 - 10^1000 + 10^-2000.
  expected='9'*1999+'0'*1000+'.'+'0'*1999+'1'
  inverse=big|dict(trade_id=114,qty='-1e1000')
  with localcontext() as context:
   context.prec=6;context.traps[Inexact]=True;context.traps[Rounded]=True
   for rows,total in (([big,tiny],expected),([tiny,big],expected),
                      ([big,tiny,inverse],'0.'+'0'*1999+'1')):
    with self.subTest(order=[r['trade_id'] for r in rows]):
     result=build(policy(),rows)
     self.assertEqual(result['gross_pnl'],total)
     self.assertTrue(result['evidence']['gross_input_complete'])
     self.assertFalse(result['accounting_complete']);self.assertIsNone(result['net_pnl'])
  self.assertEqual(G._gross_sum([Decimal(0),Decimal(0)]),Decimal(0))

 def test_price_difference_spans_admitted_input_domain(self):
  from decimal import localcontext, Inexact, Rounded
  x=row(112,incomplete=False)|dict(qty='1e1000',entry_price='1e-1000',exit_price='1'+'0'*999+'e1000')
  # (10^1999 - 10^-1000) * 10^1000 = 10^2999 - 1.
  with localcontext() as context:
   context.prec=6;context.traps[Inexact]=True;context.traps[Rounded]=True
   self.assertEqual(build(policy(),[x])['gross_pnl'],'9'*2999)

 def test_missing_invalid_inputs_fail_without_zero_fallback(self):
  for field,values in {'qty':[None,0,True,'NaN','Infinity'],
      'entry_price':[None,0,-1,False,'NaN'],'exit_price':[None,0,-1,False,'Infinity'],
      'entry_time':[None,'2026-04-16T15:00:00'],'exit_time':[None,'invalid']}.items():
   for value in values:
    with self.subTest(field=field,value=value),self.assertRaises(D.TradePolicyError):
     x=row(112,incomplete=False);x[field]=value;build(policy(),[x])
  with self.assertRaises(D.TradePolicyError):build(policy(),[row(112,incomplete=False)|{'status':'OPEN'}])

 def test_identity_revision_scope_and_overlay_fail_closed(self):
  p=policy((allocation(),))
  for changes in ({'expected_identity':('other-paper',IDENTITY[1])},{'expected_revision':5}):
   with self.subTest(changes=changes),self.assertRaises(D.TradePolicyError):build(p,[row()],**changes)
  for damaged in (replace(p,scope='ACTIVE_RECONCILIATION'),replace(p,revision=True),replace(p,allocation_metadata_json='{}')):
   with self.subTest(policy=damaged),self.assertRaises(D.TradePolicyError):build(damaged,[row()])
  meta=D.document(p.allocation_metadata_json);meta['allocations'][0]['gross_result']='999'
  with self.assertRaises(D.TradePolicyError):build(replace(p,allocation_metadata_json=D.canonical(meta)),[row()])

 def test_allocation_population_quantity_and_order_ownership_guarded(self):
  a=allocation()
  for bad in (replace(a,state='REVOKED'),replace(a,financial_eligible=True),
              replace(a,net=Decimal('-63.38')),replace(a,exit_quantity=Decimal(29))):
   with self.subTest(allocation=bad),self.assertRaises(D.TradePolicyError):build(policy((bad,)),[row()])
  with self.assertRaises(D.TradePolicyError):build(policy((a,)),[])
  with self.assertRaises(D.TradePolicyError):build(policy((a,)),[row()|{'qty':29}])
  p=policy((a,));meta=D.document(p.allocation_metadata_json);meta['reserved_order_owners']['exit-one']=111
  meta['binding_fingerprint']=D.fingerprint({k:v for k,v in meta.items() if k!='binding_fingerprint'})
  with self.assertRaises(D.TradePolicyError):build(replace(p,allocation_metadata_json=D.canonical(meta)),[row()])

 def test_cache_revision_tracks_policy_allocation_and_full_row_changes(self):
  p=policy((allocation(),));r=build(p,[row()]);same=build(p,[row()])
  self.assertEqual(r['cache_revision'],same['cache_revision'])
  variants=[build(policy((allocation(),),revision=5),[row()]),
   build(policy((replace(allocation(),revision=2),)),[row()]),build(p,[row()|{'realized_pnl':'observed-change'}])]
  for v in variants:
   with self.subTest(cache=v['cache_revision']):self.assertNotEqual(v['cache_revision'],r['cache_revision'])

 def test_empty_population_is_explicit_and_caps_never_truncate(self):
  r=build(policy(),[]);self.assertEqual(r['total_trades'],0);self.assertEqual(r['gross_pnl'],'0');self.assertIsNone(r['net_pnl'])
  with self.assertRaisesRegex(D.TradePolicyError,'ROW_CAP'):build(policy(),[{}]*(G.MAX_ROWS+1))

class GrossDatabaseAdapterTests(unittest.TestCase):
 def setUp(self):
  self.rows=[row()];self.p=policy((allocation(),));self.sql=[];self.policy_calls=[]
  test=self
  class Cursor:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def execute(self,sql):test.sql.append(sql)
   def fetchall(self):return [(D.canonical(x),) for x in test.rows]
  class Connection:
   autocommit=False
   def get_transaction_status(self):return 0
   def cursor(self):return Cursor()
   def __enter__(self):return self
   def __exit__(self,*a):pass
  self.conn=Connection()
  @contextmanager
  def maybe_conn(engine):self.assertIs(engine,self.conn);yield engine
  def validated_policy(conn,**kw):
   self.assertIs(conn,self.conn);self.policy_calls.append(kw)
   if kw.get('expected_revision') not in (None,self.p.revision):D.fail('REVISION_CHANGED')
   return self.p
  tree=ast.parse((ROOT/'scripts/db.py').read_bytes())
  names={'gross_paper_performance_snapshot','_policy_read_transaction','_paper_disposition_identity','performance_input'}
  defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
  self.ns=dict(contextmanager=contextmanager,disposition=D,os=os,_maybe_conn=maybe_conn,
   trade_policy=validated_policy,_PAPER_DISPOSITION_BINDING=(*IDENTITY,'synthetic-context-evidence'),
   configure_disposition_identity=lambda *a,**kw:IDENTITY)
  exec(compile(ast.Module(body=defs,type_ignores=[]),'actual_db_gross_adapter','exec'),self.ns)
  self.env=mock.patch.dict(os.environ,{'APCA_API_BASE_URL':'https://paper-api.alpaca.markets','JBR_EXEC_PAPER':'true'})
  self.env.start();self.addCleanup(self.env.stop)

 def test_actual_readonly_adapter_uses_eligible_query_before_cap(self):
  r=self.ns['gross_paper_performance_snapshot'](self.conn,expected_revision=4)
  self.assertEqual(r['gross_pnl'],'-63.38')
  self.assertEqual(self.sql[:2],['SET TRANSACTION ISOLATION LEVEL READ COMMITTED','SET TRANSACTION READ ONLY'])
  query=self.sql[-1];self.assertIn(D.eligible_sql(scope='VALIDATED_PERFORMANCE'),query)
  self.assertLess(query.index('NOT EXISTS'),query.index('LIMIT 10001'))
  self.assertEqual(self.policy_calls,[{'scope':'VALIDATED_PERFORMANCE','expected_revision':4}])
  self.assertTrue(all(x.startswith(('SET ','SELECT ')) for x in self.sql))

 def test_live_false_context_and_policy_failure_have_no_fallback(self):
  for endpoint,flag in [('https://api.alpaca.markets','true'),('https://paper-api.alpaca.markets','false'),('https://paper-api.alpaca.markets','')]:
   with self.subTest(endpoint=endpoint,flag=flag),mock.patch.dict(os.environ,{'APCA_API_BASE_URL':endpoint,'JBR_EXEC_PAPER':flag}),self.assertRaises(D.TradePolicyError):
    self.ns['gross_paper_performance_snapshot'](self.conn)
  self.assertEqual(self.sql,[])
  with self.assertRaisesRegex(D.TradePolicyError,'REVISION_CHANGED'):self.ns['gross_paper_performance_snapshot'](self.conn,expected_revision=5)

 def test_cap_and_original_net_gate_still_fail(self):
  self.rows=[{}]*(G.MAX_ROWS+1)
  with self.assertRaisesRegex(D.TradePolicyError,'ROW_CAP'):self.ns['gross_paper_performance_snapshot'](self.conn)
  with self.assertRaisesRegex(D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):self.ns['performance_input'](self.conn,self.p)


if __name__=='__main__':unittest.main()
