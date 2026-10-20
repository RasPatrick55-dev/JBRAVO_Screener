"""Offline actual pure helper and selected gateway/caller functions; no application imports."""
import ast
import copy
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import logging
from pathlib import Path
import sys
import types
import unittest
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {name: (ROOT / 'scripts' / (name + '.py')).read_text(encoding='utf-8')
          for name in ('trade_disposition', 'trade_allocation', 'db', 'execute_trades', 'metrics', 'trade_performance')}
SOURCE.update({name:(ROOT/'dashboards'/(name+'.py')).read_text(encoding='utf-8') for name in ('data_io','dashboard_app')})
D = types.ModuleType('paper_disposition_fixture')
sys.modules[D.__name__] = D
exec(compile(SOURCE['trade_disposition'], 'actual_trade_disposition', 'exec'), D.__dict__)
# Bind the actual pure optional-store dependency without package/app startup.
# The real read_active/schema_present execute against the explicit SQL protocol.
A = types.ModuleType('paper_allocation_fixture')
sys.modules[A.__name__] = A
exec(compile(SOURCE['trade_allocation'], 'actual_trade_allocation', 'exec'), A.__dict__)
A.policy_module = lambda: D
D._read_allocation_overlay = lambda cursor, identity, revision, excluded: A.read_active(cursor, identity, revision, excluded)

IDENTITY = ('paper-fixture', '00000000-0000-4000-8000-000000000001')
STAMP = '2026-01-01T00:00:00+00:00'
LOG = logging.getLogger('paper-fixture')
LOG.addHandler(logging.NullHandler())

def pending_allocation_policy(*, scope='VALIDATED_PERFORMANCE'):
    # Pure real metadata construction only: do not add earlier SQL/settings to
    # the publication-ordering fixture's trace or fabricate an empty overlay.
    overlay = A.result_read('MIGRATION_PENDING', (), IDENTITY, 0)
    return D.Policy(*IDENTITY, 0, (), scope, (), D.canonical(overlay.metadata_dict()))


def selected(module, names, env):
    tree = ast.parse(SOURCE[module])
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names), (module, names)
    if module == 'dashboard_app':
        for node in nodes:node.decorator_list=[]  # No Dash registration/application startup.
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), module, 'exec'), env)
    return env


def row(trade_id=110, pnl=None):
    return dict(trade_id=trade_id, symbol='CRC', status='CLOSED', qty=Decimal('2'),
        entry_price=Decimal('10'), exit_price=Decimal('11') if pnl is not None else None,
        realized_pnl=pnl, entry_time=STAMP, exit_time=STAMP if pnl is not None else None,
        updated_at=STAMP, entry_order_id='entry-' + str(trade_id), exit_order_id=None)


def approval(snapshot, scopes=('ACTIVE_RECONCILIATION', 'VALIDATED_PERFORMANCE')):
    manifest = dict(case_id='00000000-0000-4000-8000-000000000110', trade_id=snapshot['trade_id'],
        deployment_id=IDENTITY[0], database_id=IDENTITY[1], row_version=snapshot['updated_at'],
        row_fingerprint=D.fingerprint(snapshot), classification='UNRESOLVED', provenance_proved=False,
        operational_scopes=list(scopes), rationale='Owner-scoped operational hold; recovery unresolved',
        evidence=[{'reference':'fixture-only', 'sha256':'a' * 64}])
    request = {k: manifest[k] for k in ('case_id','trade_id','deployment_id','database_id','row_version','row_fingerprint')}
    request.update(action='APPLY', expected_policy_revision=0, expected_case_revision=0,
        actor='fixture-operator', approved_by='fixture-owner', approved_at=STAMP,
        approval_ref='fixture-approval', manifest_sha256=D.fingerprint(manifest))
    return request, manifest


class Store:
    """Inert SQL protocol double; models transaction rollback/CAS, not PostgreSQL locks or triggers."""
    def __init__(self):
        self.rows = {110: row()}; self.cases={}; self.audit={}; self.revision=0
        self.trace=[]; self.autocommit=False; self.fail_audit=False; self.closed=False
        self.read_rows=[]; self.metrics_writes=[]; self.direct_rows=[]
    def get_transaction_status(self): return 0
    def cursor(self, **kwargs): return Cursor(self)
    def __enter__(self):
        self.backup=copy.deepcopy((self.cases,self.audit,self.revision,self.metrics_writes)); return self
    def __exit__(self, typ, value, tb):
        if typ: self.cases,self.audit,self.revision,self.metrics_writes=self.backup
    def close(self): self.closed=True

class Cursor:
    def __init__(self, store): self.s=store; self.one=None; self.many=[]; self.rowcount=1
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def execute(self, sql, params=None):
        self.s.trace.append((sql,copy.deepcopy(params))); self.one=None; self.many=[]
        q=' '.join(sql.split()); s=self.s
        if q.startswith('SELECT pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s),'): self.one=(None,None,None,None)
        elif q.startswith('SELECT a.atttypid='): self.one=None  # Never-installed optional-store fixture.
        elif q.startswith('SHOW transaction_isolation') or q == "SELECT pg_catalog.current_setting('transaction_isolation')": self.one=('read committed',)
        elif q.startswith('SELECT deployment_id'): self.one=(*IDENTITY,'jbravo',s.revision,'jbravo')
        elif q.startswith('SELECT c.request'):
            for c in s.cases.values():
                if c[0]!='ACTIVE': continue
                req,snap,manifest=c[2:]; a=s.audit.get(req['approval_ref'])
                self.many.append((D.canonical(req),D.canonical(snap),D.canonical(manifest),c[1],
                    a[0] if a else None,a[2] if a else None,
                    D.canonical(s.rows[req['trade_id']]) if req['trade_id'] in s.rows else None,
                    D.canonical(a[3]) if a else None,D.canonical(a[1]) if a else None))
        elif q.startswith('SELECT request_digest'):
            a=s.audit.get(params[0]); self.one=(a[0],D.canonical(a[1])) if a else None
        elif q.startswith('SELECT pg_catalog.to_jsonb(t)'):
            snap=s.rows.get(params[0]); self.one=(D.canonical(snap),) if snap else None
        elif q.startswith('SELECT status, qty, entry_price, symbol, exit_order_id'):
            snap=s.rows.get(params['trade_id'])
            self.one=tuple(snap.get(key) for key in ('status','qty','entry_price','symbol','exit_order_id',
                'exit_time','exit_price','exit_reason','realized_pnl','entry_order_id','entry_time')) if snap else None
        elif q.startswith('SELECT last_after, last_ran_at FROM public.reconcile_state WHERE id=1'):
            self.one=copy.deepcopy(s.watermark)
        elif q.startswith('SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity'):
            self.many=[(name,'r',False,False,False) for name in ('order_events','trades')]
        elif q.startswith('SELECT pg_catalog.pg_try_advisory_xact_lock'):
            self.one=(True,)
        elif q.startswith('SELECT state, case_revision'):
            c=s.cases.get(params[0]); self.one=(c[0],c[1],D.canonical(c[2]),D.canonical(c[3]),D.canonical(c[4])) if c else None
        elif q.startswith('INSERT INTO public.trade_disposition_cases'):
            cid,tid,req,snap,manifest=params
            s.cases[cid]=['ACTIVE',1,D.document(req),D.document(snap),D.document(manifest)]
        elif q.startswith('UPDATE public.trade_disposition_cases'):
            c=s.cases[params[0]]; self.rowcount=int(c[0]=='ACTIVE' and c[1]==params[1])
            if self.rowcount: c[0]='REVOKED';c[1]+=1
        elif q.startswith('UPDATE public.trade_disposition_policy'):
            self.rowcount=int(s.revision==params[0]);s.revision+=self.rowcount
        elif q.startswith('INSERT INTO public.trade_disposition_audit'):
            if s.fail_audit: raise RuntimeError('synthetic audit storage failure')
            ref,digest,req,manifest,result,rev=params
            s.audit[ref]=(digest,D.document(result),rev,D.document(req))
        elif q.startswith('SELECT pg_catalog.to_jsonb(trades)'):
            scope='VALIDATED_PERFORMANCE'
            excluded={c[2]['trade_id'] for c in s.cases.values() if c[0]=='ACTIVE' and scope in c[4]['operational_scopes']}
            self.many=[(D.canonical(r),) for tid,r in sorted(s.rows.items()) if r['status']=='CLOSED' and tid not in excluded]
        elif q.startswith('INSERT INTO public.metrics_daily'): s.metrics_writes.append(copy.deepcopy(params))
        elif q.startswith('SELECT run_date, total_trades'):
            if s.metrics_writes:
                last=s.metrics_writes[-1]
                self.one=tuple(last[k] for k in ('run_date','total_trades','win_rate','net_pnl','expectancy','profit_factor','max_drawdown','sharpe','sortino','trade_policy'))
        elif q.startswith('SELECT trade_id, symbol'):
            self.many=copy.deepcopy(s.direct_rows)
        elif q.startswith(('SET ','LOCK TABLE ONLY public.','SELECT pg_catalog.pg_advisory')): pass
        else: raise AssertionError('Unmodeled SQL: '+q)
    def fetchone(self): return self.one
    def fetchall(self): return self.many
    def fetchmany(self,count): return self.many[:count]

@contextmanager
def conn_context(conn): yield conn


def db_functions(conn):
    env=dict(disposition=D, contextmanager=contextmanager, logger=LOG, Decimal=Decimal, math=__import__('math'),
        os=types.SimpleNamespace(environ={'APCA_API_BASE_URL':'https://paper-api.alpaca.markets','JBR_EXEC_PAPER':'true'}),
        _PAPER_DISPOSITION_BINDING=(*IDENTITY,'fixture-only'),
        datetime=datetime,timezone=timezone, Mapping=Mapping, _maybe_conn=lambda supplied:conn_context(supplied if supplied is not None else conn),
        extras=types.SimpleNamespace(RealDictCursor=object), _conn_or_none=lambda:conn,
        parser=types.SimpleNamespace(isoparse=lambda value:datetime.fromisoformat(value.replace('Z','+00:00'))),
        _coerce_date=lambda v:str(v), _log_write_result=lambda *args:None,
        normalize_ts=lambda value,**kw: datetime.fromisoformat(value) if isinstance(value,str) else value,
        _json_dumps_or_none=lambda v:json.dumps(v))
    return selected('db', {'_policy_write_lock','configure_disposition_identity','_paper_disposition_identity','trade_policy','_policy_read_transaction','_reconciliation_read_transaction',
        'reconciliation_policy_status','reconciliation_publication','_performance_input_records',
        'performance_input','performance_hold_visibility','performance_evidence','bind_performance_summary',
        'performance_summary_values','validated_performance_snapshot','validated_performance_artifact','validated_metrics_summary',
        'upsert_metrics_daily','_reconciliation_trade_records','get_open_trades','get_closed_trades_missing_exit',
        'close_trade','reconcile_sell_fill','_open_exit_lineage_guard','get_reconcile_state',
        '_lineage_uuid','_lineage_number','_lineage_time','_lineage_rows',
        '_open_exit_lineage_record','_require_open_exit_lineage'},env)

class PolicyTests(unittest.TestCase):
    def setUp(self):
        D._RUNTIME_IDENTITY=None
        D.bind_runtime_identity(*IDENTITY,environment='paper',evidence_ref='fixture-only',verified=True)
        self.s=Store();self.req,self.manifest=approval(self.s.rows[110]);self.db=db_functions(self.s)
    def apply(self): return D.change_disposition(self.s,self.req,self.manifest,IDENTITY)
    def test_unbound_and_nonpaper_identity_refuse(self):
        D._RUNTIME_IDENTITY=None
        with self.assertRaises(D.TradePolicyError): D.runtime_identity()
        for env,verified in [('live',True),('paper',False)]:
            with self.assertRaises(D.TradePolicyError):D.bind_runtime_identity(*IDENTITY,environment=env,evidence_ref='x',verified=verified)
    def test_identity_binding_cannot_change(self):
        with self.assertRaises(D.TradePolicyError):D.bind_runtime_identity(IDENTITY[0]+'x',IDENTITY[1],environment='paper',evidence_ref='fixture-only',verified=True)
    def test_scope_classification_and_provenance_are_bound(self):
        for key,value in [('classification','PROBABLE_TEST_GENERATED'),('provenance_proved',True),
            ('operational_scopes',[]),('operational_scopes',['ACTIVE_RECONCILIATION']*2),('operational_scopes',['HISTORY'])]:
            with self.subTest(key=key,value=value):
                m=copy.deepcopy(self.manifest);m[key]=value;r=dict(self.req,manifest_sha256=D.fingerprint(m))
                with self.assertRaises(D.TradePolicyError):D.change_disposition(self.s,r,m,IDENTITY)
        self.assertEqual(self.s.cases,{})
    def test_apply_replay_revoke_preserves_all_history(self):
        before=copy.deepcopy(self.s.rows);first=self.apply();again=self.apply()
        self.assertEqual(first,again);self.assertEqual(self.s.revision,1);self.assertEqual(len(self.s.audit),1)
        revoke=dict(self.req,action='REVOKE',approval_ref='fixture-revoke',expected_policy_revision=1,expected_case_revision=1)
        result=D.change_disposition(self.s,revoke,self.manifest,IDENTITY)
        self.assertEqual(result['state'],'REVOKED');self.assertEqual(self.s.revision,2)
        self.assertEqual(self.s.rows,before);self.assertEqual(len(self.s.audit),2)
    def test_failed_audit_rolls_back_case_and_revision(self):
        self.s.fail_audit=True
        with self.assertRaises(RuntimeError):self.apply()
        self.assertEqual((self.s.cases,self.s.audit,self.s.revision),({},{},0))
    def test_policy_cas_and_exact_row_version_refuse(self):
        self.s.revision=1
        with self.assertRaises(D.TradePolicyError):self.apply()
        self.s.revision=0;self.s.rows[110]['qty']=Decimal('2.000000000000000001')
        with self.assertRaises(D.TradePolicyError):self.apply()
    def test_scope_only_filters_selected_work_and_all_cases_protect_writes(self):
        self.req,self.manifest=approval(self.s.rows[110],('VALIDATED_PERFORMANCE',));self.apply()
        policy=D.read_policy(self.s.cursor(),IDENTITY,scope='ACTIVE_RECONCILIATION')
        self.assertEqual(policy.excluded_ids,frozenset());self.assertEqual(policy.all_excluded_ids,{110})
        self.assertFalse(policy.metadata()['recovery_complete'])
        with self.assertRaises(D.TradePolicyError):D.require_writable(policy,110)
        self.assertIn('VALIDATED_PERFORMANCE',D.eligible_sql(scope='VALIDATED_PERFORMANCE'))
    def test_active_receipt_and_current_snapshot_are_revalidated(self):
        self.apply();self.s.rows[110]['updated_at']='2026-01-02T00:00:00+00:00'
        with self.assertRaises(D.TradePolicyError):D.read_policy(self.s.cursor(),IDENTITY,scope='ACTIVE_RECONCILIATION')
    def test_audit_read_required_even_no_active_cases_and_trusted_namespace(self):
        D.read_policy(self.s.cursor(),IDENTITY,scope='ACTIVE_RECONCILIATION')
        sql='\n'.join(q for q,p in self.s.trace)
        self.assertIn('SET LOCAL search_path = pg_catalog',sql)
        self.assertIn('public.trade_disposition_audit',sql);self.assertIn('public.trades',sql)
        self.assertNotIn('FROM trades',sql)
    def test_strict_read_predicate_precedes_limit_history_unfiltered(self):
        self.db['get_open_trades'](self.s,strict=True,expected_policy_revision=0)
        q=next(q for q,p in reversed(self.s.trace) if 'SELECT trade_id, symbol' in q)
        self.assertLess(q.index('ACTIVE_RECONCILIATION'),q.index('LIMIT'))
        self.s.trace=[];self.db['get_open_trades'](self.s,strict=False)
        q=next(q for q,p in self.s.trace if 'SELECT trade_id, symbol' in q)
        self.assertIn("status='OPEN' AND TRUE",q);self.assertNotIn('disposition.manifest',q)
    def test_strict_saturation_and_revision_refuse_no_fallback(self):
        self.s.direct_rows=[{'trade_id':1,'symbol':'CRC'}]
        with self.assertRaises(RuntimeError):self.db['get_open_trades'](self.s,limit=1,strict=True)
        with self.assertRaises(D.TradePolicyError):self.db['get_open_trades'](self.s,strict=True,expected_policy_revision=7)
    def test_default_watermark_read_uses_public_namespace_under_policy_transaction(self):
        self.s.watermark={'last_after':STAMP,'last_ran_at':STAMP}
        result=self.db['get_reconcile_state'](self.s,strict=True)
        self.assertEqual(result,dict.fromkeys(('last_after','last_ran_at'),datetime.fromisoformat(STAMP)))
        queries=[q for q,_ in self.s.trace]
        actual='SELECT last_after, last_ran_at FROM public.reconcile_state WHERE id=1'
        self.assertIn(actual,queries)
        self.assertLess(queries.index('SET LOCAL search_path = pg_catalog'),queries.index(actual))
    def test_approved_rows_never_reach_legacy_update_or_fill_event(self):
        self.apply();self.s.trace=[]
        self.assertFalse(self.db['close_trade'](self.s,110,None,STAMP,None,'POSITION_CLOSED',expected_policy_revision=1))
        self.assertFalse(self.db['reconcile_sell_fill'](self.s,110,symbol='CRC',qty='2',order_id='sell',status='filled',
            event_time=STAMP,raw={},exit_price='11',exit_reason='SELL',expected_policy_revision=1))
        self.assertFalse(any('UPDATE public.trades' in q or 'INSERT INTO public.order_events' in q for q,p in self.s.trace))
    def test_performance_unknown_pnl_refuses_without_zero(self):
        policy=D.Policy(*IDENTITY,0,(), 'VALIDATED_PERFORMANCE')
        with self.assertRaises(D.TradePolicyError):self.db['performance_input'](self.s,policy)
    def test_performance_empty_and_exact_zero_input_success(self):
        self.apply();policy=D.read_policy(self.s.cursor(),IDENTITY,scope='VALIDATED_PERFORMANCE')
        records=self.db['performance_input'](self.s,policy);self.assertEqual(records,[])
        ev=self.db['performance_evidence'](policy,records);self.assertEqual(ev['input_count'],0)
        self.s=Store();self.db=db_functions(self.s);self.s.rows[110]=row(pnl=Decimal('0'))
        records=self.db['performance_input'](self.s,D.Policy(*IDENTITY,0,(), 'VALIDATED_PERFORMANCE'))
        self.assertEqual(records[0]['realized_pnl'],Decimal('0'))
    def test_publication_binds_input_summary_and_policy(self):
        self.s.rows[110]=row(pnl=Decimal('2'))
        policy=pending_allocation_policy();records=self.db['performance_input'](self.s,policy)
        summary={'total_trades':1,'net_pnl':2.0,'profit_factor':2.0}
        ev=self.db['bind_performance_summary'](self.db['performance_evidence'](policy,records),summary)
        self.assertTrue(self.db['upsert_metrics_daily'](summary,date(2026,1,1),trade_policy_metadata=ev))
        self.assertEqual(len(self.s.metrics_writes),1)
        self.s.rows[110]['realized_pnl']=Decimal('3')
        with self.assertRaises(D.TradePolicyError):self.db['upsert_metrics_daily'](summary,date(2026,1,1),trade_policy_metadata=ev)
        self.assertEqual(len(self.s.metrics_writes),1)
    def test_publication_timeouts_precede_first_exclusive_policy_lock(self):
        self.s.rows={};summary={'total_trades':0,'net_pnl':0}
        policy=pending_allocation_policy()
        ev=self.db['bind_performance_summary'](self.db['performance_evidence'](policy,[]),summary)
        self.assertTrue(self.db['upsert_metrics_daily'](summary,date(2026,1,1),trade_policy_metadata=ev))
        queries=[q for q,p in self.s.trace]
        first_lock=next(i for i,q in enumerate(queries) if 'pg_advisory_xact_lock(' in q)
        for setting in ("SET LOCAL search_path = pg_catalog", "SET LOCAL lock_timeout = '1500ms'", "SET LOCAL statement_timeout = '5000ms'"):
            self.assertLess(queries.index(setting), first_lock)
    def test_publication_changed_payload_or_epoch_and_unbound_csv_refuse(self):
        self.s.rows={};policy=pending_allocation_policy();summary={'total_trades':0,'net_pnl':0}
        ev=self.db['bind_performance_summary'](self.db['performance_evidence'](policy,[]),summary)
        with self.assertRaises(D.TradePolicyError):self.db['upsert_metrics_daily'](dict(summary,net_pnl=9),date(2026,1,1),trade_policy_metadata=ev)
        self.s.revision=1
        with self.assertRaises(D.TradePolicyError):self.db['upsert_metrics_daily'](summary,date(2026,1,1),trade_policy_metadata=ev)
        with self.assertRaises(D.TradePolicyError):self.db['upsert_metrics_daily'](summary,date(2026,1,1))
        self.assertEqual(self.s.metrics_writes,[])
    def test_infinite_derived_ratio_is_explicitly_unsupported(self):
        with self.assertRaises(D.TradePolicyError):self.db['bind_performance_summary']({}, {'profit_factor':float('inf')})


def normalized_time(value, **kw):
    if value is None:return None
    return datetime.fromisoformat(value.replace('Z','+00:00')) if isinstance(value,str) else value

class CallerTests(unittest.TestCase):
    def setUp(self):
        case=D.ActiveCase('case110',110,'approval','UNRESOLVED',('ACTIVE_RECONCILIATION',))
        self.policy=D.Policy(*IDENTITY,1,(case,), 'ACTIVE_RECONCILIATION').metadata()
        self.writes=[];self.orders=[];self.open=[];self.closed=[];self.policy_calls=0
        self.provider_error=False;self.final_changed=False;self.publication_held=False;self.publication_exit_error=False
        def status(*args,**kw):
            self.policy_calls+=1
            if self.final_changed and self.policy_calls>1:raise D.TradePolicyError('TRADE_POLICY_REVISION_CHANGED')
            return copy.deepcopy(self.policy)
        @contextmanager
        def publication(*args,**kw):
            current=status(*args,**kw)
            self.publication_held=True
            try:
                yield current
                if self.publication_exit_error:raise D.TradePolicyError('TRADE_POLICY_TRANSACTION_FAILED')
            finally:self.publication_held=False
        self.db=types.SimpleNamespace(db_enabled=lambda:True,get_db_conn=lambda:object(),
            reconciliation_policy_status=status,reconciliation_publication=publication,
            get_reconcile_state=lambda *a,**kw:{'last_after':None},
            get_open_trade_exit_lineage=lambda *a,**kw:{},
            get_open_trades=lambda *a,**kw:self.open,get_closed_trades_missing_exit=lambda *a,**kw:self.closed,
            normalize_ts=normalized_time,set_reconcile_state=lambda *a:self.writes.append(a),
            reconcile_sell_fill=lambda *a,**kw:False,close_trade=lambda *a,**kw:False)
        def fetch(**kw):
            self.assertFalse(self.publication_held,'provider I/O must precede the publication lock')
            if self.provider_error:raise RuntimeError('synthetic unavailable')
            return self.orders
        env=dict(db=self.db,LOGGER=LOG,logging=logging,Mapping=Mapping,datetime=datetime,timezone=timezone,
            timedelta=timedelta,Decimal=Decimal,InvalidOperation=InvalidOperation,
            isoparse=normalized_time,_order_snapshot=lambda order:dict(order),
            alpaca_list_orders_http=fetch,log_info=lambda *a,**kw:None,
            _RECONCILE_DB_DISABLED_LOGGED=False)
        selected('execute_trades',{'_operational_reconciliation_complete','_run_auto_reconcile'},env)
        method=next(n for n in ast.parse(SOURCE['execute_trades']).body if isinstance(n,ast.ClassDef) and n.name=='TradeExecutor')
        method=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='reconcile_closed_trades')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])),'actual_reconciler','exec'),env)
        self.env=env
        self.executor=types.SimpleNamespace(metrics=types.SimpleNamespace(reconciliation=None),config=types.SimpleNamespace(
            reconcile_use_watermark=True,reconcile_limit=5,reconcile_lookback_days=7,reconcile_overlap_secs=300,
            reconcile_auto=True),client=object())
        self.executor.reconcile_closed_trades=lambda **kw:env['reconcile_closed_trades'](self.executor,**kw)
    def test_actual_reconcile_all_scoped_work_complete_recovery_held(self):
        self.assertTrue(self.executor.reconcile_closed_trades())
        status=self.executor.metrics.reconciliation
        self.assertTrue(status['operational_complete']);self.assertFalse(status['recovery_complete'])
        self.assertFalse(status['activity_ingestion_complete']);self.assertFalse(status['accounting_complete'])
        self.assertFalse(status['watermark_advanced']);self.assertEqual(self.writes,[])
        self.assertEqual(status['recovery_pending_case_ids'],['case110'])
    def test_actual_auto_requires_scoped_result_not_legacy_true(self):
        self.assertTrue(self.env['_run_auto_reconcile'](self.executor,self.executor.config,'start'))
        self.executor.reconcile_closed_trades=lambda **kw:True
        self.assertFalse(self.env['_run_auto_reconcile'](self.executor,self.executor.config,'start'))
    def test_provider_failure_saturation_and_future_facts_never_operational_success(self):
        valid=dict(id='sell',symbol='CRC',side='sell',status='filled',filled_at=STAMP,filled_qty='2',filled_avg_price='11')
        for mode in ('failure','saturated','future'):
            with self.subTest(mode=mode):
                self.provider_error=mode=='failure'
                self.orders=[dict(valid,id=str(i)) for i in range(5)] if mode=='saturated' else [dict(valid,updated_at='2999-01-01T00:00:00+00:00')]
                self.assertFalse(self.executor.reconcile_closed_trades());self.assertEqual(self.writes,[])
    def test_unresolved_nonexcluded_and_final_policy_change_hold(self):
        self.closed=[dict(trade_id=42,symbol='OTHER',qty='2',entry_price='10')]
        self.assertFalse(self.executor.reconcile_closed_trades())
        self.closed=[];self.policy_calls=0;self.final_changed=True
        self.assertFalse(self.executor.reconcile_closed_trades());self.assertEqual(self.writes,[])
    def test_success_metrics_and_logs_publish_while_revision_is_locked(self):
        calls=[];fixture=self
        class Receipt(dict):
            def update(self,*args,**kwargs):
                proposed=dict(*args,**kwargs)
                if proposed.get('operational_complete') is True:
                    fixture.assertTrue(fixture.publication_held);calls.append('metrics_locked')
                return super().update(proposed)
        class Metrics:
            @property
            def reconciliation(self):return self.value
            @reconciliation.setter
            def reconciliation(self,value):self.value=Receipt(value)
        class ReceiptLogger:
            def info(self,message,*args):
                if 'RECONCILE_OPERATIONAL_COMPLETE' in message or 'operational_complete=true' in message:
                    fixture.assertTrue(fixture.publication_held);calls.append('operator_waits_at_success')
            def warning(self,*args):pass
        self.executor.metrics=Metrics();self.env['LOGGER']=ReceiptLogger()
        self.assertTrue(self.executor.reconcile_closed_trades())
        self.assertEqual(calls,['metrics_locked','operator_waits_at_success','operator_waits_at_success'])
        self.assertFalse(self.publication_held);self.assertEqual(self.writes,[])
    def test_publication_transaction_failure_clears_operational_success(self):
        self.publication_exit_error=True
        self.assertFalse(self.executor.reconcile_closed_trades())
        self.assertFalse(self.executor.metrics.reconciliation['operational_complete'])
        self.assertEqual(self.executor.metrics.reconciliation['reason'],'POLICY_CHANGED_OR_UNAVAILABLE')
        self.assertFalse(self.publication_held);self.assertEqual(self.writes,[])
    def test_unbound_historical_crc_never_assigns_latest_partial_order(self):
        self.closed=[dict(trade_id=110,symbol='CRC',qty='30',entry_price='65',exit_order_id=None)]
        self.orders=[dict(id='second-crc-exit',symbol='CRC',side='sell',status='filled',filled_at=STAMP,
            filled_qty='15',filled_avg_price='63.26')]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.assertFalse(self.executor.reconcile_closed_trades());self.assertEqual(self.writes,[])
    def test_bound_historical_partial_order_is_held_before_db_write(self):
        self.closed=[dict(trade_id=110,symbol='CRC',qty='30',entry_price='65',exit_order_id='second-crc-exit')]
        self.orders=[dict(id='second-crc-exit',symbol='CRC',side='sell',status='filled',filled_at=STAMP,
            filled_qty='15',filled_avg_price='63.26')]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.assertFalse(self.executor.reconcile_closed_trades());self.assertEqual(self.writes,[])
    def test_historical_lookup_uses_exact_bound_order_instead_of_latest_symbol_order(self):
        self.closed=[dict(trade_id=42,symbol='OTHER',qty='2',entry_price='10',exit_order_id='bound')]
        self.orders=[dict(id=order_id,symbol='OTHER',side='sell',status='filled',filled_at=stamp,
            filled_qty='2',filled_avg_price='11') for order_id,stamp in
            [('different-later-order','2026-01-02T00:00:00+00:00'),('bound',STAMP)]]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.assertTrue(self.executor.reconcile_closed_trades())
        self.assertEqual([write['order_id'] for write in self.writes],['bound'])
    def test_open_equal_quantity_unbound_order_does_not_create_financial_ownership(self):
        self.open=[dict(trade_id=42,symbol='OTHER',qty='2',entry_time=STAMP,exit_order_id=None)]
        self.executor.client=types.SimpleNamespace(get_all_positions=lambda:[types.SimpleNamespace(symbol='OTHER',qty='2')])
        self.orders=[dict(id='unbound-order',symbol='OTHER',side='sell',status='filled',filled_at=STAMP,
            filled_qty='2',filled_avg_price='11')]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.assertFalse(self.executor.reconcile_closed_trades());self.assertEqual(self.writes,[])
    def test_open_equal_quantity_orders_match_existing_ids_instead_of_trade_sort_order(self):
        self.open=[dict(trade_id=trade_id,symbol='OTHER',qty='2',entry_time=stamp,exit_order_id=order_id)
            for trade_id,stamp,order_id in [(41,STAMP,'bound-one'),(42,'2026-01-02T00:00:00+00:00','bound-two')]]
        self.executor.client=types.SimpleNamespace(get_all_positions=lambda:[types.SimpleNamespace(symbol='OTHER',qty='4')])
        self.orders=[dict(id=order_id,symbol='OTHER',side='sell',status='filled',filled_at=stamp,
            filled_qty='2',filled_avg_price='11') for order_id,stamp in
            [('bound-one','2026-01-04T00:00:00+00:00'),('bound-two','2026-01-03T00:00:00+00:00')]]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.assertTrue(self.executor.reconcile_closed_trades())
        self.assertEqual([(write['trade_id'],write['order_id']) for write in self.writes],[(41,'bound-one'),(42,'bound-two')])
    def test_flat_open_lineage_child_ignores_newer_unrelated_sell_and_never_precloses(self):
        self.open=[dict(trade_id=42,symbol='OTHER',qty='2',entry_time=STAMP,exit_order_id=None)]
        self.db.get_open_trade_exit_lineage=lambda *a,**kw:{42:'exact-child'}
        self.executor.client=types.SimpleNamespace(get_all_positions=lambda:[])
        self.orders=[dict(id=order_id,symbol='OTHER',side='sell',status='filled',filled_at=stamp,
            filled_qty='2',filled_avg_price='11') for order_id,stamp in
            [('unrelated-newer','2026-01-03T00:00:00+00:00'),('exact-child','2026-01-02T00:00:00+00:00')]]
        self.db.reconcile_sell_fill=lambda *a,**kw:self.writes.append(kw) or True
        self.db.close_trade=lambda *a,**kw:self.fail('exact child must be processed while OPEN')
        self.assertTrue(self.executor.reconcile_closed_trades())
        self.assertEqual([write['order_id'] for write in self.writes],['exact-child'])
        self.assertIsNone(self.open[0]['exit_order_id'],'candidate mapping must not fabricate stored ownership')
        self.assertFalse(self.executor.metrics.reconciliation['watermark_advanced'])
    def test_reconciliation_never_advances_watermark_or_claims_full_coverage(self):
        tree=ast.parse(SOURCE['execute_trades']); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TradeExecutor')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reconcile_closed_trades')
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='set_reconcile_state' for n in ast.walk(node)))
        self.assertNotIn('RECONCILE_COVERAGE_COMPLETE',ast.get_source_segment(SOURCE['execute_trades'],node))
    def test_actual_metrics_main_policy_error_returns_two(self):
        def fail(argv):raise D.TradePolicyError('TRADE_POLICY_ROW_CHANGED')
        env=selected('metrics',{'main'},dict(db=types.SimpleNamespace(disposition=D),logger=LOG,_run_metrics=fail))
        self.assertEqual(env['main']([]),2)
    def test_global_ownership_query_is_not_scope_filtered(self):
        node=next(n for n in ast.parse(SOURCE['db']).body if isinstance(n,ast.FunctionDef) and n.name=='reconcile_sell_fill')
        sqls=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        owner=next(q for q in sqls if 'WHERE exit_order_id=' in q)
        self.assertIn('public.trades',owner);self.assertNotIn('disposition',owner)


class Series(list):
    def mean(self): return sum(self)/len(self) if self else 0
    def astype(self, kind): return self
    @property
    def str(self): return self
    def upper(self): return Series([str(x).upper() for x in self])

class Frame:
    """Only pandas leaf protocol needed by actual load/workflow; no arithmetic mocked as measured."""
    def __init__(self, rows=None, columns=None):
        self.rows=copy.deepcopy(rows or []);self.columns=list(columns or (self.rows[0].keys() if self.rows else []));self.attrs={}
    @property
    def empty(self): return not self.rows
    @property
    def shape(self): return (len(self.rows),len(self.columns))
    def __len__(self): return len(self.rows)
    def __getitem__(self,key):return Series([r.get(key) for r in self.rows])
    def __setitem__(self,key,values):
        if key not in self.columns:self.columns.append(key)
        for row,value in zip(self.rows,values):row[key]=value
    def rename(self,*,columns):return Frame([{columns.get(k,k):v for k,v in row.items()} for row in self.rows], [columns.get(k,k) for k in self.columns])
    def to_dict(self,orient='records'):return copy.deepcopy(self.rows)
    def copy(self):
        result=Frame(self.rows,self.columns);result.attrs=copy.deepcopy(self.attrs);return result

class MetricsWorkflowTests(unittest.TestCase):
    def setUp(self):
        D._RUNTIME_IDENTITY=None;D.bind_runtime_identity(*IDENTITY,environment='paper',evidence_ref='fixture-only',verified=True)
        self.s=Store();self.s.rows={110:row(pnl=Decimal('0'))};self.actualdb=db_functions(self.s)
        self.pd=types.SimpleNamespace(DataFrame=Frame,to_numeric=lambda values,**kw:values,to_datetime=lambda values,**kw:values)
        self.loader=selected('metrics',{'_load_trades_from_db'},dict(db=types.SimpleNamespace(**self.actualdb),pd=self.pd))['_load_trades_from_db']
    def test_actual_performance_loader_binds_exact_finite_zero_snapshot(self):
        df=self.loader(self.s)
        self.assertEqual(df.rows[0]['net_pnl'],Decimal('0'));self.assertEqual(df.attrs['trade_policy']['input_count'],1)
        self.assertEqual(df.attrs['trade_policy']['input_fingerprint'],D.fingerprint({'rows':[self.s.rows[110]]}))
    def test_actual_loader_missing_result_refuses_instead_of_zero_or_csv(self):
        self.s.rows[110]=row()
        with self.assertRaises(D.TradePolicyError):self.loader(self.s)
    def test_actual_loader_empty_after_approved_performance_scope_is_explicit(self):
        req,manifest=approval(self.s.rows[110]);D.change_disposition(self.s,req,manifest,IDENTITY)
        df=self.loader(self.s)
        self.assertTrue(df.empty);self.assertEqual(df.attrs['trade_policy']['input_count'],0)
        self.assertEqual(df.attrs['trade_policy']['policy']['excluded_count'],1)
        self.assertFalse(df.attrs['trade_policy']['recovery_complete'])
    def workflow(self, *, backtest=True, publication=True, mutate=False):
        effects=[];frame=self.loader(self.s)
        def publish(summary,day,**kw):
            if mutate:self.s.rows[110]['realized_pnl']=Decimal('4')
            effects.append('summary')
            return self.actualdb['upsert_metrics_daily'](summary,day,**kw) if publication else False
        db=types.SimpleNamespace(disposition=D,db_enabled=lambda:True,get_db_conn=lambda:self.s,
            bind_performance_summary=self.actualdb['bind_performance_summary'],upsert_metrics_daily=publish,
            insert_top_candidates=lambda *a:effects.append('candidates'))
        class NoArtifact:
            def __init__(self,*args):pass
            def __truediv__(self,other):return self
            def exists(self):return False
        env=dict(db=db,pd=self.pd,logger=LOG,parse_args=lambda argv:types.SimpleNamespace(run_date='2026-01-01'),
            _coerce_run_date=lambda v:date(2026,1,1),load_results=lambda **kw:Frame([],['net_pnl'] if backtest else []),
            get_latest_screener_candidates=lambda day:(Frame(),None),_load_trades_from_db=lambda conn:frame,
            rank_and_filter_candidates=lambda *a:Frame(),validate_numeric=lambda value,*a:value,
            calculate_metrics=lambda value:{'total_trades':len(value),'net_pnl':0,'profit_factor':0},
            BASE_DIR='fixture',Path=NoArtifact,ensure_canonical_metrics=lambda value:value,
            derive_prefix_counts_from_scored_candidates=lambda value:{},write_screener_metrics_json=lambda *a:effects.append('artifact'))
        selected('metrics',{'_run_metrics','main'},env)
        import io
        output=io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(output): result=env['main']([])
        return result,effects,output.getvalue()
    def test_actual_metrics_workflow_success_is_scoped_and_payload_bound(self):
        result,effects,output=self.workflow()
        self.assertEqual(result,0);self.assertEqual(effects,['candidates','summary','artifact'])
        self.assertEqual(len(self.s.metrics_writes),1);self.assertIn('recovery_complete=false',output)
    def test_actual_workflow_unsupported_trade_ranking_refuses_before_any_publication(self):
        result,effects,output=self.workflow(backtest=False)
        self.assertEqual(result,2);self.assertEqual(effects,[]);self.assertNotIn('METRICS COMPLETE',output)
    def test_actual_workflow_write_false_and_changed_snapshot_do_not_complete(self):
        for kw in ({'publication':False},{'mutate':True}):
            with self.subTest(kw=kw):
                result,effects,output=self.workflow(**kw)
                self.assertEqual(result,2);self.assertNotIn('artifact',effects);self.assertNotIn('METRICS COMPLETE',output)


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        D._RUNTIME_IDENTITY=None;D.bind_runtime_identity(*IDENTITY,environment='paper',evidence_ref='fixture-only',verified=True)
        self.s=Store();self.s.rows={63:row(63,Decimal('0'))};self.functions=db_functions(self.s)
        self.db=types.SimpleNamespace(**self.functions)
        self.pd=types.SimpleNamespace(DataFrame=Frame,to_numeric=lambda values,**kw:values,to_datetime=lambda values,**kw:values)
        self.memory={};self.writes=[]
        outer=self
        class MemoryPath:
            def __init__(self,path='cache'):self.path=path.path if isinstance(path,MemoryPath) else str(path)
            def read_text(self,**kw):
                if self.path not in outer.memory:raise FileNotFoundError()
                return outer.memory[self.path]
            @property
            def parent(self):return self
            def mkdir(self,**kw):outer.writes.append('mkdir')
        self.Path=MemoryPath
        def atomic(path,data):self.writes.append('cache');self.memory[path.path]=data.decode('utf-8')
        self.perf=selected('trade_performance',{'load_trades_log','write_cache','read_cache'},dict(db=self.db,pd=self.pd,
            Path=MemoryPath,BASE_DIR='fixture',CACHE_PATH='cache',datetime=datetime,timezone=timezone,
            _coerce_json=lambda value:value,atomic_write_bytes=atomic))
    def write(self):
        frame=self.perf['load_trades_log']()
        self.perf['write_cache'](frame,{'ALL':{'trades':len(frame),'net_pnl':0}},'cache')
        return frame
    def test_runtime_revision_boolean_cannot_satisfy_integer_cas(self):
        self.s.revision=1
        with self.assertRaises(D.TradePolicyError):self.functions['trade_policy'](self.s,expected_revision=True)
    def test_canonical_trade_performance_loader_has_current_input_basis_no_csv(self):
        frame=self.perf['load_trades_log']('legacy-base-must-not-be-read')
        self.assertEqual(frame.rows[0]['trade_id'],63);self.assertEqual(frame.rows[0]['pnl'],Decimal('0'))
        self.assertEqual(frame.attrs['trade_policy']['policy']['scope'],'VALIDATED_PERFORMANCE')
        self.assertEqual(self.writes,[])
    def test_canonical_loader_missing_pnl_holds_without_dropping_or_zero(self):
        self.s.rows[63]['realized_pnl']=None
        with self.assertRaises(D.TradePolicyError):self.perf['load_trades_log']()
    def test_cache_roundtrip_is_bound_to_exact_current_input_and_result(self):
        self.write();payload=self.perf['read_cache']('cache')
        self.assertEqual(payload['validation_status'],'VALIDATED_PERFORMANCE')
        self.assertEqual(payload['df'].attrs['trade_policy']['input_count'],1)
        raw=D.document(self.memory['cache']);raw['trades'][0]['pnl']=9
        self.memory['cache']=D.canonical(raw)
        with self.assertRaises(D.TradePolicyError):self.perf['read_cache']('cache')
    def test_unbound_legacy_cache_refuses_and_unbound_write_has_no_file_effect(self):
        self.memory['cache']='{"trades":[],"summary":{}}'
        with self.assertRaises(D.TradePolicyError):self.perf['read_cache']('cache')
        with self.assertRaises(D.TradePolicyError):self.perf['write_cache'](Frame(),{},'other')
        self.assertEqual(self.writes,[]);self.assertNotIn('other',self.memory)
    def test_cache_apply_and_revoke_revision_each_invalidate_saved_input(self):
        self.write();req,manifest=approval(self.s.rows[63]);D.change_disposition(self.s,req,manifest,IDENTITY)
        with self.assertRaises(D.TradePolicyError):self.perf['read_cache']('cache')
        revoke=dict(req,action='REVOKE',approval_ref='revoke63',expected_policy_revision=1,expected_case_revision=1)
        D.change_disposition(self.s,revoke,manifest,IDENTITY)
        with self.assertRaises(D.TradePolicyError):self.perf['read_cache']('cache')
    def test_current_metrics_reader_rejects_stale_input_and_accepts_exact_zero(self):
        records,evidence=self.functions['validated_performance_snapshot']()
        summary={'total_trades':1,'net_pnl':0,'profit_factor':0}
        self.functions['upsert_metrics_daily'](summary,date(2026,1,1),trade_policy_metadata=self.functions['bind_performance_summary'](evidence,summary))
        result=self.functions['validated_metrics_summary']()
        self.assertEqual(result['validation_status'],'VALIDATED_PERFORMANCE');self.assertEqual(result['net_pnl'],0)
        self.s.rows[63]['realized_pnl']=Decimal('2')
        with self.assertRaises(D.TradePolicyError):self.functions['validated_metrics_summary']()
    def test_dashboard_summary_holds_unknown_without_csv_or_zero(self):
        env=selected('data_io',{'metrics_summary_snapshot'},dict(db=self.db,logger=LOG))
        result=env['metrics_summary_snapshot']()
        self.assertEqual(result['validation_status'],'HELD_UNVALIDATED_PERFORMANCE')
        self.assertIsNone(result['net_pnl']);self.assertIsNone(result['total_trades'])
    def test_dashboard_analytics_uses_scoped_loader_and_never_legacy_fallback(self):
        frame=self.perf['load_trades_log']()
        env=selected('dashboard_app',{'_load_trades_analytics_frame'},dict(db=self.db,pd=self.pd,logger=LOG,
            load_validated_performance_trades=lambda:frame,_normalize_trades_api_frame=lambda value:value))
        result,source,detail=env['_load_trades_analytics_frame']()
        self.assertEqual(detail,'validated-performance');self.assertEqual(result.attrs['trade_policy']['input_count'],1)
        def unavailable():raise D.TradePolicyError('TRADE_POLICY_UNAVAILABLE')
        env['load_validated_performance_trades']=unavailable
        result,source,detail=env['_load_trades_analytics_frame']()
        self.assertTrue(result.empty);self.assertEqual(detail,'trades-db-held-unvalidated-performance')
    def test_dashboard_cache_and_callback_ignore_stale_browser_store_and_show_hold(self):
        class Components:
            def __getattr__(self,name):return lambda *args,**kw:{'component':name,'args':args,'props':kw}
        def unvalidated(*args):raise D.TradePolicyError('TRADE_POLICY_ARTIFACT_INPUT_CHANGED')
        env=selected('dashboard_app',{'_load_trade_perf_cache','_performance_hold_message','update_trade_performance_tab'},dict(db=self.db,logger=LOG,
            read_validated_performance_cache=unvalidated,TRADE_PERFORMANCE_CACHE='legacy',dbc=Components(),
            _empty_trade_perf_fig=lambda title:{'title':title}))
        result=env['update_trade_performance_tab']('ALL',{'trades':[{'trade_id':63,'pnl':99999}],
            'summary':{'ALL':{'net_pnl':99999}},'validation_status':'VALIDATED_PERFORMANCE'})
        self.assertEqual(len(result),9);self.assertEqual(result[7],[])
        self.assertIn('held',str(result[0]).lower());self.assertNotIn('99999',str(result))
    def test_stats_held_returns_no_synthetic_zero_aggregate_rows(self):
        env=selected('dashboard_app',{'api_trades_stats','_trades_api_db_ready'},dict(request=types.SimpleNamespace(args={}),
            _parse_trades_range=lambda value,**kw:'all',_load_trades_analytics_frame=lambda:(Frame(),'postgres','trades-db-held-unvalidated-performance'),
            _TRADES_RANGE_ORDER=['d','all'],_build_range_metrics=lambda *a:(_ for _ in ()).throw(AssertionError('must not aggregate held as zero')),
            _record_trades_api_request=lambda **kw:False,_json_no_store=lambda payload:payload))
        result=env['api_trades_stats']()
        self.assertFalse(result['ok']);self.assertEqual(result['rows'],[])
        self.assertEqual(result['performance_status'],'HELD_UNVALIDATED_PERFORMANCE')
    def test_held_consumers_keep_current_policy_audit_and_unresolved_crc_visible(self):
        self.s.rows={63:dict(row(63),symbol='AAPL'),110:row(110)}
        req,manifest=approval(self.s.rows[63]);D.change_disposition(self.s,req,manifest,IDENTITY)
        summary=selected('data_io',{'metrics_summary_snapshot'},dict(db=self.db,logger=LOG))['metrics_summary_snapshot']()
        env=selected('dashboard_app',{'_load_trades_analytics_frame','_load_trade_perf_cache'},
            dict(db=self.db,pd=self.pd,logger=LOG,load_validated_performance_trades=self.perf['load_trades_log'],
                read_validated_performance_cache=self.perf['read_cache'],TRADE_PERFORMANCE_CACHE='cache',
                _normalize_trades_api_frame=lambda frame:frame))
        frame,source,detail=env['_load_trades_analytics_frame']()
        cache=env['_load_trade_perf_cache']()
        for visible in (summary,frame.attrs,cache):
            self.assertEqual(visible['policy_validation_status'],'VALIDATED_POLICY')
            self.assertEqual(visible['policy_revision'],1);self.assertEqual(visible['excluded_count'],1)
            self.assertEqual(visible['unresolved_trade_ids'],[110])
            policy=visible['trade_policy']['policy']
            self.assertEqual(policy['scope'],'VALIDATED_PERFORMANCE')
            self.assertEqual(policy['exclusions'][0]['trade_id'],63)
            self.assertEqual(policy['exclusions'][0]['approval_ref'],req['approval_ref'])
        self.assertIsNone(summary['net_pnl']);self.assertTrue(frame.empty);self.assertEqual(cache['trades'],[])
        api=selected('dashboard_app',{'api_trades_stats','_trades_api_db_ready'},dict(request=types.SimpleNamespace(args={}),
            _parse_trades_range=lambda value,**kw:'all',_load_trades_analytics_frame=lambda:(frame,source,detail),
            _TRADES_RANGE_ORDER=['all'],_build_range_metrics=lambda *a:(_ for _ in ()).throw(AssertionError('HELD aggregate')),
            _record_trades_api_request=lambda **kw:False,_json_no_store=lambda payload:payload))['api_trades_stats']()
        self.assertFalse(api['ok']);self.assertEqual(api['rows'],[])
        self.assertEqual(api['excluded_count'],1);self.assertEqual(api['unresolved_trade_ids'],[110])
    def test_policy_read_failure_keeps_visibility_counts_unknown(self):
        def unavailable(*args,**kwargs):raise D.TradePolicyError('TRADE_POLICY_UNAVAILABLE')
        self.functions['trade_policy']=unavailable
        summary=selected('data_io',{'metrics_summary_snapshot'},dict(db=self.db,logger=LOG))['metrics_summary_snapshot']()
        self.assertEqual(summary['policy_validation_status'],'UNAVAILABLE')
        for key in ('excluded_count','policy_revision','trade_policy','unresolved_trade_ids','net_pnl'):
            self.assertIsNone(summary[key])
    def test_held_cache_visibility_tracks_apply_and_revoke_revisions(self):
        self.write()
        req,manifest=approval(self.s.rows[63]);D.change_disposition(self.s,req,manifest,IDENTITY)
        self.s.rows[110]=row(110)
        env=selected('dashboard_app',{'_load_trade_perf_cache'},dict(db=self.db,logger=LOG,
            read_validated_performance_cache=self.perf['read_cache'],TRADE_PERFORMANCE_CACHE='cache'))
        held=env['_load_trade_perf_cache']()
        self.assertEqual(held['policy_revision'],1);self.assertEqual(held['excluded_count'],1)
        self.assertEqual(held['unresolved_trade_ids'],[110]);self.assertEqual(held['trades'],[])
        revoke=dict(req,action='REVOKE',expected_policy_revision=1,expected_case_revision=1,approval_ref='revoke-visible')
        D.change_disposition(self.s,revoke,manifest,IDENTITY)
        held=env['_load_trade_perf_cache']()
        self.assertEqual(held['policy_revision'],2);self.assertEqual(held['excluded_count'],0)
        self.assertEqual(held['unresolved_trade_ids'],[110]);self.assertEqual(held['trades'],[])
    def test_held_message_distinguishes_valid_policy_from_unknown_count(self):
        visible=self.functions['performance_hold_visibility']()
        message=selected('dashboard_app',{'_performance_hold_message'},{} )['_performance_hold_message']
        self.assertIn('excluded trades 0',message(visible))
        self.assertIn('excluded count unknown',message({'policy_validation_status':'UNAVAILABLE'}))
        self.assertNotIn('excluded trades 0',message({'policy_validation_status':'UNAVAILABLE'}))
    def test_physical_history_loader_remains_unfiltered_and_named_history(self):
        source=SOURCE['data_io'];tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_trades_db')
        self.assertNotIn('eligible_sql',ast.get_source_segment(source,node))
        self.assertIn('PHYSICAL_HISTORY_UNFILTERED',SOURCE['dashboard_app'])

class ExactFillAllocationTests(unittest.TestCase):
    def setUp(self):
        D._RUNTIME_IDENTITY=None
        self.s=Store();self.functions=db_functions(self.s)
        self.s.rows[110]=dict(row(110),qty=Decimal('30'),entry_price=Decimal('65'),
            exit_time='2026-01-02T00:00:00+00:00',exit_reason='POSITION_CLOSED')
    def fill(self,qty,decorate=True):
        return self.functions['reconcile_sell_fill'](self.s,110,symbol='CRC',qty=qty,
            order_id='second-crc-exit',status='filled',event_time=STAMP,raw={},exit_price='63.26',
            exit_reason='SELL_FILL',decorate=decorate)
    def test_partial_quantity_never_writes_or_computes_whole_trade_result(self):
        before=copy.deepcopy(self.s.rows)
        for state,decorate in [('CLOSED',True),('OPEN',False)]:
            with self.subTest(state=state):
                self.s.rows[110]['status']=state
                self.assertFalse(self.fill('15',decorate))
        self.s.rows[110]['status']='CLOSED'
        self.assertEqual(self.s.rows,before)
        self.assertFalse(any(sql.lstrip().startswith(('INSERT','UPDATE','DELETE')) for sql,_ in self.s.trace))
    def test_equal_quantity_does_not_authorize_unbound_historical_allocation(self):
        before=copy.deepcopy(self.s.rows)
        for state,decorate in [('CLOSED',True),('OPEN',False)]:
            with self.subTest(state=state):
                self.s.rows[110]['status']=state
                self.assertFalse(self.fill('30',decorate))
        self.s.rows[110]['status']='CLOSED'
        self.assertEqual(self.s.rows,before)
        self.assertFalse(any(sql.lstrip().startswith(('INSERT','UPDATE','DELETE')) for sql,_ in self.s.trace))

class PaperBindingTests(unittest.TestCase):
    def setUp(self):
        D._RUNTIME_IDENTITY=None
        self.s=Store();self.functions=db_functions(self.s)

    def test_compiled_target_is_exact_reviewed_public_binding(self):
        binding=next(n for n in ast.parse(SOURCE['db']).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='_PAPER_DISPOSITION_BINDING' for t in n.targets))
        self.assertEqual(ast.literal_eval(binding.value),(
            'jbravo-pythonanywhere-production','0dbad265-9d1c-4de9-9eaa-1b234a533a05',
            'metadata-owner-003:sha256:f6084a05d141c316850fe76b58ec36c802d585f5e45b6274dc0224ad0710d793'))

    def test_missing_or_live_context_refuses_before_policy_sql(self):
        for environment in ({},{'APCA_API_BASE_URL':'https://paper-api.alpaca.markets'},
            {'APCA_API_BASE_URL':'https://api.alpaca.markets','JBR_EXEC_PAPER':'true'},
            {'APCA_API_BASE_URL':'https://paper-api.alpaca.markets','JBR_EXEC_PAPER':'false'}):
            with self.subTest(environment=environment):
                self.functions['os'].environ=environment
                with self.assertRaisesRegex(D.TradePolicyError,'PAPER_CONTEXT_REQUIRED'):
                    self.functions['trade_policy'](self.s)
                self.assertEqual(self.s.trace,[])
                self.assertIsNone(D._RUNTIME_IDENTITY)

    def test_every_operation_rechecks_paper_context(self):
        self.functions['trade_policy'](self.s)
        count=len(self.s.trace)
        self.functions['os'].environ['JBR_EXEC_PAPER']='false'
        with self.assertRaisesRegex(D.TradePolicyError,'PAPER_CONTEXT_REQUIRED'):
            self.functions['trade_policy'](self.s)
        self.assertEqual(len(self.s.trace),count)

    def test_fixed_binding_refuses_conflicting_prebound_identity(self):
        D.bind_runtime_identity('other-paper',IDENTITY[1],environment='paper',evidence_ref='fixture-only',verified=True)
        with self.assertRaisesRegex(D.TradePolicyError,'ALREADY_BOUND'):
            self.functions['trade_policy'](self.s)
        self.assertEqual(self.s.trace,[])

    def test_connected_policy_identity_remains_independent_and_required(self):
        class OtherTarget(Cursor):
            def execute(self,sql,params=None):
                super().execute(sql,params)
                if sql.startswith('SELECT deployment_id'):
                    self.one=('other-deployment',IDENTITY[1],'jbravo',0,'jbravo')
        self.s.cursor=lambda **kw:OtherTarget(self.s)
        with self.assertRaisesRegex(D.TradePolicyError,'IDENTITY_MISMATCH'):
            self.functions['trade_policy'](self.s)

if __name__ == '__main__': unittest.main(verbosity=2)
