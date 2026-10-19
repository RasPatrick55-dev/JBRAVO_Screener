"""Normal-repository actual-function ML policy regression tests.
No pandas/psycopg/app import. SQL/frame/connection/file doubles, not PG proof.
Run with the bounded socket/process-denying launcher.
"""
from __future__ import annotations
import ast
import __future__
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
START=datetime(2026,10,1,tzinfo=timezone.utc)
END=datetime(2026,10,19,tzinfo=timezone.utc)


def selected(rel,names,ns):
    tree=ast.parse((ROOT/rel).read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise AssertionError('source_inventory_changed')
    flags=__future__.annotations.compiler_flag if any(isinstance(n,ast.ImportFrom) and n.module=='__future__' and any(a.name=='annotations' for a in n.names) for n in tree.body) else 0
    exec(compile(ast.Module(body=nodes,type_ignores=[]),rel,'exec',flags=flags,dont_inherit=True),ns)
    return ns


class Cursor:
    rowcount=1
    def __init__(self,conn):self.conn=conn;self.rows=[]
    def __enter__(self):return self
    def __exit__(self,*exc):return False
    def execute(self,sql,params=None):
        self.conn.calls.append(('SQL',sql,params));self.rows=self.conn.handler(sql,params) or []
    def fetchall(self):return self.rows
    def fetchone(self):return self.rows[0] if self.rows else None


class Conn:
    autocommit=False
    def __init__(self):self.calls=[];self.handler=lambda *_:[];self.commits=0;self.rollbacks=0
    def get_transaction_status(self):return 0
    def cursor(self,**kwargs):return Cursor(self)
    def __enter__(self):return self
    def __exit__(self,typ,*exc):
        if typ:self.rollbacks+=1
        else:self.commits+=1
        return False
    def close(self):self.calls.append(('CLOSE',))


class Frame:
    empty=True;shape=(0,0)
    def __init__(self,rows=None):self.attrs={}
    def copy(self):return self
    def to_csv(self,**kwargs):return ''


class Series:
    def __init__(self,rows):self.rows=rows
    def astype(self,*args):return self
    def dropna(self):return self
    def tolist(self):return self.rows


class Missing:
    empty=False
    def __getitem__(self,key):return Series([111] if key=='trade_id' else ['owned-entry'])
    def iterrows(self):return iter([(0,{'trade_id':111,'entry_order_id':'owned-entry','symbol':'CRC','entry_time':START})])


class MLDispositionParity(unittest.TestCase):
    def setUp(self):
        for target in ['socket.socket','subprocess.Popen']:
            guard=patch(target,side_effect=AssertionError('EXTERNAL_EFFECT_FORBIDDEN'))
            guard.start();self.addCleanup(guard.stop)
        pure={'__name__':'prepared_ml_disposition'}
        exec(compile((ROOT/'scripts/trade_disposition.py').read_text(encoding='utf8'),
            'pure-disposition','exec',dont_inherit=True),pure)
        self.d=SimpleNamespace(**pure);self.conn=Conn();self.conn.handler=lambda sql,params:[(1,)] if sql.startswith('SELECT pg_catalog.to_regclass') else []
        self.policy=SimpleNamespace(scope='VALIDATED_PERFORMANCE',revision=0,
            metadata=lambda:{'deployment_id':'fixture','database_id':'fixture','policy_revision':0})
        self.db=SimpleNamespace(disposition=self.d,trade_policy=self.policy_read,
            ml_trade_order_owners=Mock(),db_enabled=lambda:True)
        self.ns={'Any':Any,'Mapping':Mapping,'datetime':datetime,'timezone':timezone,
            'contextmanager':contextmanager,'json':json,'Decimal':Decimal,'disposition':self.d,'trade_policy':self.policy_read,
            'normalize_ts':lambda value:datetime.fromisoformat(value) if isinstance(value,str) else value,
            'db':self.db,'pd':SimpleNamespace(DataFrame=Frame),'extras':SimpleNamespace(RealDictCursor=object),
            '_ensure_columns':lambda x:x,'LOG':Mock(),'Path':Path}
    def policy_read(self,conn,**kwargs):
        conn.calls.append(('POLICY',kwargs))
        if kwargs.get('expected_revision',0)!=0:raise self.d.TradePolicyError('TRADE_POLICY_REVISION_CHANGED')
        return self.policy
    def helpers(self):
        @contextmanager
        def maybe(engine=None):yield self.conn
        @contextmanager
        def readonly(conn,**kw):
            with conn:yield self.policy_read(conn,**kw)
        names=['_ml_trade_policy_lock','_ml_trade_lock_inputs','ml_trade_read_transaction','ml_trade_input','ml_trade_match_inputs','ml_trade_evidence',
            '_ml_trade_current','ml_trade_order_owners','_ml_trade_kinds','_ml_trade_write_artifact',
            'publish_ml_trade_artifacts','validated_ml_trade_artifact']
        return selected('scripts/db.py',names,dict(self.ns,_maybe_conn=maybe,_policy_read_transaction=readonly))
    def evidence(self,ns):
        matches=ns['ml_trade_match_inputs'](self.conn,[],oos_run_date=None)
        return ns['ml_trade_evidence'](self.policy,[],start_dt=START,end_dt=END,match_inputs=matches)
    def test_four_ranker_queries_scope_before_cte_order_and_limit(self):
        names=['_query_closed_trades_base','_query_entry_context','_query_scores_direct','_query_run_map']
        ns=selected('scripts/ranker_trade_attribution.py',names,dict(self.ns))
        for name in names:
            with self.subTest(name=name):
                self.conn.calls=[];ns[name](self.conn,start_dt=START,end_dt=END)
                self.assertEqual(self.conn.calls[0][0],'POLICY')
                sql=self.conn.calls[1][1]
                self.assertIn('FROM public.trades t',sql)
                self.assertIn('VALIDATED_PERFORMANCE',sql)
                self.assertLess(sql.index('public.trade_disposition_cases'),sql.index('ORDER BY'))
                if 'LIMIT' in sql:self.assertLess(sql.index('public.trade_disposition_cases'),sql.index('LIMIT'))
    def test_table_lookup_permission_failure_propagates(self):
        self.conn.handler=lambda *_:(_ for _ in ()).throw(RuntimeError('permission_fixture'))
        ns=selected('scripts/ranker_trade_attribution.py',['_table_exists'],dict(self.ns))
        with self.assertRaisesRegex(RuntimeError,'permission_fixture'):ns['_table_exists'](self.conn,'trades')
        self.assertEqual(self.conn.calls[1][2],('public.trades',))
        self.assertIn('pg_catalog.to_regclass',self.conn.calls[1][1])
    def test_auto_db_policy_failure_never_reaches_csv(self):
        load=Mock(side_effect=self.d.TradePolicyError('policy_failure'));csv=Mock()
        ns=selected('scripts/ranker_trade_attribution.py',['_load_trades'],dict(self.ns,
            _load_closed_trades_db=load,_load_closed_trades_fs=csv))
        args=SimpleNamespace(source='auto',match_mode='auto',run_date=None)
        with self.assertRaisesRegex(self.d.TradePolicyError,'policy_failure'):
            ns['_load_trades'](args,start_dt=START,end_dt=END)
        csv.assert_not_called()
    def test_disabled_db_and_explicit_fs_refuse_population_fallback(self):
        load=Mock();self.db.db_enabled=lambda:False
        ns=selected('scripts/ranker_trade_attribution.py',['_load_trades'],dict(self.ns,_load_closed_trades_db=load))
        for source in ['fs','db','auto']:
            with self.subTest(source=source),self.assertRaises(self.d.TradePolicyError):
                ns['_load_trades'](SimpleNamespace(source=source),start_dt=START,end_dt=END)
        load.assert_not_called()
    def test_initial_backfill_filters_before_limit(self):
        ns=selected('scripts/trade_entry_ml_context_backfill.py',['_load_closed_trades'],dict(self.ns))
        ns['_load_closed_trades'](self.conn,start_dt=START,end_dt=END,limit=1)
        sql=self.conn.calls[1][1]
        self.assertEqual(self.conn.calls[0][0],'POLICY')
        self.assertLess(sql.index('public.trade_disposition_cases'),sql.index('ORDER BY'))
        self.assertLess(sql.index('public.trade_disposition_cases'),sql.index('LIMIT'))
        self.assertEqual(self.conn.calls[1][2]['limit'],1)
    def test_secondary_backfill_paths_bind_physical_id_and_scope(self):
        ns=selected('scripts/trade_entry_ml_context_backfill.py',
            ['_raw_context_candidates','_scores_asof_candidates'],dict(self.ns))
        ns['_raw_context_candidates'](self.conn,Missing())
        ns['_scores_asof_candidates'](self.conn,Missing(),max_lag_hours=24)
        self.assertEqual(self.db.ml_trade_order_owners.call_count,2)
        for call in [x for x in self.conn.calls if x[0]=='SQL']:
            self.assertIn('t.trade_id=ANY(%(trade_ids)s)',call[1])
            self.assertIn('public.trade_disposition_cases',call[1])
            self.assertEqual(call[2]['trade_ids'],[111])
    def test_excluded_physical_owner_still_prevents_reuse(self):
        ns=self.helpers();self.conn.handler=lambda *_:[(111,'owned-entry'),(63,'owned-entry')]
        with self.assertRaisesRegex(self.d.TradePolicyError,'OWNER_CHANGED'):
            ns['ml_trade_order_owners'](self.conn,[(111,'owned-entry')])
        sql=self.conn.calls[-1][1]
        self.assertNotIn('trade_disposition',sql);self.assertNotIn('status',sql)
    def test_load_policy_lock_precedes_all_context_and_artifact_locks(self):
        ns=self.helpers()
        with ns['ml_trade_read_transaction'](self.conn):
            ns['ml_trade_input'](self.conn,self.policy,start_dt=START,end_dt=END)
        policy_call=next(x for x in self.conn.calls if x[0]=='POLICY')
        self.assertFalse(policy_call[1].get('exclusive',False),'runtime policy read must not request FOR UPDATE')
        statements=[x[1] for x in self.conn.calls if x[0]=='SQL']
        advisory="SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('jbravo:trade-disposition:policy'))"
        self.assertLess(statements.index("SET LOCAL lock_timeout = '1500ms'"),statements.index(advisory))
        self.assertLess(statements.index("SET LOCAL statement_timeout = '5000ms'"),statements.index(advisory))
        self.assertLess(self.conn.calls.index(next(x for x in self.conn.calls if x[0]=='SQL' and x[1]==advisory)),
            self.conn.calls.index(policy_call))
        locks=[x[1] for x in self.conn.calls if x[0]=='SQL' and x[1].startswith('LOCK')]
        self.assertEqual(locks,['LOCK TABLE public.'+table+' IN SHARE MODE' for table in
            ('trades','trade_entry_ml_context_app','executed_trades','screener_ranker_scores_app','screener_run_map_app','ml_artifacts')])
        self.assertLess(self.conn.calls.index(next(x for x in self.conn.calls if x[0]=='POLICY')),
            self.conn.calls.index(next(x for x in self.conn.calls if x[0]=='SQL' and x[1].startswith('LOCK'))))
    def test_zero_row_publication_is_bound_atomic_pair_not_complete_accounting(self):
        ns=self.helpers();original={'status':'no_data'}
        result=ns['publish_ml_trade_artifacts'](self.evidence(ns),kind='ranker_trade_attribution',
            run_date=END.date(),payload=original,rows_csv='',rows_count=0)
        writes=[x for x in self.conn.calls if x[0]=='SQL' and x[1].startswith('INSERT')]
        self.assertEqual(len(writes),2);self.assertEqual(self.conn.commits,1)
        self.assertEqual(original,{'status':'no_data'})
        self.assertEqual(result['artifact_binding']['rows_sha256'],hashlib.sha256(b'').hexdigest())
        self.assertFalse(result['trade_policy']['accounting_complete'])
    def test_runtime_policy_reader_rejects_operator_authority_in_load_and_publication(self):
        ns=self.helpers()
        def select_only_policy(conn,**kwargs):
            if kwargs.get('exclusive',False):raise AssertionError('POLICY_UPDATE_AUTHORITY_REQUESTED')
            return self.policy_read(conn,**kwargs)
        ns['trade_policy']=select_only_policy
        with ns['ml_trade_read_transaction'](self.conn):pass
        result=ns['publish_ml_trade_artifacts'](self.evidence(ns),kind='ranker_trade_attribution',
            run_date=END.date(),payload={'status':'no_data'},rows_csv='',rows_count=0)
        self.assertIn('artifact_binding',result)
        self.assertGreaterEqual(len([x for x in self.conn.calls if x[0]=='POLICY']),3)
        self.assertTrue(all(not x[1].get('exclusive',False) for x in self.conn.calls if x[0]=='POLICY'))
    def test_ml_advisory_lock_refusal_rolls_back_before_policy_or_inputs(self):
        ns=self.helpers()
        def blocked(sql,params):
            if 'pg_advisory_xact_lock(' in sql:raise RuntimeError('fixture_lock_timeout')
            return []
        self.conn.handler=blocked
        with self.assertRaisesRegex(RuntimeError,'fixture_lock_timeout'):
            with ns['ml_trade_read_transaction'](self.conn):self.fail('LOCK_REFUSAL_YIELDED')
        self.assertEqual(self.conn.rollbacks,1)
        self.assertFalse(any(x[0]=='POLICY' or x[0]=='SQL' and x[1].startswith(('LOCK TABLE','INSERT','UPDATE')) for x in self.conn.calls))
    def test_stale_full_row_input_refuses_before_writes(self):
        ns=self.helpers();evidence=self.evidence(ns)
        self.conn.handler=lambda sql,params:[(1,)] if sql.startswith('SELECT pg_catalog.to_regclass') else ([(json.dumps({'trade_id':111}),)] if 'to_jsonb(t)' in sql else [])
        with self.assertRaisesRegex(self.d.TradePolicyError,'INPUT_CHANGED'):
            ns['publish_ml_trade_artifacts'](evidence,kind='ranker_trade_attribution',
                run_date=END.date(),payload={},rows_csv='',rows_count=0)
        self.assertEqual(self.conn.rollbacks,1)
        self.assertFalse(any(x[0]=='SQL' and x[1].startswith('INSERT') for x in self.conn.calls))
    def test_context_epoch_change_refuses_even_unchanged_trade_rows(self):
        ns=self.helpers();evidence=self.evidence(ns)
        self.conn.handler=lambda sql,params:[('2026-10-19','new predictions')] if sql.startswith('SELECT run_date,csv_data') else []
        with self.assertRaisesRegex(self.d.TradePolicyError,'INPUT_CHANGED'):
            ns['_ml_trade_current'](self.conn,evidence)
    def test_policy_revision_change_refuses_without_publication(self):
        ns=self.helpers();evidence=self.evidence(ns);evidence['policy']['policy_revision']=1
        with self.assertRaisesRegex(self.d.TradePolicyError,'REVISION_CHANGED'):
            ns['publish_ml_trade_artifacts'](evidence,kind='ranker_trade_attribution',
                run_date=END.date(),payload={},rows_csv='',rows_count=0)
        self.assertEqual(self.conn.rollbacks,1)
    def test_second_artifact_failure_rolls_back_whole_publication(self):
        ns=self.helpers();evidence=self.evidence(ns);inserts=[]
        def fail(sql,params):
            if sql.startswith('SELECT pg_catalog.to_regclass'):return [(1,)]
            if sql.startswith('INSERT'):
                inserts.append(params[0])
                if len(inserts)==2:raise RuntimeError('second_insert_failed')
            return []
        self.conn.handler=fail
        with self.assertRaisesRegex(RuntimeError,'second_insert_failed'):
            ns['publish_ml_trade_artifacts'](evidence,kind='ranker_trade_attribution',
                run_date=END.date(),payload={},rows_csv='',rows_count=0)
        self.assertEqual(len(inserts),2);self.assertEqual(self.conn.commits,0);self.assertEqual(self.conn.rollbacks,1)
    def test_dashboard_hold_never_reads_filesystem_fallback(self):
        self.db.validated_ml_trade_artifact=Mock(side_effect=self.d.TradePolicyError('stale'))
        ns=selected('dashboards/ml_overview_api.py',['_artifact_record'],dict(self.ns))
        class Forbidden:
            def exists(self):raise AssertionError('FS_REACHED')
        result=ns['_artifact_record']('ranker_trade_attribution',base_dir=Forbidden(),fs_json=Forbidden())
        self.assertFalse(result['present']);self.assertEqual(result['source'],'HELD')
    def test_backfill_disabled_fails_before_any_output(self):
        self.db.db_enabled=lambda:False;output=Mock(side_effect=AssertionError('OUTPUT_REACHED'))
        ns=selected('scripts/trade_entry_ml_context_backfill.py',['run_backfill'],dict(self.ns,
            _window_bounds=lambda *_:(START,END),_ensure_output_dir=output))
        args=SimpleNamespace(lookback_days=252,max_lag_hours=24,dry_run=True,limit=1,output_dir=Path('INERT'))
        with self.assertRaisesRegex(self.d.TradePolicyError,'DB_UNAVAILABLE'):ns['run_backfill'](args)
        output.assert_not_called()
    def test_legacy_publication_helper_cannot_bypass_input_validation(self):
        ns=selected('scripts/trade_entry_ml_context_backfill.py',['_write_fs_outputs'],dict(self.ns))
        with self.assertRaisesRegex(self.d.TradePolicyError,'PUBLICATION_UNVALIDATED'):
            ns['_write_fs_outputs'](None,{},None)


    def consumer_fixture(self, mutation=None):
        ns=self.helpers();evidence=self.evidence(ns);payload={'status':'ok','trade_policy':evidence}
        binding={'kind':'ranker_trade_attribution','rows_kind':'ranker_trade_attribution_trades',
            'rows_count':0,'rows_sha256':hashlib.sha256(b'').hexdigest(),
            'payload_fingerprint':self.d.fingerprint(payload)}
        payload['artifact_binding']=binding
        state={'payload':payload,'pair':{'trade_policy':evidence,'artifact_binding':binding},'csv':''}
        if mutation:mutation(state)
        def handler(sql,params):
            if sql.startswith('SELECT pg_catalog.to_regclass'):return [(1,)]
            if sql.startswith('SELECT run_date,payload'):return [(END.date(),self.d.canonical(state['payload']))]
            if sql.startswith('SELECT payload::pg_catalog.text,csv_data'):
                return [(self.d.canonical(state['pair']),state['csv'],0)]
            return []
        self.conn.handler=handler
        return ns
    def test_actual_consumer_accepts_current_bound_zero_row_artifact(self):
        ns=self.consumer_fixture()
        self.assertEqual(ns['validated_ml_trade_artifact']('ranker_trade_attribution')['status'],'ok')
    def test_consumer_rejects_changed_payload(self):
        ns=self.consumer_fixture(lambda s:s['payload'].update(status='forged'))
        with self.assertRaisesRegex(self.d.TradePolicyError,'PAYLOAD_CHANGED'):
            ns['validated_ml_trade_artifact']('ranker_trade_attribution')
    def test_consumer_rejects_changed_csv_without_fs_fallback(self):
        ns=self.consumer_fixture(lambda s:s.update(csv='excluded63\n'))
        with self.assertRaisesRegex(self.d.TradePolicyError,'ROWS_CHANGED'):
            ns['validated_ml_trade_artifact']('ranker_trade_attribution')

    def context_publication_fixture(self, change_score=False):
        ns=self.helpers();record={'trade_id':111,'entry_order_id':'owned-entry','symbol':'CRC'}
        stored={'context':None,'artifacts':{},'score_changed':False}
        def handler(sql,params):
            if sql.startswith('SELECT pg_catalog.to_regclass'):return [(1,)]
            if 'to_jsonb(t)' in sql:return [(self.d.canonical(record),)]
            if sql.startswith('SELECT trade_id,entry_order_id'):return [(111,'owned-entry')]
            if sql.startswith('SELECT pg_catalog.to_jsonb(x)'):
                if 'public.trade_entry_ml_context_app' in sql and stored['context'] is not None:
                    return [(self.d.canonical(stored['context']),)]
                if 'public.screener_ranker_scores_app' in sql and stored['score_changed']:
                    return [(self.d.canonical({'symbol':'CRC','model_score_5d':Decimal('0.9')}),)]
                return []
            if sql.startswith('INSERT INTO public.trade_entry_ml_context_app'):
                row=(params[0],params[1],datetime.fromisoformat(params[2]),datetime.fromisoformat(params[3]),
                    params[4],params[5],params[6],params[7],params[8])
                stored['context']={'order_id':params[0],'symbol':params[1],'model_score':params[4],
                    'raw':self.d.document(params[8])}
                stored['score_changed']=change_score
                return [row]
            if sql.startswith('INSERT INTO public.ml_artifacts'):
                stored['artifacts'][params[0]]=(params[1],params[3],params[4],params[2])
            if sql.startswith('SELECT run_date,payload'):
                artifact=stored['artifacts'].get(params[0])
                return [(artifact[0],artifact[1])] if artifact else []
            if sql.startswith('SELECT payload::pg_catalog.text,csv_data'):
                artifact=stored['artifacts'].get(params[0])
                return [(artifact[1],artifact[2],artifact[3])] if artifact else []
            return []
        self.conn.handler=handler
        matches=ns['ml_trade_match_inputs'](self.conn,[record])
        evidence=ns['ml_trade_evidence'](self.policy,[record],start_dt=START,end_dt=END,match_inputs=matches)
        context={'trade_id':111,'order_id':'owned-entry','symbol':'CRC','entry_time':START.isoformat(),
            'screener_run_ts_utc':START.isoformat(),'model_score':Decimal('0.5'),'model_score_5d':Decimal('0.5'),
            'score_col':'model_score','score_source':'backfill.scores_direct','raw':{'source_kind':'scores_asof'}}
        return ns,evidence,context,stored
    def test_own_context_insert_binds_poststate_without_self_invalidating(self):
        ns,evidence,context,stored=self.context_publication_fixture()
        result=ns['publish_ml_trade_artifacts'](evidence,kind='trade_entry_ml_context_backfill',
            run_date=END.date(),payload={'status':'ok'},rows_csv='order_id\nowned-entry\n',rows_count=1,context_rows=[context])
        self.assertEqual(self.conn.commits,1);self.assertEqual(result['rows_upserted'],1)
        self.assertEqual(result['pre_write_input'],evidence)
        self.assertNotEqual(result['trade_policy']['match_inputs'],evidence['match_inputs'])
        self.assertEqual(ns['validated_ml_trade_artifact']('trade_entry_ml_context_backfill'),result)
    def test_score_change_during_own_context_insert_cannot_be_laundered(self):
        ns,evidence,context,stored=self.context_publication_fixture(change_score=True)
        with self.assertRaisesRegex(self.d.TradePolicyError,'NONCONTEXT_INPUT_CHANGED'):
            ns['publish_ml_trade_artifacts'](evidence,kind='trade_entry_ml_context_backfill',
                run_date=END.date(),payload={'status':'ok'},rows_csv='order_id\nowned-entry\n',rows_count=1,context_rows=[context])
        self.assertEqual(self.conn.commits,0);self.assertEqual(self.conn.rollbacks,1)
        self.assertEqual(stored['artifacts'],{})
        # Connection double counts rollback; only real native PG can prove durable row rollback.

if __name__=='__main__':unittest.main(verbosity=2)
