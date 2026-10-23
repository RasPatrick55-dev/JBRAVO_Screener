"""Actual gross primary stage with real pandas/pure code and strict transactional fake DB.

Source extraction avoids importing credential-loading application modules. This
does not qualify native PostgreSQL or the full Linux Pipeline process.
"""
import argparse, ast, io, json, logging, math, shutil, tempfile, unittest
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Optional
from unittest import mock
import numpy as np
import pandas as pd
from scripts import gross_paper_accounting as G, trade_disposition as D
from scripts import pipeline_postflight as P, primary_pipeline as Primary

ROOT=Path(__file__).resolve().parents[1]
F=ModuleType('recovery_pure_fixtures')
fixture=Path(__file__).with_name('test_gross_paper_accounting.py')
F.__file__=str(fixture);exec(compile(fixture.read_bytes(),str(fixture),'exec'),F.__dict__)

def extract(path,names,ns):
    tree=ast.parse(path.read_bytes())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

class State:
    def __init__(self):
        self.trades=[F.row()];self.policy=F.policy((F.allocation(),));self.top=[];self.daily={}
        self.sql=[];self.commits=0;self.rollbacks=0;self.closed=0;self.locks=[]
        self.refuse=None;self.on_trade_read=None;self.trade_reads=0
        self.advisory=False;self.policy_calls=[];self.gross_schema=True
    def connection(self):return Connection(self)

class Connection:
    autocommit=False
    def __init__(self,state):self.s=state
    def get_transaction_status(self):return 0
    def __enter__(self):self.saved=deepcopy((self.s.top,self.s.daily));return self
    def __exit__(self,typ,*args):
        self.s.advisory=False
        if typ:self.s.top,self.s.daily=self.saved;self.s.rollbacks+=1
        else:self.s.commits+=1
    def close(self):self.s.closed+=1
    def cursor(self):return Cursor(self.s)

class Cursor:
    def __init__(self,state):self.s=state;self.answer=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute(self,sql,params=None):
        self.s.sql.append((sql,params))
        if self.s.refuse and self.s.refuse in sql:raise RuntimeError('synthetic SQL refusal')
        if sql.startswith('SET '):return
        if sql.startswith('SELECT pg_catalog.pg_advisory_xact_lock('):
            self.s.advisory=True;self.answer=[(None,)];return
        if sql.startswith('CREATE '):raise AssertionError('Runtime must not perform DDL')
        if sql.startswith('SELECT pg_catalog.to_jsonb(trades)'):
            self.s.trade_reads+=1
            if self.s.on_trade_read:self.s.on_trade_read(self.s.trade_reads)
            self.answer=[(D.canonical(x),) for x in self.s.trades];return
        if sql.startswith('SELECT pg_catalog.to_jsonb(candidate)'):
            self.answer=[(D.canonical(x),) for x in sorted(self.s.top,key=lambda x:x['symbol']) if x['run_date']==params[0]];return
        if sql.startswith('DELETE FROM public.top_candidates WHERE run_date='):
            self.s.top=[x for x in self.s.top if x['run_date']!=params[0]];return
        if sql.startswith('INSERT INTO public.paper_gross_metrics_daily'):
            if not self.s.gross_schema:raise RuntimeError('synthetic missing derived schema')
            self.s.daily[params[0]]=params[1];return
        if sql.startswith('SELECT publication::pg_catalog.text FROM public.paper_gross_metrics_daily'):
            self.answer=[(self.s.daily[params[0]],)] if params[0] in self.s.daily else [];return
        raise AssertionError('Unrecognized SQL in strict fake: '+sql[:100])
    def fetchall(self):return self.answer
    def fetchone(self):return self.answer[0] if self.answer else None

def batch(cursor,sql,rows,**kw):
    assert sql.startswith('INSERT INTO public.top_candidates (')
    cursor.s.sql.append((sql,deepcopy(rows)));cursor.s.top.extend(deepcopy(rows))

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.storage=tempfile.TemporaryDirectory();self.addCleanup(self.storage.cleanup)
        self.root=Path(self.storage.name)
        for name in set(G.SOURCE_PATHS) | set(P.SOURCE_FILES):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        self.state=State();s=self.state
        @contextmanager
        def maybe_conn(engine):
            c=engine if engine is not None else s.connection()
            try:yield c
            finally:
                if engine is None:c.close()
        @contextmanager
        def policy_read(c,**kw):
            s.locks.append('shared')
            try:yield s.policy
            finally:s.locks.pop()
        def policy(c,**kw):
            self.assertTrue(s.advisory);self.assertFalse(kw.get('exclusive',False))
            s.policy_calls.append(dict(kw,sql_count=len(s.sql)))
            self.assertEqual(kw['scope'],'VALIDATED_PERFORMANCE')
            self.assertEqual(kw['expected_revision'],s.policy.revision)
            return s.policy
        self.db=dict(__file__=str(self.root/'scripts/db.py'),Path=Path,contextmanager=contextmanager,
            Any=Any,pd=pd,math=math,Decimal=Decimal,disposition=D,datetime=datetime,date=date,
            _paper_disposition_identity=lambda:F.IDENTITY,trade_policy=policy,
            _conn_or_none=s.connection,_maybe_conn=maybe_conn,_policy_read_transaction=policy_read,
            extras=SimpleNamespace(execute_batch=batch))
        extract(ROOT/'scripts/db.py',('_coerce_date','_gross_pipeline_date','_gross_pipeline_current','_gross_pipeline_candidates',
            '_gross_pipeline_candidate_rows','publish_paper_gross_pipeline','paper_gross_pipeline_evidence',
            'gross_paper_performance_snapshot','_paper_gross_artifact','publish_paper_gross_report'),self.db)
        self.rank=pd.DataFrame([dict(symbol='ONE',score=0.75,win_rate=50.0,net_pnl=5.0,trades=2,
            wins=1,losses=1,entry_price=50.0,exchange='TEST',source='synthetic-backtest')])
        self.day='2026-10-23';self.original=deepcopy(s.trades)
    def snapshot(self):return F.build(self.state.policy,self.state.trades)
    def publish(self):return self.db['publish_paper_gross_pipeline'](self.day,self.rank,self.snapshot())
    def evidence(self,**kw):return self.db['paper_gross_pipeline_evidence'](self.day,**kw)
    def test_atomic_db_publication_retains_originals_unknown_net_and_backtest_ranking(self):
        result=self.publish();self.assertEqual(self.state.commits,1);self.assertEqual(self.state.rollbacks,0)
        self.assertEqual(self.state.trades,self.original);self.assertEqual(self.state.top[0]['net_pnl'],5.0)
        doc=D.document(self.state.daily[self.day]);self.assertEqual(doc['summary']['gross_pnl'],'-63.38')
        self.assertIsNone(doc['summary']['net_pnl']);self.assertIsNone(doc['summary']['fees'])
        self.assertFalse(result['entry_ready']);self.assertFalse(result['accounting_complete'])
        self.assertEqual(doc['candidate_basis'],'BACKTEST_RESULTS_UNCHANGED')
        self.assertFalse(any('public.metrics_daily' in q or 'UPDATE public.trades' in q for q,_ in self.state.sql))
        with self.evidence(expected_publication=result['publication_fingerprint']) as read:
            self.assertEqual(self.state.locks,['shared']);self.assertEqual(read,result)
        self.assertEqual(self.state.locks,[])
    def test_runtime_serializes_before_select_only_policy_and_never_runs_ddl(self):
        self.publish();queries=[q for q,_ in self.state.sql]
        lock=next(i for i,q in enumerate(queries) if q.startswith('SELECT pg_catalog.pg_advisory_xact_lock('))
        self.assertIn("pg_catalog.hashtext('jbravo:trade-disposition:policy')",queries[lock])
        self.assertLess(lock,self.state.policy_calls[0]['sql_count'])
        self.assertFalse(self.state.policy_calls[0].get('exclusive',False))
        self.assertFalse(any(q.startswith(('CREATE ','ALTER ','GRANT ')) or 'FOR UPDATE' in q for q in queries))
        self.assertFalse(self.state.advisory)
    def test_missing_schema_rolls_back_without_creating_it_or_changing_originals(self):
        self.state.top=[dict(run_date=self.day,symbol='OLD',score=2)]
        before=deepcopy((self.state.top,self.state.daily,self.state.trades));self.state.gross_schema=False
        with self.assertRaises(RuntimeError):self.publish()
        self.assertEqual((self.state.top,self.state.daily,self.state.trades),before)
        self.assertEqual(self.state.commits,0);self.assertEqual(self.state.rollbacks,1)
        self.assertFalse(self.state.gross_schema);self.assertFalse(self.state.advisory)
        self.assertFalse(any(q.startswith(('CREATE ','ALTER ','GRANT ')) for q,_ in self.state.sql))
    def test_db_write_refusal_rolls_back_candidates_and_daily_publication(self):
        self.state.top=[dict(run_date=self.day,symbol='OLD',score=2)]
        old=deepcopy(self.state.top);self.state.refuse='INSERT INTO public.paper_gross_metrics_daily'
        with self.assertRaises(RuntimeError):self.publish()
        self.assertEqual(self.state.top,old);self.assertEqual(self.state.daily,{})
        self.assertEqual(self.state.rollbacks,1);self.assertEqual(self.state.commits,0);self.assertEqual(self.state.closed,1)
        self.assertEqual(self.state.trades,self.original)
    def test_precommit_trade_and_source_drift_reject_and_rollback(self):
        for kind in ('trade','source'):
            with self.subTest(kind=kind):
                self.state.trade_reads=0
                def drift(n):
                    if n==2:
                        if kind=='trade':self.state.trades[0]['realized_pnl']='changed'
                        else:
                            p=self.root/'scripts/metrics.py';p.write_bytes(p.read_bytes()+b'\n# drift\n')
                self.state.on_trade_read=drift
                with self.assertRaises(D.TradePolicyError):self.publish()
                self.assertEqual(self.state.top,[]);self.assertEqual(self.state.daily,{})
                self.state.trades=deepcopy(self.original);self.state.on_trade_read=None
    def test_read_rejects_row_policy_candidate_and_source_drift_without_fallback(self):
        self.publish();original=deepcopy((self.state.trades,self.state.policy,self.state.top))
        for kind in ('row','policy','revoke','candidate','source'):
            with self.subTest(kind=kind):
                self.state.trades,self.state.policy,self.state.top=deepcopy(original)
                if kind=='row':self.state.trades[0]['realized_pnl']='unrelated change'
                elif kind=='policy':self.state.policy=F.policy((F.allocation(),),revision=5)
                elif kind=='revoke':self.state.policy=F.policy(revision=5)
                elif kind=='candidate':self.state.top[0]['score']=0.123
                else:
                    p=self.root/'scripts/metrics.py';p.write_bytes(p.read_bytes()+b'\n# drift\n')
                with self.assertRaises(D.TradePolicyError):
                    with self.evidence():pass
                self.assertEqual(self.state.locks,[])
    def test_old_and_wrong_publications_missing_table_and_cap_failure_are_incomplete(self):
        result=self.publish()
        for kwargs in ({'expected_publication':'f'*64},{'not_before':datetime.now(timezone.utc)+timedelta(seconds=1)}):
            with self.subTest(kwargs=kwargs),self.assertRaises(D.TradePolicyError):
                with self.evidence(**kwargs):pass
        self.state.daily.clear()
        with self.assertRaises(D.TradePolicyError):
            with self.evidence():pass
        self.state.trades=[F.row(i,incomplete=False) for i in range(1,G.MAX_ROWS+2)]
        with self.assertRaises(D.TradePolicyError):self.snapshot()
    def test_candidate_identity_values_empty_and_limits_refuse_before_writes(self):
        cases=[pd.DataFrame(),pd.concat([self.rank,self.rank]),self.rank.assign(score=float('inf')),
            self.rank.assign(symbol='bad symbol'),self.rank.assign(score=None),self.rank.assign(trades=True)]
        for frame in cases:
            with self.subTest(frame=str(frame.head(1))),self.assertRaises(D.TradePolicyError):
                self.db['publish_paper_gross_pipeline'](self.day,frame,self.snapshot())
        self.assertEqual(self.state.sql,[])
    def test_existing_no_loss_backtest_profit_factor_is_preserved_but_not_infinite_scores(self):
        frame=self.rank.assign(profit_factor=float('inf'))
        columns,rows=self.db['_gross_pipeline_candidate_rows'](frame,self.day)
        self.assertEqual(rows[0]['profit_factor'],float('inf'))
        self.assertEqual(rows[0]['net_pnl'],5.0)
        for field in ('score','net_pnl','entry_price','profit_factor'):
            value=float('-inf') if field=='profit_factor' else float('inf')
            with self.subTest(field=field),self.assertRaises(D.TradePolicyError):
                self.db['_gross_pipeline_candidate_rows'](frame.assign(**{field:value}),self.day)
    def test_bad_run_dates_are_refused_before_connection_or_queries(self):
        for bad in (None,True,'not-a-date','2026-02-30'):
            with self.subTest(bad=bad):
                with self.assertRaises(D.TradePolicyError):self.db['publish_paper_gross_pipeline'](bad,self.rank,self.snapshot())
                with self.assertRaises(D.TradePolicyError):
                    with self.db['paper_gross_pipeline_evidence'](bad):pass
        self.assertEqual(self.state.sql,[]);self.assertEqual(self.state.closed,0)
    def metrics(self,**overrides):
        db=SimpleNamespace(**{k:v for k,v in self.db.items() if not k.startswith('__')})
        db.db_enabled=lambda:True;db.fetch_latest_run_date=lambda table:date(2026,10,23)
        db.disposition=D
        results=self.rank.drop(columns=['score','entry_price','exchange','source']).copy()
        results['avg_return']=0.1
        candidates=pd.DataFrame([dict(symbol='ONE',entry_price=50.0,exchange='TEST',source='screener')])
        ns=dict(argparse=argparse,Optional=Optional,Iterable=Iterable,logger=mock.Mock(),db=db,
            pd=pd,np=np,date=date,load_results=lambda **kw:results.copy(),
            get_latest_screener_candidates=lambda day:(candidates.copy(),None))
        extract(ROOT/'scripts/metrics.py',('parse_args','rank_candidates','rank_and_filter_candidates',
            '_run_paper_gross_metrics','_run_metrics','main'),ns)
        ns.update(overrides);return ns,db
    def test_actual_metrics_caller_runs_ranking_db_artifact_and_validation_then_returns_zero(self):
        ns,db=self.metrics()
        with mock.patch('sys.stdout',new=io.StringIO()):self.assertEqual(ns['main'](['--paper-gross-pipeline']),0)
        self.assertEqual(self.state.top[0]['symbol'],'ONE');self.assertEqual(self.state.top[0]['net_pnl'],5.0)
        self.assertEqual(self.state.trades,self.original);self.assertEqual(self.state.locks,[])
        self.assertTrue((self.root/G.REPORT_PATH).is_file())
        with self.evidence() as proof:
            P._validate_paper_gross_evidence(proof,datetime.now(timezone.utc)-timedelta(minutes=1),datetime.now(timezone.utc))
            self.assertFalse(proof['net_eligible']);self.assertFalse(proof['entry_ready'])
        self.assertTrue(any('METRICS_PAPER_GROSS_COMPLETE' in str(c) for c in ns['logger'].mock_calls))
    def test_metrics_policy_upstream_publication_and_artifact_failures_cannot_pass(self):
        for kind in ('context','date','empty','ranking','db','artifact','final'):
            with self.subTest(kind=kind):
                ns,db=self.metrics()
                if kind=='context':db._paper_disposition_identity=mock.Mock(side_effect=D.TradePolicyError('PAPER_CONTEXT_REQUIRED'))
                if kind=='date':db.fetch_latest_run_date=lambda table:date(2026,10,22) if table=='backtest_results' else date(2026,10,23)
                if kind=='empty':ns['load_results']=lambda **kw:pd.DataFrame()
                if kind=='ranking':ns['rank_and_filter_candidates']=lambda *a:None
                if kind=='db':db.publish_paper_gross_pipeline=mock.Mock(side_effect=RuntimeError('private database error'))
                if kind=='artifact':db.publish_paper_gross_report=mock.Mock(side_effect=RuntimeError('private artifact error'))
                if kind=='final':db.paper_gross_pipeline_evidence=mock.Mock(side_effect=D.TradePolicyError('GROSS_INPUT_CHANGED'))
                with mock.patch('sys.stdout',new=io.StringIO()):self.assertEqual(ns['main'](['--paper-gross-pipeline']),2)
                self.assertFalse(any('private ' in str(c) for c in ns['logger'].mock_calls))
    def test_mode_conflict_and_run_date_override_refuse(self):
        ns,db=self.metrics()
        for args in (['--paper-gross-pipeline','--paper-gross-report-only'],
                     ['--paper-gross-pipeline','--run-date','2026-10-22']):
            with self.subTest(args=args):self.assertEqual(ns['main'](args),2)
        self.assertEqual(self.state.sql,[])
    def test_actual_primary_date_injection_command_and_metrics_wrapper_complete(self):
        from typing import Sequence
        pipeline_source=ast.parse((ROOT/'scripts/run_pipeline.py').read_bytes())
        main=next(n for n in pipeline_source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        injected=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id=='_inject_run_date_arg'
            and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='extras'
                and isinstance(t.slice,ast.Constant) and t.slice.value=='metrics' for t in n.targets)]
        self.assertEqual(len(injected),1)
        flag=next(arg.split('=',1)[1] for arg in Primary.build_arguments([]) if arg.startswith('--metrics-args='))
        extras={'metrics':[flag]};ns={'extras':extras,'pipeline_run_date':date(2026,10,23),'date':date,
            'Sequence':Sequence,'Optional':Optional}
        extract(ROOT/'scripts/run_pipeline.py',('_inject_run_date_arg','_run_step_metrics'),ns)
        exec(compile(ast.Module(body=injected,type_ignores=[]),'actual_metrics_date_injection','exec'),ns)
        self.assertEqual(extras['metrics'],['--paper-gross-pipeline','--run-date','2026-10-23'])
        cmd_assignments=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and isinstance(n.value,ast.List)
            and any(isinstance(v,ast.Constant) and v.value=='scripts.metrics' for v in n.value.elts)]
        self.assertEqual(len(cmd_assignments),1)
        ns['sys']=SimpleNamespace(executable='synthetic-python')
        exec(compile(ast.Module(body=cmd_assignments,type_ignores=[]),'actual_metrics_command','exec'),ns)
        command=ns['cmd']+extras['metrics'];metrics,db=self.metrics()
        def run_step(name,cmd,**kw):
            self.assertEqual(name,'metrics');self.assertEqual(kw['timeout'],180)
            self.assertTrue(kw['tee_to_stdout']);self.assertTrue(kw['tee_to_logger'])
            self.assertEqual(cmd[:3],['synthetic-python','-m','scripts.metrics'])
            return metrics['main'](cmd[3:]),1.0
        ns['run_step']=run_step
        with mock.patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(ns['_run_step_metrics'](command,timeout=180),(0,1.0))
        self.assertIn(self.day,self.state.daily);self.assertEqual(self.state.trades,self.original)
    def test_explicit_pipeline_run_date_must_match_both_upstream_populations(self):
        for bad in ('2026-10-22','2026-02-30','20261023','bad-date'):
            with self.subTest(date=bad):
                ns,db=self.metrics();self.assertEqual(ns['main'](['--paper-gross-pipeline','--run-date',bad]),2)
        self.assertEqual(self.state.sql,[])
    def test_postflight_rejects_unknown_stale_tampered_or_financially_promoted_proof(self):
        proof=self.publish();now=datetime.now(timezone.utc);start=now-timedelta(minutes=1)
        P._validate_paper_gross_evidence(proof,start,now)
        for field in ('net_eligible','entry_ready','accounting_complete','activity_ingestion_complete','recovery_complete','operating_acceptance'):
            for value in (True,None,0):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    P._validate_paper_gross_evidence(dict(proof,**{field:value}),start,now)
        for field,value in (('fees_status','ZERO'),('net_status','KNOWN'),('schema_version',True),
            ('gross_input_complete',False),('candidate_count',0),('candidate_count',10001),
            ('policy_revision',True),('publication_fingerprint','x'),('published_at',(start-timedelta(seconds=1)).isoformat())):
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                P._validate_paper_gross_evidence(dict(proof,**{field:value}),start,now)
    def test_primary_fixes_gross_mode_and_forbids_metrics_override(self):
        args=Primary.build_arguments([])
        self.assertIn('--metrics-args=--paper-gross-pipeline',args)
        for flag in ('--metrics-args=--paper-gross-report-only','--metrics-args-split','--metric','--metrics-args'):
            with self.subTest(flag=flag),self.assertRaises(ValueError):Primary.build_arguments([flag])
    def test_postflight_cross_checks_gross_proof_against_its_own_source_population(self):
        proof=self.publish();now=datetime.now(timezone.utc);start=now-timedelta(minutes=1)
        P._validate_paper_gross_evidence(proof,start,now,sources=G.source_bindings(self.root))
        with self.assertRaises((ValueError,KeyError)):
            P._validate_paper_gross_evidence(proof,start,now,sources={})
        p=self.root/'scripts/primary_pipeline.py';p.write_bytes(p.read_bytes()+b'\n# source changed\n')
        with self.assertRaises(ValueError):
            P._validate_paper_gross_evidence(proof,start,now,sources=G.source_bindings(self.root))
    def test_actual_pipeline_recording_hook_holds_policy_lock_and_truthfully_reports_proof_failure(self):
        baseline=ModuleType('recovery_recording_fixtures')
        fixture=Path(__file__).with_name('test_pipeline_postflight.py')
        baseline.__file__=str(fixture)
        exec(compile(fixture.read_bytes(),str(fixture),'exec'),baseline.__dict__)
        facts=baseline.PostflightTests('test_actual_pipeline_recording_hook_success_and_failure')
        facts.setUp();self.addCleanup(facts.doCleanups)
        self.publish()
        tree=ast.parse((ROOT/'scripts/run_pipeline.py').read_bytes())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        hooks=[n for n in ast.walk(main) if isinstance(n,ast.If) and isinstance(n.test,ast.Call)
            and isinstance(n.test.func,ast.Name) and n.test.func.id=='getattr'
            and any(isinstance(a,ast.Constant) and a.value=='postflight_report' for a in n.test.args)]
        self.assertEqual(len(hooks),1)
        namespace=dict(extras={'metrics':['--paper-gross-pipeline']},pipeline_run_date=self.day,
            db=SimpleNamespace(paper_gross_pipeline_evidence=self.db['paper_gross_pipeline_evidence']),
            base_dir=facts.root,args=SimpleNamespace(postflight_report=True,pipeline_readiness_only=True),
            session_context=None,started_dt=facts.started,datetime=datetime,timezone=timezone,rc=0,
            steps=facts.facts['steps'],step_rcs=facts.facts['step_rcs'],stage_times=facts.facts['stage_times'],
            degraded=False,labels_rows=120,enrichment_freshness=facts.facts['freshness'],
            model_score_coverage_summary=facts.facts['coverage'],ml_health_summary=facts.facts['ml_health'],LOG=mock.Mock(),
            **{key:True for key in ('strict_predictions_meta','strict_auto_refresh_predictions','auto_refresh_features',
                'auto_refresh_predictions','use_champion','ml_health_guard_enabled','refresh_predictions_for_candidates')})
        actual=P.publish_health
        def checked(*a,**kw):
            self.assertEqual(self.state.locks,['shared']);return actual(*a,**kw)
        executable=compile(ast.Module(body=hooks,type_ignores=[]),'actual_pipeline_gross_recording_hook','exec')
        with mock.patch.object(P,'publish_health',side_effect=checked):exec(executable,namespace)
        namespace['LOG'].error.assert_not_called();self.assertEqual(namespace['rc'],0)
        self.assertEqual(self.state.locks,[])
        facts.now=datetime.now(timezone.utc)
        result=facts.check()
        self.assertEqual(result['status'],'ok',result);self.assertTrue(result['pipeline_ready']);self.assertFalse(result['entry_ready'])
        self.state.trades[0]['realized_pnl']='changed'
        namespace['rc']=0
        exec(executable,namespace)
        self.assertEqual(namespace['rc'],1)
        facts.now=datetime.now(timezone.utc)
        result=facts.check();self.assertEqual(result['status'],'failed',result)
        self.assertIn('paper_gross_performance_unproved',result['failures'])
        self.assertIn('pipeline_failed_or_unknown',result['failures'])
    def test_legacy_financial_gate_and_missing_allocation_stay_closed(self):
        with self.assertRaises(D.TradePolicyError):D.require_financial_eligible(self.state.policy)
        self.state.policy=F.policy()
        with self.assertRaises(D.TradePolicyError):self.publish()
        self.assertEqual(self.state.sql,[]);self.assertEqual(self.state.trades,self.original)

class GrossSchemaSetupTests(unittest.TestCase):
    def migration(self,refuse=False):
        source=ROOT/'scripts/db_migrate.py';tree=ast.parse(source.read_bytes())
        assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAPER_GROSS_METRICS_DDL' for t in n.targets))
        ddl=ast.literal_eval(assign.value);calls=[]
        class Conn:
            closed=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def close(self):self.closed=True
            def cursor(self):return self
            def execute(self,statement):
                calls.append(statement)
                if refuse:raise RuntimeError('synthetic schema refusal')
        conn=Conn();ns=dict(PAPER_GROSS_METRICS_DDL=ddl,logger=mock.Mock(),argparse=argparse,
            logging=SimpleNamespace(INFO=20,basicConfig=mock.Mock()),load_env=mock.Mock(),
            db=SimpleNamespace(get_db_conn=lambda:conn),run_upgrade=mock.Mock(return_value=True))
        extract(source,('_execute_statement','ensure_paper_gross_schema','main'),ns)
        return ns,conn,calls,tree
    def test_targeted_setup_contains_only_original_additive_table_contract(self):
        ns,conn,calls,tree=self.migration();self.assertTrue(ns['ensure_paper_gross_schema'](conn))
        self.assertEqual(calls,[ns['PAPER_GROSS_METRICS_DDL']]);statement=calls[0]
        self.assertIn('CREATE TABLE IF NOT EXISTS public.paper_gross_metrics_daily',statement)
        self.assertIn('run_date pg_catalog.date PRIMARY KEY',statement)
        self.assertIn('publication pg_catalog.jsonb NOT NULL',statement)
        self.assertIn("CHECK (publication->>'namespace' = 'PAPER_GROSS_PIPELINE_V1')",statement)
        self.assertFalse(any(word in statement.upper() for word in ('GRANT ','ALTER ','INSERT ','UPDATE ','DELETE ','TRADES','ORDER_EVENTS')))
        tables=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TABLE_STATEMENTS' for t in n.targets))
        self.assertEqual(sum(isinstance(n,ast.Name) and n.id=='PAPER_GROSS_METRICS_DDL' for n in tables.elts),1)
    def test_targeted_cli_only_executes_schema_statement_and_closes(self):
        ns,conn,calls,_=self.migration()
        with mock.patch('sys.argv',['db_migrate','--action','paper-gross-schema']):ns['main']()
        self.assertTrue(conn.closed);self.assertEqual(calls,[ns['PAPER_GROSS_METRICS_DDL']])
        ns['run_upgrade'].assert_not_called()
    def test_schema_refusal_remains_failure_and_closes_without_other_statements(self):
        ns,conn,calls,_=self.migration(refuse=True)
        with mock.patch('sys.argv',['db_migrate','--action','paper-gross-schema']),self.assertRaises(RuntimeError):ns['main']()
        self.assertTrue(conn.closed);self.assertEqual(calls,[ns['PAPER_GROSS_METRICS_DDL']])
        ns['run_upgrade'].assert_not_called()
        self.assertFalse(ns['ensure_paper_gross_schema'](None))
    def test_existing_default_upgrade_dispatch_is_retained(self):
        ns,conn,calls,_=self.migration()
        with mock.patch('sys.argv',['db_migrate']):ns['main']()
        ns['run_upgrade'].assert_called_once_with(conn);self.assertTrue(conn.closed);self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
