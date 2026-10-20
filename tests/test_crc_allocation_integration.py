"""Actual pure-policy/selected-consumer tests with a typed allocation-reader seam.
No applications, configuration, provider, DB drivers or real SQL execution.
Core validation/DDL/locking are tested separately; these fixtures do not prove PG.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Optional
import unittest

ROOT=Path(__file__).resolve().parents[1]
IDENTITY=('paper-fixture','00000000-0000-4000-8000-000000000001')
STAMP='2026-01-01T00:00:00+00:00'
CRC_ENTRY='7ad70913-5833-4f98-a3cc-d9a590da02a6'
CRC_EXITS=('da569a6d-2be6-4d1e-a3ae-9c639fce3668','1e248c11-d86f-431c-9c1a-ddd6e36365ff')
CRC_ORDERS=(CRC_ENTRY,)+CRC_EXITS


def selected(rel,names,env):
    tree=ast.parse((ROOT/rel).read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise AssertionError('source_inventory_changed')
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),rel,'exec'),env)
    return env


@dataclass(frozen=True)
class Allocation:
    trade_id:int=110
    financial_eligible:bool=False
    order_ids:tuple=CRC_ORDERS


def snapshot(trade_id=110,complete=False):
    return dict(trade_id=trade_id,symbol='CRC' if trade_id==110 else 'AAPL',status='CLOSED',
        qty=Decimal('30'),entry_price=Decimal('65'),entry_time=STAMP,updated_at=STAMP,
        exit_price=Decimal('66') if complete else None,exit_time=STAMP if complete else None,
        exit_reason='fixture' if complete else None,realized_pnl=Decimal('30') if complete else None,
        entry_order_id='entry-'+str(trade_id),exit_order_id=None)


class ReaderResult:
    def __init__(self,allocations,metadata):self.allocations=tuple(allocations);self._metadata=copy.deepcopy(metadata)
    def metadata_dict(self):return copy.deepcopy(self._metadata)


class PolicyCursor:
    def __init__(self,D,*,revision=2,case=True):
        self.D=D;self.revision=revision;self.case=case;self.trace=[];self.one=None;self.rows=[]
        self.snapshot=snapshot();self.owned_connection=None;self.allocation_approval=False
    def __enter__(self):return self
    def __exit__(self,*unused):return False
    def execute(self,sql,params=None):
        self.trace.append((sql,params));self.one=None;self.rows=[]
        if sql.startswith('SHOW transaction_isolation'):self.one=('read committed',)
        elif sql.startswith('SELECT deployment_id'):self.one=(*IDENTITY,'fixture',self.revision,'fixture')
        elif sql.startswith('SELECT c.request') and self.case:
            req,manifest=approval(self.D,snapshot(63),revision=0)
            receipt=dict(case_id=req['case_id'],trade_id=63,state='ACTIVE',case_revision=1,policy_revision=1,approval_ref=req['approval_ref'])
            snap=self.D.canonical(snapshot(63))
            self.rows=[(self.D.canonical(req),snap,self.D.canonical(manifest),1,self.D.fingerprint(req),1,snap,
                self.D.canonical(req),self.D.canonical(receipt))]
        elif sql.startswith('SELECT approval_ref FROM public.trade_allocation_audit') and self.allocation_approval:self.one=(params[0],)
        elif sql.startswith('SELECT pg_catalog.to_jsonb(t)'):self.one=(self.D.canonical(self.snapshot),)
        elif sql.startswith('INSERT') or sql.startswith('UPDATE'):raise AssertionError('UNEXPECTED_MUTATION')
    def fetchone(self):return self.one
    def fetchall(self):return self.rows


def approval(D,row,revision=2):
    manifest=dict(case_id='00000000-0000-4000-8000-'+format(row['trade_id'],'012d'),trade_id=row['trade_id'],
        deployment_id=IDENTITY[0],database_id=IDENTITY[1],row_version=STAMP,row_fingerprint=D.fingerprint(row),
        classification='UNRESOLVED',provenance_proved=False,
        operational_scopes=['ACTIVE_RECONCILIATION','VALIDATED_PERFORMANCE'],rationale='fixture approval',
        evidence=[{'reference':'fixture-only','sha256':'b'*64}])
    request={k:manifest[k] for k in ('case_id','trade_id','deployment_id','database_id','row_version','row_fingerprint')}
    request.update(action='APPLY',expected_policy_revision=revision,expected_case_revision=0,
        actor='fixture',approved_by='fixture',approved_at=STAMP,approval_ref='fixture-'+str(row['trade_id']),
        manifest_sha256=D.fingerprint(manifest))
    return request,manifest


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.D=types.ModuleType('crc_integration_disposition')
        sys.modules[self.D.__name__]=self.D
        exec(compile((ROOT/'scripts/trade_disposition.py').read_bytes(),'CRC_ACTUAL_PURE_POLICY','exec'),self.D.__dict__)
        self.reader_calls=[]
        self.metadata={'status':'VALIDATED','deployment_id':IDENTITY[0],'database_id':IDENTITY[1],
            'policy_revision':2,'active_count':1,'allocation_schema_version':1,'binding_fingerprint':'a'*64,
            'reserved_order_owners':{order:110 for order in CRC_ORDERS},
            'allocations':[{'trade_id':110,'allocation_id':'fixture-allocation','allocation_revision':1,
                'state':'ACTIVE','entry_quantity':'30','exit_quantity':'30','gross_result':'-63.38',
                'fees':None,'net':None,'fees_status':'UNKNOWN','net_status':'UNKNOWN',
                'financial_eligible':False,'audit_refs':['fixture-audit'],'source_refs':['fixture-native'],
                'order_ids':list(CRC_ORDERS),'activity_ids':['fixture-native']} ]}
        self.overlay=ReaderResult((Allocation(),),self.metadata)
        def read(cursor,identity,revision,excluded):
            self.reader_calls.append((identity,revision,excluded))
            self.assertTrue(any('pg_advisory_xact_lock' in sql for sql,_ in cursor.trace))
            return self.overlay
        self.D._read_allocation_overlay=read
        self.cursor=PolicyCursor(self.D)
        self.policy=self.D.read_policy(self.cursor,IDENTITY,scope='ACTIVE_RECONCILIATION')

    def policy_for(self,scope='VALIDATED_PERFORMANCE',*,revision=2,allocations=None,metadata=None):
        return self.D.Policy(*IDENTITY,revision,self.policy.active,scope,
            self.policy.allocations if allocations is None else allocations,
            self.D.canonical(self.metadata if metadata is None else metadata))

    def db(self,names,**extra):
        env=dict(disposition=self.D,Mapping=Mapping,datetime=datetime,timezone=timezone,
            contextmanager=contextmanager,logger=logging.getLogger('crc-fixture'),**extra)
        return selected('scripts/db.py',set(names),env)

    def test_policy_preserves_exclusion_and_protects_allocation_original(self):
        self.assertEqual(self.policy.excluded_ids,{63})
        self.assertEqual(self.policy.all_excluded_ids,{63,110})
        self.assertEqual(self.reader_calls,[(IDENTITY,2,frozenset({63}))])
        self.assertEqual(self.policy.metadata()['excluded_count'],1)
        self.assertFalse(self.policy.metadata()['accounting_complete'])
        with self.assertRaisesRegex(self.D.TradePolicyError,'TRADE_EXCLUDED'):
            self.D.require_writable(self.policy,110)

    def test_optional_migration_absence_keeps_existing_aapl_scope(self):
        meta=dict(self.metadata,status='MIGRATION_PENDING',active_count=0,allocation_schema_version=0,allocations=[],reserved_order_owners={})
        self.overlay=ReaderResult((),meta)
        policy=self.D.read_policy(PolicyCursor(self.D),IDENTITY,scope='ACTIVE_RECONCILIATION')
        self.assertEqual(policy.excluded_ids,{63});self.assertEqual(policy.all_excluded_ids,{63})
        self.assertEqual(policy.metadata()['allocation_overlay']['status'],'MIGRATION_PENDING')
        self.assertNotIn('NOT IN',self.D.reconciliation_eligible_sql(policy))

    def test_permission_failure_is_not_empty_overlay(self):
        def denied(*unused):raise PermissionError('fixture permission denied')
        self.D._read_allocation_overlay=denied
        with self.assertRaises(PermissionError):self.D.read_policy(PolicyCursor(self.D),IDENTITY)

    def test_disposition_allocation_collision_refuses(self):
        self.overlay=ReaderResult((Allocation(63),),self.metadata)
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_CASE_CONFLICT'):
            self.D.read_policy(PolicyCursor(self.D),IDENTITY)

    def test_overlay_identity_revision_mismatch_refuses(self):
        for key,value in [('database_id','wrong'),('deployment_id','wrong'),('policy_revision',1)]:
            with self.subTest(key=key):
                self.overlay=ReaderResult((Allocation(),),dict(self.metadata,**{key:value}))
                with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_IDENTITY_MISMATCH'):
                    self.D.read_policy(PolicyCursor(self.D),IDENTITY)

    def test_returned_metadata_cannot_mutate_retained_binding(self):
        before=self.policy.metadata()
        result=self.policy.metadata();result['allocation_overlay']['allocations'][0]['gross_result']='999'
        self.metadata['allocations'][0]['source_refs'].append('changed caller')
        self.assertEqual(self.policy.metadata(),before)

    def selection(self,policy,strict=True):
        calls=[];rows=[snapshot(110),snapshot(111)]
        class Query:
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def execute(inner,sql,params):calls.append((sql,params))
            def fetchall(inner):
                # Synthetic SQL protocol: missing overlay returns the saturated
                # two-row result and would fail the real strict caller.
                return copy.deepcopy(rows[1:] if 'trades.trade_id NOT IN (110)' in calls[-1][0] else rows)
        conn=types.SimpleNamespace(cursor=lambda **unused:Query())
        @contextmanager
        def maybe(*unused):yield conn
        @contextmanager
        def transaction(*args,**kwargs):yield policy
        env=self.db({'get_closed_trades_missing_exit','_reconciliation_trade_records'},
            _maybe_conn=maybe,_reconciliation_read_transaction=transaction,
            extras=types.SimpleNamespace(RealDictCursor=object()))
        result=env['get_closed_trades_missing_exit'](conn,datetime(2026,1,1,tzinfo=timezone.utc),
            limit=2 if strict else 3,strict=strict,expected_policy_revision=policy.revision)
        self.assertEqual(rows,[snapshot(110),snapshot(111)])
        return result,calls

    def test_strict_resolution_is_before_limit_not_a_saturated_postfilter(self):
        rows,calls=self.selection(self.policy)
        self.assertEqual([r['trade_id'] for r in rows],[111])
        sql,params=calls[0]
        self.assertLess(sql.index('trades.trade_id NOT IN (110)'),sql.index('ORDER BY'))
        self.assertLess(sql.index('trades.trade_id NOT IN (110)'),sql.index('LIMIT'))
        self.assertTrue(params['strict']);self.assertEqual(params['limit'],2)

    def test_nonstrict_history_remains_unfiltered_by_allocation(self):
        rows,calls=self.selection(self.policy,strict=False)
        self.assertEqual([r['trade_id'] for r in rows],[110,111]);self.assertNotIn('NOT IN',calls[0][0])

    def test_revocation_restores_unresolved_query_and_preserves_history(self):
        metadata=dict(self.metadata,policy_revision=3,active_count=0,allocations=[])
        policy=self.policy_for('ACTIVE_RECONCILIATION',revision=3,allocations=(),metadata=metadata)
        # Two unresolved rows saturate strict limit2 again; no values are filled.
        with self.assertRaisesRegex(RuntimeError,'RECONCILE_DB_INCOMPLETE'):self.selection(policy)
        self.assertNotIn('NOT IN',self.D.reconciliation_eligible_sql(policy))
        self.assertEqual(policy.excluded_ids,{63})
        self.assertEqual(policy.all_excluded_ids,{63})
        self.assertIsNone(self.D.require_writable(policy,110))
        # Historical reservations still protect the orders against other trades,
        # while allowing their rightful owner and a separately approved case.
        for child in CRC_EXITS:
            self.assertIsNone(self.D.require_order_owner(policy,110,child))
            with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_OWNER_CONFLICT'):
                self.D.require_order_owner(policy,111,child)
        self.overlay=ReaderResult((),metadata)
        class ApplyingCursor(PolicyCursor):
            def execute(self,sql,params=None):
                if sql.startswith(('INSERT INTO public.trade_disposition_cases',
                        'UPDATE public.trade_disposition_policy','INSERT INTO public.trade_disposition_audit')):
                    self.trace.append((sql,params));self.one=None;self.rows=[];self.rowcount=1
                else:super().execute(sql,params)
        cursor=ApplyingCursor(self.D,revision=3)
        class Connection:
            autocommit=False
            def get_transaction_status(self):return 0
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def cursor(self):return cursor
        request,manifest=approval(self.D,snapshot(),revision=3)
        result=self.D.change_disposition(Connection(),request,manifest,IDENTITY)
        self.assertEqual(result['state'],'ACTIVE');self.assertEqual(result['policy_revision'],4)
        self.assertTrue(any(sql.startswith('INSERT INTO public.trade_disposition_audit') for sql,_ in cursor.trace))
        self.assertFalse(any(sql.startswith(('UPDATE public.trades','INSERT INTO public.order_events',
            'UPDATE public.alpaca_activities')) for sql,_ in cursor.trace))

    def test_financial_input_never_substitutes_audited_gross_even_complete_scalars(self):
        policy=self.policy_for()
        def read(*unused):raise AssertionError('FINANCIAL_QUERY_BEFORE_HOLD')
        env=self.db({'performance_input'},_performance_input_records=read)
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):
            env['performance_input'](object(),policy)
        self.assertEqual(self.policy.metadata()['allocation_overlay']['allocations'][0]['gross_result'],'-63.38')

    def test_ml_input_holds_affected_window_but_preserves_unaffected_window(self):
        calls=[];rows=[];D=self.D
        class Query:
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def execute(self,sql,params):calls.append((sql,params))
            def fetchall(self):return [(D.canonical(row),) for row in rows]
        conn=types.SimpleNamespace(cursor=lambda:Query())
        env=self.db({'ml_trade_input'})
        start=datetime(2026,1,1,tzinfo=timezone.utc);end=datetime(2026,2,1,tzinfo=timezone.utc)
        rows[:]=[snapshot(110,True)]
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):
            env['ml_trade_input'](conn,self.policy_for(),start_dt=start,end_dt=end)
        rows[:]=[snapshot(111,True)]
        result=env['ml_trade_input'](conn,self.policy_for(),start_dt=start,end_dt=end)
        self.assertEqual(result,rows);self.assertEqual(calls[-1][1],(start,end))
        self.assertIn('COALESCE(t.exit_time,t.entry_time)>=%s',calls[-1][0])

    def test_evidence_builders_cannot_certify_nonfinancial_allocation(self):
        env=self.db({'performance_evidence','ml_trade_evidence'})
        for invoke in [lambda:env['performance_evidence'](self.policy_for(),[]),
                lambda:env['ml_trade_evidence'](self.policy_for(),[snapshot(110,True)],start_dt=None,end_dt=None,match_inputs={})]:
            with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):invoke()
        start=datetime(2026,1,1,tzinfo=timezone.utc);end=datetime(2026,2,1,tzinfo=timezone.utc)
        evidence=env['ml_trade_evidence'](self.policy_for(),[snapshot(111,True)],
            start_dt=start,end_dt=end,match_inputs={})
        self.assertEqual(evidence['input_count'],1)
        self.assertEqual(evidence['policy']['allocation_overlay']['active_count'],1)
        self.assertFalse(evidence['accounting_complete'])

    def test_hold_visibility_keeps_gross_and_unknown_net_even_complete_original(self):
        @contextmanager
        def maybe(*unused):yield object()
        @contextmanager
        def transaction(*args,**kwargs):yield self.policy_for()
        original=snapshot(110,True)
        env=self.db({'performance_hold_visibility'},_maybe_conn=maybe,_policy_read_transaction=transaction,
            _performance_input_records=lambda *unused:[copy.deepcopy(original)])
        result=env['performance_hold_visibility']()
        self.assertEqual(result['policy_validation_status'],'VALIDATED_POLICY')
        self.assertEqual(result['excluded_count'],1);self.assertEqual(result['unresolved_trade_ids'],[110])
        self.assertEqual(result['accounting_input_status'],'INCOMPLETE')
        alloc=result['allocation_visibility']['allocations'][0]
        self.assertEqual(alloc['gross_result'],'-63.38');self.assertIsNone(alloc['net']);self.assertIsNone(alloc['fees'])
        self.assertEqual(original,snapshot(110,True))

    def test_prior_financial_artifact_cannot_be_consumed_after_activation(self):
        @contextmanager
        def maybe(*unused):yield object()
        @contextmanager
        def transaction(*args,**kwargs):yield self.policy_for()
        env=self.db({'validated_performance_artifact','performance_input','performance_evidence'},
            _maybe_conn=maybe,_policy_read_transaction=transaction)
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ACCOUNTING_INCOMPLETE'):
            with env['validated_performance_artifact']({'policy':{'policy_revision':1}}):
                self.fail('STALE_ARTIFACT_YIELDED')

    def test_dashboard_message_names_gross_unknown_net_and_audit(self):
        env=selected('dashboards/dashboard_app.py',{'_performance_hold_message'}, {})
        message=env['_performance_hold_message']({'policy_validation_status':'VALIDATED_POLICY',
            'trade_policy':{'policy':self.policy_for().metadata()},'unresolved_trade_ids':[110]})
        for text in ['Policy revision 2','excluded trades 1','-63.38','fees and net UNKNOWN','fixture-audit']:
            self.assertIn(text,message)
        self.assertNotIn('net -63.38',message)

    def test_ml_overview_held_payload_carries_overlay_without_legacy_fallback(self):
        def refused(*unused):raise self.D.TradePolicyError('HELD')
        def legacy():raise AssertionError('LEGACY_FALLBACK')
        visibility={'policy_validation_status':'VALIDATED_POLICY','allocation_visibility':copy.deepcopy(self.metadata)}
        env=selected('dashboards/ml_overview_api.py',{'_artifact_record'},dict(
            db=types.SimpleNamespace(validated_ml_trade_artifact=refused,
                performance_hold_visibility=lambda:visibility,db_enabled=legacy)))
        result=env['_artifact_record']('ranker_trade_attribution',base_dir=Path('UNUSED'))
        self.assertFalse(result['present']);self.assertEqual(result['payload']['allocation_visibility'],self.metadata)
        self.assertIsNone(result['rows_count'])

    def test_alias_and_scope_are_not_untrusted_sql_inputs(self):
        with self.assertRaises(self.D.TradePolicyError):self.D.reconciliation_eligible_sql(self.policy,'trades;DROP')
        with self.assertRaises(self.D.TradePolicyError):self.D.reconciliation_eligible_sql(self.policy_for())

    def test_competing_disposition_apply_cannot_claim_allocated_trade(self):
        D=self.D;cursor=PolicyCursor(D);request,manifest=approval(D,snapshot())
        class Connection:
            autocommit=False
            def get_transaction_status(self):return 0
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def cursor(self):return cursor
        with self.assertRaisesRegex(D.TradePolicyError,'CASE_CONFLICT'):
            D.change_disposition(Connection(),request,manifest,IDENTITY)
        self.assertFalse(any(sql.startswith(('INSERT','UPDATE')) for sql,_ in cursor.trace))

    def test_disposition_cannot_reuse_allocation_approval_reference(self):
        D=self.D;cursor=PolicyCursor(D);cursor.snapshot=snapshot(111);cursor.allocation_approval=True
        request,manifest=approval(D,snapshot(111))
        class Connection:
            autocommit=False
            def get_transaction_status(self):return 0
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def cursor(self):return cursor
        with self.assertRaisesRegex(D.TradePolicyError,'APPROVAL_REUSED'):
            D.change_disposition(Connection(),request,manifest,IDENTITY)
        self.assertFalse(any(sql.startswith(('INSERT','UPDATE')) for sql,_ in cursor.trace))


    def lineage(self,policy,child,*,physical_conflict=False):
        parent='11111111-1111-4111-8111-111111111111'
        at=datetime(2026,1,1,tzinfo=timezone.utc)
        opened=dict(trade_id=111,status='OPEN',symbol='CRC',qty=Decimal('2'),entry_price=Decimal('10'),
            entry_time=at,entry_order_id=parent,exit_order_id=None,exit_time=None,exit_price=None,
            exit_reason=None,realized_pnl=None)
        event=dict(event_id=901,event_type='TRAIL_SUBMIT',symbol='CRC',qty='2',order_id=child,
            parent_order_id=parent,envelope_order_id=child,envelope_event_type='TRAIL_SUBMIT',
            metadata_event_type='TRAIL_SUBMIT',raw_shape_valid=True,top_parent_key_present=False,
            top_parent_order_id=None,recorded_submitted_at=at,event_time=at)
        physical=dict(trade_id=111,entry_order_id=parent,exit_order_id=None,identity_oversize=False)
        if physical_conflict:physical.update(trade_id=222,exit_order_id=CRC_EXITS[0])
        answers=[[opened],[event],[event],[physical]];calls=[]
        def rows(cursor,sql,params,cap):
            calls.append((sql,params,cap));return copy.deepcopy(answers.pop(0))
        class Query:
            def __enter__(self):return self
            def __exit__(self,*unused):return False
        conn=types.SimpleNamespace(cursor=lambda:Query())
        @contextmanager
        def maybe(*unused):yield conn
        @contextmanager
        def transaction(*args,**kwargs):yield policy
        tree=ast.parse((ROOT/'scripts/db.py').read_text(encoding='utf8'))
        constants={target.id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            for target in node.targets if isinstance(target,ast.Name)
            and target.id in {'_OPEN_EXIT_LINEAGE_CAP','_OPEN_EXIT_EVENT_COLUMNS'}}
        env=self.db({'get_open_trade_exit_lineage','_require_open_exit_lineage','_lineage_uuid',
                '_lineage_number','_lineage_time','_open_exit_lineage_record'},
            Decimal=Decimal,parser=types.SimpleNamespace(isoparse=lambda text:datetime.fromisoformat(text)),
            _maybe_conn=maybe,_reconciliation_read_transaction=transaction,
            _lineage_rows=rows,_open_exit_lineage_guard=lambda cursor:None,**constants)
        return env,conn,opened,calls

    def test_lineage_reader_refuses_each_crc_exit_active_and_revoked_before_offering(self):
        for active in (True,False):
            metadata=dict(self.metadata,active_count=int(active),allocations=self.metadata['allocations'] if active else [])
            policy=self.policy_for('ACTIVE_RECONCILIATION',allocations=self.policy.allocations if active else (),metadata=metadata)
            for child in CRC_EXITS:
                with self.subTest(active=active,child=child):
                    env,conn,opened,calls=self.lineage(policy,child)
                    with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_OWNER_CONFLICT'):
                        env['get_open_trade_exit_lineage'](conn,[opened],observed_before=datetime(2026,1,2,tzinfo=timezone.utc))
                    self.assertEqual(len(calls),4)
                    self.assertTrue(all('ONLY public.' in query for query,_,_ in calls))
                    self.assertIsNone(opened['exit_order_id'])

    def test_lineage_writer_refuses_reserved_crc_exit_before_any_write_proof_query(self):
        for active in (True,False):
            metadata=dict(self.metadata,active_count=int(active),allocations=self.metadata['allocations'] if active else [])
            policy=self.policy_for('ACTIVE_RECONCILIATION',allocations=self.policy.allocations if active else (),metadata=metadata)
            for child in CRC_EXITS:
                with self.subTest(active=active,child=child):
                    env,conn,opened,calls=self.lineage(policy,child)
                    with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_OWNER_CONFLICT'):
                        env['_require_open_exit_lineage'](object(),opened,child,{},None,None,policy=policy)
                    self.assertEqual(calls,[])
                    self.assertIsNone(opened['exit_order_id'])

    def test_lineage_unclaimed_child_still_works_and_physical_contradiction_raises(self):
        child='22222222-2222-4222-8222-222222222222'
        env,conn,opened,calls=self.lineage(self.policy,child)
        self.assertEqual(env['get_open_trade_exit_lineage'](conn,[opened],
            observed_before=datetime(2026,1,2,tzinfo=timezone.utc)),{111:child})
        env,conn,opened,calls=self.lineage(self.policy,child,physical_conflict=True)
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_OWNER_CONFLICT'):
            env['get_open_trade_exit_lineage'](conn,[opened],observed_before=datetime(2026,1,2,tzinfo=timezone.utc))

    def test_reserved_claim_metadata_missing_or_inconsistent_is_not_empty_ownership(self):
        metadata=dict(self.metadata);metadata.pop('reserved_order_owners')
        policy=self.policy_for(allocations=(),metadata=metadata)
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_CLAIMS_UNAVAILABLE'):
            self.D.require_order_owner(policy,111,CRC_EXITS[0])
        metadata=dict(self.metadata,reserved_order_owners={CRC_ENTRY:110})
        with self.assertRaisesRegex(self.D.TradePolicyError,'ALLOCATION_ORDER_CLAIMS_INVALID'):
            self.D.require_order_owner(self.policy_for(metadata=metadata),111,CRC_EXITS[0])



if __name__=='__main__':unittest.main()
