"""Offline contract tests for safe reconciliation failure evidence."""
import ast
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import types
import unittest
from collections.abc import Mapping


ROOT = Path(__file__).resolve().parents[1]
HEALTH_SOURCE = (ROOT / 'scripts/executor_health.py').read_text(encoding='utf-8')
EXECUTE_SOURCE = (ROOT / 'scripts/execute_trades.py').read_text(encoding='utf-8')


def functions(source, names, scope):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<safe-source-fixture>', 'exec'), scope)
    return scope


def health_fixture():
    import os, stat, uuid
    scope = dict(json=json, re=re, os=os, stat=stat, uuid=uuid, Path=Path, hashlib=hashlib,
                 datetime=datetime, timezone=timezone)
    tree = ast.parse(HEALTH_SOURCE)
    names = {'RECONCILIATION_FAILURE_CODES', 'MAX_RECONCILIATION_FAILURE_FINDINGS',
             'MAX_RECONCILIATION_FAILURE_BYTES', '_RECONCILIATION_COUNT_MAX',
             'ROLES', 'COUNTS', 'MAX_OBSERVATIONS', 'MAX_BYTES', 'SOURCES',
             '_POLICY_REFUSAL_CODES', '_POLICY_SQLSTATE_CATEGORIES'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(a, ast.Name) and a.id in names for a in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<fixed-constants>', 'exec'), scope)
    functions(HEALTH_SOURCE, {'new_reconciliation_failure_evidence', 'record_reconciliation_failure',
              'reconciliation_failure_payload', '_validated_reconciliation_failure_payload', 'binding', 'ordinary', 'read', 'utc',
              'assess_outcome', '_policy_failure_payload', 'publish_receipt'}, scope)
    return types.SimpleNamespace(**scope)


H = health_fixture()


class CollectorTests(unittest.TestCase):
    def test_empty_has_explicit_zero_failure_occurrences_not_financial_success(self):
        out = H.reconciliation_failure_payload(H.new_reconciliation_failure_evidence())
        self.assertEqual(out['total_count'], 0)
        self.assertEqual(out['findings'], [])
        self.assertTrue(out['evidence_complete'])
        self.assertEqual(out['count_semantics'], 'GUARD_OCCURRENCES')
        self.assertNotIn('operational_complete', out)
        self.assertNotIn('accounting_complete', out)

    def test_duplicates_are_counted_and_stored_once(self):
        state = H.new_reconciliation_failure_evidence()
        for _ in range(3):H.record_reconciliation_failure(state, 'HISTORICAL_EXIT_UNBOUND', 1)
        out = H.reconciliation_failure_payload(state)
        self.assertEqual(out['total_count'], 3)
        self.assertEqual(out['findings'], [{'reason_code': 'HISTORICAL_EXIT_UNBOUND', 'trade_id': 1, 'occurrences': 3}])
        self.assertEqual(out['reason_counts'], {'HISTORICAL_EXIT_UNBOUND': 3})

    def test_truncation_retains_total_counts_and_represented_duplicates(self):
        state = H.new_reconciliation_failure_evidence()
        for i in range(1, 71):H.record_reconciliation_failure(state, 'HISTORICAL_EXIT_UNBOUND', i)
        H.record_reconciliation_failure(state, 'HISTORICAL_EXIT_UNBOUND', 70)
        H.record_reconciliation_failure(state, 'HISTORICAL_EXIT_UNBOUND', 1)
        out = H.reconciliation_failure_payload(state)
        self.assertEqual(len(out['findings']), 64)
        self.assertEqual(out['total_count'], 72)
        self.assertEqual(out['omitted_count'], 7)
        self.assertEqual(sum(x['occurrences'] for x in out['findings']), 65)
        self.assertTrue(out['truncated']);self.assertFalse(out['evidence_complete'])

    def test_invalid_ids_are_unattributed_without_coercion(self):
        for value in (None, True, False, '1', -1, 0, 2**63, 1.0, {'account': 'SYNTHETIC_SECRET'}):
            with self.subTest(value_type=type(value).__name__):
                state = H.new_reconciliation_failure_evidence()
                H.record_reconciliation_failure(state, 'DB_UNAVAILABLE', value)
                out = H.reconciliation_failure_payload(state)
                self.assertEqual(out['unattributed_count'], 1)
                self.assertIsNone(out['findings'][0]['trade_id'])
                self.assertNotIn('SYNTHETIC_SECRET', json.dumps(out))

    def test_unknown_reason_does_not_publish_raw_text_or_replace_result(self):
        state = H.new_reconciliation_failure_evidence()
        H.record_reconciliation_failure(state, 'SYNTHETIC_SECRET SQL account=123', 1)
        out = H.reconciliation_failure_payload(state)
        self.assertEqual(out, {'schema_version': 1, 'status': 'INVALID', 'evidence_complete': False})
        H.record_reconciliation_failure(None, 'DB_UNAVAILABLE')

    def test_payload_is_a_deep_copy(self):
        state = H.new_reconciliation_failure_evidence()
        H.record_reconciliation_failure(state, 'DB_UNAVAILABLE')
        out = H.reconciliation_failure_payload(state)
        state['findings'][0]['occurrences'] = 999
        state['reason_counts']['DB_UNAVAILABLE'] = 999
        self.assertEqual(out['findings'][0]['occurrences'], 1)
        self.assertEqual(out['reason_counts']['DB_UNAVAILABLE'], 1)

    def test_malformed_counts_ids_fields_and_relations_are_redacted(self):
        base = H.new_reconciliation_failure_evidence()
        H.record_reconciliation_failure(base, 'DB_UNAVAILABLE')
        variants = []
        for key, value in [('total_count', True), ('total_count', -1), ('total_count', 2**63),
                           ('omitted_count', 1), ('unattributed_count', 0), ('truncated', True),
                           ('schema_version', True), ('invalid', True), ('token', 'SYNTHETIC_SECRET')]:
            changed = copy.deepcopy(base);changed[key] = value;variants.append(changed)
        changed = copy.deepcopy(base);changed['reason_counts'] = {'RAW_SECRET': 1};variants.append(changed)
        changed = copy.deepcopy(base);changed['findings'][0]['trade_id'] = True;variants.append(changed)
        changed = copy.deepcopy(base);changed['findings'][0]['raw_sql'] = 'SYNTHETIC_SECRET';variants.append(changed)
        changed = copy.deepcopy(base);changed['findings'].append(changed['findings'][0].copy());variants.append(changed)
        for i, value in enumerate(variants):
            with self.subTest(index=i):
                self.assertEqual(H.reconciliation_failure_payload(value), {'schema_version': 1, 'status': 'INVALID', 'evidence_complete': False})

    def test_worst_allowed_integer_payload_remains_under_16k(self):
        state = H.new_reconciliation_failure_evidence()
        code = max(H.RECONCILIATION_FAILURE_CODES, key=len)
        for i in range(64):H.record_reconciliation_failure(state, code, 2**63 - 1 - i)
        out = H.reconciliation_failure_payload(state)
        self.assertEqual(out['status'], 'VALID')
        self.assertLessEqual(len(json.dumps(out, separators=(',', ':')).encode()), 16384)


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):return cls(2026, 10, 22, 22, 15, 32, tzinfo=timezone.utc)


POLICY = dict(scope='ACTIVE_RECONCILIATION', policy_revision=1, excluded_count=1, recovery_pending_cases=[])
OPEN = dict(trade_id=1, symbol='SYN', qty=30, entry_price=65,
            entry_time=datetime(2026, 10, 21, tzinfo=timezone.utc), exit_order_id='order-a')
CLOSED = dict(OPEN, trade_id=2, status='CLOSED', exit_price=None, realized_pnl=None)
ORDER = dict(id='order-a', symbol='SYN', side='sell', status='filled',
             filled_qty='30', qty='30', filled_avg_price='63', type='limit',
             filled_at='2026-10-22T21:00:00+00:00', updated_at='2026-10-22T21:00:00+00:00')
CASES = (
    'success', 'db_disabled', 'db_engine_missing', 'policy_read_failed', 'watermark_read_failed',
    'watermark_state_invalid', 'watermark_time_invalid', 'watermark_time_future',
    'orders_not_list', 'orders_saturated', 'orders_record_invalid', 'orders_future', 'orders_fill_invalid', 'orders_failed',
    'open_read_failed', 'lineage_failed', 'client_missing', 'positions_invalid',
    'position_close_refused', 'ambiguous_fill', 'unbound_fill_with_unrelated_bound_row',
    'open_partial', 'open_close_refused', 'flat_exact_child_unresolved', 'closed_read_failed',
    'closed_unbound', 'closed_fill_missing', 'closed_partial', 'closed_decorate_refused',
    'policy_revision_changed', 'policy_publication_failed', 'open_close_exception', 'closed_decorate_exception',
)


def exercise(source, case, covered=None):
    calls, logs = [], []
    open_rows, closed_rows, orders = [], [], []
    position_cases = {'positions_invalid', 'position_close_refused', 'ambiguous_fill', 'unbound_fill_with_unrelated_bound_row',
                      'open_partial', 'open_close_refused', 'open_close_exception', 'flat_exact_child_unresolved'}
    if case in position_cases:open_rows = [OPEN.copy()]
    if case == 'position_close_refused':open_rows[0]['exit_order_id'] = None
    if case == 'ambiguous_fill':open_rows.append(dict(OPEN, trade_id=3));orders = [ORDER.copy()]
    if case == 'unbound_fill_with_unrelated_bound_row':
        open_rows = [dict(OPEN, exit_order_id=None), dict(OPEN, trade_id=3, exit_order_id='unrelated-order')];orders=[ORDER.copy()]
    if case in {'open_partial', 'open_close_refused', 'open_close_exception'}:orders=[ORDER.copy()]
    if case == 'open_partial':orders[0]['filled_qty']='15'
    if case.startswith('closed_') and case != 'closed_read_failed':closed_rows=[CLOSED.copy()]
    if case == 'closed_unbound':closed_rows[0]['exit_order_id']=None
    if case in {'closed_partial', 'closed_decorate_refused', 'closed_decorate_exception'}:orders=[ORDER.copy()]
    if case == 'closed_partial':orders[0]['filled_qty']='15'
    if case == 'orders_not_list':orders={}
    if case == 'orders_saturated':orders=[ORDER.copy() for _ in range(2)]
    if case == 'orders_record_invalid':orders=[{}]
    if case == 'orders_future':orders=[dict(ORDER, updated_at='2026-10-22T23:00:00+00:00')]
    if case == 'orders_fill_invalid':orders=[dict(ORDER, filled_qty='not-numeric')]
    def fail():raise RuntimeError('SYNTHETIC_SECRET sql/account credentials must not escape')
    class DB:
        def db_enabled(self):calls.append(('db_enabled',));return case != 'db_disabled'
        def get_db_conn(self):calls.append(('get_db_conn',));return None if case=='db_engine_missing' else 'synthetic-engine'
        def reconciliation_policy_status(self, engine):
            calls.append(('policy',engine))
            if case=='policy_read_failed':fail()
            return POLICY.copy()
        def get_reconcile_state(self, engine, **kwargs):
            calls.append(('state',kwargs))
            if case=='watermark_read_failed':fail()
            if case=='watermark_state_invalid':return []
            if case=='watermark_time_invalid':return dict(last_after='bad')
            if case=='watermark_time_future':return dict(last_after='2026-10-23T00:00:00+00:00')
            return dict(last_after='2026-10-21T00:00:00+00:00')
        def normalize_ts(self, value, **kwargs):
            if value is None:return None
            if isinstance(value,datetime):return value
            try:return datetime.fromisoformat(str(value).replace('Z','+00:00'))
            except ValueError:return None
        def get_open_trades(self, engine, **kwargs):
            calls.append(('open',kwargs))
            if case=='open_read_failed':fail()
            return copy.deepcopy(open_rows)
        def get_open_trade_exit_lineage(self, engine, rows, **kwargs):
            calls.append(('lineage',copy.deepcopy(rows),kwargs))
            if case=='lineage_failed':fail()
            return {}
        def close_trade(self, **kwargs):calls.append(('close_trade',kwargs));return False
        def get_closed_trades_missing_exit(self, **kwargs):
            calls.append(('closed',kwargs))
            if case=='closed_read_failed':fail()
            return copy.deepcopy(closed_rows)
        def reconcile_sell_fill(self, **kwargs):
            calls.append(('sell_fill',kwargs))
            if case in {'open_close_exception','closed_decorate_exception'}:fail()
            return False
        @contextmanager
        def reconciliation_publication(self, engine, **kwargs):
            calls.append(('publication',kwargs))
            if case=='policy_publication_failed':fail()
            yield dict(POLICY,policy_revision=2) if case=='policy_revision_changed' else POLICY.copy()
    class Client:
        def get_all_positions(self):
            calls.append(('positions',))
            if case=='positions_invalid':return None
            return [] if case in {'position_close_refused','flat_exact_child_unresolved'} else [types.SimpleNamespace(symbol='SYN',qty='30')]
    def list_orders(**kwargs):
        calls.append(('list_orders',kwargs))
        if case=='orders_failed':fail()
        return copy.deepcopy(orders)
    def capture(message,*args):logs.append(message % args if args else message)
    logger=types.SimpleNamespace(warning=capture,info=capture)
    scope=dict(datetime=FrozenDateTime,timedelta=timedelta,timezone=timezone,Mapping=Mapping,Decimal=Decimal,
               InvalidOperation=InvalidOperation,Any=object,Dict=dict,db=DB(),LOGGER=logger,
               get_trading_client=lambda:None,alpaca_list_orders_http=list_orders,
               isoparse=lambda v:datetime.fromisoformat(v.replace('Z','+00:00')),
               log_info=lambda *args,**kwargs:None,_order_snapshot=lambda order:copy.deepcopy(order),
               executor_health=types.SimpleNamespace(new_reconciliation_failure_evidence=H.new_reconciliation_failure_evidence,
                    record_reconciliation_failure=H.record_reconciliation_failure,
                    policy_failure_diagnostic=lambda e:dict(refusal_code='UNKNOWN',sqlstate_category='UNKNOWN',paper_endpoint_exact=True,paper_flag_exact_true=True)))
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='reconcile_closed_trades')
    name='<candidate-reconciliation>' if 'failure_evidence=' in source else '<original-reconciliation>'
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),scope)
    obj=types.SimpleNamespace(metrics=types.SimpleNamespace(),config=types.SimpleNamespace(reconcile_use_watermark=True,reconcile_lookback_days=14,reconcile_limit=2,reconcile_overlap_secs=300),client=None if case=='client_missing' else Client())
    def trace(frame,event,arg):
        if covered is not None and event=='line' and frame.f_code.co_filename==name:covered.add(frame.f_lineno)
        return trace
    old=sys.gettrace()
    try:
        if covered is not None:sys.settrace(trace)
        result=scope['reconcile_closed_trades'](obj)
    finally:sys.settrace(old)
    return result,obj.metrics.reconciliation,calls,logs


class RuntimeContractTests(unittest.TestCase):
    # The baseline test runner supplies exact originalsource as a classattribute;
    # repository execution compares an AST with diagnostics removed below.
    before_source = None

    def test_every_added_guard_site_executes_with_unchanged_results_and_io_sequence(self):
        covered=set();observed_codes=set()
        for case in CASES:
            with self.subTest(case=case):
                if self.before_source is not None:
                    old,old_meta,old_calls,old_logs=exercise(self.before_source,case)
                new,new_meta,new_calls,new_logs=exercise(EXECUTE_SOURCE,case,covered)
                evidence=new_meta.pop('failure_evidence')
                self.assertEqual(new,case=='success')
                if self.before_source is not None:
                    self.assertEqual(new,old);self.assertEqual(new_meta,old_meta)
                    self.assertEqual(new_calls,old_calls);self.assertEqual(new_logs,old_logs)
                out=H.reconciliation_failure_payload(evidence)
                self.assertEqual(out['status'],'VALID');self.assertNotIn('SYNTHETIC_SECRET',json.dumps(out))
                observed_codes.update(out['reason_counts'])
                if new is False:self.assertGreater(out['total_count'],0)
                else:self.assertEqual(out['total_count'],0)
                if case=='unbound_fill_with_unrelated_bound_row':
                    ids={f['trade_id'] for f in out['findings'] if f['reason_code']=='FILL_ALLOCATION_UNBOUND_OR_AMBIGUOUS'}
                    self.assertEqual(ids,{1})
        tree=ast.parse(EXECUTE_SOURCE)
        sites={n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'record_failure','record_failure_group'} and n.args and isinstance(n.args[0],ast.Constant)}
        self.assertEqual(len(sites),31);self.assertFalse(sites-covered,sorted(sites-covered))
        self.assertEqual(observed_codes,H.RECONCILIATION_FAILURE_CODES)



class ReceiptTests(unittest.TestCase):
    def publish(self, evidence, role='reconciliation', rc=1, include=True):
        import tempfile
        temp_base=ROOT/'.test-reconciliation-receipts'
        temp_base.mkdir(exist_ok=True)
        self.assertTrue(temp_base.resolve().is_relative_to(ROOT.resolve()))
        with tempfile.TemporaryDirectory(dir=temp_base) as name:
            root=Path(name)
            for source in H.SOURCES:
                path=root/source;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic source binding only\n')
            metrics=dict(status='failed' if rc else 'ok',exit_reason='RECONCILE_INCOMPLETE' if rc else 'NO_ACTION',
                         symbols_in=0,orders_submitted=0,orders_filled=0,orders_canceled=0,trailing_attached=0,api_failures=0,in_window=False,skips={})
            if include:metrics['reconciliation']=dict(reason='OPERATIONAL_WORK_PENDING',failure_evidence=evidence)
            started=datetime(2026,10,22,22,15,tzinfo=timezone.utc)
            record=H.publish_receipt(root,role=role,started_at=started,finished_at=started,rc=rc,metrics=metrics)
            self.assertEqual(json.loads((root/'reports/executor'/role/('run-'+record['run_id']+'.json')).read_bytes()),record)
            self.assertEqual(json.loads((root/'reports/executor'/role/'latest.json').read_bytes()),record)
            self.assertTrue(root.resolve().is_relative_to(temp_base.resolve()))
            return record

    def test_failure_evidence_survives_immutable_receipt_and_health_is_unchanged(self):
        evidence=H.new_reconciliation_failure_evidence();H.record_reconciliation_failure(evidence,'HISTORICAL_EXIT_UNBOUND',1)
        record=self.publish(evidence)
        self.assertEqual(record['metrics']['reconciliation_failure']['findings'][0]['trade_id'],1)
        self.assertEqual(record['process_rc'],1);self.assertEqual(record['health'],'failed')
        self.assertFalse(record['operating_acceptance']);self.assertFalse(record['research_qualified'])
        baseline=self.publish(None,include=False)
        for key in ('health','process_rc','pending_terminal_confirmations','unobserved_submissions','operating_acceptance'):
            self.assertEqual(record[key],baseline[key])

    def test_old_no_evidence_and_other_roles_keep_original_schema(self):
        self.assertNotIn('reconciliation_failure',self.publish(None,include=False)['metrics'])
        self.assertNotIn('reconciliation_failure',self.publish(H.new_reconciliation_failure_evidence(),role='dry_run')['metrics'])

    def test_failed_zero_guard_run_is_unclassified_and_never_claims_complete(self):
        empty=H.new_reconciliation_failure_evidence()
        failure=self.publish(empty)
        out=failure['metrics']['reconciliation_failure']
        self.assertEqual(out['status'],'UNCLASSIFIED')
        self.assertEqual(out['total_count'],0)
        self.assertFalse(out['evidence_complete'])
        self.assertEqual(failure['health'],'failed')
        success=self.publish(empty,rc=0)
        baseline=self.publish(None,rc=0,include=False)
        self.assertEqual(success['health'],baseline['health'])
        self.assertEqual(success['metrics']['reconciliation_failure']['status'],'VALID')

    def test_malformed_evidence_is_visible_as_invalid_without_raw_payload(self):
        record=self.publish({'token':'SYNTHETIC_SECRET','raw_sql':'SYNTHETIC_SECRET'})
        self.assertEqual(record['metrics']['reconciliation_failure'],{'schema_version':1,'status':'INVALID','evidence_complete':False})
        self.assertNotIn('SYNTHETIC_SECRET',json.dumps(record))


if __name__ == '__main__':
    unittest.main()
