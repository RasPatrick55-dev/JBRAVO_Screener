"""Actual-function lock-order regressions with inert SQL/connection/frame doubles.

No repository application imports, database, provider, subprocess or socket use.
These checks complement, and do not replace, separate-session native PG tests.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime(2026, 10, 18, tzinfo=timezone.utc)
LOCK = "SELECT pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'))"
INPUTS = ('trades', 'trade_entry_ml_context_app', 'executed_trades',
          'screener_ranker_scores_app', 'screener_run_map_app', 'ml_artifacts', 'alpaca_activities')


class Refused(RuntimeError): pass


class Cursor:
    rowcount = 1
    def __init__(self, conn): self.conn = conn; self.one = None
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, sql, params=None):
        q = ' '.join(sql.split())
        self.conn.calls.append((q, params))
        self.one = None
        if q == "SELECT pg_catalog.current_setting('transaction_isolation')":
            self.one = (self.conn.isolation,)
        elif q == LOCK:
            if self.conn.fail_lock: raise Refused('synthetic_lock_timeout')
            self.conn.locked = True
        elif 'FROM information_schema.columns' in q:
            self.one = (1,)
        if q.startswith(('INSERT ', 'UPDATE ', 'DELETE ', 'CREATE ', 'ALTER ', 'LOCK TABLE ')):
            normalized = q.replace('public.', '')
            if any(('INTO ' + name in normalized or 'UPDATE ' + name in normalized or
                    'TABLE IF NOT EXISTS ' + name in normalized or 'TABLE ' + name in normalized)
                   for name in INPUTS):
                if not self.conn.locked:
                    raise AssertionError('INPUT_RELATION_ACCESSED_BEFORE_POLICY_LOCK: ' + q.split()[0])
                self.conn.input_statements.append(q)
    def fetchone(self): return self.one
    def fetchall(self): return []


class Conn:
    autocommit = False
    isolation = 'read committed'
    fail_lock = False
    def __init__(self):
        self.calls = []; self.input_statements = []; self.locked = False
        self.commits = self.rollbacks = self.closes = 0
    def cursor(self, **kwargs): return Cursor(self)
    def __enter__(self): return self
    def __exit__(self, typ, value, traceback):
        if typ: self.rollback()
        else: self.commit()
        return False
    def commit(self): self.commits += 1; self.locked = False
    def rollback(self): self.rollbacks += 1; self.locked = False
    def close(self): self.closes += 1


class Frame:
    empty = False
    columns = ('symbol', 'model_score_5d')
    def copy(self): return self
    def iterrows(self): return iter([(0, {'symbol':'SYNTHETIC', 'model_score_5d':0.7})])


def selected(relative, names, namespace):
    parsed = ast.parse((ROOT / relative).read_text(encoding='utf8'))
    body = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names]
    if {node.name for node in body} != set(names):
        raise AssertionError('ACTUAL_SOURCE_FUNCTION_INVENTORY')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[])),
                 relative, 'exec', dont_inherit=True), namespace)
    return namespace


class PolicyLockOrderTests(unittest.TestCase):
    def setUp(self):
        self.conn = Conn()
        def batch(cursor, sql, rows, **kwargs):
            for row in rows: cursor.execute(sql, row)
        self.extras = SimpleNamespace(execute_batch=batch, Json=lambda value:value)
        self.ns = dict(disposition=SimpleNamespace(TradePolicyError=Refused),
            datetime=datetime, timezone=timezone, Decimal=Decimal, json=json,
            logger=Mock(), extras=self.extras, _conn_or_none=lambda:self.conn,
            normalize_ts=lambda value, **kwargs:value, _log_write_result=Mock(),
            db_enabled=lambda:True, pd=SimpleNamespace(isna=lambda value:value is None),
            _json_dumps_or_none=lambda value:json.dumps(value,default=str) if value is not None else None,
            _coerce_date=lambda value:value.isoformat() if hasattr(value,'isoformat') else str(value),
            ensure_trade_entry_ml_context_app_table=lambda:None,
            ensure_screener_ranker_scores_app_table=lambda:None,
            ensure_ml_artifacts_table=lambda:None,
            ensure_latest_screener_candidates_view=lambda:None,
            _repair_screener_run_map_app_schema=lambda cursor:None,
            _env_truthy=lambda name:False, _DUMPED_CANDIDATES=False,
            normalize_score_breakdown=lambda value, **kwargs:value,
            normalize_gate_fail_reason=lambda value:value, _coerce_bool=lambda value:value)
        selected('scripts/db.py', {'_policy_write_lock'}, self.ns)

    def load_db(self, *names):
        return selected('scripts/db.py', set(names), self.ns)

    def assert_actual_input_prelock(self, table):
        writes = [q for q in self.conn.input_statements if table in q]
        self.assertTrue(writes, 'actual writer did not reach its intended input table')
        sql = [q for q, _ in self.conn.calls]
        self.assertIn(LOCK, sql)
        self.assertLess(sql.index(LOCK), sql.index(writes[0]))
        self.assertFalse(any('search_path' in q for q in sql))

    def test_helper_checks_isolation_sets_bounds_then_locks_without_commit(self):
        self.ns['_policy_write_lock'](self.conn)
        self.assertEqual([q for q, _ in self.conn.calls], [
            "SELECT pg_catalog.current_setting('transaction_isolation')",
            "SET LOCAL lock_timeout = '1500ms'", "SET LOCAL statement_timeout = '5000ms'", LOCK])
        self.assertTrue(self.conn.locked)
        self.assertEqual((self.conn.commits,self.conn.rollbacks,self.conn.closes),(0,0,0))

    def test_helper_refuses_missing_and_autocommit_connections_without_sql(self):
        for value in (None, SimpleNamespace(autocommit=True)):
            with self.subTest(connection=value), self.assertRaises(Refused):
                self.ns['_policy_write_lock'](value)
        self.assertEqual(self.conn.calls, [])

    def test_helper_refuses_wrong_isolation_before_lock_or_input_access(self):
        self.conn.isolation = 'repeatable read'
        with self.assertRaisesRegex(Refused, 'ISOLATION'):
            self.ns['_policy_write_lock'](self.conn)
        self.assertEqual(len(self.conn.calls),1)
        self.assertEqual(self.conn.input_statements,[])
        self.assertFalse(self.conn.locked)

    def test_actual_buy_upsert_prelocks_same_connection_before_insert(self):
        self.load_db('upsert_trade_on_buy_fill')
        self.assertTrue(self.ns['upsert_trade_on_buy_fill']('SYNTHETIC', '2', 'new-order', STAMP, '10'))
        self.assert_actual_input_prelock('trades')
        self.assertEqual((self.conn.commits,self.conn.rollbacks,self.conn.closes),(1,0,1))
        self.assertIn('ON CONFLICT (entry_order_id) DO UPDATE', self.conn.input_statements[0])

    def test_actual_buy_lock_failure_rolls_back_before_any_input_write(self):
        self.load_db('upsert_trade_on_buy_fill')
        self.conn.fail_lock = True
        self.assertFalse(self.ns['upsert_trade_on_buy_fill']('SYNTHETIC', '2', 'new-order', STAMP, '10'))
        self.assertEqual(self.conn.input_statements, [])
        self.assertEqual((self.conn.commits,self.conn.rollbacks,self.conn.closes),(0,1,1))

    def activity(self):
        env = dict(db=SimpleNamespace(_policy_write_lock=self.ns['_policy_write_lock']),
            extras=self.extras, Decimal=Decimal, datetime=datetime, timezone=timezone,
            date_parser=SimpleNamespace(isoparse=lambda text:datetime.fromisoformat(text)),json=json)
        return selected('scripts/fetch_account_activities.py',
            {'_parse_decimal','_parse_timestamp','_normalize_activity','insert_activities'},env)

    @staticmethod
    def activity_payload():
        return dict(id='native-fixture',activity_type='FILL',transaction_time=STAMP.isoformat(),
            symbol='SYNTHETIC',side='buy',qty='2',price='10',order_id='new-order')

    def test_actual_activity_producer_prelocks_before_real_batch_call(self):
        api = self.activity()
        result = api['insert_activities'](self.conn,[self.activity_payload()])
        self.assertEqual(result,(1,STAMP.isoformat(),STAMP.isoformat()))
        self.assert_actual_input_prelock('alpaca_activities')
        self.assertEqual(self.conn.commits,1)
        self.assertEqual(self.conn.closes,0)

    def test_actual_activity_lock_failure_has_no_batch_write_and_rolls_back(self):
        api = self.activity(); self.conn.fail_lock=True
        with self.assertRaisesRegex(RuntimeError,'db_insert_failed'):
            api['insert_activities'](self.conn,[self.activity_payload()])
        self.assertEqual(self.conn.input_statements,[])
        self.assertEqual(self.conn.rollbacks,1)
        self.assertEqual(self.conn.commits,0)

    def test_actual_ml_input_writers_each_prelock_their_own_connection(self):
        cases = (
            ('upsert_trade_entry_ml_context','trade_entry_ml_context_app',
             ({'order_id':'new-order','symbol':'SYNTHETIC','entry_time':STAMP},),{}),
            ('insert_executed_trade','executed_trades',
             ({'order_id':'new-order','symbol':'SYNTHETIC','qty':'2','entry_time':STAMP},),{}),
            ('upsert_screener_ranker_scores_frame','screener_ranker_scores_app',(STAMP,Frame()),{}),
            ('upsert_ml_artifact','ml_artifacts',('synthetic-kind','2026-10-18'),{'payload':{'fixture':True}}),
        )
        for name, table, args, kwargs in cases:
            with self.subTest(actual_writer=name):
                self.setUp(); self.load_db(name)
                result = self.ns[name](*args,**kwargs)
                self.assertTrue(result)
                self.assert_actual_input_prelock(table)
                self.assertEqual(self.conn.commits,1)
                self.assertEqual(self.conn.rollbacks,0)

    def test_actual_run_map_writer_reacquires_after_candidate_transaction(self):
        self.load_db('insert_screener_candidates')
        self.ns['insert_screener_candidates']('2026-10-18',Frame(),run_ts_utc=STAMP)
        self.assert_actual_input_prelock('screener_run_map_app')
        self.assertEqual(self.conn.commits,2)
        sql = [q for q,_ in self.conn.calls]
        candidate = next(i for i,q in enumerate(sql) if q.startswith('INSERT INTO screener_candidates'))
        self.assertGreater(sql.index(LOCK),candidate)
        self.assertFalse(self.conn.locked)

    def test_actual_schema_ensure_paths_prelock_before_affected_relation_ddl(self):
        for name, table in (('ensure_ml_artifacts_table','ml_artifacts'),
                ('ensure_screener_ranker_scores_app_table','screener_ranker_scores_app'),
                ('ensure_trade_entry_ml_context_app_table','trade_entry_ml_context_app')):
            with self.subTest(actual_writer=name):
                self.setUp(); self.load_db(name); self.ns[name]()
                self.assert_actual_input_prelock(table)
                self.assertEqual(self.conn.commits,1)
                self.assertEqual(self.conn.rollbacks,0)

    def test_caller_can_roll_back_lock_refusal_and_reuse_without_hidden_commit(self):
        self.conn.fail_lock = True
        with self.assertRaises(Refused):
            with self.conn: self.ns['_policy_write_lock'](self.conn)
        self.assertEqual(self.conn.rollbacks,1)
        self.conn.fail_lock = False
        with self.conn: self.ns['_policy_write_lock'](self.conn)
        self.assertEqual(self.conn.commits,1)
        self.assertFalse(self.conn.locked)


if __name__ == '__main__': unittest.main()
