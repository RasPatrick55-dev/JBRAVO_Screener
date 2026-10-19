"""Offline selected actual functions with inert SQL and transaction fixtures."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import unittest
# Inert dependency fixture only. Actual production uses dateutil.parser;
# the separate real native producer path validates that installed dependency.
def fixture_isoparse(value):
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError('fixture_iso_invalid')
    return datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value)

parser = SimpleNamespace(isoparse=fixture_isoparse)

DB = Path(__file__).resolve().parents[1] / 'scripts/db.py'
PARENT = '11111111-1111-4111-8111-111111111111'
CHILD = '22222222-2222-4222-8222-222222222222'
OTHER = '33333333-3333-4333-8333-333333333333'
START = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)


@contextmanager
def maybe(connection):
    yield connection


@contextmanager
def read_transaction(connection, **kwargs):
    connection.actions.append('read_policy')
    yield SimpleNamespace()


def load_actual():
    raw = DB.read_bytes()
    tree = ast.parse(raw)
    names = {'_lineage_uuid', '_lineage_number', '_lineage_time', '_lineage_rows',
        '_open_exit_lineage_guard', '_open_exit_lineage_record', 'get_open_trade_exit_lineage',
        '_require_open_exit_lineage', 'reconcile_sell_fill'}
    constants = {'_OPEN_EXIT_LINEAGE_CAP', '_OPEN_EXIT_EVENT_COLUMNS'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    if {n.name for n in nodes if isinstance(n, ast.FunctionDef)} != names:
        raise AssertionError('selected_actual_functions_missing')
    ns = dict(Decimal=Decimal, datetime=datetime, timezone=timezone, parser=parser,
        Mapping=Mapping, Any=Any, Optional=Optional, PGConnection=object, math=math,
        _maybe_conn=maybe, _reconciliation_read_transaction=read_transaction,
        normalize_ts=lambda value, **kwargs: parser.isoparse(value) if isinstance(value, str) else value,
        _json_dumps_or_none=lambda value: json.dumps(value, default=str),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        _log_write_result=lambda *a, **k: None,
        disposition=SimpleNamespace(require_writable=lambda policy, trade_id: None))
    def policy(connection, **kwargs):
        connection.actions.append('write_policy')
        return SimpleNamespace()
    ns['trade_policy'] = policy
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), str(DB), 'exec'), ns)
    return SimpleNamespace(**ns)


def trade(**changes):
    value = dict(trade_id=1, symbol='FIXTURE', qty=Decimal('2'), status='OPEN',
        entry_order_id=PARENT, entry_time=START, entry_price=Decimal('10'),
        exit_order_id=None, exit_time=None, exit_price=None, exit_reason=None, realized_pnl=None)
    value.update(changes)
    return value


def event(**changes):
    value = dict(event_id=1, event_type='TRAIL_SUBMIT', symbol='FIXTURE', qty='2',
        order_id=CHILD, parent_order_id=PARENT, envelope_order_id=CHILD,
        envelope_event_type='TRAIL_SUBMIT', metadata_event_type='TRAIL_SUBMIT',
        raw_shape_valid=True, top_parent_key_present=False, top_parent_order_id=None,
        recorded_submitted_at=(START + timedelta(seconds=1)).isoformat(),
        event_time=START + timedelta(seconds=1))
    value.update(changes)
    return value


def provider(**changes):
    value = dict(id=CHILD, symbol='FIXTURE', side='sell', status='filled',
        type='trailing_stop', qty='2', filled_qty='2', filled_avg_price='11',
        created_at=(START + timedelta(seconds=1)).isoformat(),
        filled_at=(START + timedelta(seconds=2)).isoformat())
    value.update(changes)
    return value


class Connection:
    def __init__(self):
        self.autocommit = False
        self.trades = [trade()]
        self.events = [event()]
        self.fills = []
        self.actions = []
        self.queries = []
        self.table_busy = False
        self.order_busy = False
    def get_transaction_status(self):
        return 0
    def cursor(self):
        return Cursor(self)
    def __enter__(self):
        self.saved = deepcopy((self.trades, self.events, self.fills))
        return self
    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.trades, self.events, self.fills = self.saved
            self.actions.append('rollback')
        else:
            self.actions.append('commit')
        return False


class Cursor:
    def __init__(self, connection):
        self.c = connection
        self.rows = []
        self.description = []
        self.rowcount = 0
    def __enter__(self):
        return self
    def __exit__(self, *unused):
        return False
    def setrows(self, rows, names=None):
        names = names or (list(rows[0]) if rows else [])
        self.description = [SimpleNamespace(name=name) for name in names]
        self.rows = [tuple(row.get(name) for name in names) for row in rows]
    def execute(self, sql, args=None):
        c = self.c
        c.queries.append((sql, deepcopy(args)))
        if sql.startswith('SET '):
            self.setrows([])
        elif sql.startswith('LOCK TABLE'):
            c.actions.append(sql)
            if c.table_busy:
                raise RuntimeError('fixture_table_lock_busy')
            self.setrows([])
        elif 'pg_try_advisory_xact_lock' in sql:
            c.actions.append('order_try_lock')
            self.setrows([{'locked': not c.order_busy}])
        elif 'FROM pg_catalog.pg_class' in sql:
            c.actions.append('physical_guard')
            self.setrows([dict(name=name, kind='r', rls=False, forced=False, inheritance=False)
                for name in ('order_events', 'trades')])
        elif 'SELECT status, qty, entry_price, symbol, exit_order_id' in sql:
            c.actions.append('trade_for_update')
            values = [t for t in c.trades if t['trade_id'] == args['trade_id']]
            self.setrows(values, ['status', 'qty', 'entry_price', 'symbol', 'exit_order_id',
                'exit_time', 'exit_price', 'exit_reason', 'realized_pnl', 'entry_order_id', 'entry_time'])
        elif 'FROM ONLY public.trades' in sql and 'binding_ids' in sql:
            c.actions.append('all_physical_binding_owners')
            values = [t for t in c.trades if t.get('entry_order_id') in args['binding_ids']
                or t.get('exit_order_id') in args['binding_ids']]
            self.setrows([dict(trade_id=t['trade_id'], entry_order_id=t.get('entry_order_id'),
                exit_order_id=t.get('exit_order_id'), identity_oversize=False) for t in values])
        elif 'FROM ONLY public.trades' in sql and 'trade_ids' in sql:
            self.setrows([t for t in c.trades if t['trade_id'] in args['trade_ids']
                or t.get('entry_order_id') in args['parents']])
        elif 'FROM ONLY public.order_events' in sql:
            c.actions.append('lineage_event_read')
            if 'parents' in args:
                values = [e for e in c.events if e.get('parent_order_id') in args['parents']]
            elif 'children' in args:
                values = [e for e in c.events if e.get('order_id') in args['children']]
            else:
                values = [e for e in c.events if e.get('order_id') == args['child']
                    or e.get('parent_order_id') == args['parent']]
            self.setrows(values)
        elif 'SELECT trade_id FROM public.trades' in sql:
            c.actions.append('all_physical_exit_and_entry_owners')
            self.setrows([t for t in c.trades if t.get('exit_order_id') == args['order_id']
                or t.get('entry_order_id') == args['order_id']], ['trade_id'])
        elif 'SELECT symbol, qty, status, event_time FROM public.order_events' in sql:
            self.setrows([f for f in c.fills if f['order_id'] == args['order_id']],
                ['symbol', 'qty', 'status', 'event_time'])
        elif 'INSERT INTO public.order_events' in sql:
            c.actions.append('insert_fill')
            c.fills.append(deepcopy(args))
            self.rowcount = 1
        elif 'UPDATE public.trades SET exit_order_id' in sql:
            c.actions.append('financial_update')
            row = next(t for t in c.trades if t['trade_id'] == args['trade_id'])
            row.update(exit_order_id=args['order_id'], status='CLOSED', exit_time=args['event_time'],
                exit_price=args['exit_price'], exit_reason=args['exit_reason'], realized_pnl=args['realized_pnl'])
            self.rowcount = 1
        else:
            raise AssertionError('unexpected_fixture_sql')
    def fetchone(self):
        return self.rows[0] if self.rows else None
    def fetchall(self):
        return self.rows
    def fetchmany(self, count):
        return self.rows[:count]


class OpenExitLineageTests(unittest.TestCase):
    def setUp(self):
        self.db = load_actual()
        self.connection = Connection()
    def write(self, **changes):
        args = dict(engine=self.connection, trade_id=1, symbol='FIXTURE', qty='2',
            order_id=CHILD, status='filled', event_time=START + timedelta(seconds=2),
            raw=provider(), exit_price='11', exit_reason='TRAIL_STOP')
        args.update(changes)
        return self.db.reconcile_sell_fill(**args)
    def assert_no_financial_effect(self):
        self.assertEqual(self.connection.fills, [])
        self.assertNotIn('financial_update', self.connection.actions)
    def test_actual_reader_returns_separate_mapping_and_writer_binds_atomically(self):
        original = deepcopy(self.connection.trades)
        mapping = self.db.get_open_trade_exit_lineage(self.connection, deepcopy(original),
            observed_before=START + timedelta(seconds=3))
        self.assertEqual(mapping, {1: CHILD})
        self.assertEqual(self.connection.trades, original)
        self.assertTrue(self.write())
        self.assertEqual(self.connection.trades[0]['exit_order_id'], CHILD)
        self.assertEqual(self.connection.trades[0]['status'], 'CLOSED')
        self.assertEqual(len(self.connection.fills), 1)
    def test_selected_child_contrary_parent_symbol_or_quantity_is_not_hidden(self):
        for changes in [dict(parent_order_id=OTHER), dict(symbol='OTHER'), dict(qty='1')]:
            with self.subTest(changes=changes):
                self.connection = Connection()
                self.connection.events.append(event(event_id=2, **changes))
                self.assertFalse(self.write())
                self.assert_no_financial_effect()
    def test_duplicate_identical_producer_events_are_ambiguous(self):
        self.connection.events.append(event(event_id=2))
        self.assertFalse(self.write())
        self.assert_no_financial_effect()
    def test_all_physical_closed_excluded_entry_or_exit_ownership_refuses(self):
        for changes in [dict(entry_order_id=CHILD), dict(exit_order_id=CHILD),
                dict(exit_order_id=PARENT), dict(entry_order_id=PARENT)]:
            with self.subTest(changes=changes):
                self.connection = Connection()
                other = dict(trade_id=2, status='CLOSED', excluded=True, entry_order_id=OTHER)
                other.update(changes)
                self.connection.trades.append(trade(**other))
                self.assertFalse(self.write())
                self.assert_no_financial_effect()
    def test_partial_or_provider_identity_conflict_refuses(self):
        for raw in [provider(filled_qty='1'), provider(qty='3'), provider(id=OTHER),
                provider(side='buy'), provider(filled_avg_price='12')]:
            with self.subTest(raw=raw):
                self.connection = Connection()
                self.assertFalse(self.write(raw=raw))
                self.assert_no_financial_effect()
    def test_naive_or_conflicting_chronology_refuses(self):
        for raw in [provider(created_at=(START - timedelta(seconds=1)).isoformat()),
                provider(filled_at=START.isoformat()), provider(filled_at='2026-01-01T10:00:02')]:
            with self.subTest(raw=raw):
                self.connection = Connection()
                self.assertFalse(self.write(raw=raw))
                self.assert_no_financial_effect()
    def test_missing_null_or_contrary_top_parent_is_distinct_from_matching(self):
        for changes in [dict(parent_order_id=None), dict(raw_shape_valid=False),
                dict(top_parent_key_present=True, top_parent_order_id=OTHER)]:
            with self.subTest(changes=changes):
                self.connection = Connection()
                self.connection.events = [event(**changes)]
                self.assertFalse(self.write())
                self.assert_no_financial_effect()
    def test_closed_unbound_is_never_first_bound(self):
        self.connection.trades[0]['status'] = 'CLOSED'
        self.assertFalse(self.write(decorate=True))
        self.assertIsNone(self.connection.trades[0]['exit_order_id'])
        self.assert_no_financial_effect()
    def test_busy_locks_refuse_without_retry_or_writes(self):
        for field in ('table_busy', 'order_busy'):
            with self.subTest(field=field):
                self.connection = Connection()
                setattr(self.connection, field, True)
                self.assertFalse(self.write())
                self.assert_no_financial_effect()
                self.assertEqual(self.connection.actions.count('order_try_lock'), 0 if field == 'table_busy' else 1)
    def test_strong_fixed_nowait_locks_precede_owner_proof_and_financial_write(self):
        self.assertTrue(self.write())
        actions = self.connection.actions
        trades_lock = 'LOCK TABLE ONLY public.trades IN SHARE ROW EXCLUSIVE MODE NOWAIT'
        events_lock = 'LOCK TABLE ONLY public.order_events IN SHARE ROW EXCLUSIVE MODE NOWAIT'
        ordered = ['write_policy', trades_lock, events_lock, 'physical_guard', 'order_try_lock',
            'trade_for_update', 'all_physical_binding_owners', 'lineage_event_read', 'insert_fill', 'financial_update']
        positions = [actions.index(action) for action in ordered]
        self.assertEqual(positions, sorted(positions))
    def test_mapping_is_not_authority_writer_rechecks_changed_proof(self):
        mapping = self.db.get_open_trade_exit_lineage(self.connection, deepcopy(self.connection.trades),
            observed_before=START + timedelta(seconds=3))
        self.assertEqual(mapping, {1: CHILD})
        self.connection.events[0]['parent_order_id'] = OTHER
        self.assertFalse(self.write())
        self.assert_no_financial_effect()
    def test_reader_checks_all_selected_child_facts_and_saturation(self):
        self.connection.events.append(event(event_id=2, parent_order_id=OTHER, symbol='OTHER'))
        self.assertEqual(self.db.get_open_trade_exit_lineage(self.connection, deepcopy(self.connection.trades),
            observed_before=START + timedelta(seconds=3)), {})
        self.connection = Connection()
        self.connection.events = [event(event_id=i + 1) for i in range(513)]
        with self.assertRaises(ValueError):
            self.db.get_open_trade_exit_lineage(self.connection, deepcopy(self.connection.trades),
                observed_before=START + timedelta(seconds=3))
        self.assert_no_financial_effect()

    def test_null_optional_parent_is_unverified_not_a_contrary_uuid(self):
        with self.assertRaisesRegex(ValueError, 'optional_parent_unverified'):
            self.db._open_exit_lineage_record(event(top_parent_key_present=True, top_parent_order_id=None),
                trade(), observed_before=START + timedelta(seconds=3))
        with self.assertRaisesRegex(ValueError, 'contrary_parent'):
            self.db._open_exit_lineage_record(event(parent_order_id=OTHER),
                trade(), observed_before=START + timedelta(seconds=3))
