"""Synthetic exchange sessions and actual inert producer/consumer paths.

Legacy pipeline hook extraction is explicit: it does not qualify its full
initialization, live provider/DB, model preparation or host scheduling.
"""
import ast
import copy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import io
import contextlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
import pytest
from scripts import signal_handoff as signals
from scripts import executor_health as executor
from scripts import pipeline_postflight as postflight
from scripts import primary_pipeline as primary
from scripts.utils.calendar import calc_daily_window, completed_signal_session
from alpaca.trading.models import Calendar

pytestmark = pytest.mark.alpaca_optional
SOURCE = Path(__file__).resolve().parents[1]


def calendar(*days):
    return [{'date': day, 'open': '09:30', 'close': '16:00'} for day in days]


class SessionPolicyTests(unittest.TestCase):
    def test_real_calendar_sdk_records_and_explicit_data_window(self):
        entries = [Calendar(date=day, open='09:30', close='16:00')
                   for day in ('2026-10-08', '2026-10-09')]
        client = SimpleNamespace(get_calendar=mock.Mock(return_value=entries))
        result = completed_signal_session(client, now=signals.utc('2026-10-09T00:30:00Z'))
        self.assertEqual(result['signal_session'], '2026-10-08')
        client.get_calendar.return_value = entries[:1]
        start, end, day = calc_daily_window(client, 5, end_date=date(2026, 10, 8), session_cutoff=True)
        self.assertEqual((start, end, day), ('2026-10-08T04:00:00+00:00',
                                           '2026-10-09T00:00:00+00:00', '2026-10-08'))
        self.assertEqual(calc_daily_window(client, 5, end_date=date(2026, 10, 8))[:2],
                         ('2026-10-08T00:00:00Z', '2026-10-08T23:59:59Z'))

    def test_completed_cutoff_does_not_fall_back_to_yesterday(self):
        cal = calendar('2026-10-08', '2026-10-09', '2026-10-12')
        for instant in ('2026-10-09T19:59:59-04:00', '2026-10-09T20:14:59-04:00'):
            with self.subTest(instant=instant), self.assertRaisesRegex(ValueError, 'not_completed'):
                signals.session_contract(cal, now=signals.utc(instant))
        result = signals.session_contract(cal, now=signals.utc('2026-10-09T20:15:00-04:00'))
        self.assertEqual(result['signal_session'], '2026-10-09')
        self.assertEqual(result['execution_session'], '2026-10-12')
        self.assertEqual(result['entry_open_utc'], '2026-10-12T08:00:00+00:00')
        after_midnight = signals.session_contract(cal, now=signals.utc('2026-10-09T01:00:00-04:00'))
        self.assertEqual(after_midnight['signal_session'], '2026-10-08')
        self.assertEqual(after_midnight['execution_session'], '2026-10-09')

    def test_holiday_and_dst_use_returned_sessions_not_weekday_or_fixed_utc(self):
        for days, now, expected in (
            (('2026-01-16', '2026-01-20'), '2026-01-17T02:00:00Z', '2026-01-20T09:00:00+00:00'),
            (('2026-03-06', '2026-03-09'), '2026-03-07T02:00:00Z', '2026-03-09T08:00:00+00:00'),
            (('2026-10-30', '2026-11-02'), '2026-10-31T01:00:00Z', '2026-11-02T09:00:00+00:00')):
            with self.subTest(days=days):
                result = signals.session_contract(calendar(*days), now=signals.utc(now))
                self.assertEqual(result['entry_open_utc'], expected)

    def test_early_close_unknown_next_and_duplicate_calendar_block(self):
        cal = calendar('2026-11-27', '2026-11-30')
        cal[0]['close'] = '13:00'
        for population in (cal, calendar('2026-10-08'), calendar('2026-10-08', '2026-10-08', '2026-10-09')):
            with self.subTest(population=population), self.assertRaises(ValueError):
                signals.session_contract(population, now=signals.utc('2026-11-28T02:00:00Z'))

    def test_explicit_opening_expiry_and_preparation_must_finish_before_open(self):
        started = signals.utc('2026-10-09T00:30:00Z')
        session = signals.session_contract(calendar('2026-10-08', '2026-10-09'), now=started)
        finished = started + timedelta(minutes=20)
        for now, entry, expected in (
            ('2026-10-09T07:59:59Z', False, True), ('2026-10-09T07:59:59Z', True, False),
            ('2026-10-09T08:00:00Z', True, True), ('2026-10-09T13:29:59Z', True, True),
            ('2026-10-09T13:30:00Z', True, False)):
            with self.subTest(now=now, entry=entry):
                if expected:
                    signals.validate_session(session, started=started, finished=finished,
                                             now=signals.utc(now), for_entry=entry)
                else:
                    with self.assertRaises(ValueError):
                        signals.validate_session(session, started=started, finished=finished,
                                                 now=signals.utc(now), for_entry=entry)
        with self.assertRaises(ValueError):
            signals.validate_session(session, started=started, finished=signals.utc(session['entry_open_utc']),
                                     now=signals.utc(session['entry_open_utc']))
        for key in ('entry_open_utc', 'completed_cutoff_utc', 'execution_session', 'arrival_allowance_minutes'):
            changed = copy.deepcopy(session); changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                signals.validate_session(changed, started=started, finished=finished, now=finished)

    def test_price_cap_frozen_reference_precision_and_invalid_settings(self):
        snapshot = {'references': [{'symbol': 'AAA', 'price': '10.0099'}]}
        self.assertEqual(signals.price_limit(snapshot, 'AAA', buffer_pct=.5, max_gap_pct=3), 10.05)
        self.assertEqual(signals.price_limit(snapshot, 'AAA', buffer_pct=99, max_gap_pct=3), 10.31)
        self.assertEqual(signals.price_cap(snapshot, 'AAA', 3), 10.31)
        for value in (float('inf'), float('nan'), -1, 'bad', None, True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                signals.price_limit(snapshot, 'AAA', buffer_pct=value, max_gap_pct=3)
        with self.assertRaises(ValueError):
            signals.price_cap(snapshot, 'UNKNOWN', 3)


class BoundHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for relative in set(postflight.SOURCE_FILES + executor.SOURCES):
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, path)
        self.started = signals.utc('2026-10-09T00:30:00Z')
        self.finished = self.started + timedelta(minutes=30)
        self.batch = self.started + timedelta(minutes=5)
        self.now = signals.utc('2026-10-09T08:00:00Z')
        self.session = signals.session_contract(calendar('2026-10-08', '2026-10-09'), now=self.started)
        self.rows = [{'symbol': s, 'timestamp': '2026-10-08T04:00:00Z', 'run_date': '2026-10-08',
                      'run_ts_utc': self.batch.isoformat(), 'close': Decimal('10.125'), 'score': 7,
                      'entry_price': 20.0, 'source': 'synthetic_daily'} for s in ('AAA', 'BBB')]
        self.invocation = postflight.begin_primary(self.root, self.started)
        self.facts = dict(started_at=self.started, finished_at=self.finished, pipeline_rc=0,
            steps=postflight.REQUIRED_STEPS, step_rcs={s: 0 for s in postflight.REQUIRED_STEPS},
            stage_times={s: 1.0 for s in postflight.REQUIRED_STEPS}, degraded=False, labels_rows=5,
            freshness={'stale': False, 'pred_compatible': True, 'predict_rc': None, 'snapshot_date': '2026-10-08'},
            coverage={'total': 2, 'non_null': 2, 'pct': 100.0, 'run_ts_utc': self.batch.isoformat()},
            ml_health={'decision': 'allow', 'reasons': []},
            controls={k: True for k in ('strict_predictions_meta', 'strict_auto_refresh_predictions',
                'auto_refresh_features', 'auto_refresh_predictions', 'use_champion', 'ml_health_guard',
                'refresh_predictions_for_candidates')}, session=self.session,
            frozen_candidates=signals.freeze_candidates(self.rows, self.session, run_ts=self.batch),
            invocation_id=self.invocation['id'])
        for target in ('socket.create_connection', 'socket.socket.connect', 'subprocess.run', 'subprocess.Popen'):
            patch = mock.patch(target, side_effect=AssertionError('external_effect_forbidden'))
            patch.start(); self.addCleanup(patch.stop)
        self.publish()

    def publish(self, **changes):
        return postflight.publish_health(self.root, **(self.facts | changes))

    def complete(self):
        result = postflight.check_health(self.root, now=self.now, require_session=True)
        postflight.complete_primary(self.root, self.invocation, result)

    def gate(self, **changes):
        return executor.preflight(self.root, **(dict(now=self.now, rows=self.rows,
                                                    require_session=True, for_entry=True) | changes))

    def test_postflight_completion_required_and_failed_next_invocation_cannot_borrow_it(self):
        self.assertEqual(self.gate()['status'], 'blocked')
        self.complete(); self.assertEqual(self.gate()['status'], 'pass')
        postflight.begin_primary(self.root, self.finished)
        self.assertEqual(self.gate()['status'], 'blocked')

    def test_candidate_membership_prices_rank_values_and_optional_presence_are_frozen(self):
        self.complete()
        for change in ({'close': Decimal('10.126')}, {'entry_price': 21}, {'score': 8},
                       {'symbol': 'CCC'}, {'new_optional': None}, {'run_date': '2026-10-07'},
                       {'timestamp': '2026-10-07T04:00:00Z'}, {'timestamp': '2026-10-09T01:00:00Z'}):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.rows); changed[0].update(change)
                self.assertEqual(self.gate(rows=changed)['status'], 'blocked')
        self.assertEqual(self.gate(rows=list(reversed(self.rows)))['status'], 'pass')
        self.assertEqual(self.gate(rows=self.rows[:1])['status'], 'blocked')
        self.assertEqual(self.gate(rows=[self.rows[0], self.rows[0]])['status'], 'blocked')
        self.assertEqual(self.rows[0]['close'], Decimal('10.125'))

    def test_cross_utc_midnight_and_weekend_age_use_session_not_utc_day(self):
        self.complete()
        self.assertEqual(self.gate()['status'], 'pass')
        friday = signals.utc('2026-10-10T00:30:00Z')
        session = signals.session_contract(calendar('2026-10-09', '2026-10-12'), now=friday)
        rows = [r | {'run_date': '2026-10-09', 'timestamp': '2026-10-09T04:00:00Z',
                     'run_ts_utc': (friday + timedelta(minutes=5)).isoformat()} for r in self.rows]
        self.invocation = postflight.begin_primary(self.root, friday)
        self.now = signals.utc('2026-10-12T08:00:00Z')
        self.facts.update(started_at=friday, finished_at=friday + timedelta(minutes=30),
            session=session, frozen_candidates=signals.freeze_candidates(rows, session, run_ts=rows[0]['run_ts_utc']),
            invocation_id=self.invocation['id'], coverage=self.facts['coverage'] | {'run_ts_utc': rows[0]['run_ts_utc']})
        self.rows = rows; self.publish(); self.complete()
        self.assertEqual(self.gate()['status'], 'pass')
        self.assertEqual(self.gate(now=signals.utc('2026-10-12T13:30:00Z'))['status'], 'blocked')

    def test_completion_binding_mutation_and_source_mutation_block(self):
        self.complete()
        path = self.root / 'reports/pipeline_postflight/completed-primary.json'
        path.write_text('{"id":"wrong","report_binding":{}}')
        self.assertEqual(self.gate()['status'], 'blocked')
        self.complete()
        (self.root / 'scripts/signal_handoff.py').write_text('changed')
        self.assertEqual(self.gate()['status'], 'blocked')

    def test_primary_normal_import_runs_once_and_completes_bound_postflight(self):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return self.started if tz else self.started.replace(tzinfo=None)
        calls = []
        module = ModuleType('scripts.run_pipeline')
        def run(arguments):
            calls.append(arguments)
            invocation = arguments[arguments.index('--handoff-invocation') + 1]
            self.publish(finished_at=self.started, coverage=self.facts['coverage'] | {'run_ts_utc': self.started.isoformat()},
                         frozen_candidates=signals.freeze_candidates(
                             [r | {'run_ts_utc': self.started.isoformat()} for r in self.rows], self.session, run_ts=self.started),
                         invocation_id=invocation)
            return 0
        module.main = run
        actual_check = postflight.check_health
        with mock.patch.object(primary, 'datetime', Clock), mock.patch.dict(sys.modules, {'scripts.run_pipeline': module}), \
             mock.patch.object(postflight, 'check_health', side_effect=lambda *a, **k: actual_check(*a, now=self.started, **k)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primary.main([], base_dir=self.root), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn('--session-handoff', calls[0])
        self.assertEqual(actual_check(self.root, now=self.now, require_session=True, require_completion=True)['status'], 'ok')
        with mock.patch.object(primary, 'datetime', Clock), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primary.main([], pipeline_main=lambda args: 1, base_dir=self.root), 1)
        self.assertEqual(actual_check(self.root, now=self.now, require_session=True, require_completion=True)['status'], 'failed')

    def test_actual_pipeline_publication_hook_freezes_same_db_query_and_closes_snapshot(self):
        tree = ast.parse((SOURCE / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        hook = next(n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Name) and n.test.func.id == 'getattr'
            and any(isinstance(a, ast.Constant) and a.value == 'postflight_report' for a in n.test.args))
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[hook], type_ignores=[])), 'pipeline_hook', 'exec')
        conn = SimpleNamespace(set_session=mock.Mock(), rollback=mock.Mock(), close=mock.Mock())
        query = mock.Mock(return_value=(pd.DataFrame(self.rows), self.batch))
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None): return self.finished
        namespace = dict(args=SimpleNamespace(postflight_report=True, handoff_invocation=self.invocation['id']),
            session_context=self.session, pipeline_run_date=date(2026, 10, 8), db=SimpleNamespace(get_db_conn=lambda: conn),
            get_latest_screener_candidates=query, base_dir=self.root, started_dt=self.started,
            datetime=Clock, timezone=timezone, rc=0, steps=self.facts['steps'], step_rcs=self.facts['step_rcs'],
            stage_times=self.facts['stage_times'], degraded=False, labels_rows=5, enrichment_freshness=self.facts['freshness'],
            model_score_coverage_summary=self.facts['coverage'], ml_health_summary=self.facts['ml_health'], LOG=mock.Mock(),
            **{k: True for k in ('strict_predictions_meta', 'strict_auto_refresh_predictions', 'auto_refresh_features',
                'auto_refresh_predictions', 'use_champion', 'ml_health_guard_enabled', 'refresh_predictions_for_candidates')})
        exec(compiled, namespace)
        self.assertEqual(namespace['rc'], 0)
        query.assert_called_once_with(date(2026, 10, 8), connection=conn, exact_run_ts=self.batch.isoformat())
        conn.set_session.assert_called_once_with(isolation_level='REPEATABLE READ', readonly=True, autocommit=False)
        conn.rollback.assert_called_once(); conn.close.assert_called_once()
        record = json.loads((self.root / 'reports/pipeline_postflight/latest.json').read_bytes())
        self.assertEqual(record['frozen_candidates'], self.facts['frozen_candidates'])
        self.assertEqual(self.gate()['status'], 'blocked')  # Pipeline record alone is not completed postflight.


if __name__ == '__main__':
    unittest.main()
