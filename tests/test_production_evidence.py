"""Synthetic qualification regressions. No retained data, DB, model or provider."""
from contextlib import ExitStack
import copy
from datetime import date, datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import scripts
from scripts import production_evidence as evidence
from scripts import production_score_capture as capture

ROOT = Path(__file__).resolve().parents[1]


class ProductionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundaries = ExitStack()
        db = ModuleType('scripts.db')
        db.db_enabled = Mock(return_value=False)
        db.fetch_latest_ml_artifact = Mock(side_effect=AssertionError('database_forbidden'))
        db.upsert_ml_artifact = Mock(side_effect=AssertionError('database_forbidden'))
        env = ModuleType('utils.env')
        env.load_env = Mock()
        cls.boundaries.enter_context(patch.dict(sys.modules, {'scripts.db': db, 'utils.env': env}))
        cls.boundaries.enter_context(patch.object(scripts, 'db', db, create=True))
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen', 'subprocess.run'):
            cls.boundaries.enter_context(patch(target, side_effect=AssertionError('external_forbidden')))
        cls.monitor = importlib.import_module('scripts.ranker_monitor')
        cls.guard = importlib.import_module('scripts.utils.ml_health_guard')

    @classmethod
    def tearDownClass(cls):
        cls.boundaries.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 11, 20, 23, tzinfo=timezone.utc)
        dates = pd.bdate_range('2026-01-05', '2026-11-23')
        self.calendar = {'schema': 'jbravo.market-session-calendar.v1', 'venue': 'XNYS',
            'sessions': [{'session': d.date().isoformat(), 'open_utc': d.date().isoformat() + 'T14:30:00Z',
                          'close_utc': d.date().isoformat() + 'T21:00:00Z'} for d in dates]}
        # This synthetic calendar is NOT an authoritative exchange calendar.
        start = date(2026, 1, 5)
        bend = start + timedelta(days=251)
        rstart = bend + timedelta(days=1)
        rend = rstart + timedelta(days=62)
        self.symbols = ['S' + str(i) for i in range(60)]
        self.model = capture.scoring_binding({'fixture': 1}, None, ['x'], evidence.TARGET, None)
        self.plan = {'schema': 'jbravo.production-evidence-plan.v1', 'policy': evidence.POLICY,
            'window_lengths': {'baseline_calendar_days': 252, 'recent_calendar_days': 63},
            'approval': {'approval_id': 'synthetic-plan-only', 'reviewer': 'test-fixture',
                         'approved_at_utc': '2026-01-02T12:00:00Z'},
            'model_binding': self.model, 'producer_sources': capture.producer_binding(),
            'population_policy': 'all_reviewed_symbols_every_session', 'symbols': self.symbols,
            'windows': {'baseline_start': start.isoformat(), 'baseline_end': bend.isoformat(),
                        'recent_start': rstart.isoformat(), 'recent_end': rend.isoformat()},
            'calendar': self.save('calendar.json', self.calendar)}

    def save(self, relative, value):
        content = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + '\n').encode()
        file = self.root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return {'path': relative, **capture.binding(content)}

    def validate(self, plan=None):
        return evidence.validate_plan(plan or self.plan, self.calendar, now=self.now)

    def observation(self, session='2026-01-05', symbols=None):
        symbols = self.symbols if symbols is None else symbols
        times = [session + 'T00:00:00Z'] * len(symbols)
        features = pd.DataFrame({'symbol': symbols, 'timestamp': times, 'x': np.arange(len(symbols))})
        predictions = features[['symbol', 'timestamp']].assign(score_5d=.2)
        generated = datetime.fromisoformat(session + 'T22:00:00+00:00')
        result = capture.capture(self.root / 'fixture-producer', features, predictions,
            snapshot_date=session, model_binding=self.model, generated_at=generated)
        file = Path(result['manifest_path'])
        cid = file.parent.name
        manifest = json.loads(file.read_bytes())
        for name in ('features.csv', 'predictions.csv'):
            self.save('captures/' + cid + '/' + name, (file.parent / name).read_bytes())
        ref = self.save('captures/' + cid + '/manifest.json', manifest)
        return ref, manifest

    def consume(self, references, plan=None):
        _, sessions, cohorts = self.validate(plan)
        return evidence.capture_rows(evidence.Reader(self.root), references, plan or self.plan,
                                     sessions, cohorts, self.now)

    def tiny_outcomes(self):
        _, sessions, _ = self.validate()
        rows = pd.DataFrame([{'symbol': 'S0', 'timestamp': '2026-01-05T00:00:00+00:00',
            'signal_session': '2026-01-05', 'observed_at_utc': '2026-01-05T22:00:00Z',
            'capture_id': 'a' * 32, evidence.SCORE: .2, 'x': 1}])
        bars = pd.DataFrame([{'symbol': 'S0', 'timestamp': s['session'].isoformat() + 'T00:00:00Z',
                             'close': 100 if i < 5 else 104} for i, s in enumerate(sessions[:6])])
        return rows, bars, sessions

    def join(self, rows, bars, sessions, cutoff='2026-01-12T22:00:00Z'):
        when = evidence.utc(cutoff)
        return evidence.mature_rows(rows, bars, sessions, observed_at=when, cutoff=when, now=self.now)

    def test_same_model_population_with_missing_sessions_stays_incomplete(self):
        ref, manifest = self.observation()
        rows, missing = self.consume([ref])
        self.assertEqual(len(rows), 60)
        self.assertGreater(len(missing), 200)
        self.assertNotIn(date(2026, 1, 5), missing)
        self.assertEqual(manifest['schema'], 'jbravo.production-score-observation.v2')
        self.assertEqual(manifest['producer_sources'], self.plan['producer_sources'])

    def test_legacy_capture_and_model_or_source_rebinding_rejected(self):
        ref, original = self.observation()
        for mutate in ('schema', 'model_binding', 'producer_sources'):
            value = copy.deepcopy(original)
            if mutate == 'schema':
                value['schema'] = 'jbravo.production-score-observation.v1'
            elif mutate == 'model_binding':
                value[mutate]['sha256'] = 'b' * 64
            else:
                value[mutate]['scripts/ranker_predict.py']['sha256'] = 'b' * 64
            changed = self.save(ref['path'], value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.consume([changed])

    def test_late_score_cannot_masquerade_as_forward_observation(self):
        ref, manifest = self.observation()
        manifest['observed_at_utc'] = '2026-01-06T15:00:00Z'
        changed = self.save(ref['path'], manifest)
        with self.assertRaisesRegex(ValueError, 'future_market_data'):
            self.consume([changed])

    def test_changed_population_and_duplicate_capture_sessions_rejected(self):
        ref, _ = self.observation(symbols=self.symbols[:-1])
        with self.assertRaisesRegex(ValueError, 'changed_capture_population'):
            self.consume([ref])
        ref, _ = self.observation()
        with self.assertRaisesRegex(ValueError, 'duplicate_or_outside'):
            self.consume([ref, ref])

    def test_captured_bytes_are_verified_before_use(self):
        ref, _ = self.observation()
        (self.root / ref['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes_changed'):
            self.consume([ref])

    def test_plan_rejects_overlap_unreviewed_lengths_and_changed_thresholds(self):
        for mutation in ('overlap', 'duration', 'threshold', 'late_registration'):
            plan = copy.deepcopy(self.plan)
            if mutation == 'overlap': plan['windows']['recent_start'] = plan['windows']['baseline_end']
            elif mutation == 'duration': plan['windows']['baseline_end'] = '2026-02-01'
            elif mutation == 'threshold': plan['policy']['psi_warn'] = .9
            else: plan['approval']['approved_at_utc'] = '2026-01-05T22:00:00Z'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(plan)

    def test_plan_accepts_reviewed_lengths_and_original_monitor_defaults(self):
        bounds, _, _ = self.validate()
        self.assertEqual((bounds['baseline_end'] - bounds['baseline_start']).days + 1, 252)
        plan = copy.deepcopy(self.plan)
        plan['window_lengths'] = {'baseline_calendar_days': 70, 'recent_calendar_days': 49}
        plan['windows'] = {'baseline_start': '2026-07-20', 'baseline_end': '2026-09-27',
                           'recent_start': '2026-09-28', 'recent_end': '2026-11-15'}
        self.validate(plan)

    def test_plan_requires_explicit_positive_integer_window_lengths(self):
        for value in (None, {}, {'baseline_calendar_days': 70},
                      {'baseline_calendar_days': 70, 'recent_calendar_days': 49, 'extra': 1}):
            plan = copy.deepcopy(self.plan)
            plan['window_lengths'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'window_lengths_required'):
                self.validate(plan)
        for value in (0, -1, True, 49.0, '49'):
            plan = copy.deepcopy(self.plan)
            plan['window_lengths']['recent_calendar_days'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'window_lengths_required'):
                self.validate(plan)
        plan = copy.deepcopy(self.plan)
        del plan['window_lengths']
        with self.assertRaisesRegex(ValueError, 'window_lengths_required'):
            self.validate(plan)

    def test_join_uses_five_sessions_across_weekend_and_recomputes_label(self):
        rows, bars, sessions = self.tiny_outcomes()
        mature, missing = self.join(rows, bars, sessions)
        self.assertFalse(missing)
        self.assertAlmostEqual(mature.iloc[0][evidence.RETURN], .04)
        self.assertEqual(mature.iloc[0][evidence.TARGET], 1)
        self.assertEqual(mature.iloc[0]['outcome_available_at_utc'], '2026-01-12T21:00:00+00:00')

    def test_finite_return_cannot_clear_unmatured_or_missing_bar_outcome(self):
        rows, bars, sessions = self.tiny_outcomes()
        with self.assertRaisesRegex(ValueError, 'future_outcome_bar'):
            self.join(rows, bars, sessions, cutoff='2026-01-12T20:00:00Z')
        mature, missing = self.join(rows, bars.iloc[:-1], sessions, cutoff='2026-01-12T20:00:00Z')
        self.assertTrue(mature.empty)
        self.assertEqual(missing[0]['reason'], 'outcome_not_mature')

    def test_missing_intermediate_bar_cannot_shift_the_fifth_session(self):
        rows, bars, sessions = self.tiny_outcomes()
        mature, missing = self.join(rows, bars.drop(index=2), sessions)
        self.assertTrue(mature.empty)
        self.assertEqual(missing[0]['reason'], 'outcome_bar_gap')

    def test_future_timestamp_cannot_enter_a_closed_session_snapshot(self):
        rows, bars, sessions = self.tiny_outcomes()
        bars.loc[5, 'timestamp'] = '2026-01-12T23:00:00Z'
        with self.assertRaisesRegex(ValueError, 'future_outcome_bar_timestamp'):
            self.join(rows, bars, sessions)

    def test_duplicate_or_mismatched_entry_or_naive_bar_time_rejected(self):
        rows, bars, sessions = self.tiny_outcomes()
        for value in (pd.concat([bars, bars.iloc[[0]]]),
                      bars.assign(timestamp=['2026-01-05T01:00:00Z', *bars.timestamp.iloc[1:]]),
                      bars.assign(timestamp=bars.timestamp.str.replace('Z', '', regex=False))):
            with self.subTest(), self.assertRaises(ValueError):
                self.join(rows, value, sessions)

    def test_constant_scores_do_not_hide_a_large_probability_shift(self):
        stable = evidence.production_psi([.2] * 50, [.2] * 50, probability=True)
        shifted = evidence.production_psi([.2] * 50, [.8] * 50, probability=True)
        self.assertEqual(stable['psi'], 0)
        self.assertEqual(shifted['level'], 'alert')
        feature = evidence.production_psi([1] * 50, [-1] * 50)
        self.assertEqual(feature['level'], 'alert')

    def test_readers_reject_traversal_duplicate_json_and_duplicate_csv_headers(self):
        for path in ('../plan.json', '/plan.json', '.env', 'data\\plan.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                evidence.Reader(self.root).bound({'path': path, 'bytes': 1, 'sha256': 'a' * 64})
        with self.assertRaises(ValueError): evidence.strict_json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError): evidence.read_csv(b'symbol,timestamp,symbol\nA,time,A\n')

    def test_health_retains_real_numerical_thresholds_and_never_orders_retraining(self):
        days = list(pd.bdate_range('2026-01-05', periods=40)) + list(pd.bdate_range('2026-10-01', periods=40))
        rows = []
        for i, stamp in enumerate(days):
            for j in range(60):
                score = 1. if j < 30 else 0.
                rows.append({'symbol': 'S' + str(j), 'timestamp': stamp.isoformat(), 'score_5d': score,
                    evidence.TARGET: int(score), evidence.RETURN: .04 + .01 * (i % 2) if score else -.01,
                    'x': float(j)})
        bounds = {'baseline_start': date(2026, 1, 1), 'baseline_end': date(2026, 9, 1),
                  'recent_start': date(2026, 10, 1), 'recent_end': date(2026, 12, 1)}
        stable = evidence.health(pd.DataFrame(rows), bounds)
        self.assertEqual(stable['recommended_action'], 'none')
        self.assertTrue(stable['calibration_applicable'])
        self.assertGreater(stable['recent_strategy']['sharpe'], .3)
        shifted = pd.DataFrame(rows)
        shifted.loc[pd.to_datetime(shifted.timestamp).dt.month.ge(10), 'score_5d'] = .5
        drift = evidence.health(shifted, bounds)
        self.assertEqual(drift['recommended_action'], 'investigate')
        self.assertEqual(drift['diagnostic_recommended_action'], 'retrain')

    def test_human_review_required_before_evaluation_or_output(self):
        ref = self.save('review.json', {'schema': 'jbravo.production-evidence-review.v1'})
        with self.assertRaisesRegex(ValueError, 'independent_review_required'):
            evidence.evaluate(self.root, ref, now=self.now)
        self.assertEqual(list(self.root.glob('*/monitor_candidate.json')), [])

    def test_saved_candidate_does_not_publish_or_grant_acceptance(self):
        report = {'monitor_candidate_eligible': False, 'operating_acceptance': False}
        result = evidence.save_result(self.root, report, pd.DataFrame({'symbol': ['fixture']}))
        output = Path(result['packet']) / 'monitor_candidate.json'
        self.assertFalse(result['operating_acceptance'])
        self.assertEqual(capture.binding(output.read_bytes()), result['monitor_binding'])
        first = output.read_bytes()
        evidence.save_result(self.root, report, pd.DataFrame({'symbol': ['fixture']}))
        self.assertEqual(output.read_bytes(), first)

    def test_guard_cannot_clear_an_unpublished_or_wrong_current_model_candidate(self):
        candidate = {'monitor_contract_version': 3, 'recommended_action': 'none',
            'qualified_for_monitoring': True, 'monitor_candidate_eligible': True,
            'publication_approved': False, 'policy': evidence.POLICY,
            'window_lengths': self.plan['window_lengths'],
            'model_binding': self.model, 'producer_sources': self.plan['producer_sources'],
            'model_comparison': {'scope': 'reviewed_fixed_production_model',
                'production_drift_established': True, 'common_model_sha256': [self.model['sha256']]},
            'missing_capture_sessions': [], 'incomplete_outcome_rows': 0,
            'windows': {**self.plan['windows'], 'baseline_rows': 3000, 'recent_rows': 3000},
            **{key: capture.binding(b'fixture') for key in ('review_binding', 'plan_binding',
                'calendar_binding', 'bars_binding', 'qualified_rows_binding')}}
        current = {'model_binding': self.model, 'producer_sources': self.plan['producer_sources']}
        self.assertEqual(self.guard._extract_health(candidate, current)['recommended_action'], 'investigate')
        # This synthetic boolean is not a real publication approval.
        candidate['publication_approved'] = True
        self.assertEqual(self.guard._extract_health(candidate, current)['recommended_action'], 'none')
        for mutation in ('missing', 'mismatch', 'overlap'):
            invalid = copy.deepcopy(candidate)
            if mutation == 'missing': del invalid['window_lengths']
            elif mutation == 'mismatch': invalid['window_lengths']['recent_calendar_days'] += 1
            else: invalid['windows']['recent_start'] = invalid['windows']['baseline_end']
            result = self.guard._extract_health(invalid, current)
            with self.subTest(mutation=mutation):
                self.assertEqual(result['recommended_action'], 'investigate')
                self.assertIn('production_window_design_invalid', result['monitor_production_qualification_reasons'])
        changed = copy.deepcopy(current)
        changed['model_binding']['sha256'] = 'a' * 64
        result = self.guard._extract_health(candidate, changed)
        self.assertEqual(result['recommended_action'], 'investigate')
        self.assertIn('current_prediction_model_or_producer_not_comparable',
                      result['monitor_production_qualification_reasons'])

    def test_guard_unknown_contract_is_investigation(self):
        self.assertEqual(self.guard._extract_health({'monitor_contract_version': 99,
                         'recommended_action': 'none'})['recommended_action'], 'investigate')

    def test_reserved_feature_names_cannot_overwrite_outcome_provenance(self):
        for name in ('capture_id', 'outcome_timestamp', 'signal_session', 'fwd_ret_10d'):
            model = copy.deepcopy(self.model)
            model['feature_columns'] = [name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'feature_identity'):
                evidence.model_identity(model)

    def test_metric_imports_leave_authentication_loading_at_runtime_boundary(self):
        import ast
        for relative, runtime in (('scripts/ranker_monitor.py', 'run_monitor'),
                                  ('scripts/ranker_strategy_eval.py', 'run_strategy_eval')):
            tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
            top_calls = [node.value for node in tree.body if isinstance(node, ast.Expr)
                         and isinstance(node.value, ast.Call)]
            self.assertFalse(any(isinstance(node.func, ast.Name) and node.func.id == 'load_env'
                                 for node in top_calls))
            # Existing actual entry points still load authentication normally.
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == runtime)
            self.assertEqual(ast.unparse(function.body[0]), 'load_env()')

    def test_full_bound_packet_reaches_only_a_healthy_unpublished_candidate(self):
        # Shorter lengths are synthetic examples, not a recommended production study.
        self.plan['window_lengths'] = {'baseline_calendar_days': 70, 'recent_calendar_days': 49}
        self.plan['windows'] = {'baseline_start': '2026-07-20', 'baseline_end': '2026-09-27',
                                'recent_start': '2026-09-28', 'recent_end': '2026-11-15'}
        bounds, sessions, cohorts = self.validate()
        references = []
        for index, session in enumerate(cohorts['baseline'] + cohorts['recent']):
            # Complete synthetic packet fixtures, never retained production data.
            # Normal capture/predictor execution is verified by separate tests.
            cid = format(index + 1, '032x')
            frame = pd.DataFrame({'symbol': self.symbols,
                'timestamp': [session.isoformat() + 'T00:00:00+00:00'] * 60, 'x': np.arange(60)})
            predictions = frame[['symbol', 'timestamp']].assign(score_5d=[1.] * 30 + [0.] * 30)
            manifest = {'schema': 'jbravo.production-score-observation.v2', 'capture_id': cid,
                'observed_at_utc': session.isoformat() + 'T22:00:00Z', 'signal_session': session.isoformat(),
                'rows': 60, 'model_binding': self.model, 'producer_sources': self.plan['producer_sources'],
                'feature_representation': 'normalized_imputed_predictor_input',
                'outcomes_attached': False, 'research_qualified': False, 'operating_acceptance': False}
            for kind, value in (('features', frame), ('predictions', predictions)):
                ref = self.save('captures/' + cid + '/' + kind + '.csv', value.to_csv(index=False).encode())
                manifest[kind] = {key: ref[key] for key in ('bytes', 'sha256')}
            references.append(self.save('captures/' + cid + '/manifest.json', manifest))
        bars = []
        for i, item in enumerate(sessions):
            if item['close'] > self.now:
                continue
            for j, symbol in enumerate(self.symbols):
                log_price = i * (.009 if j < 30 else -.002) + .002 * np.sin(i)
                bars.append({'symbol': symbol, 'timestamp': item['session'].isoformat() + 'T00:00:00Z',
                             'close': 100 * np.exp(log_price)})
        review = {'schema': 'jbravo.production-evidence-review.v1',
            'approval': {'approval_id': 'synthetic-review-only', 'reviewer': 'test-fixture',
                         'approved_at_utc': '2026-11-20T22:30:00Z'},
            'attestations': {key: True for key in ('model_deployment_identity_reviewed',
                'causal_capture_reviewed', 'calendar_completeness_reviewed',
                'population_comparability_reviewed', 'split_adjusted_outcomes_reviewed')},
            'plan': self.save('plan.json', self.plan), 'captures': references,
            'bars': self.save('bars.csv', pd.DataFrame(bars).to_csv(index=False).encode()),
            'bars_observed_at_utc': '2026-11-20T22:00:00Z',
            'outcome_cutoff_utc': '2026-11-20T21:00:00Z', 'bars_adjustment': 'split',
            'signal_session': '2026-11-20'}
        ref = self.save('review.json', review)
        report, rows = evidence.evaluate(self.root, ref, now=self.now)
        self.assertTrue(report['qualified_for_monitoring'])
        self.assertTrue(report['monitor_candidate_eligible'], report['recommendation_reasons'])
        self.assertFalse(report['publication_approved'])
        self.assertFalse(report['operating_acceptance'])
        self.assertEqual(report['recommended_action'], 'none')
        self.assertEqual(report['incomplete_outcome_rows'], 0)
        self.assertEqual(report['window_lengths'], self.plan['window_lengths'])
        self.assertGreater(report['windows']['recent_rows'], 2000)
        saved = evidence.save_result(self.root, report, rows)
        self.assertTrue(saved['monitor_candidate_eligible'])
        # Preserve the known failed packet. Removing one capture cannot be
        # silently replaced by overlapping windows or a row-count-only check.
        review['captures'] = references[:-1]
        rejected, _ = evidence.evaluate(self.root, self.save('incomplete-review.json', review), now=self.now)
        self.assertFalse(rejected['monitor_candidate_eligible'])
        self.assertIn('missing_production_score_sessions', rejected['recommendation_reasons'])
        # A complete shorter plan still cannot waive the existing population minimum.
        short_plan = copy.deepcopy(self.plan)
        short_plan['window_lengths'] = {'baseline_calendar_days': 2, 'recent_calendar_days': 2}
        short_plan['windows'] = {'baseline_start': '2026-11-10', 'baseline_end': '2026-11-11',
                                 'recent_start': '2026-11-12', 'recent_end': '2026-11-13'}
        review['plan'] = self.save('short-plan.json', short_plan)
        review['captures'] = [ref for session, ref in zip(cohorts['baseline'] + cohorts['recent'], references)
                              if date(2026, 11, 10) <= session <= date(2026, 11, 13)]
        rejected, _ = evidence.evaluate(self.root, self.save('short-review.json', review), now=self.now)
        self.assertEqual(rejected['missing_capture_sessions'], [])
        self.assertEqual(rejected['incomplete_outcome_rows'], 0)
        self.assertFalse(rejected['monitor_candidate_eligible'])
        self.assertIn('insufficient_comparable_rows', rejected['recommendation_reasons'])


if __name__ == '__main__':
    unittest.main()
