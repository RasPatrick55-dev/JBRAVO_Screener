"""Normal guard import with inert DB/package boundaries; no live monitoring.

The pipeline summary assignment is source-extracted, not a full pipeline import.
The actual postflight producer/checker run against temporary files.
Run using unittest so the repository's credential-dependent conftest is not loaded.
"""
import ast
import copy
from datetime import date, datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import scripts
from scripts import pipeline_postflight as postflight

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
DAY = NOW.date()


def payload():
    return {
        'run_utc': '2026-10-09T10:00:00Z', 'run_date': '2026-10-08',
        'data_source': 'db://ml_artifacts/ranker_oos_predictions',
        'recommended_action': 'none',
        'windows': {'dataset_start': '2026-01-01', 'dataset_end': '2026-10-08',
                    'baseline_start': '2026-01-01', 'baseline_end': '2026-09-09',
                    'recent_start': '2026-08-07', 'recent_end': '2026-10-08',
                    'population_rows': 1000, 'baseline_rows': 600, 'recent_rows': 250},
    }


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.socket.connect', 'socket.create_connection',
                       'subprocess.Popen', 'subprocess.run'):
            patcher = mock.patch(target, side_effect=AssertionError('external_execution_forbidden'))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.db = ModuleType('scripts.db')
        self.db.db_enabled = mock.Mock(return_value=True)
        self.db.fetch_latest_ml_artifact = mock.Mock(return_value=None)
        utils = ModuleType('scripts.utils')
        utils.__path__ = [str(ROOT / 'scripts/utils')]
        # Do not import the production utils initializer (SDK imports) or DB.
        with mock.patch.dict(sys.modules, {'scripts.db': self.db, 'scripts.utils': utils}), \
             mock.patch.object(scripts, 'db', self.db, create=True), \
             mock.patch.object(scripts, 'utils', utils, create=True):
            self.guard = importlib.import_module('scripts.utils.ml_health_guard')

    def loaded(self, value=None, *, record_date='2026-10-08'):
        self.db.fetch_latest_ml_artifact.return_value = {
            'payload': payload() if value is None else value, 'run_date': record_date,
        }
        return self.guard.load_latest_ml_health(base_dir=self.root, source='db')

    def decide(self, value=None, **kw):
        return self.guard.decide_ml_enrichment(
            self.loaded() if value is None else value, mode=kw.pop('mode', 'warn'),
            max_age_days=kw.pop('max_age_days', 7), pipeline_run_date=kw.pop('pipeline_run_date', DAY),
            now=kw.pop('now', NOW), require_monitor_provenance=kw.pop('require_monitor_provenance', True), **kw)

    def test_fresh_report_and_inputs_are_separate(self):
        result = self.decide()
        self.assertEqual(result['decision'], 'allow')
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['monitor_generated_at'], '2026-10-09T10:00:00+00:00')
        self.assertEqual(result['monitor_generation_age_seconds'], 7200)
        self.assertEqual(result['monitor_input_age_days'], 1)
        self.assertEqual(result['monitor_input_coverage'], payload()['windows'])
        self.assertEqual(result['monitor_artifact_run_date'], '2026-10-08')
        self.assertEqual(result['monitor_observed_at'], NOW.isoformat())

    def test_db_record_date_never_substitutes_for_generation_or_coverage(self):
        legacy = {'recommended_action': 'none', 'run_date': None}
        result = self.decide(self.loaded(legacy))
        self.assertEqual(result['monitor_run_date'], '2026-10-08')
        self.assertEqual(result['monitor_run_date_source'], 'db_record')
        self.assertIsNone(result['monitor_generated_at'])
        self.assertIsNone(result['monitor_input_coverage']['dataset_end'])
        self.assertIn('missing_monitor_generation_time', result['reasons'])
        self.assertIn('missing_monitor_input_coverage', result['reasons'])
        self.assertEqual(result['decision'], 'warn')

    def test_fs_payload_date_has_distinct_provenance(self):
        path = self.root / 'monitor.json'
        path.write_text(json.dumps(payload()), encoding='utf-8')
        health = self.guard.load_latest_ml_health(base_dir=self.root, source='fs', path=path)
        self.db.fetch_latest_ml_artifact.assert_not_called()
        self.assertEqual(health['source'], 'fs')
        self.assertEqual(health['run_date_source'], 'payload')
        self.assertIsNone(health['artifact_run_date'])
        self.assertEqual(self.decide(health)['decision'], 'allow')

    def test_db_precedence_retains_its_stale_inputs(self):
        path = self.root / 'monitor.json'
        path.write_text(json.dumps(payload()), encoding='utf-8')
        old = payload()
        old['windows']['dataset_end'] = old['windows']['recent_end'] = '2026-09-09'
        old['windows']['recent_start'] = '2026-07-09'
        self.loaded(old)
        health = self.guard.load_latest_ml_health(base_dir=self.root, source='auto', path=path)
        self.assertEqual(health['source'], 'db')
        self.assertIn('stale_monitor_input_dataset_end', self.decide(health)['reasons'])

    def test_fresh_generation_over_stale_inputs_cannot_allow(self):
        value = payload()
        value['windows']['dataset_end'] = value['windows']['recent_end'] = '2026-09-09'
        value['windows']['recent_start'] = '2026-07-09'
        result = self.decide(self.loaded(value))
        self.assertNotIn('stale_monitor_generation', result['reasons'])
        self.assertIn('stale_monitor_input_dataset_end', result['reasons'])
        self.assertIn('stale_monitor_input_recent_end', result['reasons'])
        self.assertEqual(result['decision'], 'warn')

    def test_stale_generation_with_fresh_inputs_cannot_allow(self):
        value = payload()
        value['run_utc'] = '2026-09-30T10:00:00Z'
        result = self.decide(self.loaded(value))
        self.assertIn('stale_monitor_generation', result['reasons'])
        self.assertIn('monitor_input_after_generation', result['reasons'])
        self.assertNotIn('stale_monitor_input_dataset_end', result['reasons'])

    def test_generation_threshold_is_elapsed_time(self):
        for delta, stale in ((timedelta(days=7), False),
                             (timedelta(days=7, microseconds=1), True)):
            with self.subTest(delta=delta):
                value = payload()
                value['run_utc'] = (NOW - delta).isoformat()
                result = self.decide(self.loaded(value))
                self.assertEqual('stale_monitor_generation' in result['reasons'], stale)

    def test_input_threshold_is_calendar_days(self):
        for days, stale in ((7, False), (8, True)):
            with self.subTest(days=days):
                value = payload()
                value['windows']['dataset_end'] = value['windows']['recent_end'] = (DAY - timedelta(days=days)).isoformat()
                result = self.decide(self.loaded(value))
                self.assertEqual('stale_monitor_input_dataset_end' in result['reasons'], stale)

    def test_missing_generation_and_coverage(self):
        value = payload()
        del value['run_utc']
        del value['windows']
        result = self.decide(self.loaded(value))
        self.assertIn('missing_monitor_generation_time', result['reasons'])
        self.assertIn('missing_monitor_input_coverage', result['reasons'])

    def test_invalid_and_naive_generation(self):
        for raw in ('2026-10-09', '2026-10-09T10:00:00', 'bad', 0, True, {}):
            with self.subTest(raw=raw):
                value = payload()
                value['run_utc'] = raw
                self.assertIn('invalid_monitor_generation_time', self.decide(self.loaded(value))['reasons'])

    def test_generation_offsets_are_normalized(self):
        value = payload()
        value['run_utc'] = '2026-10-09T05:00:00-05:00'
        self.assertEqual(self.decide(self.loaded(value))['monitor_generated_at'], '2026-10-09T10:00:00+00:00')

    def test_future_generation(self):
        value = payload()
        value['run_utc'] = (NOW + timedelta(microseconds=1)).isoformat()
        self.assertIn('future_monitor_generation_time', self.decide(self.loaded(value))['reasons'])

    def test_missing_and_invalid_input_fields(self):
        for field in payload()['windows']:
            for raw, prefix in ((None, 'missing'), ('bad', 'invalid')):
                with self.subTest(field=field, raw=raw):
                    value = payload()
                    value['windows'][field] = raw
                    self.assertIn(prefix + '_monitor_input_' + field, self.decide(self.loaded(value))['reasons'])

    def test_invalid_coverage_container(self):
        for raw in ([], 'bad', False):
            value = payload()
            value['windows'] = raw
            self.assertIn('invalid_monitor_input_coverage', self.decide(self.loaded(value))['reasons'])

    def test_inconsistent_window_order_and_bounds(self):
        for field, raw in (('dataset_start', '2026-10-09'), ('baseline_start', '2025-12-31'),
                           ('baseline_end', '2026-10-09'), ('recent_start', '2026-10-09'),
                           ('recent_end', '2026-10-09')):
            with self.subTest(field=field):
                value = payload()
                value['windows'][field] = raw
                self.assertIn('inconsistent_monitor_input_windows', self.decide(self.loaded(value))['reasons'])

    def test_nominal_recent_window_can_precede_short_dataset(self):
        value = payload()
        value['windows'].update(dataset_start='2026-10-01', baseline_start='2026-10-01',
                                baseline_end='2026-10-08', population_rows=20, baseline_rows=20, recent_rows=20)
        self.assertEqual(self.decide(self.loaded(value))['decision'], 'allow')

    def test_future_coverage_and_coverage_after_generation(self):
        value = payload()
        value['windows']['dataset_end'] = value['windows']['recent_end'] = '2026-10-10'
        result = self.decide(self.loaded(value))
        self.assertIn('future_monitor_input_dataset_end', result['reasons'])
        self.assertIn('future_monitor_input_recent_end', result['reasons'])
        self.assertIn('monitor_input_after_generation', result['reasons'])

    def test_empty_invalid_and_excessive_row_counts(self):
        for raw, reason in ((0, 'empty_monitor_input_recent_rows'),
                            (True, 'invalid_monitor_input_recent_rows'),
                            (-1, 'invalid_monitor_input_recent_rows'),
                            (1.5, 'invalid_monitor_input_recent_rows'),
                            (1001, 'inconsistent_monitor_input_row_counts')):
            with self.subTest(raw=raw):
                value = payload()
                value['windows']['recent_rows'] = raw
                self.assertIn(reason, self.decide(self.loaded(value))['reasons'])

    def test_coverage_dates_are_not_truncated_or_coerced(self):
        for raw in ('2026-10-08junk', '2026-10-08T23:00:00Z', '20261008', True):
            value = payload()
            value['windows']['dataset_end'] = raw
            self.assertIn('invalid_monitor_input_dataset_end', self.decide(self.loaded(value))['reasons'])

    def test_legacy_run_date_remains_explicit_and_stale(self):
        value = payload()
        value['run_date'] = '2026-02-27'
        result = self.decide(self.loaded(value))
        self.assertIn('stale_monitor', result['reasons'])
        self.assertEqual(result['monitor_run_date'], '2026-02-27')
        self.assertEqual(result['monitor_generated_at'], '2026-10-09T10:00:00+00:00')

    def test_malformed_and_future_run_date(self):
        for raw, reason in (('2026-10-08junk', 'invalid_monitor_run_date'),
                            ('2026-10-10', 'future_monitor_run_date')):
            value = payload()
            value['run_date'] = raw
            self.assertIn(reason, self.decide(self.loaded(value))['reasons'])

    def test_missing_source_and_reference_are_not_green(self):
        value = payload()
        del value['data_source']
        result = self.decide(self.loaded(value), pipeline_run_date=None)
        self.assertIn('missing_monitor_input_source', result['reasons'])
        self.assertIn('missing_or_invalid_monitor_reference_date', result['reasons'])

    def test_missing_monitor(self):
        self.db.fetch_latest_ml_artifact.return_value = None
        health = self.guard.load_latest_ml_health(base_dir=self.root, source='db')
        result = self.decide(health)
        self.assertIn('missing_monitor', result['reasons'])
        self.assertEqual(result['decision'], 'warn')

    def test_modes_actions_and_prediction_reasons_are_preserved(self):
        for mode, expected in (('warn', 'warn'), ('block', 'block')):
            value = payload()
            value['recommended_action'] = 'recalibrate'
            result = self.decide(self.loaded(value), mode=mode, predictions_stale=True)
            self.assertEqual(result['decision'], expected)
            self.assertIn('action_recalibrate', result['reasons'])
            self.assertIn('stale_predictions', result['reasons'])

    def test_reference_time_requires_explicit_timezone(self):
        with self.assertRaisesRegex(ValueError, 'aware_monitor_reference_time_required'):
            self.decide(now=NOW.replace(tzinfo=None))

    def test_legacy_caller_decision_is_not_changed_by_new_provenance_gaps(self):
        legacy = self.loaded({'recommended_action': 'none', 'run_date': '2026-10-08'})
        result = self.decide(legacy, require_monitor_provenance=False)
        self.assertEqual(result['decision'], 'allow')
        self.assertEqual(result['reasons'], [])
        self.assertFalse(result['monitor_provenance_required'])
        self.assertIn('missing_monitor_generation_time', result['monitor_provenance_reasons'])
        # Historical parsing behavior is retained only for the legacy policy.
        legacy['run_date'] = '2026-10-08legacy'
        self.assertEqual(self.decide(legacy, require_monitor_provenance=False)['decision'], 'allow')
        self.assertIn('invalid_monitor_run_date', self.decide(legacy)['reasons'])

    def test_primary_pipeline_explicitly_requires_provenance(self):
        tree = ast.parse((ROOT / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'decide_ml_enrichment']
        self.assertEqual(len(calls), 1)
        flags = [k.value for k in calls[0].keywords if k.arg == 'require_monitor_provenance']
        self.assertEqual(len(flags), 1)
        self.assertIs(ast.literal_eval(flags[0]), True)

    def test_generation_overflow_is_invalid_not_an_exception(self):
        value = payload()
        value['run_utc'] = '0001-01-01T00:00:00+01:00'
        self.assertIn('invalid_monitor_generation_time', self.decide(self.loaded(value))['reasons'])

    def test_input_payload_and_windows_are_immutable(self):
        value = payload()
        before = copy.deepcopy(value)
        self.decide(self.loaded(value))
        self.assertEqual(value, before)

    def pipeline_summary(self, decision):
        tree = ast.parse((ROOT / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'ml_health_summary' for t in n.targets)
                       and isinstance(n.value, ast.Dict) and any(isinstance(k, ast.Constant)
                       and k.value == 'monitor_generated_at' for k in n.value.keys)]
        self.assertEqual(len(assignments), 1)
        scope = {'decision': decision, 'enrichment_freshness': {'stale': False}}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), 'pipeline_health_summary', 'exec'), scope)
        return scope['ml_health_summary']

    def publish(self, health):
        for relative in postflight.SOURCE_FILES:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        started = NOW - timedelta(minutes=1)
        return postflight.publish_health(
            self.root, started_at=started, finished_at=NOW - timedelta(seconds=1),
            pipeline_rc=0, steps=postflight.REQUIRED_STEPS,
            step_rcs={s: 0 for s in postflight.REQUIRED_STEPS},
            stage_times={s: 1.0 for s in postflight.REQUIRED_STEPS}, degraded=False, labels_rows=10,
            freshness={'stale': False, 'pred_compatible': True, 'predict_rc': None, 'snapshot_date': DAY.isoformat()},
            coverage={'total': 2, 'non_null': 2, 'pct': 100, 'run_ts_utc': started.isoformat()},
            ml_health=health, controls={k: True for k in ('strict_predictions_meta', 'strict_auto_refresh_predictions',
                'auto_refresh_features', 'auto_refresh_predictions', 'use_champion', 'ml_health_guard',
                'refresh_predictions_for_candidates')})

    def test_pipeline_summary_and_saved_postflight_carry_both_clocks(self):
        summary = self.pipeline_summary(self.decide())
        record = self.publish(summary)
        saved = json.loads((self.root / 'reports/pipeline_postflight/latest.json').read_bytes())
        for field in postflight.ML_PROVENANCE_FIELDS:
            if field in summary:
                self.assertEqual(saved['ml_health'][field], summary[field])
        self.assertEqual(record['ml_health']['monitor_generated_at'], '2026-10-09T10:00:00+00:00')
        self.assertEqual(record['ml_health']['monitor_input_coverage']['dataset_end'], '2026-10-08')
        result = postflight.check_health(self.root, now=NOW)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['ml_health'], saved['ml_health'])

    def test_stale_inputs_reach_postflight_as_degraded(self):
        value = payload()
        value['windows']['dataset_end'] = value['windows']['recent_end'] = '2026-09-09'
        value['windows']['recent_start'] = '2026-07-09'
        self.publish(self.pipeline_summary(self.decide(self.loaded(value))))
        health = postflight.check_health(self.root, now=NOW)
        self.assertEqual(health['status'], 'degraded')
        self.assertIn('ml_health_reason_stale_monitor_input_dataset_end', health['qualifications'])
        self.assertFalse(health['research_qualified'])
        self.assertFalse(health['operating_acceptance'])

    def test_changed_guard_source_rejects_saved_report(self):
        self.publish(self.pipeline_summary(self.decide()))
        path = self.root / 'scripts/utils/ml_health_guard.py'
        path.write_bytes(path.read_bytes() + b'\n# synthetic source mutation\n')
        result = postflight.check_health(self.root, now=NOW)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('ml_health', result)


if __name__ == '__main__':
    unittest.main()
