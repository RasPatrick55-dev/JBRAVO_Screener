"""Offline ML log parsing/display evidence regressions; all source ports mocked."""
import unittest
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from test_dashboard_status_provenance import NOW, ROOT, artifact, source_module


class MlLogEvidenceTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen', 'subprocess.run'):
            block = patch(target, side_effect=AssertionError('external_execution_forbidden'))
            block.start()
            self.addCleanup(block.stop)
        self.ml = source_module('dashboards/ml_overview_api.py')
        self.ml._artifact_record = lambda *a, **kw: artifact()
        self.ml._latest_status_payload = lambda *a: {}
        self.ml._latest_model_snapshot = lambda *a: artifact()
        self.ml._expected_prediction_session = lambda *a: '2026-11-06'
        self.ml.load_latest_champion = lambda *a: {}
        self.ml.champion_execution_overrides = lambda *a: {}
        self.ml.safe_tail_lines = lambda *a, **kw: []

    def overview(self, lines):
        self.ml.safe_tail_lines = lambda *a, **kw: lines
        return self.ml.build_ml_overview(ROOT, now=NOW)

    def test_scalar_preserves_numeric_boolean_and_unknown_types(self):
        for raw, expected, kind in [
            ('0', 0, int), ('1', 1, int), ('-1', -1, int), ('7', 7, int),
            ('0.0', 0.0, float), ('1.0', 1.0, float), ('0.25,', 0.25, float),
            ('true', True, bool), ('False', False, bool), ('yes', True, bool),
            ('null', None, type(None)), ('None', None, type(None)),
            ('Unknown', None, type(None)), ('db:present', 'db:present', str),
        ]:
            with self.subTest(raw=raw):
                actual = self.ml._coerce_scalar(raw)
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), kind)

    def test_boolean_values_are_not_numeric_evidence(self):
        for raw in (True, False):
            self.assertIsNone(self.ml._coerce_int(raw))
            self.assertIsNone(self.ml._coerce_float(raw))
        for raw in ('false', 'null', 'Unknown', 'true', '0.5', '0', '1'):
            result = self.overview([f'2026-11-06 23:00:01 [INFO] RANKER_PREDICT rc={raw}'])
            self.assertEqual(result['fresh_success_count'], 0)
            self.assertIsNone(result['last_successful_execution'])

    def test_real_audit_dates_are_not_joined_into_one_run(self):
        result = self.overview([
            "2026-10-24 03:57:41 [INFO] MODEL_SCORE_OVERLAP_SAMPLE missing_symbols=['AAA', 'BBB'] reason=no_prediction_for_symbol",
            '2026-10-24 03:58:31 [INFO] AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_DONE rc=0 predictions_source=db:present',
            '2026-10-25 04:04:58 [INFO] MODEL_SCORE_OVERLAP_DIAG candidates=7 prediction_symbols=9 overlap=7 run_date=2026-10-23',
        ])
        refresh = result['enrichment']['candidate_refresh_done']
        self.assertIs(type(refresh['rc']), int)
        self.assertEqual(refresh['rc'], 0)
        self.assertEqual(refresh['evidence']['timestamp'], '2026-10-24T03:58:31+00:00')
        self.assertEqual(refresh['evidence']['scope'], 'historical_diagnostic')
        self.assertEqual(refresh['evidence']['source'], 'logs/pipeline.log')
        self.assertEqual(refresh['evidence']['run_binding'], 'unverified')
        overlap = result['overlap']
        self.assertEqual(overlap['overlap'], 7)
        self.assertEqual(overlap['evidence']['timestamp'], '2026-10-25T04:04:58+00:00')
        self.assertIsNone(overlap['sample_reason'])
        self.assertIsNone(overlap['missing_symbols'])
        sample = overlap['latest_sample']
        self.assertEqual(sample['missing_symbols'], ['AAA', 'BBB'])
        self.assertEqual(sample['evidence']['timestamp'], '2026-10-24T03:57:41+00:00')
        self.assertEqual(sample['evidence']['run_binding'], 'unverified')
        self.assertIsNone(result['last_successful_execution'])
        self.assertEqual(result['fresh_success_count'], 0)

    def test_same_second_is_not_an_invented_run_binding(self):
        result = self.overview([
            '2026-11-06 23:00:01 [INFO] MODEL_SCORE_OVERLAP_DIAG candidates=1 prediction_symbols=0 overlap=0',
            "2026-11-06 23:00:01 [INFO] MODEL_SCORE_OVERLAP_SAMPLE missing_symbols=['AAA'] reason=no_prediction_for_symbol",
        ])
        self.assertIs(type(result['overlap']['candidates']), int)
        self.assertIs(type(result['overlap']['overlap']), int)
        self.assertIsNone(result['overlap']['missing_symbols'])
        self.assertEqual(result['overlap']['latest_sample']['evidence']['run_binding'], 'unverified')

    def test_absent_logs_preserve_unknown(self):
        result = self.overview([])
        self.assertIsNone(result['enrichment']['candidate_refresh_done'])
        self.assertIsNone(result['overlap']['latest_sample'])
        self.assertIsNone(result['overlap']['evidence'])
        self.assertIsNone(result['overlap']['missing_symbols'])

    def test_old_log_checks_do_not_override_newer_prediction_metadata(self):
        result = self.overview([
            '2026-10-24 03:57:41 [INFO] PREDICTIONS_FRESHNESS stale=true pred_compatible=false pred_features_set=old pred_features_signature=old model_path=old.pkl pred_model_path=old.pkl pred_missing_frac=1',
            '2026-10-24 03:57:42 [INFO] ML_ENRICHMENT_DECISION action=retrain',
        ])
        fresh = result['freshness']
        self.assertFalse(fresh['stale'])
        self.assertTrue(fresh['pred_compatible'])
        self.assertIsNone(fresh['pred_features_set'])
        self.assertIsNone(fresh['pred_features_signature'])
        self.assertIsNone(fresh['pred_model_path'])
        self.assertIsNone(fresh['pred_missing_frac'])
        self.assertEqual(fresh['source'], 'db')
        self.assertTrue(fresh['latest_log_check']['stale'])
        self.assertEqual(fresh['latest_log_check']['evidence']['timestamp'], '2026-10-24T03:57:41+00:00')
        self.assertIsNone(result['monitor']['recommended_action'])
        self.assertEqual(result['enrichment']['latest_decision']['action'], 'retrain')

    def assert_execution_unknown(self, result):
        self.assertIsNone(result['last_successful_execution'])
        self.assertEqual(result['fresh_success_count'], 0)
        self.assertEqual(result['execution_evidence_scope'], 'UNVERIFIED_NO_PROVIDER_RECEIPT')
        self.assertEqual(result['status']['label'], 'Unknown')
        self.assertEqual(result['status']['tone'], 'neutral')
        for stage in result['pipeline_stages']:
            self.assertEqual(stage['execution_status'], 'unverified')
            self.assertFalse(stage['fresh_success'])
            self.assertNotEqual(stage['status'], 'success')
            self.assertNotIn('execution verified', stage['detail'])

    def test_fs_and_unbound_rc_zero_cannot_verify_current_state(self):
        self.ml._artifact_record = lambda *a, **kw: dict(artifact(), source='fs')
        result = self.overview([
            '2026-11-06 23:00:01 [INFO] RANKER_PREDICT rc=0',
            '2026-10-01 10:00:00 [INFO] MODEL_SCORE_COVERAGE total=1 non_null=1 pct=100',
        ])
        self.assert_execution_unknown(result)
        self.assertEqual(result['freshness']['status'], 'unknown')
        self.assertIsNone(result['freshness']['stale'])
        self.assertEqual(result['freshness']['metadata_status'], 'fresh')
        self.assertEqual(result['freshness']['source_authority'], 'diagnostic_only')
        self.assertEqual(result['coverage']['evidence']['timestamp'], '2026-10-01T10:00:00+00:00')
        self.assertTrue(result['coverage']['diagnostic_only'])
        prediction = next(s for s in result['pipeline_stages'] if s['label'] == 'Predict')
        self.assertEqual(prediction['status'], 'diagnostic')
        self.assertEqual(prediction['timestamp'], '2026-11-06T23:00:00+00:00')

    def test_db_metadata_freshness_is_not_a_provider_execution_receipt(self):
        for rc in (0, 1):
            result = self.overview([
                f'2026-11-06 23:00:01 [INFO] RANKER_PREDICT rc={rc}',
                '2026-10-01 10:00:00 [INFO] MODEL_SCORE_COVERAGE total=1 non_null=1 pct=100',
            ])
            self.assert_execution_unknown(result)
            self.assertEqual(result['freshness']['status'], 'fresh')
            self.assertEqual(result['freshness']['source_authority'], 'postgresql')
            self.assertEqual(result['freshness']['evidence_scope'], 'artifact_metadata_only')
            self.assertEqual(result['enrichment']['latest_prediction_event']['rc'], rc)

    def test_filesystem_fallback_cannot_self_declare_database_authority(self):
        # Exercise the actual fallback loader rather than only the overview mocks.
        ml = source_module('dashboards/ml_overview_api.py')
        ml.db = types.SimpleNamespace(db_enabled=lambda: False)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / 'latest_meta.json'
            path.write_text(json.dumps(dict(artifact()['payload'], source='db')), encoding='utf-8')
            record = ml._artifact_record('predictions', base_dir=base, fs_json=path)
            self.assertEqual(record['source'], 'fs')
            self.assertTrue(record['diagnostic_only'])
            timing = ml.artifact_freshness(record, now=NOW, expected_session='2026-11-06')
            self.assertEqual(timing['status'], 'unknown')
            self.assertEqual(timing['metadata_status'], 'fresh')
            self.assertEqual(timing['artifact_timestamp'], '2026-11-06T23:00:00+00:00')

    def test_log_updates_do_not_retimestamp_database_artifacts(self):
        result = self.overview([
            '2026-11-08 11:00:00 [INFO] RANKER_RECALIBRATE_END rc=0',
            '2026-11-08 11:00:01 [INFO] AUTOREMEDIATE_END rc=0',
            '2026-11-08 11:00:02 [INFO] TRADE_ATTRIBUTION_END rc=0',
        ])
        for key in ('last_eval', 'last_monitor', 'last_recalibrate', 'last_autoremediate', 'last_trade_attribution'):
            self.assertEqual(result['timestamps'][key], '2026-11-06T23:00:00+00:00')
        self.assertEqual(result['recent_events'][0]['timestamp'], '2026-11-08T11:00:02+00:00')

    def test_fs_monitor_and_coverage_cannot_promote_current_summary(self):
        def record(*a, **kw):
            value = dict(artifact(), source='fs')
            value['payload'] = dict(value['payload'], recommended_action='none')
            return value
        self.ml._artifact_record = record
        self.ml._latest_status_payload = lambda *a: dict(
            timestamp='2026-10-01T10:00:00Z', _diagnostic_source='data/pipeline_status.json',
            model_score_coverage=dict(pct=100), ml_health=dict(decision='allow'))
        result = self.overview([])
        self.assert_execution_unknown(result)
        self.assertIsNone(result['monitor']['recommended_action'])
        self.assertEqual(result['monitor']['guard_decision'], 'allow')
        self.assertEqual(result['monitor']['guard_evidence']['scope'], 'historical_diagnostic')
        self.assertEqual(result['coverage']['evidence']['source'], 'data/pipeline_status.json')
        self.assertEqual(result['coverage']['evidence']['timestamp'], '2026-10-01T10:00:00+00:00')
        # Preserve conservative native guard holds, without inventing execution.
        self.ml._latest_status_payload = lambda *a: dict(ml_health=dict(decision='block'))
        held = self.overview([])
        self.assertEqual(held['status']['label'], 'Guard held')
        self.assertIsNone(held['last_successful_execution'])
        self.assertEqual(held['fresh_success_count'], 0)


if __name__ == '__main__':
    unittest.main()
