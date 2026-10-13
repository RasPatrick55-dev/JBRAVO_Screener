"""Inert session-label and forward score-history regressions. No DB or model loading."""
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch, Mock

import pandas as pd
import scripts
from scripts import production_score_capture as capture

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('external_execution_forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = pd.DataFrame({'symbol': ['A', 'A', 'B'],
            'timestamp': ['2026-10-08T20:00:00Z', '2026-10-09T20:00:00Z', '2026-10-09T20:00:00Z'],
            'x': [1., 2., 3.]})
        self.scores = self.features[['symbol', 'timestamp']].assign(score_5d=[.1, .2, .3])
        self.identity = capture.scoring_binding({'actual_model': 1}, {'scaler': 2}, ['x'], 'label_5d_pos_300bp', None)

    def save(self, **kwargs):
        return capture.capture(self.root, self.features, self.scores,
            snapshot_date='2026-10-09', model_binding=self.identity,
            generated_at=datetime(2026, 10, 10, 1, tzinfo=timezone.utc), **kwargs)

    def test_current_session_only_complete_bytes_no_outcomes_or_acceptance(self):
        result = self.save()
        path = Path(result['manifest_path'])
        manifest = json.loads(path.read_bytes())
        self.assertEqual(manifest['rows'], 2)
        self.assertFalse(manifest['outcomes_attached'])
        self.assertFalse(manifest['operating_acceptance'])
        self.assertFalse(manifest['research_qualified'])
        for filename, field in (('features.csv', 'features'), ('predictions.csv', 'predictions')):
            raw = (path.parent / filename).read_bytes()
            self.assertEqual(capture.binding(raw), manifest[field])
            self.assertNotIn(b'2026-10-08', raw)
            self.assertNotIn(b'label_', raw)

    def test_capture_never_overwrites_an_earlier_observation(self):
        first = self.save()
        raw = Path(first['manifest_path']).read_bytes()
        second = self.save()
        self.assertNotEqual(first['manifest_path'], second['manifest_path'])
        self.assertEqual(Path(first['manifest_path']).read_bytes(), raw)

    def test_actual_model_scaler_features_target_and_calibrator_bind_identity(self):
        base = self.identity['sha256']
        for args in (({'actual_model': 9}, {'scaler': 2}, ['x'], 'label_5d_pos_300bp', None),
                     ({'actual_model': 1}, {'scaler': 9}, ['x'], 'label_5d_pos_300bp', None),
                     ({'actual_model': 1}, {'scaler': 2}, ['y'], 'label_5d_pos_300bp', None),
                     ({'actual_model': 1}, {'scaler': 2}, ['x'], 'other_target', None),
                     ({'actual_model': 1}, {'scaler': 2}, ['x'], 'label_5d_pos_300bp', {'calibrator': 1})):
            self.assertNotEqual(capture.scoring_binding(*args)['sha256'], base)

    def test_missing_or_duplicate_or_mismatched_scores_do_not_create_a_packet(self):
        for replacement in (self.scores.iloc[:1], pd.concat([self.scores, self.scores.iloc[[1]]]),
                            self.scores.assign(symbol=['A', 'A', 'C'])):
            with self.subTest(), patch.object(self, 'scores', replacement), self.assertRaises(ValueError):
                self.save()
        self.assertFalse((self.root / 'data').exists())

    def test_invalid_scores_and_features_are_rejected_before_writes(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with patch.object(self, 'scores', self.scores.assign(score_5d=[.1, value, .3])), self.assertRaises(ValueError):
                self.save()
        with patch.object(self, 'features', self.features.assign(x=[1., float('inf'), 3.])), self.assertRaises(ValueError):
            self.save()
        self.assertFalse((self.root / 'data').exists())

    def test_reused_predictions_cannot_be_relabelled_as_a_new_session(self):
        with self.assertRaisesRegex(ValueError, 'population_ineligible'):
            capture.capture(self.root, self.features, self.scores, snapshot_date='2026-10-10',
                model_binding=self.identity, generated_at=datetime(2026, 10, 10, 1, tzinfo=timezone.utc))

    def test_future_same_session_features_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'future_feature_timestamp'):
            capture.capture(self.root, self.features, self.scores, snapshot_date='2026-10-09',
                model_binding=self.identity, generated_at=datetime(2026, 10, 9, 1, tzinfo=timezone.utc))

    def test_both_pipeline_label_commands_receive_the_canonical_session(self):
        tree = ast.parse((ROOT / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        commands = [node for node in ast.walk(tree) if isinstance(node, ast.List)
                    and any(isinstance(v, ast.Constant) and v.value == 'scripts.label_generator' for v in node.elts)]
        self.assertEqual(len(commands), 2)
        for node in commands:
            text = ast.unparse(node)
            self.assertIn("'--run-date', pipeline_run_date.isoformat()", text)

    def test_weekend_label_generation_publishes_signal_session(self):
        db = ModuleType('scripts.db')
        db.db_enabled = Mock(return_value=True)
        db.upsert_ml_artifact_frame = Mock(return_value=True)
        utils = ModuleType('scripts.utils')
        utils.__path__ = [str(ROOT / 'scripts/utils')]
        with patch.dict(sys.modules, {'scripts.db': db, 'scripts.utils': utils}), patch.object(scripts, 'db', db, create=True):
            labels = importlib.import_module('scripts.label_generator')
        bars = pd.DataFrame({'symbol': ['A'] * 15, 'timestamp': pd.date_range('2026-09-25', periods=15, tz='UTC'),
                             'close': [100. + i for i in range(15)]})
        argv = ['label_generator', '--bars-path', 'inert.csv', '--output-dir', str(self.root), '--run-date', '2026-10-09']
        with patch.object(sys, 'argv', argv), patch.object(labels, 'load_bars', return_value=bars), \
                patch.object(labels, '_current_run_date', side_effect=AssertionError('wall_date_forbidden')):
            self.assertEqual(labels.main(), 0)
        self.assertEqual(str(db.upsert_ml_artifact_frame.call_args.args[1]), '2026-10-09')
        self.assertTrue((self.root / 'labels_20261009.csv').exists())

    def run_predictor(self, mutate=False, fail_capture=False, pending_labels=False):
        db = ModuleType('scripts.db')
        db.db_enabled = Mock(return_value=False)
        train = ModuleType('scripts.ranker_train')
        train.FEATURE_COLUMNS = ['x']
        train._extract_features_date = lambda path: '2026-10-09'
        def predict(model, inputs):
            if mutate:
                model['weight'] = 99
            return inputs['x'].to_numpy() / 10.
        train._predict_proba = predict
        env = ModuleType('utils.env')
        env.load_env = Mock()
        schema = ModuleType('scripts.utils.feature_schema')
        schema.compute_feature_signature = lambda names: 'fixture-signature'
        schema.load_features_meta_for_path = lambda *a, **k: ({'feature_set': 'v1', 'feature_signature': 'fixture-signature'}, 'fixture')
        schema.meta_matches_features_path = lambda *a: True
        selection = ModuleType('scripts.utils.model_selection')
        selection.select_compatible_model = lambda *a, **k: (self.root / 'trusted.pkl', {'selected_model_meta': {}, 'selection_mode': 'fixture'})
        selection.summary_path_for_model = lambda path: None
        features_path = self.root / 'features_2026-10-09.csv'
        self.features.assign(label_5d_pos_300bp=float('nan') if pending_labels else 0, close=100.).to_csv(features_path, index=False)
        spec = importlib.util.spec_from_file_location('inert_predictor', ROOT / 'scripts/ranker_predict.py')
        module = importlib.util.module_from_spec(spec)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'scripts.db': db, 'scripts.ranker_train': train,
                'utils.env': env, 'scripts.utils.feature_schema': schema, 'scripts.utils.model_selection': selection}))
            stack.enter_context(patch.object(scripts, 'db', db, create=True))
            stack.enter_context(patch.dict(os.environ, {'JBR_CAPTURE_PRODUCTION_SCORES': 'true', 'JBR_PREDICT_SYMBOLS_PATH': ''}))
            spec.loader.exec_module(module)
            stack.enter_context(patch.object(module, 'BASE_DIR', self.root))
            stack.enter_context(patch.object(module, '_load_model', return_value=({'weight': 1}, None, ['x'], 'v1', 'fixture-signature', False, 'none', None)))
            if fail_capture:
                stack.enter_context(patch.object(capture, 'save', side_effect=OSError('fixture write failure')))
            return module.main(['--features-path', str(features_path), '--output-dir', str(self.root / 'predictions')])

    def test_actual_predictor_preserves_scores_and_references_saved_capture(self):
        self.assertEqual(self.run_predictor(), 0)
        output = pd.read_csv(self.root / 'predictions/predictions_2026-10-09.csv')
        self.assertEqual(list(output['score_5d']), [.1, .2, .3])
        meta = json.loads((self.root / 'predictions/latest_meta.json').read_bytes())
        proof = meta['production_score_capture']
        self.assertEqual(capture.binding(Path(proof['manifest_path']).read_bytes()), proof['manifest_binding'])
        self.assertEqual(proof['rows'], 2)

    def test_current_prediction_and_capture_do_not_require_future_labels(self):
        self.assertEqual(self.run_predictor(pending_labels=True), 0)
        output = pd.read_csv(self.root / 'predictions/predictions_2026-10-09.csv')
        self.assertEqual(list(output['score_5d']), [.1, .2, .3])
        self.assertTrue(output['label_5d_pos_300bp'].isna().all())
        meta = json.loads((self.root / 'predictions/latest_meta.json').read_bytes())
        proof = meta['production_score_capture']
        features = pd.read_csv(Path(proof['manifest_path']).parent / 'features.csv')
        self.assertNotIn('label_5d_pos_300bp', features)
        self.assertEqual(proof['rows'], 2)

    def test_capture_errors_export_only_allowlisted_reasons(self):
        self.assertEqual(capture.failure_code(ValueError('capture_population_ineligible')), 'capture_population_ineligible')
        self.assertEqual(capture.failure_code(OSError('private-example-value')), 'capture_io_failure')
        self.assertEqual(capture.failure_code(ValueError('private-example-value')), 'capture_details_withheld')

    def test_feature_refresh_commands_bind_pipeline_session(self):
        tree = ast.parse((ROOT / 'scripts/run_pipeline.py').read_text())
        commands = [node for node in ast.walk(tree) if isinstance(node, ast.List)
                    and any(isinstance(v, ast.Constant) and v.value == 'scripts.feature_generator' for v in node.elts)]
        self.assertEqual(len(commands), 2)
        for node in commands:
            self.assertIn("'--run-date', pipeline_run_date.isoformat()", ast.unparse(node))

    def test_mutated_bundle_or_failed_capture_prevents_prediction_publication(self):
        for mutate, fail in ((True, False), (False, True)):
            with self.subTest(mutate=mutate, fail=fail):
                self.assertEqual(self.run_predictor(mutate, fail), 2)
                self.assertFalse((self.root / 'predictions').exists())

    def test_changed_producer_prevents_prediction_publication(self):
        original = capture.producer_binding()
        changed = {key: dict(value) for key, value in original.items()}
        changed['scripts/ranker_predict.py']['sha256'] = 'a' * 64
        with patch.object(capture, 'producer_binding', side_effect=[original, changed]):
            self.assertEqual(self.run_predictor(), 2)
        self.assertFalse((self.root / 'predictions').exists())


if __name__ == '__main__':
    unittest.main()
