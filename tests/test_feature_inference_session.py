"""Offline regressions for current-session inputs with immature outcomes."""
import ast
from datetime import date, datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import scripts

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'label_5d_pos_300bp'


class FeatureSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = ModuleType('scripts.db')
        self.db.db_enabled = Mock(return_value=True)
        self.db.upsert_ml_artifact_frame = Mock(return_value=True)
        spec = importlib.util.spec_from_file_location('inert_features', ROOT / 'scripts/feature_generator.py')
        self.generator = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'scripts.db': self.db}), patch.object(scripts, 'db', self.db, create=True):
            spec.loader.exec_module(self.generator)
        dates = pd.bdate_range('2026-09-01', '2026-10-09', tz='UTC')
        self.bars = pd.DataFrame({'symbol': 'AAA', 'timestamp': dates,
            'open': 100. + np.arange(len(dates)), 'high': 102. + np.arange(len(dates)),
            'low': 99. + np.arange(len(dates)), 'close': 101. + np.arange(len(dates)),
            'volume': 1000. + np.arange(len(dates))})
        self.labels = self.bars[['symbol', 'timestamp']].assign(
            label_5d_pos_300bp=1., label_10d_pos_300bp=0.)
        self.labels.loc[self.labels.index[-10:], ['label_5d_pos_300bp', 'label_10d_pos_300bp']] = np.nan
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('external_execution_forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def build(self, labels=None):
        with patch.object(self.generator, '_prepare_bars', return_value=self.bars), \
                patch.object(self.generator, '_load_labels', return_value=(self.labels if labels is None else labels, {})):
            return self.generator.build_feature_set(Path('inert.csv'), None, keep_na=False, feature_set='v1')

    def test_pending_outcomes_retain_current_features_without_fabricated_labels(self):
        frame, bars_day, _, _ = self.build()
        self.assertEqual(bars_day, date(2026, 10, 9))
        self.assertEqual(frame.timestamp.max().date(), bars_day)
        self.assertTrue(pd.isna(frame.iloc[-1][TARGET]))
        self.assertFalse(frame[['mom_5d', 'mom_10d', 'vol_10d', 'vol_raw', 'vol_avg_10d', 'vol_rvol_10d']].isna().any().any())

    def test_absent_current_labels_do_not_remove_current_bars(self):
        frame, _, _, _ = self.build(self.labels.iloc[:-10])
        self.assertEqual(frame.timestamp.max().date(), date(2026, 10, 9))
        self.assertTrue(pd.isna(frame.iloc[-1][TARGET]))

    def test_duplicate_label_identity_is_rejected(self):
        with self.assertRaises(pd.errors.MergeError):
            self.build(pd.concat([self.labels, self.labels.iloc[-1:]]))

    def run_generator(self, requested=None, build=None):
        frame, bars_day, labels_day, metadata = self.build() if build is None else build
        argv = ['feature_generator', '--output-dir', str(self.root), '--feature-set', 'v1']
        if requested:
            argv.extend(['--run-date', requested])
        with patch.object(sys, 'argv', argv), \
                patch.object(self.generator, 'build_feature_set', return_value=(frame, bars_day, labels_day, metadata)), \
                patch.object(self.generator, '_current_run_date', side_effect=AssertionError('wall_date_forbidden')):
            self.generator.main()

    def test_weekend_generation_publishes_actual_session_and_pending_count(self):
        self.run_generator('2026-10-09')
        self.assertTrue((self.root / 'features_2026-10-09.csv').exists())
        self.assertEqual(self.db.upsert_ml_artifact_frame.call_args.args[1], date(2026, 10, 9))
        payload = json.loads((self.root / 'latest_meta.json').read_text())
        self.assertEqual(payload['signal_session'], '2026-10-09')
        self.assertEqual(payload['outcomes_pending'], 10)

    def test_default_date_is_input_session_without_wall_clock(self):
        self.run_generator()
        self.assertEqual(self.db.upsert_ml_artifact_frame.call_args.args[1], date(2026, 10, 9))

    def test_unavailable_requested_session_fails_before_writes(self):
        with self.assertRaises(SystemExit):
            self.run_generator('2026-10-12')
        self.db.upsert_ml_artifact_frame.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_current_computed_features_fail_before_writes(self):
        frame, bars_day, labels_day, metadata = self.build()
        with self.assertRaises(SystemExit):
            self.run_generator(build=(frame.iloc[:-1], bars_day, labels_day, metadata))
        self.db.upsert_ml_artifact_frame.assert_not_called()

    def test_training_still_discards_unknown_targets(self):
        # Execute the actual training normalizer with inert feature inference.
        tree = ast.parse((ROOT / 'scripts/ranker_train.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_features_frame')
        namespace = {'pd': pd, 'np': np, 'ID_COLUMNS': {'symbol', 'timestamp'},
            'DROP_FEATURE_COLUMNS': set(), 'infer_feature_columns_for_ml': lambda *a, **k: ['mom_5d']}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<training_normalizer>', 'exec'), namespace)
        frame, *_ = self.build()
        training, _ = namespace['_normalize_features_frame'](frame, TARGET)
        self.assertFalse(training[TARGET].isna().any())
        self.assertLess(training.timestamp.max(), frame.timestamp.max())

    def test_pipeline_refreshes_legacy_or_wrong_session_features(self):
        tree = ast.parse((ROOT / 'scripts/run_pipeline.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_ensure_features_freshness')
        model = {'model_path': 'fixture.pkl', 'feature_set': 'v1', 'feature_signature': 'fixture'}
        for session, stale in ((None, True), ('2026-10-09', True), ('2026-10-12', False)):
            with self.subTest(session=session):
                meta = {'feature_set': 'v1', 'feature_signature': 'fixture', 'feature_columns': ['x']}
                if session is not None:
                    meta['signal_session'] = session
                namespace = {'_latest_model_meta': lambda: model, '_load_features_meta': lambda: (meta, 'fixture'),
                    'pipeline_run_date': date(2026, 10, 12), 'compute_feature_signature': lambda _: 'fixture',
                    'LOG': Mock(), 'Any': Any}
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<pipeline_freshness>', 'exec'), namespace)
                self.assertEqual(namespace['_ensure_features_freshness']('ranker_eval')['stale'], stale)


if __name__ == '__main__':
    unittest.main()
