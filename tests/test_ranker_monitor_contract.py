"""Synthetic monitoring contracts; normal imports with inert external boundaries."""
from contextlib import ExitStack
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scripts

ROOT = Path(__file__).resolve().parents[1]
TARGET = "label_5d_pos_300bp"
RETURN = "fwd_ret_5d"


class FakeEstimator:
    def __init__(self, **kwargs):
        self.options = kwargs

    def fit(self, x, y):
        self.prevalence = float(y.mean())
        return self

    def predict_proba(self, x):
        return np.tile([1 - self.prevalence, self.prevalence], (len(x), 1))


class FakeScaler:
    def fit_transform(self, x):
        self.mean = list(x.mean())
        return np.array(x)

    def transform(self, x):
        return np.array(x)


class FakeCalibrator(FakeEstimator):
    pass


class MonitorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imports = ExitStack()
        cls.db = ModuleType("scripts.db")
        cls.db.db_enabled = mock.Mock(return_value=False)
        cls.db.fetch_ml_artifact = mock.Mock()
        cls.db.fetch_latest_ml_artifact = mock.Mock()
        cls.db.upsert_ml_artifact = mock.Mock(return_value=True)
        cls.db.upsert_ml_artifact_frame = mock.Mock(return_value=True)
        utils = ModuleType("scripts.utils")
        utils.__path__ = [str(ROOT / "scripts/utils")]
        env = ModuleType("utils.env")
        env.load_env = mock.Mock()
        predictor = ModuleType("scripts.ranker_predict")
        predictor._find_latest = mock.Mock(return_value=None)
        predictor._mtime_iso = mock.Mock(return_value=None)
        cls.imports.enter_context(mock.patch.dict(sys.modules, {
            "scripts.db": cls.db, "scripts.utils": utils, "utils.env": env,
            "scripts.ranker_predict": predictor,
        }))
        cls.imports.enter_context(mock.patch.object(scripts, "db", cls.db, create=True))
        cls.imports.enter_context(mock.patch.object(scripts, "utils", utils, create=True))
        for target in ("socket.socket.connect", "socket.create_connection",
                       "subprocess.Popen", "subprocess.run"):
            cls.imports.enter_context(mock.patch(target, side_effect=AssertionError("external_forbidden")))
        cls.monitor = importlib.import_module("scripts.ranker_monitor")
        cls.walkforward = importlib.import_module("scripts.ranker_walkforward")
        cls.guard = importlib.import_module("scripts.utils.ml_health_guard")
        cls.auto = importlib.import_module("scripts.ranker_autoremediate")

    @classmethod
    def tearDownClass(cls):
        cls.imports.close()

    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.db.db_enabled.return_value = False
        self.db.upsert_ml_artifact.reset_mock()

    def calibration(self, frame, **kwargs):
        return self.monitor._compute_calibration_window(
            frame, label_col=TARGET, score_col="score_oos", fwd_ret_col=RETURN,
            bins=10, min_rows=kwargs.get("min_rows", 1),
        )

    def binary(self, baseline, recent):
        return self.monitor._compute_binary_psi(pd.Series(baseline), pd.Series(recent),
                                                warn=0.1, alert=0.25)

    def population(self, incomplete=0):
        rows = []
        for day, labels, score, fold, digest in (
            ("2026-01-01", [0] * 80 + [1] * 20, 0.2, 1, "a" * 64),
            ("2026-04-01", [0] * 20 + [1] * 80, 0.4, 2, "b" * 64),
        ):
            for index, label in enumerate(labels):
                rows.append({"symbol": f"S{index}", "timestamp": day, "close": 10,
                             TARGET: label, RETURN: 0.1 if label else -0.1,
                             "score_oos": score, "fold_id": fold,
                             "model_sha256": digest, "model_binding_bytes": 128,
                             "model_binding_kind": "scoring_bundle_pickle_sha256_v2",
                             "model_score_column": "score_oos", "model_target": TARGET,
                             "model_role": "walkforward_fold"})
        for index in range(incomplete):
            rows.append({**rows[-1], "symbol": f"U{index}", TARGET: 0, RETURN: np.nan})
        return pd.DataFrame(rows)

    def run_cli(self, frame):
        path = self.root / "input.csv"
        frame.to_csv(path, index=False)
        args = ["--input-path", str(path), "--output-dir", str(self.root / "monitor"),
                "--baseline-days", "10", "--recent-days", "5",
                "--calibration-min-rows", "10", "--run-date", "2026-04-01"]
        with mock.patch.object(self.monitor, "_run_strategy_metrics", return_value={"sharpe": 1.0}), \
             mock.patch.object(self.monitor, "load_latest_champion", return_value=None):
            self.assertEqual(self.monitor.main(args), 0)
        saved = (self.root / "monitor/latest.json").read_text(encoding="utf-8")
        result = json.loads(saved, parse_constant=lambda value: self.fail(f"nonfinite JSON {value}"))
        content = path.read_bytes()
        self.assertEqual(result["input_binding"]["bytes"], len(content))
        self.assertEqual(result["input_binding"]["sha256"], hashlib.sha256(content).hexdigest())
        self.db.upsert_ml_artifact.assert_not_called()
        return result

    def test_binary_frequency_change_cannot_collapse(self):
        result = self.binary([0] * 80 + [1] * 20, [0] * 20 + [1] * 80)
        expected = 0.6 * np.log(4) + (-0.6) * np.log(0.25)
        self.assertAlmostEqual(result["psi"], expected)
        self.assertEqual(result["bins_used"], 2)
        self.assertEqual(result["baseline"]["counts"], [80, 20])
        self.assertEqual(result["recent"]["counts"], [20, 80])
        self.assertAlmostEqual(result["prevalence_delta"], 0.6)

    def test_binary_identical_and_constant_populations(self):
        self.assertEqual(self.binary([0, 1], [0, 1])["psi"], 0)
        self.assertEqual(self.binary([0] * 10, [0] * 3)["psi"], 0)
        result = self.binary([0] * 10, [1] * 10)
        self.assertTrue(np.isfinite(result["psi"]))
        self.assertGreater(result["psi"], 20)

    def test_invalid_or_empty_labels_are_insufficient(self):
        for values in ([0, 2], [0, np.nan], [0, np.inf], [0, "bad"], []):
            with self.subTest(values=values):
                self.assertIsNone(self.binary([0, 1], values)["psi"])
                self.assertEqual(self.binary([0, 1], values)["level"], "insufficient")

    def test_calibration_excludes_42_incomplete_outcomes(self):
        frame = self.population(incomplete=42).iloc[100:]
        result = self.calibration(frame)
        self.assertEqual(result["input_rows"], 142)
        self.assertEqual(result["rows"], 100)
        self.assertEqual(result["excluded_unmatured_rows"], 42)
        self.assertAlmostEqual(result["ece"], 0.4)
        self.assertEqual(sum(row["count"] for row in result["reliability_table"]), 100)

    def test_maturity_and_invalid_exclusions_are_disjoint(self):
        frame = pd.DataFrame({TARGET: [1, 0, 0, 2, 1, 1],
                              "score_oos": [0.8, 0.2, 0.2, 0.2, np.inf, np.nan],
                              RETURN: [0.1, np.inf, np.nan, 0.1, 0.1, 0.1]})
        result = self.calibration(frame)
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["excluded_unmatured_rows"], 2)
        self.assertEqual(result["excluded_invalid_label_rows"], 1)
        self.assertEqual(result["excluded_invalid_score_rows"], 2)
        self.assertAlmostEqual(result["ece"], 0.2)

    def test_missing_returns_never_fall_back_to_labels(self):
        result = self.calibration(pd.DataFrame({TARGET: [0], "score_oos": [0.2]}))
        self.assertFalse(result["applicable"])
        self.assertEqual(result["skip_reason"], "missing_forward_return")

    def test_only_incomplete_outcomes_and_minimum_mature_count(self):
        frame = pd.DataFrame({TARGET: [0, 1], "score_oos": [0.2, 0.8], RETURN: [np.nan, 0.1]})
        self.assertEqual(self.calibration(frame, min_rows=2)["skip_reason"], "insufficient_rows")
        frame[RETURN] = np.nan
        self.assertEqual(self.calibration(frame)["rows"], 0)

    def test_out_of_range_probability_still_rejected(self):
        result = self.calibration(pd.DataFrame({TARGET: [1], "score_oos": [1.2], RETURN: [0.1]}))
        self.assertFalse(result["applicable"])
        self.assertEqual(result["skip_reason"], "score_out_of_range")

    def test_normal_cli_reports_maturity_binary_psi_and_cross_model_warning(self):
        result = self.run_cli(self.population(incomplete=42))
        self.assertTrue(result["calibration_applicable"] is True)
        self.assertEqual(result["calibration"]["recent"]["rows"], 100)
        self.assertEqual(result["calibration"]["recent"]["excluded_unmatured_rows"], 42)
        self.assertEqual(result["outcome_maturity"]["incomplete_forward_return_rows"], 42)
        binary = result["drift"]["columns"][TARGET]
        self.assertEqual(binary["recent"]["counts"], [20, 80])
        self.assertGreater(binary["psi"], 1)
        context = result["model_comparison"]
        self.assertEqual(context["scope"], "cross_model_diagnostic")
        self.assertEqual(context["common_model_sha256"], [])
        self.assertFalse(context["production_drift_established"])
        self.assertEqual(result["recommended_action"], "investigate")
        self.assertEqual(result["diagnostic_recommended_action"], "retrain")

    def test_legacy_identity_gap_cannot_clear_warning_even_with_stable_scores(self):
        frame = self.population().drop(columns=["model_sha256"])
        frame["score_oos"] = 0.2
        frame.loc[100:, TARGET] = frame.loc[:99, TARGET].to_numpy()
        frame.loc[100:, RETURN] = frame.loc[:99, RETURN].to_numpy()
        result = self.run_cli(frame)
        self.assertEqual(result["model_comparison"]["scope"], "unknown_model_diagnostic")
        self.assertEqual(result["recommended_action"], "investigate")
        self.assertEqual(result["diagnostic_recommended_action"], "none")
        self.assertIn("score_model_identity_unbound", result["recommendation_reasons"])

    def test_model_identity_conflicts_within_and_across_windows(self):
        frame = self.population()
        frame.loc[0, "model_sha256"] = "c" * 64
        result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:])
        self.assertIn("conflicting_fold_model_identity", result["reasons"])
        frame = self.population()
        frame["fold_id"] = 1
        result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:])
        self.assertIn("conflicting_fold_model_identity_across_windows", result["reasons"])

    def test_invalid_identity_and_fake_production_role_are_not_promoted(self):
        for field, value in (("model_sha256", "not-a-hash"), ("model_binding_bytes", 0),
                             ("model_binding_bytes", 1.5), ("model_role", "production"),
                             ("model_binding_kind", "latest_file_name")):
            with self.subTest(field=field, value=value):
                frame = self.population()
                frame[field] = value
                result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:])
                self.assertEqual(result["scope"], "unknown_model_diagnostic")
                self.assertFalse(result["production_drift_established"])

    def test_nullable_binding_and_same_digest_with_conflicting_sizes(self):
        frame = self.population()
        frame["model_binding_kind"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:])
        self.assertEqual(result["windows"]["baseline"]["unbound_rows"], 100)
        frame = self.population()
        frame.loc[0, "model_binding_bytes"] = 129
        result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:])
        self.assertIn("conflicting_model_binding_size", result["reasons"])

    def test_binding_cannot_be_reused_for_another_score_or_target(self):
        frame = self.population()
        for kwargs in ({"score_col": "score_5d"}, {"target": "label_10d_pos_300bp"}):
            with self.subTest(kwargs=kwargs):
                result = self.monitor._model_comparison(frame.iloc[:100], frame.iloc[100:], **kwargs)
                self.assertEqual(result["scope"], "unknown_model_diagnostic")
                self.assertEqual(result["windows"]["baseline"]["unbound_rows"], 100)

    def test_legacy_v1_fingerprint_remains_unbound_and_investigation_only(self):
        frame = self.population()
        frame["model_binding_kind"] = "scoring_bundle_pickle_sha256_v1"
        result = self.run_cli(frame)
        context = result["model_comparison"]
        self.assertEqual(context["scope"], "unknown_model_diagnostic")
        for window in context["windows"].values():
            self.assertEqual(window["bound_rows"], 0)
            self.assertEqual(window["unbound_rows"], window["rows"])
        self.assertIn("score_model_identity_unbound", context["reasons"])
        self.assertFalse(context["production_drift_established"])
        self.assertEqual(result["recommended_action"], "investigate")

    def test_overlap_is_reported_not_silently_deduplicated(self):
        frame = self.population()
        baseline = pd.concat([frame.iloc[:100], frame.iloc[:100]], ignore_index=True)
        result = self.monitor._model_comparison(baseline, frame.iloc[100:])
        self.assertEqual(result["windows"]["baseline"]["rows"], 200)
        self.assertEqual(result["windows"]["baseline"]["unique_symbol_sessions"], 100)
        self.assertEqual(result["weighting"], "fold_rows_no_silent_deduplication")

    def test_fingerprint_binds_model_scaler_features_and_target(self):
        bind = self.walkforward._scoring_model_binding
        first = bind({"coefficient": 1}, {"mean": 0}, ["x", "y"], TARGET)
        self.assertEqual(first, bind({"coefficient": 1}, {"mean": 0}, ["x", "y"], TARGET))
        for model, scaler, features, target in (
            ({"coefficient": 2}, {"mean": 0}, ["x", "y"], TARGET),
            ({"coefficient": 1}, {"mean": 1}, ["x", "y"], TARGET),
            ({"coefficient": 1}, {"mean": 0}, ["y", "x"], TARGET),
            ({"coefficient": 1}, {"mean": 0}, ["x", "y"], "other_target"),
        ):
            self.assertNotEqual(first["model_sha256"], bind(model, scaler, features, target)["model_sha256"])

    def test_fingerprint_binds_scored_column_without_changing_scores(self):
        train = pd.DataFrame({"x": range(8), TARGET: [0] * 8})
        bindings, scores = [], []
        for column in ("score_oos", "other_score", "score_oos"):
            with mock.patch.object(self.walkforward, "OOS_SCORE_COL", column):
                sink = {}
                predictions, _, _ = self.walkforward._predict_fold_proba_retrained(
                    train, train.iloc[:2], ["x"], TARGET, "none", binding_sink=sink)
                bindings.append(sink)
                scores.append(predictions)
                self.assertEqual(sink["model_score_column"], column)
                self.assertEqual(sink["model_binding_kind"], "scoring_bundle_pickle_sha256_v2")
        self.assertNotEqual(bindings[0]["model_sha256"], bindings[1]["model_sha256"])
        self.assertEqual(bindings[0], bindings[2])
        for predictions in scores:
            np.testing.assert_array_equal(predictions, [0.0, 0.0])

    def test_binding_failure_does_not_assert_identity(self):
        result = self.walkforward._scoring_model_binding(lambda: None, None, ["x"], TARGET)
        self.assertIsNone(result["model_sha256"])
        self.assertEqual(result["model_binding_kind"], "unbound")

    def test_actual_scoring_function_binds_constant_uncalibrated_and_calibrated_models(self):
        for labels, calibrate, expected_type in (
            ([0] * 8, "none", "constant_baseline"),
            ([0, 1] * 4, "none", "logistic_regression"),
            ([0, 1] * 4, "sigmoid", "logistic_regression_calibrated"),
        ):
            with self.subTest(calibrate=calibrate, expected_type=expected_type), \
                 mock.patch.object(self.walkforward, "LogisticRegression", FakeEstimator), \
                 mock.patch.object(self.walkforward, "StandardScaler", FakeScaler), \
                 mock.patch.object(self.walkforward, "CalibratedClassifierCV", FakeCalibrator), \
                 mock.patch.object(self.walkforward, "SKLEARN_AVAILABLE", True):
                train = pd.DataFrame({"x": range(8), TARGET: labels})
                sink = {}
                scores, model_type, _ = self.walkforward._predict_fold_proba_retrained(
                    train, train.iloc[:2], ["x"], TARGET, calibrate, binding_sink=sink)
                np.testing.assert_array_equal(scores, [np.mean(labels)] * 2)
                self.assertEqual(model_type, expected_type)
                self.assertEqual(len(sink["model_sha256"]), 64)
                self.assertGreater(sink["model_binding_bytes"], 0)

    def test_walkforward_saved_csv_and_fold_report_share_actual_binding(self):
        frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=25, tz="UTC"),
                              "symbol": "S", "close": 10, TARGET: 0, RETURN: -0.01, "x": 1})
        args = self.walkforward.parse_args(["--output-dir", str(self.root / "folds"),
                                           "--train-window-days", "5", "--test-window-days", "4",
                                           "--step-days", "4", "--retrain-per-fold"])
        with mock.patch.object(self.walkforward, "_load_inputs", return_value=(frame, frame, frame, RETURN, None)):
            result = self.walkforward.run_walkforward(args)
        saved = pd.read_csv(self.root / "folds/oos_predictions.csv")
        for fold in result["folds"]:
            rows = saved.loc[saved["fold_id"] == fold["fold"]]
            self.assertEqual(set(rows["model_sha256"]), {fold["model_sha256"]})
            self.assertEqual(set(rows["model_binding_bytes"]), {fold["model_binding_bytes"]})
            self.assertEqual(fold["model_binding_kind"], "scoring_bundle_pickle_sha256_v2")
            self.assertEqual(set(rows["model_binding_kind"]), {fold["model_binding_kind"]})
            self.assertEqual(fold["model_score_column"], self.walkforward.OOS_SCORE_COL)
            self.assertEqual(set(rows["model_score_column"]), {fold["model_score_column"]})

    def test_database_csv_value_gets_exact_input_binding_without_live_database(self):
        value = self.population().to_csv(index=False)
        self.db.db_enabled.return_value = True
        self.db.fetch_ml_artifact.return_value = {"csv_data": value, "run_date": "2026-04-01"}
        frame, source, _ = self.monitor._load_oos_inputs(self.monitor.parse_args([]))
        self.assertEqual(frame.attrs["input_binding"]["sha256"], hashlib.sha256(value.encode()).hexdigest())
        self.assertEqual(source, "db://ml_artifacts/ranker_oos_predictions")

    def test_investigation_preserves_guard_warning_and_never_dispatches_remediation(self):
        result = self.run_cli(self.population())
        now = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
        health = self.guard._extract_health({**result, "run_utc": now.isoformat()})
        health.update(present=True, source="fs")
        decision = self.guard.decide_ml_enrichment(health, mode="warn", max_age_days=7,
                                                   pipeline_run_date=date(2026, 4, 1), now=now)
        self.assertEqual(decision["decision"], "warn")
        self.assertIn("action_investigate", decision["reasons"])
        args = self.auto.parse_args(["--output-dir", str(self.root / "auto"),
                                    "--run-date", "2026-04-01", "--refresh-predictions", "true",
                                    "--refresh-features", "true"])
        with mock.patch.object(self.auto, "load_latest_ml_health", return_value=health), \
             mock.patch.object(self.auto, "decide_ml_enrichment", return_value=decision), \
             mock.patch.object(self.auto, "_latest_model_identity", return_value={}), \
             mock.patch.object(self.auto, "_predictions_source_state", return_value={}), \
             mock.patch.object(self.auto, "_champion_pointer", return_value=None):
            payload = self.auto.run_autoremediate(args)
        self.assertFalse(payload["executed"])
        self.assertFalse(payload["autotune"]["requested"])
        self.assertFalse(payload["recalibrate"]["requested"])
        self.assertFalse(payload["repredict"]["executed"])
        self.assertFalse(payload["repredict"]["features_refresh"]["attempted"])
        self.assertEqual(payload["remediation_skipped_reason"], "investigation_required")


if __name__ == "__main__":
    unittest.main()
