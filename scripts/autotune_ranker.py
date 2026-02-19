from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - support ``python scripts/...`` execution
    from .eval_ranker import (
        EvaluationConfig,
        _series_from_breakdown,
        label_predictions,
        load_prediction_history,
        load_price_history,
        roc_auc_score,
    )
    from .ranking import DEFAULT_WEIGHTS
except Exception:  # pragma: no cover - fallback when executed as a script
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    from scripts.eval_ranker import (  # type: ignore
        EvaluationConfig,
        _series_from_breakdown,
        label_predictions,
        load_prediction_history,
        load_price_history,
        roc_auc_score,
    )
    from scripts.ranking import DEFAULT_WEIGHTS  # type: ignore

from utils.io_utils import atomic_write_bytes

TREND_COMPONENTS = {"trend", "momentum", "breakout", "pivot_trend", "multi_horizon"}


def _load_ranker_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _extract_weights(config: Dict[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {key: float(value) for key, value in DEFAULT_WEIGHTS.items()}
    provided = config.get("weights", {}) if isinstance(config, dict) else {}
    if isinstance(provided, dict):
        for component, value in provided.items():
            try:
                result[component] = float(value)
            except (TypeError, ValueError):
                continue
    return result


def _resolve_components(weights: Dict[str, float]) -> List[str]:
    if weights:
        return list(weights.keys())
    return list(DEFAULT_WEIGHTS.keys())


def _fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    samples, features = X.shape
    weights = np.zeros(features + 1, dtype=float)
    for _ in range(max_iter):
        logits = weights[0] + X @ weights[1:]
        preds = 1.0 / (1.0 + np.exp(-logits))
        error = preds - y

        grad_bias = error.mean()
        grad_weights = (X.T @ error) / samples + l2 * weights[1:]

        W = preds * (1.0 - preds)
        H = (X.T * W) @ X / samples
        H += l2 * np.eye(features)

        try:
            delta_weights = np.linalg.solve(H, grad_weights)
        except np.linalg.LinAlgError:
            delta_weights = np.linalg.lstsq(H, grad_weights, rcond=None)[0]

        h_bias = W.sum() / samples
        if h_bias <= 0:
            h_bias = 1e-6
        delta_bias = grad_bias / h_bias

        update_norm = math.sqrt(float(delta_bias**2 + np.dot(delta_weights, delta_weights)))
        weights[0] -= delta_bias
        weights[1:] -= delta_weights

        if update_norm < tol:
            break
    return weights


def _apply_constraints(
    old: np.ndarray,
    proposed: np.ndarray,
    components: Sequence[str],
    max_change: float,
) -> np.ndarray:
    constrained = old.copy()
    for idx, component in enumerate(components):
        old_value = float(old[idx])
        target = float(proposed[idx])
        if old_value == 0.0:
            new_value = 0.0
        else:
            delta = max_change * abs(old_value)
            lower = old_value - delta
            upper = old_value + delta
            new_value = float(np.clip(target, lower, upper))
        if component in TREND_COMPONENTS and old_value != 0.0:
            if old_value > 0:
                new_value = abs(new_value) if new_value != 0.0 else abs(old_value)
            else:
                new_value = -abs(new_value) if new_value != 0.0 else -abs(old_value)
        constrained[idx] = new_value
    return constrained


def _prepare_feature_matrix(
    labelled: pd.DataFrame,
    components: Sequence[str],
) -> pd.DataFrame:
    if "score_breakdown" not in labelled.columns:
        raise ValueError(
            "score_breakdown column missing; nightly pipeline must export breakdown JSON"
        )
    features = _series_from_breakdown(labelled["score_breakdown"], components)
    features.index = labelled.index
    return features


def _ensure_date_series(values: Iterable[object]) -> pd.Series:
    parsed = []
    for value in values:
        if isinstance(value, date):
            parsed.append(value)
        elif isinstance(value, pd.Timestamp):
            parsed.append(value.date())
        elif isinstance(value, str):
            try:
                parsed.append(datetime.strptime(value, "%Y-%m-%d").date())
            except ValueError:
                parsed.append(pd.NaT)
        else:
            parsed.append(pd.NaT)
    return pd.Series(parsed)


def _write_state(path: Path, payload: Dict[str, object]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Autotune ranker weights via logistic regression")
    parser.add_argument(
        "--train-days", type=int, default=120, help="Number of prediction days to use for training"
    )
    parser.add_argument(
        "--label-horizon", type=int, default=3, help="Forward days for label computation"
    )
    parser.add_argument(
        "--hit-threshold", type=float, default=0.04, help="Hit threshold for label computation"
    )
    parser.add_argument(
        "--drawdown-threshold",
        type=float,
        default=None,
        help="Drawdown threshold for labels (defaults to hit threshold)",
    )
    parser.add_argument("--predictions-dir", type=Path, default=EvaluationConfig.predictions_dir)
    parser.add_argument("--prices", type=Path, default=EvaluationConfig.prices_path)
    parser.add_argument("--config", type=Path, default=Path("config") / "ranker.yml")
    parser.add_argument("--state-path", type=Path, default=Path("data") / "model_state.json")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Minimum AUC improvement required to accept weights",
    )
    parser.add_argument(
        "--min-sample", type=int, default=200, help="Minimum validation rows required"
    )
    parser.add_argument(
        "--l2", type=float, default=1.0, help="L2 regularisation strength for logistic fit"
    )
    parser.add_argument(
        "--max-change", type=float, default=0.2, help="Maximum proportional weight adjustment"
    )
    parser.add_argument("--as-of", type=str, default=None, help="Optional cutoff date (YYYY-MM-DD)")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the most recent samples reserved for validation",
    )
    parser.add_argument(
        "--gated-only",
        action="store_true",
        help="Only use rows that passed nightly gates when fitting",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None

    cfg = EvaluationConfig(
        days=max(1, args.train_days),
        label_horizon=max(1, args.label_horizon),
        hit_threshold=float(args.hit_threshold),
        drawdown_threshold=args.drawdown_threshold,
        predictions_dir=args.predictions_dir,
        prices_path=args.prices,
        as_of=as_of,
    )

    ranker_cfg = _load_ranker_config(args.config)
    weights_map = _extract_weights(ranker_cfg)
    components = _resolve_components(weights_map)
    weight_vector = np.array(
        [float(weights_map.get(component, 0.0)) for component in components], dtype=float
    )

    predictions = load_prediction_history(cfg)
    prices = load_price_history(cfg.prices_path)
    labelled = label_predictions(predictions, prices, cfg)
    if args.gated_only and "gates_passed" in labelled.columns:
        labelled = labelled[labelled["gates_passed"] == True]  # noqa: E712

    labelled = labelled.dropna(subset=["label"])
    if labelled.empty:
        _write_state(
            args.state_path,
            {
                "accepted": False,
                "reason": "no_samples",
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 0

    features = _prepare_feature_matrix(labelled, components)
    labels = labelled["label"].astype(float).to_numpy()

    labelled_dates = _ensure_date_series(labelled.get("as_of"))
    labelled["as_of_date"] = labelled_dates
    valid_mask = labelled["as_of_date"].notna()
    labelled = labelled[valid_mask]
    features = features.loc[labelled.index]
    labels = labels[valid_mask.to_numpy(dtype=bool)]

    unique_dates = sorted(set(labelled["as_of_date"]))
    if not unique_dates:
        _write_state(
            args.state_path,
            {
                "accepted": False,
                "reason": "no_dates",
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 0

    validation_fraction = float(args.validation_fraction)
    if not math.isfinite(validation_fraction) or validation_fraction <= 0.0:
        validation_fraction = 0.2
    elif validation_fraction >= 1.0:
        validation_fraction = 0.5

    cutoff_idx = max(1, int(math.ceil(len(unique_dates) * (1.0 - validation_fraction))))
    train_dates = set(unique_dates[:cutoff_idx])
    val_dates = set(unique_dates[cutoff_idx:]) or {unique_dates[-1]}

    train_mask = labelled["as_of_date"].isin(train_dates)
    val_mask = labelled["as_of_date"].isin(val_dates)

    X_train = features.loc[train_mask].to_numpy(dtype=float)
    y_train = labels[train_mask.to_numpy(dtype=bool)]
    X_val = features.loc[val_mask].to_numpy(dtype=float)
    y_val = labels[val_mask.to_numpy(dtype=bool)]

    if len(y_train) == 0 or len(y_val) == 0:
        _write_state(
            args.state_path,
            {
                "accepted": False,
                "reason": "insufficient_split",
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 0

    base_auc = roc_auc_score(y_val, X_val @ weight_vector)
    if base_auc is None:
        _write_state(
            args.state_path,
            {
                "accepted": False,
                "reason": "baseline_auc_undefined",
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 0

    logistic_weights = _fit_logistic_l2(X_train, y_train, l2=float(args.l2))
    raw_coeffs = logistic_weights[1:]
    if np.allclose(raw_coeffs, 0.0):
        _write_state(
            args.state_path,
            {
                "accepted": False,
                "reason": "degenerate_fit",
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 0

    scale = np.sum(np.abs(weight_vector))
    denom = np.sum(np.abs(raw_coeffs))
    if denom == 0:
        scaled = raw_coeffs
    else:
        scaled = raw_coeffs * (scale / denom if scale > 0 else 1.0)

    constrained = _apply_constraints(weight_vector, scaled, components, float(args.max_change))
    new_auc = roc_auc_score(y_val, X_val @ constrained)

    accepted = False
    reason = "no_improvement"
    improvement = None
    if new_auc is not None:
        improvement = float(new_auc - base_auc)
        if len(y_val) >= int(args.min_sample) and improvement >= float(args.delta):
            accepted = True
            reason = "accepted"
        else:
            reason = "threshold_not_met" if improvement is not None else "auc_undefined"
    else:
        reason = "auc_undefined"

    state_payload: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "as_of": (as_of or date.today()).isoformat(),
        "train_days": int(args.train_days),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "baseline_auc": base_auc,
        "candidate_auc": new_auc,
        "auc_improvement": improvement,
        "delta_threshold": float(args.delta),
        "min_sample": int(args.min_sample),
        "accepted": accepted,
        "reason": reason,
        "weights": {
            "old": {
                component: float(weight_vector[idx]) for idx, component in enumerate(components)
            },
            "proposed": {
                component: float(constrained[idx]) for idx, component in enumerate(components)
            },
            "raw_fit": {component: float(scaled[idx]) for idx, component in enumerate(components)},
        },
        "logistic": {
            "bias": float(logistic_weights[0]),
            "raw_coefficients": {
                component: float(raw_coeffs[idx]) for idx, component in enumerate(components)
            },
        },
        "splits": {
            "train_dates": sorted(d.isoformat() for d in train_dates),
            "validation_dates": sorted(d.isoformat() for d in val_dates),
        },
        "config_path": str(args.config),
        "predictions_dir": str(args.predictions_dir),
        "prices_path": str(args.prices),
        "gated_only": bool(args.gated_only),
        "label_horizon": int(args.label_horizon),
        "hit_threshold": float(args.hit_threshold),
        "drawdown_threshold": float(cfg.resolved_drawdown),
        "validation_fraction": validation_fraction,
    }

    if accepted:
        state_payload["preview_weights"] = state_payload["weights"]["proposed"]

    _write_state(args.state_path, state_payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
