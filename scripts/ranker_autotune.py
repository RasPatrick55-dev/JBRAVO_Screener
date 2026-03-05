"""Autotune OOS ranking strategy settings using walk-forward + strategy eval.

This script is evaluation-only (paper-mode safe). It does not alter trading
execution logic and does not auto-promote selected settings into execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.ranker_strategy_eval import StrategyArgs, run_strategy_eval  # noqa: E402
from scripts.ranker_walkforward import WalkforwardArgs, run_walkforward  # noqa: E402
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_autotune")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_TARGET = "label_5d_pos_300bp"
DEFAULT_FEATURE_SETS = "v1,v2"
DEFAULT_SPLIT_ADJUST_VALUES = "off,auto"
DEFAULT_BARS_ADJUSTMENTS = "raw,split"
DEFAULT_CALIBRATIONS = "none,sigmoid"
DEFAULT_TOPK_VALUES = "10,25,50"
DEFAULT_COST_BPS_VALUES = "0,5,10"
DEFAULT_MIN_SCORE_GRID = "0.0,0.55,0.6,0.65"
DEFAULT_HOLDOUT_DAYS = 252
DEFAULT_MIN_HOLDOUT_SHARPE = 0.3
DEFAULT_MIN_HOLDOUT_PERIODS = 30
DEFAULT_MAX_CAGR_CAP = 5.0
DEFAULT_MAX_ABS_PERIOD_RETURN_CAP = 1.0
VALID_FEATURE_SETS = {"v1", "v2"}
VALID_SPLIT_ADJUST_VALUES = {"off", "auto", "force"}
VALID_BARS_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
VALID_CALIBRATIONS = {"none", "sigmoid", "isotonic"}


@dataclass(frozen=True)
class TrialConfig:
    feature_set: str
    split_adjust: str
    bars_adjustment: str
    calibration: str
    top_k: int
    cost_bps: float
    min_score: float


@dataclass
class AutotuneArgs:
    target: str
    trials: int
    seed: int
    top_n: int
    min_sharpe: float
    max_drawdown_floor: float
    holdout_days: int
    min_holdout_sharpe: float
    min_holdout_periods: int
    max_cagr_cap: float
    max_abs_period_return_cap: float
    feature_sets: list[str]
    split_adjust_values: list[str]
    bars_adjustments: list[str]
    calibrations: list[str]
    top_k_values: list[int]
    cost_bps_values: list[float]
    min_score_values: list[float]
    train_window_days: int
    test_window_days: int
    step_days: int
    embargo_days: int | None
    max_abs_fwd_ret: float
    run_date: date | None
    bars_path: Path
    labels_output_dir: Path
    features_output_dir: Path
    output_dir: Path


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


def _parse_csv_strings(value: str) -> list[str]:
    out: list[str] = []
    for token in str(value or "").split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item not in out:
            out.append(item)
    return out


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value or "").split(","):
        item = token.strip()
        if not item:
            continue
        parsed = int(float(item))
        if parsed not in out:
            out.append(parsed)
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for token in str(value or "").split(","):
        item = token.strip()
        if not item:
            continue
        parsed = float(item)
        if parsed not in out:
            out.append(parsed)
    return out


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found under {directory}")
    return files[-1]


def _compute_window_dates(
    oos_path: Path,
    *,
    holdout_days: int,
) -> tuple[date | None, date | None]:
    if int(holdout_days) <= 0:
        return None, None
    if not oos_path.exists():
        raise FileNotFoundError(f"OOS predictions not found at {oos_path}")
    oos_df = pd.read_csv(oos_path, usecols=["timestamp"])
    if oos_df.empty:
        raise RuntimeError("OOS predictions are empty; cannot derive holdout window.")
    ts = pd.to_datetime(oos_df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise RuntimeError("OOS predictions contain no valid timestamps for holdout split.")
    max_ts = pd.Timestamp(ts.max())
    holdout_start_ts = max_ts - pd.Timedelta(days=max(int(holdout_days) - 1, 0))
    holdout_start_date = holdout_start_ts.date()
    tune_end_date = (holdout_start_ts - pd.Timedelta(days=1)).date()
    return holdout_start_date, tune_end_date


def _read_max_abs_period_return(equity_path: Path) -> float | None:
    if not equity_path.exists():
        return None
    try:
        eq = pd.read_csv(equity_path)
    except Exception:
        return None
    if eq.empty or "period_return" not in eq.columns:
        return None
    series = pd.to_numeric(eq["period_return"], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.abs().max())


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return value


def _objective(
    *,
    sharpe: float | None,
    cagr: float | None,
    max_dd: float | None,
    min_sharpe: float,
    max_drawdown_floor: float,
) -> tuple[bool, float, float, float, float]:
    sharpe_v = sharpe if sharpe is not None else float("-inf")
    cagr_v = cagr if cagr is not None else float("-inf")
    max_dd_v = max_dd if max_dd is not None else float("-inf")
    eligible = bool(sharpe is not None and sharpe >= min_sharpe and max_dd is not None and max_dd >= max_drawdown_floor)
    score = float("-inf")
    if eligible:
        # Primary objective: Sharpe, then CAGR. Drawdown contributes as a
        # penalty term (less negative is better).
        score = float((sharpe_v * 1000.0) + (cagr_v * 100.0) + (max_dd_v * 10.0))
    return eligible, score, sharpe_v, cagr_v, max_dd_v


def _run_cmd(cmd: list[str], *, env: dict[str, str], name: str) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        check=False,
    )
    if int(completed.returncode or 0) != 0:
        raise RuntimeError(f"{name} failed rc={completed.returncode} cmd={' '.join(cmd)}")


def _generate_labels_and_features(
    args: AutotuneArgs,
    *,
    trial_env: dict[str, str],
    feature_set: str,
) -> tuple[Path, Path]:
    args.labels_output_dir.mkdir(parents=True, exist_ok=True)
    args.features_output_dir.mkdir(parents=True, exist_ok=True)

    label_cmd = [
        sys.executable,
        "-m",
        "scripts.label_generator",
        "--bars-path",
        str(args.bars_path),
        "--output-dir",
        str(args.labels_output_dir),
    ]
    _run_cmd(label_cmd, env=trial_env, name="label_generator")
    labels_path = _latest_file(args.labels_output_dir, "labels_*.csv")

    feature_cmd = [
        sys.executable,
        "-m",
        "scripts.feature_generator",
        "--bars-path",
        str(args.bars_path),
        "--labels-path",
        str(labels_path),
        "--output-dir",
        str(args.features_output_dir),
        "--feature-set",
        str(feature_set),
    ]
    _run_cmd(feature_cmd, env=trial_env, name="feature_generator")
    features_path = _latest_file(args.features_output_dir, "features_*.csv")
    return labels_path, features_path


def _trial_configs(args: AutotuneArgs) -> list[TrialConfig]:
    configs = [
        TrialConfig(
            feature_set=str(feature_set),
            split_adjust=str(split_adjust),
            bars_adjustment=str(bars_adjustment),
            calibration=str(calibration),
            top_k=int(top_k),
            cost_bps=float(cost_bps),
            min_score=float(min_score),
        )
        for feature_set, split_adjust, bars_adjustment, calibration, top_k, cost_bps, min_score in product(
            args.feature_sets,
            args.split_adjust_values,
            args.bars_adjustments,
            args.calibrations,
            args.top_k_values,
            args.cost_bps_values,
            args.min_score_values,
        )
    ]
    if args.trials >= len(configs):
        return configs
    rng = random.Random(int(args.seed))
    indices = list(range(len(configs)))
    rng.shuffle(indices)
    chosen = sorted(indices[: args.trials])
    return [configs[index] for index in chosen]


def run_autotune(args: AutotuneArgs) -> dict[str, Any]:
    trial_configs = _trial_configs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    walkforward_root = args.output_dir / "_trial_walkforward"
    strategy_root = args.output_dir / "_trial_strategy"
    walkforward_root.mkdir(parents=True, exist_ok=True)
    strategy_root.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "[INFO] AUTOTUNE_START target=%s requested_trials=%d sampled_trials=%d seed=%d objective=sharpe_then_cagr_with_drawdown_penalty holdout_days=%d min_holdout_sharpe=%s min_holdout_periods=%d min_score_grid=%s",
        args.target,
        int(args.trials),
        int(len(trial_configs)),
        int(args.seed),
        int(args.holdout_days),
        float(args.min_holdout_sharpe),
        int(args.min_holdout_periods),
        ",".join(f"{float(v):g}" for v in args.min_score_values),
    )

    results: list[dict[str, Any]] = []
    for idx, config in enumerate(trial_configs, start=1):
        trial_started = time.time()
        trial_walkforward_dir = walkforward_root / f"trial_{idx:03d}"
        trial_strategy_dir = strategy_root / f"trial_{idx:03d}"
        trial_walkforward_dir.mkdir(parents=True, exist_ok=True)
        trial_strategy_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "trial": int(idx),
            "feature_set": config.feature_set,
            "split_adjust": config.split_adjust,
            "bars_adjustment": config.bars_adjustment,
            "calibration": config.calibration,
            "top_k": int(config.top_k),
            "cost_bps": float(config.cost_bps),
            "min_score": float(config.min_score),
            "status": "failed",
            "error": None,
            "eligibility_reason": "not_evaluated",
            "eligible": False,
            "tune_objective_score": float("-inf"),
            "objective_score": float("-inf"),
        }
        trial_env = os.environ.copy()
        trial_env["JBR_ML_FEATURE_SET"] = str(config.feature_set)
        trial_env["JBR_SPLIT_ADJUST"] = str(config.split_adjust)
        trial_env["JBR_BARS_ADJUSTMENT"] = str(config.bars_adjustment)

        previous_env = {
            "JBR_ML_FEATURE_SET": os.environ.get("JBR_ML_FEATURE_SET"),
            "JBR_SPLIT_ADJUST": os.environ.get("JBR_SPLIT_ADJUST"),
            "JBR_BARS_ADJUSTMENT": os.environ.get("JBR_BARS_ADJUSTMENT"),
        }
        try:
            os.environ["JBR_ML_FEATURE_SET"] = str(config.feature_set)
            os.environ["JBR_SPLIT_ADJUST"] = str(config.split_adjust)
            os.environ["JBR_BARS_ADJUSTMENT"] = str(config.bars_adjustment)

            labels_path, features_path = _generate_labels_and_features(
                args,
                trial_env=trial_env,
                feature_set=config.feature_set,
            )

            wf_args = WalkforwardArgs(
                target=str(args.target),
                test_window_days=int(args.test_window_days),
                train_window_days=int(args.train_window_days),
                step_days=int(args.step_days),
                embargo_days=args.embargo_days,
                top_k=int(config.top_k),
                score_col="score_5d",
                retrain_per_fold=True,
                calibrate=str(config.calibration),
                max_abs_fwd_ret=float(args.max_abs_fwd_ret),
                run_date=args.run_date,
                output_dir=trial_walkforward_dir,
                features_path=None if db.db_enabled() else features_path,
                labels_path=None if db.db_enabled() else labels_path,
                predictions_path=None,
            )
            wf_payload = run_walkforward(wf_args)
            oos_path = trial_walkforward_dir / "oos_predictions.csv"
            holdout_start_date, tune_end_date = _compute_window_dates(
                oos_path,
                holdout_days=int(args.holdout_days),
            )

            tune_dir = trial_strategy_dir / "tune"
            holdout_dir = trial_strategy_dir / "holdout"
            tune_dir.mkdir(parents=True, exist_ok=True)
            holdout_dir.mkdir(parents=True, exist_ok=True)

            tune_args = StrategyArgs(
                target=str(args.target),
                score_col="score_oos",
                top_k=int(config.top_k),
                horizon_days=None,
                rebalance_days=None,
                cost_bps=float(config.cost_bps),
                min_score=float(config.min_score) if float(config.min_score) > 0 else None,
                max_abs_fwd_ret=float(args.max_abs_fwd_ret),
                sweep=False,
                sweep_topk="",
                sweep_min_score="",
                sweep_cost_bps="",
                run_date=args.run_date,
                start_date=None,
                end_date=tune_end_date,
                input_path=oos_path,
                output_dir=tune_dir,
            )
            tune_payload = run_strategy_eval(tune_args)
            tune_metrics = tune_payload.get("metrics") or {}
            tune_max_abs_period_return = _read_max_abs_period_return(tune_dir / "equity_curve.csv")

            holdout_payload: dict[str, Any] = {}
            holdout_metrics: dict[str, Any] = {}
            holdout_max_abs_period_return = None
            holdout_enabled = bool(int(args.holdout_days) > 0)
            if holdout_enabled:
                holdout_args = StrategyArgs(
                    target=str(args.target),
                    score_col="score_oos",
                    top_k=int(config.top_k),
                    horizon_days=None,
                    rebalance_days=None,
                    cost_bps=float(config.cost_bps),
                    min_score=float(config.min_score) if float(config.min_score) > 0 else None,
                    max_abs_fwd_ret=float(args.max_abs_fwd_ret),
                    sweep=False,
                    sweep_topk="",
                    sweep_min_score="",
                    sweep_cost_bps="",
                    run_date=args.run_date,
                    start_date=holdout_start_date,
                    end_date=None,
                    input_path=oos_path,
                    output_dir=holdout_dir,
                )
                holdout_payload = run_strategy_eval(holdout_args)
                holdout_metrics = holdout_payload.get("metrics") or {}
                holdout_max_abs_period_return = _read_max_abs_period_return(
                    holdout_dir / "equity_curve.csv"
                )
            wf_summary = (wf_payload.get("summary") or {}).get("top_k_mean_fwd_ret") or {}
            tune_sharpe = _as_float(tune_metrics.get("sharpe"))
            tune_cagr = _as_float(tune_metrics.get("cagr"))
            tune_max_dd = _as_float(tune_metrics.get("max_drawdown"))
            tune_win_rate = _as_float(tune_metrics.get("win_rate"))
            tune_avg_period_return = _as_float(tune_metrics.get("avg_period_return"))
            holdout_sharpe = _as_float(holdout_metrics.get("sharpe"))
            holdout_cagr = _as_float(holdout_metrics.get("cagr"))
            holdout_max_dd = _as_float(holdout_metrics.get("max_drawdown"))
            holdout_win_rate = _as_float(holdout_metrics.get("win_rate"))
            holdout_avg_period_return = _as_float(holdout_metrics.get("avg_period_return"))
            holdout_periods = int(holdout_metrics.get("periods", 0) or 0)
            top_k_mean_fwd_ret = _as_float(wf_summary.get("mean"))
            top_k_win_rate = _as_float(
                ((wf_payload.get("summary") or {}).get("top_k_win_rate") or {}).get("mean")
            )
            tune_eligible, tune_objective_score, _, _, _ = _objective(
                sharpe=tune_sharpe,
                cagr=tune_cagr,
                max_dd=tune_max_dd,
                min_sharpe=float(args.min_sharpe),
                max_drawdown_floor=float(args.max_drawdown_floor),
            )
            eligibility_reasons: list[str] = []
            eligible = bool(tune_eligible)
            if not tune_eligible:
                eligibility_reasons.append("tune_objective_gate_failed")
            if holdout_enabled:
                if holdout_sharpe is None or holdout_sharpe < float(args.min_holdout_sharpe):
                    eligible = False
                    eligibility_reasons.append("holdout_sharpe_below_min")
                if holdout_periods < int(args.min_holdout_periods):
                    eligible = False
                    eligibility_reasons.append("holdout_periods_below_min")
                if holdout_max_dd is None or holdout_max_dd < float(args.max_drawdown_floor):
                    eligible = False
                    eligibility_reasons.append("holdout_max_dd_below_floor")
                if holdout_cagr is not None and holdout_cagr > float(args.max_cagr_cap):
                    eligible = False
                    eligibility_reasons.append("holdout_cagr_cap_exceeded")
                if (
                    holdout_max_abs_period_return is not None
                    and holdout_max_abs_period_return > float(args.max_abs_period_return_cap)
                ):
                    eligible = False
                    eligibility_reasons.append("holdout_period_return_cap_exceeded")
            eligibility_reason = "ok" if eligible else ",".join(eligibility_reasons) or "ineligible"
            result.update(
                {
                    "status": "ok",
                    "tune_sharpe": tune_sharpe,
                    "tune_cagr": tune_cagr,
                    "tune_max_drawdown": tune_max_dd,
                    "tune_win_rate": tune_win_rate,
                    "tune_avg_period_return": tune_avg_period_return,
                    "tune_periods": int(tune_metrics.get("periods", 0) or 0),
                    "tune_total_return": _as_float(tune_metrics.get("total_return")),
                    "tune_max_abs_period_return": tune_max_abs_period_return,
                    "holdout_sharpe": holdout_sharpe,
                    "holdout_cagr": holdout_cagr,
                    "holdout_max_drawdown": holdout_max_dd,
                    "holdout_win_rate": holdout_win_rate,
                    "holdout_avg_period_return": holdout_avg_period_return,
                    "holdout_periods": holdout_periods,
                    "holdout_total_return": _as_float(holdout_metrics.get("total_return")),
                    "holdout_max_abs_period_return": holdout_max_abs_period_return,
                    "holdout_start_date": holdout_start_date.isoformat()
                    if holdout_start_date
                    else None,
                    "tune_end_date": tune_end_date.isoformat() if tune_end_date else None,
                    "holdout_days": int(args.holdout_days),
                    "top_k_mean_fwd_ret": top_k_mean_fwd_ret,
                    "top_k_win_rate": top_k_win_rate,
                    "folds_count": int(wf_payload.get("folds_count", 0) or 0),
                    "sample_size_total": int(wf_payload.get("sample_size_total", 0) or 0),
                    "tune_objective_score": tune_objective_score,
                    "objective_score": tune_objective_score,
                    "eligibility_reason": eligibility_reason,
                    "eligible": bool(eligible),
                }
            )
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            result["eligible"] = False
            result["eligibility_reason"] = "exception"
            result["objective_score"] = float("-inf")
            result["tune_objective_score"] = float("-inf")
            LOG.warning(
                "[WARN] AUTOTUNE_TRIAL_FAILED trial=%d feature_set=%s split_adjust=%s bars_adjustment=%s calibration=%s top_k=%d cost_bps=%s min_score=%s err=%s",
                int(idx),
                config.feature_set,
                config.split_adjust,
                config.bars_adjustment,
                config.calibration,
                int(config.top_k),
                config.cost_bps,
                config.min_score,
                exc,
            )
        finally:
            for key, original in previous_env.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original

        elapsed = float(time.time() - trial_started)
        result["elapsed_secs"] = round(elapsed, 3)
        LOG.info(
            "[INFO] AUTOTUNE_TRIAL trial=%d feature_set=%s bars_adjustment=%s split_adjust=%s calibration=%s top_k=%d cost_bps=%s min_score=%s tune_sharpe=%s holdout_sharpe=%s holdout_cagr=%s holdout_max_dd=%s eligibility=%s reason=%s",
            int(idx),
            config.feature_set,
            config.bars_adjustment,
            config.split_adjust,
            config.calibration,
            int(config.top_k),
            config.cost_bps,
            config.min_score,
            result.get("tune_sharpe"),
            result.get("holdout_sharpe"),
            result.get("holdout_cagr"),
            result.get("holdout_max_drawdown"),
            str(bool(result.get("eligible"))).lower(),
            result.get("eligibility_reason"),
        )
        results.append(result)

    sweep_df = pd.DataFrame(results)
    if not sweep_df.empty and "eligible" in sweep_df.columns:
        sweep_df["eligible"] = sweep_df["eligible"].astype(bool)

    def _rank_key(row: dict[str, Any], *, include_eligibility: bool) -> tuple[Any, ...]:
        eligible = bool(row.get("eligible"))
        sharpe = _as_float(row.get("tune_sharpe"))
        cagr = _as_float(row.get("tune_cagr"))
        max_dd = _as_float(row.get("tune_max_drawdown"))
        score = _as_float(row.get("tune_objective_score"))
        prefix = (1 if eligible else 0,) if include_eligibility else tuple()
        return prefix + (
            score if score is not None else float("-inf"),
            sharpe if sharpe is not None else float("-inf"),
            cagr if cagr is not None else float("-inf"),
            max_dd if max_dd is not None else float("-inf"),
        )

    ok_rows = [row for row in results if row.get("status") == "ok"]
    eligible_rows = [row for row in ok_rows if bool(row.get("eligible"))]
    ranked_rows = sorted(ok_rows, key=lambda row: _rank_key(row, include_eligibility=True), reverse=True)
    best_tune_row = sorted(
        ok_rows,
        key=lambda row: _rank_key(row, include_eligibility=False),
        reverse=True,
    )[0] if ok_rows else {}
    champion_status = "ok"
    champion_row = {}
    if eligible_rows:
        champion_row = sorted(
            eligible_rows,
            key=lambda row: _rank_key(row, include_eligibility=False),
            reverse=True,
        )[0]
    else:
        champion_row = best_tune_row
        if int(args.holdout_days) > 0:
            champion_status = "no_holdout_pass"
            LOG.warning(
                "[WARN] AUTOTUNE_NO_HOLDOUT_PASS trials=%d eligible=0 selecting_best_tune=true",
                int(len(ok_rows)),
            )
    top_rows = ranked_rows[: max(int(args.top_n), 1)]

    latest_json_path = args.output_dir / "latest.json"
    sweep_csv_path = args.output_dir / "param_sweep.csv"
    champion_json_path = args.output_dir / "champion.json"
    sweep_df.to_csv(sweep_csv_path, index=False)

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": str(args.run_date) if args.run_date else None,
        "target": args.target,
        "champion_status": champion_status,
        "holdout_days": int(args.holdout_days),
        "trials_requested": int(args.trials),
        "trials_executed": int(len(results)),
        "trials_ok": int(len(ok_rows)),
        "trials_eligible": int(len(eligible_rows)),
        "objective": {
            "description": (
                "maximize Sharpe primarily, then CAGR, with drawdown penalty; "
                "discard configs where sharpe < min_sharpe or max_drawdown < max_drawdown_floor"
            ),
            "min_sharpe": float(args.min_sharpe),
            "max_drawdown_floor": float(args.max_drawdown_floor),
            "min_holdout_sharpe": float(args.min_holdout_sharpe),
            "min_holdout_periods": int(args.min_holdout_periods),
            "max_cagr_cap": float(args.max_cagr_cap),
            "max_abs_period_return_cap": float(args.max_abs_period_return_cap),
        },
        "search_space": {
            "feature_set": args.feature_sets,
            "split_adjust": args.split_adjust_values,
            "bars_adjustment": args.bars_adjustments,
            "calibration": args.calibrations,
            "top_k": args.top_k_values,
            "cost_bps": args.cost_bps_values,
            "min_score": args.min_score_values,
        },
        "walkforward": {
            "train_window_days": int(args.train_window_days),
            "test_window_days": int(args.test_window_days),
            "step_days": int(args.step_days),
            "embargo_days": args.embargo_days,
            "retrain_per_fold": True,
        },
        "best_config": champion_row,
        "top_configs": top_rows,
        "output_files": {
            "latest_json": str(latest_json_path),
            "param_sweep_csv": str(sweep_csv_path),
            "champion_json": str(champion_json_path),
        },
    }
    payload = _json_safe(payload)
    with latest_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    champion_payload: dict[str, Any] = {
        "run_utc": payload.get("run_utc"),
        "run_date": payload.get("run_date"),
        "champion_status": champion_status,
        "holdout_days": int(args.holdout_days),
        "champion_params": {
            "feature_set": champion_row.get("feature_set"),
            "split_adjust": champion_row.get("split_adjust"),
            "bars_adjustment": champion_row.get("bars_adjustment"),
            "calibration": champion_row.get("calibration"),
            "top_k": champion_row.get("top_k"),
            "cost_bps": champion_row.get("cost_bps"),
            "min_score": champion_row.get("min_score"),
        },
        "execution": {
            "min_model_score": champion_row.get("min_score"),
            "require_model_score": False,
        },
        "tune_metrics": {
            "sharpe": champion_row.get("tune_sharpe"),
            "cagr": champion_row.get("tune_cagr"),
            "max_drawdown": champion_row.get("tune_max_drawdown"),
            "win_rate": champion_row.get("tune_win_rate"),
            "avg_period_return": champion_row.get("tune_avg_period_return"),
            "periods": champion_row.get("tune_periods"),
            "total_return": champion_row.get("tune_total_return"),
            "max_abs_period_return": champion_row.get("tune_max_abs_period_return"),
        },
        "holdout_metrics": {
            "sharpe": champion_row.get("holdout_sharpe"),
            "cagr": champion_row.get("holdout_cagr"),
            "max_drawdown": champion_row.get("holdout_max_drawdown"),
            "win_rate": champion_row.get("holdout_win_rate"),
            "avg_period_return": champion_row.get("holdout_avg_period_return"),
            "periods": champion_row.get("holdout_periods"),
            "total_return": champion_row.get("holdout_total_return"),
            "max_abs_period_return": champion_row.get("holdout_max_abs_period_return"),
            "start_date": champion_row.get("holdout_start_date"),
            "tune_end_date": champion_row.get("tune_end_date"),
        },
        "objective": payload.get("objective"),
        "thresholds": {
            "min_sharpe": float(args.min_sharpe),
            "max_drawdown_floor": float(args.max_drawdown_floor),
            "min_holdout_sharpe": float(args.min_holdout_sharpe),
            "min_holdout_periods": int(args.min_holdout_periods),
            "max_cagr_cap": float(args.max_cagr_cap),
            "max_abs_period_return_cap": float(args.max_abs_period_return_cap),
        },
        "reproducibility": {
            "seed": int(args.seed),
            "sampled_trials": int(len(trial_configs)),
        },
    }
    champion_payload = _json_safe(champion_payload)
    with champion_json_path.open("w", encoding="utf-8") as handle:
        json.dump(champion_payload, handle, indent=2)
    LOG.info(
        "[INFO] AUTOTUNE_CHAMPION status=%s holdout_sharpe=%s holdout_cagr=%s holdout_max_dd=%s champion_min_score=%s path=%s",
        champion_status,
        champion_row.get("holdout_sharpe"),
        champion_row.get("holdout_cagr"),
        champion_row.get("holdout_max_drawdown"),
        champion_row.get("min_score"),
        champion_json_path,
    )

    run_date_for_artifact = args.run_date or datetime.now(timezone.utc).date()
    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_autotune",
            run_date_for_artifact,
            payload=payload,
            rows_count=int(len(results)),
            source="ranker_autotune",
            file_name=latest_json_path.name,
        ):
            LOG.info(
                "[INFO] AUTOTUNE_DB_WRITTEN artifact_type=ranker_autotune run_date=%s",
                run_date_for_artifact,
            )
        if db.upsert_ml_artifact_frame(
            "ranker_autotune_sweep",
            run_date_for_artifact,
            sweep_df,
            source="ranker_autotune",
            file_name=sweep_csv_path.name,
        ):
            LOG.info(
                "[INFO] AUTOTUNE_DB_WRITTEN artifact_type=ranker_autotune_sweep run_date=%s",
                run_date_for_artifact,
            )
        if db.upsert_ml_artifact(
            "ranker_champion",
            run_date_for_artifact,
            payload=champion_payload,
            rows_count=1,
            source="ranker_autotune",
            file_name=champion_json_path.name,
        ):
            LOG.info(
                "[INFO] AUTOTUNE_DB_WRITTEN artifact_type=ranker_champion run_date=%s",
                run_date_for_artifact,
            )
    else:
        LOG.warning("[WARN] DB_DISABLED autotune_using_fs_fallback=true")

    LOG.info(
        "[INFO] AUTOTUNE_END trials=%d best_sharpe=%s best_cagr=%s best_min_score=%s output=%s",
        int(len(results)),
        (champion_row or {}).get("tune_sharpe"),
        (champion_row or {}).get("tune_cagr"),
        (champion_row or {}).get("min_score"),
        latest_json_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> AutotuneArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Label target used for walk-forward and strategy eval.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of sampled configs to run from the full parameter grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling trials from the parameter grid.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best configs to include in latest.json top_configs.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=DEFAULT_HOLDOUT_DAYS,
        help=(
            "Calendar days reserved as final holdout for champion gating. "
            "Set 0 to disable holdout and use tune-only behavior."
        ),
    )
    parser.add_argument(
        "--min-holdout-sharpe",
        type=float,
        default=DEFAULT_MIN_HOLDOUT_SHARPE,
        help="Minimum holdout Sharpe required for champion eligibility.",
    )
    parser.add_argument(
        "--min-holdout-periods",
        type=int,
        default=DEFAULT_MIN_HOLDOUT_PERIODS,
        help="Minimum holdout rebalance periods required for champion eligibility.",
    )
    parser.add_argument(
        "--max-cagr-cap",
        type=float,
        default=DEFAULT_MAX_CAGR_CAP,
        help="Reject trials when holdout CAGR exceeds this cap (default: 5.0 = 500%%).",
    )
    parser.add_argument(
        "--max-abs-period-return-cap",
        type=float,
        default=DEFAULT_MAX_ABS_PERIOD_RETURN_CAP,
        help=(
            "Reject trials when any absolute holdout period return exceeds this cap "
            "(default: 1.0 = 100%%)."
        ),
    )
    parser.add_argument(
        "--feature-sets",
        type=str,
        default=DEFAULT_FEATURE_SETS,
        help="Comma-separated feature set values (default: v1,v2).",
    )
    parser.add_argument(
        "--split-adjust-values",
        type=str,
        default=DEFAULT_SPLIT_ADJUST_VALUES,
        help="Comma-separated split-adjust values (default: off,auto).",
    )
    parser.add_argument(
        "--bars-adjustments",
        type=str,
        default=DEFAULT_BARS_ADJUSTMENTS,
        help="Comma-separated bars adjustment values (default: raw,split).",
    )
    parser.add_argument(
        "--calibrations",
        type=str,
        default=DEFAULT_CALIBRATIONS,
        help="Comma-separated calibration values (default: none,sigmoid).",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default=DEFAULT_TOPK_VALUES,
        help="Comma-separated top-k values (default: 10,25,50).",
    )
    parser.add_argument(
        "--cost-bps-values",
        type=str,
        default=DEFAULT_COST_BPS_VALUES,
        help="Comma-separated transaction cost bps values (default: 0,5,10).",
    )
    parser.add_argument(
        "--min-score-grid",
        "--sweep-min-score",
        dest="min_score_grid",
        type=str,
        default=DEFAULT_MIN_SCORE_GRID,
        help="Comma-separated min-score values to sweep (default: 0.0,0.55,0.6,0.65).",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum Sharpe threshold for eligible configs (default: 0).",
    )
    parser.add_argument(
        "--max-drawdown-floor",
        type=float,
        default=-0.60,
        help="Minimum acceptable max drawdown (default: -0.60).",
    )
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=504,
        help="Walk-forward train window in calendar days (default: 504).",
    )
    parser.add_argument(
        "--test-window-days",
        type=int,
        default=63,
        help="Walk-forward test window in calendar days (default: 63).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=21,
        help="Walk-forward step size in calendar days (default: 21).",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help="Optional walk-forward embargo override in days.",
    )
    parser.add_argument(
        "--max-abs-fwd-ret",
        type=float,
        default=0.0,
        help="Optional fwd-return clipping threshold (0 disables clipping).",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional run date filter (YYYY-MM-DD) forwarded to evaluators.",
    )
    parser.add_argument(
        "--bars-path",
        type=Path,
        default=BASE_DIR / "data" / "daily_bars.csv",
        help="Bars CSV path for label generation (DB mode ignores this file content).",
    )
    parser.add_argument(
        "--labels-output-dir",
        type=Path,
        default=BASE_DIR / "data" / "labels",
        help="Output directory for labels CSV writes.",
    )
    parser.add_argument(
        "--features-output-dir",
        type=Path,
        default=BASE_DIR / "data" / "features",
        help="Output directory for features CSV writes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_autotune",
        help="Output directory for autotune summary and sweep files.",
    )

    parsed = parser.parse_args(argv)
    if int(parsed.trials) < 1:
        parser.error("--trials must be >= 1")
    if int(parsed.top_n) < 1:
        parser.error("--top-n must be >= 1")
    if int(parsed.holdout_days) < 0:
        parser.error("--holdout-days must be >= 0")
    if int(parsed.min_holdout_periods) < 0:
        parser.error("--min-holdout-periods must be >= 0")
    if int(parsed.train_window_days) < 1:
        parser.error("--train-window-days must be >= 1")
    if int(parsed.test_window_days) < 1:
        parser.error("--test-window-days must be >= 1")
    if int(parsed.step_days) < 1:
        parser.error("--step-days must be >= 1")
    if parsed.embargo_days is not None and int(parsed.embargo_days) < 0:
        parser.error("--embargo-days must be >= 0")
    if float(parsed.max_abs_fwd_ret) < 0:
        parser.error("--max-abs-fwd-ret must be >= 0")
    if float(parsed.max_cagr_cap) <= 0:
        parser.error("--max-cagr-cap must be > 0")
    if float(parsed.max_abs_period_return_cap) <= 0:
        parser.error("--max-abs-period-return-cap must be > 0")

    feature_sets = _parse_csv_strings(parsed.feature_sets)
    split_adjust_values = _parse_csv_strings(parsed.split_adjust_values)
    bars_adjustments = _parse_csv_strings(parsed.bars_adjustments)
    calibrations = _parse_csv_strings(parsed.calibrations)
    top_k_values = _parse_csv_ints(parsed.top_k_values)
    cost_bps_values = _parse_csv_floats(parsed.cost_bps_values)
    min_score_values = _parse_csv_floats(parsed.min_score_grid)
    if not feature_sets:
        parser.error("--feature-sets must contain at least one value")
    invalid_feature_sets = sorted(set(feature_sets) - VALID_FEATURE_SETS)
    if invalid_feature_sets:
        parser.error(f"--feature-sets contains unsupported values: {invalid_feature_sets}")
    if not split_adjust_values:
        parser.error("--split-adjust-values must contain at least one value")
    invalid_split = sorted(set(split_adjust_values) - VALID_SPLIT_ADJUST_VALUES)
    if invalid_split:
        parser.error(f"--split-adjust-values contains unsupported values: {invalid_split}")
    if not bars_adjustments:
        parser.error("--bars-adjustments must contain at least one value")
    invalid_bars = sorted(set(bars_adjustments) - VALID_BARS_ADJUSTMENTS)
    if invalid_bars:
        parser.error(f"--bars-adjustments contains unsupported values: {invalid_bars}")
    if not calibrations:
        parser.error("--calibrations must contain at least one value")
    invalid_cal = sorted(set(calibrations) - VALID_CALIBRATIONS)
    if invalid_cal:
        parser.error(f"--calibrations contains unsupported values: {invalid_cal}")
    if not top_k_values:
        parser.error("--top-k-values must contain at least one value")
    if not cost_bps_values:
        parser.error("--cost-bps-values must contain at least one value")
    if not min_score_values:
        parser.error("--min-score-grid must contain at least one value")
    if any(value < 1 for value in top_k_values):
        parser.error("--top-k-values must be >= 1")
    if any(value < 0 for value in min_score_values):
        parser.error("--min-score-grid values must be >= 0")

    try:
        run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits

    return AutotuneArgs(
        target=str(parsed.target),
        trials=int(parsed.trials),
        seed=int(parsed.seed),
        top_n=int(parsed.top_n),
        min_sharpe=float(parsed.min_sharpe),
        max_drawdown_floor=float(parsed.max_drawdown_floor),
        holdout_days=int(parsed.holdout_days),
        min_holdout_sharpe=float(parsed.min_holdout_sharpe),
        min_holdout_periods=int(parsed.min_holdout_periods),
        max_cagr_cap=float(parsed.max_cagr_cap),
        max_abs_period_return_cap=float(parsed.max_abs_period_return_cap),
        feature_sets=feature_sets,
        split_adjust_values=split_adjust_values,
        bars_adjustments=bars_adjustments,
        calibrations=calibrations,
        top_k_values=top_k_values,
        cost_bps_values=cost_bps_values,
        min_score_values=min_score_values,
        train_window_days=int(parsed.train_window_days),
        test_window_days=int(parsed.test_window_days),
        step_days=int(parsed.step_days),
        embargo_days=parsed.embargo_days,
        max_abs_fwd_ret=float(parsed.max_abs_fwd_ret),
        run_date=run_date,
        bars_path=Path(parsed.bars_path),
        labels_output_dir=Path(parsed.labels_output_dir),
        features_output_dir=Path(parsed.features_output_dir),
        output_dir=Path(parsed.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_autotune(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("AUTOTUNE_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
