"""Monitor OOS ranker health via drift (PSI) and recent strategy metrics.

This script is evaluation-only. It does not alter training/execution behavior
and does not place orders.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.ranker_strategy_eval import (  # noqa: E402
    _compute_summary_metrics,
    _infer_label_horizon_days,
    _prepare_frame,
    _resolve_fwd_return_column,
    _simulate_strategy,
)
from scripts.utils.champion_config import load_latest_champion  # noqa: E402
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_monitor")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_TARGET = "label_5d_pos_300bp"
DEFAULT_SCORE_COL = "score_oos"
SCORE_RANGE_EPS = 1e-6


@dataclass
class MonitorArgs:
    target: str
    recent_days: int
    baseline_days: int
    psi_bins: int
    psi_warn: float
    psi_alert: float
    score_col: str
    calibration_bins: int
    calibration_min_rows: int
    calibration_ece_warn: float
    calibration_ece_alert: float
    drift_cols: list[str]
    top_k: int
    cost_bps: float
    horizon_days: int | None
    rebalance_days: int | None
    recent_sharpe_warn: float
    recent_sharpe_alert: float
    run_date: date | None
    input_path: Path | None
    output_dir: Path


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


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
    if isinstance(value, pd.Timestamp):
        if value.tz is None:
            value = value.tz_localize("UTC")
        return value.isoformat()
    if value is None:
        return None
    return value


def _load_oos_inputs(args: MonitorArgs) -> tuple[pd.DataFrame, str, date | None]:
    if db.db_enabled():
        record = db.fetch_ml_artifact("ranker_oos_predictions", run_date=args.run_date)
        if record is None and args.run_date is not None:
            LOG.warning(
                "[WARN] RANKER_MONITOR_RUN_DATE_FALLBACK artifact=ranker_oos_predictions run_date=%s",
                args.run_date,
            )
            record = db.fetch_latest_ml_artifact("ranker_oos_predictions")
        if record is not None:
            csv_data = record.get("csv_data")
            if csv_data:
                try:
                    frame = pd.read_csv(io.StringIO(str(csv_data)))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed reading DB artifact ranker_oos_predictions CSV: {exc}"
                    ) from exc
                artifact_run_date = None
                try:
                    if record.get("run_date") is not None:
                        artifact_run_date = pd.to_datetime(record.get("run_date")).date()
                except Exception:
                    artifact_run_date = None
                return frame, "db://ml_artifacts/ranker_oos_predictions", artifact_run_date
            LOG.warning(
                "[WARN] RANKER_MONITOR_DB_ARTIFACT_EMPTY artifact=ranker_oos_predictions "
                "falling_back=filesystem"
            )

    fallback_path = args.input_path or (
        BASE_DIR / "data" / "ranker_walkforward" / "oos_predictions.csv"
    )
    if not fallback_path.exists():
        raise FileNotFoundError(
            "OOS predictions not found. Expected DB artifact 'ranker_oos_predictions' "
            f"or file {fallback_path}."
        )
    frame = pd.read_csv(fallback_path)
    return frame, str(fallback_path), args.run_date


def _resolve_score_column(frame: pd.DataFrame, requested: str) -> str:
    preferred = str(requested or "").strip()
    if preferred and preferred in frame.columns:
        return preferred
    if "score_oos" in frame.columns:
        if preferred and preferred != "score_oos":
            LOG.warning(
                "[WARN] RANKER_MONITOR_SCORE_FALLBACK from=%s to=score_oos reason=missing_column",
                preferred,
            )
        return "score_oos"
    if "score_5d" in frame.columns:
        if preferred and preferred != "score_5d":
            LOG.warning(
                "[WARN] RANKER_MONITOR_SCORE_FALLBACK from=%s to=score_5d reason=missing_column",
                preferred,
            )
        return "score_5d"
    score_like = sorted([column for column in frame.columns if column.startswith("score_")])
    raise RuntimeError(
        f"Unable to resolve score column (requested='{preferred or 'none'}'). "
        f"Available score-like columns: {score_like or 'none'}."
    )


def _window_bounds(work: pd.DataFrame, *, recent_days: int, baseline_days: int) -> dict[str, date]:
    if work.empty:
        raise RuntimeError("No OOS rows available for monitoring.")
    ts_dates = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.date
    min_date = ts_dates.min()
    max_date = ts_dates.max()
    if min_date is None or max_date is None:
        raise RuntimeError("Unable to derive date bounds from OOS timestamps.")
    recent_start = max_date - timedelta(days=max(int(recent_days) - 1, 0))
    baseline_end = min_date + timedelta(days=max(int(baseline_days) - 1, 0))
    if baseline_end > max_date:
        baseline_end = max_date
    return {
        "dataset_start": min_date,
        "dataset_end": max_date,
        "recent_start": recent_start,
        "recent_end": max_date,
        "baseline_start": min_date,
        "baseline_end": baseline_end,
    }


def _slice_window(work: pd.DataFrame, *, start: date, end: date) -> pd.DataFrame:
    ts_dates = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.date
    mask = (ts_dates >= start) & (ts_dates <= end)
    out = work.loc[mask].copy()
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _default_drift_columns(
    frame: pd.DataFrame,
    *,
    requested: list[str],
    score_col: str,
    fwd_ret_col: str,
    target: str,
) -> list[str]:
    preferred = list(requested or [])
    if not preferred:
        preferred = [score_col, fwd_ret_col, "close", target]
        if score_col != "score_oos":
            preferred.insert(0, "score_oos")
        if "score_5d" not in preferred:
            preferred.append("score_5d")

    seen: set[str] = set()
    resolved: list[str] = []
    for col in preferred:
        key = str(col or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[key], errors="coerce")
        if int(numeric.notna().sum()) <= 1:
            continue
        resolved.append(key)
    return resolved


def _compute_psi(
    ref: pd.Series,
    cur: pd.Series,
    *,
    bins: int,
    warn: float,
    alert: float,
) -> dict[str, Any]:
    ref_vals = pd.to_numeric(ref, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cur_vals = pd.to_numeric(cur, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ref_vals.empty or cur_vals.empty:
        return {
            "psi": None,
            "level": "insufficient",
            "ref_count": int(ref_vals.shape[0]),
            "cur_count": int(cur_vals.shape[0]),
            "bins_used": 0,
        }

    q = np.linspace(0.0, 1.0, int(max(bins, 2)) + 1)
    edges = np.unique(np.quantile(ref_vals, q))
    if edges.size < 2:
        vmin = float(ref_vals.min())
        vmax = float(ref_vals.max())
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        edges = np.array([vmin, vmax], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref_vals, bins=edges)
    cur_hist, _ = np.histogram(cur_vals, bins=edges)

    ref_pct = ref_hist.astype(float) / max(float(ref_hist.sum()), 1.0)
    cur_pct = cur_hist.astype(float) / max(float(cur_hist.sum()), 1.0)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()
    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    level = "stable"
    if psi_value >= float(alert):
        level = "alert"
    elif psi_value >= float(warn):
        level = "warn"
    return {
        "psi": psi_value,
        "level": level,
        "ref_count": int(ref_vals.shape[0]),
        "cur_count": int(cur_vals.shape[0]),
        "bins_used": int(ref_pct.shape[0]),
    }


def _compute_calibration_window(
    frame: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    bins: int,
    min_rows: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "applicable": False,
        "skip_reason": None,
        "rows": 0,
        "ece": None,
        "mce": None,
        "score_min": None,
        "score_max": None,
        "reliability_table": [],
    }
    if label_col not in frame.columns:
        out["skip_reason"] = "missing_label"
        return out
    if score_col not in frame.columns:
        out["skip_reason"] = "missing_score"
        return out

    labels = pd.to_numeric(frame[label_col], errors="coerce")
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    valid = labels.notna() & scores.notna()
    rows = int(valid.sum())
    out["rows"] = rows
    if rows <= 0:
        out["skip_reason"] = "insufficient_rows"
        return out

    labels = labels.loc[valid].astype(float)
    scores = scores.loc[valid].astype(float)
    score_min = float(scores.min())
    score_max = float(scores.max())
    out["score_min"] = score_min
    out["score_max"] = score_max
    if rows < int(max(min_rows, 1)):
        out["skip_reason"] = "insufficient_rows"
        return out
    if score_min < -SCORE_RANGE_EPS or score_max > (1.0 + SCORE_RANGE_EPS):
        out["skip_reason"] = "score_out_of_range"
        return out

    bin_count = int(max(2, bins))
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_ids = np.digitize(scores.to_numpy(dtype=float), edges[1:-1], right=False)
    reliability_table: list[dict[str, Any]] = []
    ece_acc = 0.0
    mce = 0.0
    total = int(len(scores))
    for idx in range(bin_count):
        mask = bin_ids == idx
        count = int(np.sum(mask))
        if count <= 0:
            continue
        bin_scores = scores.iloc[mask]
        bin_labels = labels.iloc[mask]
        avg_pred = float(bin_scores.mean())
        frac_pos = float(bin_labels.mean())
        abs_gap = abs(avg_pred - frac_pos)
        ece_acc += (count / max(total, 1)) * abs_gap
        mce = max(mce, abs_gap)
        reliability_table.append(
            {
                "bin_lo": float(edges[idx]),
                "bin_hi": float(edges[idx + 1]),
                "count": count,
                "avg_pred": avg_pred,
                "frac_pos": frac_pos,
            }
        )

    out["applicable"] = True
    out["skip_reason"] = None
    out["ece"] = float(ece_acc)
    out["mce"] = float(mce)
    out["reliability_table"] = reliability_table
    return out


def _compute_calibration_drift(
    *,
    recent: pd.DataFrame,
    baseline: pd.DataFrame,
    label_col: str,
    score_col: str,
    bins: int,
    min_rows: int,
) -> dict[str, Any]:
    recent_stats = _compute_calibration_window(
        recent,
        label_col=label_col,
        score_col=score_col,
        bins=bins,
        min_rows=min_rows,
    )
    baseline_stats = _compute_calibration_window(
        baseline,
        label_col=label_col,
        score_col=score_col,
        bins=bins,
        min_rows=min_rows,
    )
    applicable = bool(recent_stats.get("applicable")) and bool(baseline_stats.get("applicable"))
    skip_reason = None
    if not applicable:
        skip_reason = str(
            recent_stats.get("skip_reason")
            or baseline_stats.get("skip_reason")
            or "insufficient_rows"
        )
    delta_ece = None
    if applicable:
        try:
            delta_ece = float(recent_stats["ece"]) - float(baseline_stats["ece"])
        except Exception:
            delta_ece = None
    return {
        "calibration_applicable": applicable,
        "calibration_skip_reason": skip_reason,
        "calibration": {
            "bins": int(max(2, bins)),
            "min_rows": int(max(1, min_rows)),
            "score_col": score_col,
            "label_col": label_col,
            "recent": {
                "rows": int(recent_stats.get("rows") or 0),
                "ece": recent_stats.get("ece"),
                "mce": recent_stats.get("mce"),
                "score_min": recent_stats.get("score_min"),
                "score_max": recent_stats.get("score_max"),
                "reliability_table": recent_stats.get("reliability_table") or [],
            },
            "baseline": {
                "rows": int(baseline_stats.get("rows") or 0),
                "ece": baseline_stats.get("ece"),
                "mce": baseline_stats.get("mce"),
                "score_min": baseline_stats.get("score_min"),
                "score_max": baseline_stats.get("score_max"),
                "reliability_table": baseline_stats.get("reliability_table") or [],
            },
            "delta_ece": delta_ece,
        },
    }


def _run_strategy_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    fwd_ret_col: str,
    top_k: int,
    rebalance_days: int,
    cost_bps: float,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "periods": 0,
            "error": "empty_window",
        }
    try:
        periods = _simulate_strategy(
            frame,
            score_col=score_col,
            fwd_ret_col=fwd_ret_col,
            top_k=int(top_k),
            rebalance_days=int(rebalance_days),
            cost_bps=float(cost_bps),
            min_score=None,
        )
        metrics = _compute_summary_metrics(
            periods, top_k=int(top_k), rebalance_days=int(rebalance_days)
        )
        return metrics
    except Exception as exc:
        return {
            "periods": 0,
            "error": str(exc),
        }


def _recommend_action(
    *,
    max_psi: float | None,
    recent_sharpe: float | None,
    calibration_applicable: bool,
    recent_ece: float | None,
    psi_warn: float,
    psi_alert: float,
    calibration_ece_warn: float,
    calibration_ece_alert: float,
    recent_sharpe_warn: float,
    recent_sharpe_alert: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    # Severe drift/performance degradations prioritize retraining.
    if max_psi is not None and float(max_psi) >= float(psi_alert):
        reasons.append("psi_alert")
    if recent_sharpe is not None and float(recent_sharpe) < float(recent_sharpe_alert):
        reasons.append("recent_sharpe_alert")
    if reasons:
        return "retrain", reasons

    # Calibration issues can recommend recalibration when broader drift is not severe.
    if calibration_applicable and recent_ece is not None:
        if float(recent_ece) >= float(calibration_ece_alert):
            return "recalibrate", ["calibration_ece_alert"]
        if float(recent_ece) >= float(calibration_ece_warn):
            return "recalibrate", ["calibration_ece_warn"]

    if max_psi is not None and float(max_psi) >= float(psi_warn):
        reasons.append("psi_warn")
    if recent_sharpe is not None and float(recent_sharpe) < float(recent_sharpe_warn):
        reasons.append("recent_sharpe_warn")
    if reasons:
        return "recalibrate", reasons
    return "none", []


def run_monitor(args: MonitorArgs) -> dict[str, Any]:
    LOG.info(
        "[INFO] RANKER_MONITOR_START target=%s recent_days=%d baseline_days=%d psi_bins=%d psi_warn=%s psi_alert=%s calibration_bins=%d calibration_min_rows=%d calibration_ece_warn=%s calibration_ece_alert=%s",
        args.target,
        int(args.recent_days),
        int(args.baseline_days),
        int(args.psi_bins),
        args.psi_warn,
        args.psi_alert,
        int(args.calibration_bins),
        int(args.calibration_min_rows),
        args.calibration_ece_warn,
        args.calibration_ece_alert,
    )

    raw, data_source, artifact_run_date = _load_oos_inputs(args)
    horizon_days = int(args.horizon_days or 0)
    if horizon_days <= 0:
        horizon_days = _infer_label_horizon_days(args.target)
    if horizon_days <= 0:
        raise RuntimeError("Unable to infer horizon days; pass --horizon-days explicitly.")

    rebalance_days = int(args.rebalance_days or horizon_days)
    if rebalance_days <= 0:
        raise RuntimeError("--rebalance-days must be >= 1")

    score_col = _resolve_score_column(raw, args.score_col)
    fwd_ret_col = _resolve_fwd_return_column(raw, horizon_days)
    work = _prepare_frame(raw, score_col=score_col, fwd_ret_col=fwd_ret_col)
    if work.empty:
        raise RuntimeError("OOS dataset has no usable rows after normalization.")

    bounds = _window_bounds(
        work, recent_days=int(args.recent_days), baseline_days=int(args.baseline_days)
    )
    baseline = _slice_window(
        work,
        start=bounds["baseline_start"],
        end=bounds["baseline_end"],
    )
    recent = _slice_window(
        work,
        start=bounds["recent_start"],
        end=bounds["recent_end"],
    )
    if recent.empty:
        raise RuntimeError("Recent window has no rows; increase coverage or reduce --recent-days.")
    if baseline.empty:
        LOG.warning("[WARN] RANKER_MONITOR_BASELINE_EMPTY fallback=full_dataset")
        baseline = work.copy()

    drift_cols = _default_drift_columns(
        work,
        requested=args.drift_cols,
        score_col=score_col,
        fwd_ret_col=fwd_ret_col,
        target=args.target,
    )
    if not drift_cols:
        raise RuntimeError("No valid numeric drift columns resolved.")

    drift: dict[str, Any] = {}
    psi_values: list[tuple[str, float]] = []
    warn_cols: list[str] = []
    alert_cols: list[str] = []
    for col in drift_cols:
        psi_row = _compute_psi(
            baseline[col],
            recent[col],
            bins=int(args.psi_bins),
            warn=float(args.psi_warn),
            alert=float(args.psi_alert),
        )
        drift[col] = psi_row
        psi_value = psi_row.get("psi")
        if psi_value is not None:
            psi_values.append((col, float(psi_value)))
        level = str(psi_row.get("level") or "stable")
        if level in {"warn", "alert"}:
            LOG.warning(
                "[WARN] RANKER_MONITOR_PSI col=%s psi=%s level=%s warn=%s alert=%s",
                col,
                psi_row.get("psi"),
                level,
                args.psi_warn,
                args.psi_alert,
            )
            if level == "alert":
                alert_cols.append(col)
            else:
                warn_cols.append(col)

    max_psi_col = None
    max_psi = None
    if psi_values:
        max_psi_col, max_psi = max(psi_values, key=lambda item: item[1])
    LOG.info(
        "[INFO] RANKER_MONITOR_DRIFT cols=%d psi_max=%s psi_max_col=%s warn_cols=%s alert_cols=%s",
        len(drift_cols),
        max_psi,
        max_psi_col,
        ",".join(warn_cols) if warn_cols else "none",
        ",".join(alert_cols) if alert_cols else "none",
    )

    baseline_metrics = _run_strategy_metrics(
        baseline,
        score_col=score_col,
        fwd_ret_col=fwd_ret_col,
        top_k=int(args.top_k),
        rebalance_days=int(rebalance_days),
        cost_bps=float(args.cost_bps),
    )
    recent_metrics = _run_strategy_metrics(
        recent,
        score_col=score_col,
        fwd_ret_col=fwd_ret_col,
        top_k=int(args.top_k),
        rebalance_days=int(rebalance_days),
        cost_bps=float(args.cost_bps),
    )
    recent_sharpe = None
    try:
        recent_sharpe = (
            float(recent_metrics.get("sharpe"))
            if recent_metrics.get("sharpe") is not None
            else None
        )
    except (TypeError, ValueError):
        recent_sharpe = None

    calibration_source = raw.copy()
    if "symbol" in calibration_source.columns:
        calibration_source["symbol"] = calibration_source["symbol"].astype("string").str.upper()
    else:
        calibration_source["symbol"] = "__ALL__"
    calibration_source["timestamp"] = pd.to_datetime(
        calibration_source.get("timestamp"),
        utc=True,
        errors="coerce",
    )
    calibration_source = calibration_source.dropna(subset=["timestamp"]).copy()
    calibration_recent = _slice_window(
        calibration_source,
        start=bounds["recent_start"],
        end=bounds["recent_end"],
    )
    calibration_baseline = _slice_window(
        calibration_source,
        start=bounds["baseline_start"],
        end=bounds["baseline_end"],
    )
    if calibration_baseline.empty:
        calibration_baseline = calibration_source.copy()
    calibration_block = _compute_calibration_drift(
        recent=calibration_recent,
        baseline=calibration_baseline,
        label_col=args.target,
        score_col=score_col,
        bins=int(args.calibration_bins),
        min_rows=int(args.calibration_min_rows),
    )
    calibration_payload = calibration_block.get("calibration") or {}
    calibration_applicable = bool(calibration_block.get("calibration_applicable", False))
    calibration_skip_reason = calibration_block.get("calibration_skip_reason")
    recent_cal = calibration_payload.get("recent") if isinstance(calibration_payload, dict) else {}
    baseline_cal = (
        calibration_payload.get("baseline") if isinstance(calibration_payload, dict) else {}
    )
    if calibration_applicable:
        LOG.info(
            "[INFO] RANKER_MONITOR_CALIBRATION bins=%s recent_ece=%s baseline_ece=%s delta_ece=%s score_min=%s score_max=%s rows_recent=%s rows_baseline=%s",
            int(calibration_payload.get("bins", args.calibration_bins) or args.calibration_bins),
            recent_cal.get("ece") if isinstance(recent_cal, dict) else None,
            baseline_cal.get("ece") if isinstance(baseline_cal, dict) else None,
            calibration_payload.get("delta_ece"),
            recent_cal.get("score_min") if isinstance(recent_cal, dict) else None,
            recent_cal.get("score_max") if isinstance(recent_cal, dict) else None,
            recent_cal.get("rows") if isinstance(recent_cal, dict) else 0,
            baseline_cal.get("rows") if isinstance(baseline_cal, dict) else 0,
        )
    else:
        LOG.warning(
            "[WARN] RANKER_MONITOR_CALIBRATION_SKIPPED reason=%s rows_recent=%s rows_baseline=%s",
            calibration_skip_reason or "insufficient_rows",
            recent_cal.get("rows") if isinstance(recent_cal, dict) else 0,
            baseline_cal.get("rows") if isinstance(baseline_cal, dict) else 0,
        )

    recent_ece = None
    try:
        if isinstance(recent_cal, dict) and recent_cal.get("ece") is not None:
            recent_ece = float(recent_cal.get("ece"))
    except (TypeError, ValueError):
        recent_ece = None

    recommended_action, recommendation_reasons = _recommend_action(
        max_psi=max_psi,
        recent_sharpe=recent_sharpe,
        calibration_applicable=calibration_applicable,
        recent_ece=recent_ece,
        psi_warn=float(args.psi_warn),
        psi_alert=float(args.psi_alert),
        calibration_ece_warn=float(args.calibration_ece_warn),
        calibration_ece_alert=float(args.calibration_ece_alert),
        recent_sharpe_warn=float(args.recent_sharpe_warn),
        recent_sharpe_alert=float(args.recent_sharpe_alert),
    )

    champion = load_latest_champion(BASE_DIR)
    champion_summary = None
    if champion:
        champion_summary = {
            "source": champion.get("_champion_source"),
            "run_date": champion.get("_champion_run_date") or champion.get("run_date"),
            "params": champion.get("champion_params") or {},
            "status": champion.get("champion_status"),
        }

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": str(args.run_date) if args.run_date else None,
        "target": args.target,
        "data_source": data_source,
        "score_col_requested": args.score_col,
        "score_col_used": score_col,
        "fwd_return_column": fwd_ret_col,
        "horizon_days": int(horizon_days),
        "rebalance_days": int(rebalance_days),
        "top_k": int(args.top_k),
        "cost_bps": float(args.cost_bps),
        "windows": {
            "dataset_start": bounds["dataset_start"].isoformat(),
            "dataset_end": bounds["dataset_end"].isoformat(),
            "baseline_start": bounds["baseline_start"].isoformat(),
            "baseline_end": bounds["baseline_end"].isoformat(),
            "recent_start": bounds["recent_start"].isoformat(),
            "recent_end": bounds["recent_end"].isoformat(),
            "baseline_rows": int(baseline.shape[0]),
            "recent_rows": int(recent.shape[0]),
            "population_rows": int(work.shape[0]),
        },
        "drift": {
            "psi_bins": int(args.psi_bins),
            "psi_warn": float(args.psi_warn),
            "psi_alert": float(args.psi_alert),
            "columns": drift,
            "max_psi": max_psi,
            "max_psi_col": max_psi_col,
            "warn_cols": warn_cols,
            "alert_cols": alert_cols,
        },
        "calibration_applicable": calibration_applicable,
        "calibration": calibration_payload,
        "recent_strategy": recent_metrics,
        "baseline_strategy": baseline_metrics,
        "recent_sharpe_warn": float(args.recent_sharpe_warn),
        "recent_sharpe_alert": float(args.recent_sharpe_alert),
        "recommended_action": recommended_action,
        "recommendation_reasons": recommendation_reasons,
    }
    if champion_summary is not None:
        payload["champion"] = champion_summary
    payload = _json_safe(payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    run_date_for_artifact = artifact_run_date or args.run_date or datetime.now(timezone.utc).date()
    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_monitor",
            run_date_for_artifact,
            payload=payload,
            rows_count=int(recent.shape[0]),
            source="ranker_monitor",
            file_name=latest_path.name,
        ):
            LOG.info(
                "[INFO] RANKER_MONITOR_DB_WRITTEN artifact_type=ranker_monitor run_date=%s",
                run_date_for_artifact,
            )
    else:
        LOG.warning("[WARN] DB_DISABLED ranker_monitor_using_fs_fallback=true")

    LOG.info(
        "[INFO] RANKER_MONITOR_END psi_score=%s recent_sharpe=%s recommended_action=%s output=%s",
        max_psi,
        recent_metrics.get("sharpe"),
        recommended_action,
        latest_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> MonitorArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Label target used to infer horizon (default: label_5d_pos_300bp).",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=63,
        help="Calendar days in the recent monitoring window (default: 63).",
    )
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=252,
        help="Calendar days used for the baseline/reference window (default: 252).",
    )
    parser.add_argument(
        "--psi-bins",
        type=int,
        default=10,
        help="Number of bins used for PSI (default: 10).",
    )
    parser.add_argument(
        "--psi-warn",
        type=float,
        default=0.10,
        help="PSI warning threshold (default: 0.10).",
    )
    parser.add_argument(
        "--psi-alert",
        type=float,
        default=0.25,
        help="PSI alert threshold (default: 0.25).",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default=DEFAULT_SCORE_COL,
        help="Requested score column for drift/performance metrics (default: score_oos).",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Bin count for OOS calibration diagnostics (default: 10).",
    )
    parser.add_argument(
        "--calibration-min-rows",
        type=int,
        default=2000,
        help="Minimum rows required in recent and baseline windows for calibration diagnostics (default: 2000).",
    )
    parser.add_argument(
        "--calibration-ece-warn",
        type=float,
        default=0.05,
        help="ECE warning threshold for calibration drift (default: 0.05).",
    )
    parser.add_argument(
        "--calibration-ece-alert",
        type=float,
        default=0.10,
        help="ECE alert threshold for calibration drift (default: 0.10).",
    )
    parser.add_argument(
        "--drift-cols",
        type=str,
        default=None,
        help="Optional comma-separated numeric columns for PSI drift checks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-k symbols selected in strategy monitor metrics (default: 25).",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost (bps) for strategy monitor metrics (default: 5).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Optional horizon override; otherwise inferred from --target.",
    )
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=None,
        help="Optional rebalance cadence; default uses horizon days.",
    )
    parser.add_argument(
        "--recent-sharpe-warn",
        type=float,
        default=0.30,
        help="Sharpe warning threshold for recent window recommendation (default: 0.30).",
    )
    parser.add_argument(
        "--recent-sharpe-alert",
        type=float,
        default=0.00,
        help="Sharpe alert threshold for recent window recommendation (default: 0.00).",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional artifact run date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional filesystem input path for oos_predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_monitor",
        help="Output directory for monitor artifacts (default: data/ranker_monitor).",
    )
    parsed = parser.parse_args(argv)

    if parsed.recent_days < 1:
        parser.error("--recent-days must be >= 1")
    if parsed.baseline_days < 1:
        parser.error("--baseline-days must be >= 1")
    if parsed.psi_bins < 2:
        parser.error("--psi-bins must be >= 2")
    if parsed.psi_warn < 0 or parsed.psi_alert < 0:
        parser.error("--psi-warn/--psi-alert must be >= 0")
    if parsed.psi_warn > parsed.psi_alert:
        parser.error("--psi-warn must be <= --psi-alert")
    if parsed.calibration_bins < 2:
        parser.error("--calibration-bins must be >= 2")
    if parsed.calibration_min_rows < 1:
        parser.error("--calibration-min-rows must be >= 1")
    if parsed.calibration_ece_warn < 0 or parsed.calibration_ece_alert < 0:
        parser.error("--calibration-ece-warn/--calibration-ece-alert must be >= 0")
    if parsed.calibration_ece_warn > parsed.calibration_ece_alert:
        parser.error("--calibration-ece-warn must be <= --calibration-ece-alert")
    if parsed.top_k < 1:
        parser.error("--top-k must be >= 1")
    if parsed.cost_bps < 0:
        parser.error("--cost-bps must be >= 0")
    if parsed.horizon_days is not None and parsed.horizon_days < 1:
        parser.error("--horizon-days must be >= 1")
    if parsed.rebalance_days is not None and parsed.rebalance_days < 1:
        parser.error("--rebalance-days must be >= 1")

    try:
        run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits

    drift_cols: list[str] = []
    if parsed.drift_cols:
        drift_cols = [token.strip() for token in str(parsed.drift_cols).split(",") if token.strip()]

    return MonitorArgs(
        target=str(parsed.target),
        recent_days=int(parsed.recent_days),
        baseline_days=int(parsed.baseline_days),
        psi_bins=int(parsed.psi_bins),
        psi_warn=float(parsed.psi_warn),
        psi_alert=float(parsed.psi_alert),
        score_col=str(parsed.score_col),
        calibration_bins=int(parsed.calibration_bins),
        calibration_min_rows=int(parsed.calibration_min_rows),
        calibration_ece_warn=float(parsed.calibration_ece_warn),
        calibration_ece_alert=float(parsed.calibration_ece_alert),
        drift_cols=drift_cols,
        top_k=int(parsed.top_k),
        cost_bps=float(parsed.cost_bps),
        horizon_days=parsed.horizon_days,
        rebalance_days=parsed.rebalance_days,
        recent_sharpe_warn=float(parsed.recent_sharpe_warn),
        recent_sharpe_alert=float(parsed.recent_sharpe_alert),
        run_date=run_date,
        input_path=parsed.input_path,
        output_dir=Path(parsed.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_monitor(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("RANKER_MONITOR_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
