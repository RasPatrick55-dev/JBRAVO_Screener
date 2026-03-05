"""Evaluate OOS ranker predictions as a simple paper-mode strategy.

This module consumes per-row OOS predictions from ranker walk-forward output
and computes reproducible, non-overlapping rebalance metrics plus an equity
curve. It is evaluation-only and does not place orders.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.utils.fwd_return_sanity import (  # noqa: E402
    clip_forward_returns,
    log_forward_return_sanity,
)
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_strategy_eval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_TARGET = "label_5d_pos_300bp"
DEFAULT_SCORE_COL = "score_5d"


@dataclass
class StrategyArgs:
    target: str
    score_col: str
    top_k: int
    horizon_days: int | None
    rebalance_days: int | None
    cost_bps: float
    min_score: float | None
    max_abs_fwd_ret: float
    sweep: bool
    sweep_topk: str
    sweep_min_score: str
    sweep_cost_bps: str
    run_date: date | None
    start_date: date | None
    end_date: date | None
    input_path: Path | None
    output_dir: Path


def _infer_label_horizon_days(target: str) -> int:
    match = re.search(r"label_(\d+)d", str(target or ""))
    if not match:
        return 0
    try:
        return max(int(match.group(1)), 0)
    except (TypeError, ValueError):
        return 0


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


def _parse_optional_date(value: str | None, *, label: str) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid {label}: {value}")
    return parsed.date()


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for token in (value or "").split(","):
        item = token.strip()
        if not item:
            continue
        out.append(int(float(item)))
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for token in (value or "").split(","):
        item = token.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _parse_sweep_min_scores(value: str) -> list[float | None]:
    out: list[float | None] = []
    for token in (value or "").split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "na"}:
            out.append(None)
            continue
        out.append(float(item))
    return out


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _load_oos_inputs(args: StrategyArgs) -> tuple[pd.DataFrame, str, date | None]:
    if db.db_enabled():
        record = db.fetch_ml_artifact("ranker_oos_predictions", run_date=args.run_date)
        if record is None and args.run_date is not None:
            LOG.warning(
                "[WARN] STRATEGY_EVAL_RUN_DATE_FALLBACK artifact=ranker_oos_predictions run_date=%s",
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
                run_date = None
                try:
                    if record.get("run_date") is not None:
                        run_date = pd.to_datetime(record.get("run_date")).date()
                except Exception:
                    run_date = None
                return frame, "db://ml_artifacts/ranker_oos_predictions", run_date
            LOG.warning(
                "[WARN] STRATEGY_EVAL_DB_ARTIFACT_EMPTY artifact=ranker_oos_predictions "
                "falling_back=filesystem"
            )

    fallback_path = args.input_path or (BASE_DIR / "data" / "ranker_walkforward" / "oos_predictions.csv")
    if not fallback_path.exists():
        raise FileNotFoundError(
            "OOS predictions not found. Expected DB artifact 'ranker_oos_predictions' "
            f"or file {fallback_path}."
        )
    frame = pd.read_csv(fallback_path)
    return frame, str(fallback_path), args.run_date


def _resolve_score_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    if requested == DEFAULT_SCORE_COL and "score_oos" in frame.columns:
        LOG.warning(
            "[WARN] STRATEGY_EVAL_SCORE_FALLBACK from=%s to=score_oos reason=missing_column",
            requested,
        )
        return "score_oos"
    available = sorted([column for column in frame.columns if column.startswith("score_")])
    raise RuntimeError(
        f"Required score column '{requested}' not found. "
        f"Available score-like columns: {available or 'none'}."
    )


def _resolve_fwd_return_column(frame: pd.DataFrame, horizon_days: int) -> str:
    preferred = f"fwd_ret_{horizon_days}d"
    if preferred in frame.columns:
        return preferred
    candidates = sorted([column for column in frame.columns if column.startswith("fwd_ret_")])
    raise RuntimeError(
        f"Required forward return column '{preferred}' not found. "
        "Provide --horizon-days matching available columns or regenerate OOS predictions. "
        f"Available fwd return columns: {candidates or 'none'}."
    )


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    fwd_ret_col: str,
) -> pd.DataFrame:
    required = {"symbol", "timestamp", score_col, fwd_ret_col}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"OOS dataset missing required columns: {sorted(missing)}")

    work = frame.copy()
    work["symbol"] = work["symbol"].astype("string").str.upper()
    work["timestamp"] = _coerce_timestamp(work["timestamp"])
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[fwd_ret_col] = pd.to_numeric(work[fwd_ret_col], errors="coerce")
    work = work.dropna(subset=["symbol", "timestamp", score_col, fwd_ret_col])
    work = work.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return work


def _apply_date_filter(
    frame: pd.DataFrame,
    *,
    start_date: date | None,
    end_date: date | None,
) -> tuple[pd.DataFrame, int, int]:
    rows_in = int(frame.shape[0])
    if frame.empty or (start_date is None and end_date is None):
        return frame, rows_in, rows_in

    work = frame.copy()
    ts_dates = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.date
    mask = pd.Series(True, index=work.index)
    if start_date is not None:
        mask = mask & (ts_dates >= start_date)
    if end_date is not None:
        mask = mask & (ts_dates <= end_date)
    filtered = work.loc[mask].copy()
    rows_out = int(filtered.shape[0])
    return filtered, rows_in, rows_out


def _choose_rebalance_timestamps(timestamps: list[pd.Timestamp], rebalance_days: int) -> list[pd.Timestamp]:
    selected: list[pd.Timestamp] = []
    if not timestamps:
        return selected
    for ts in timestamps:
        if not selected:
            selected.append(ts)
            continue
        if ts >= selected[-1] + pd.Timedelta(days=rebalance_days):
            selected.append(ts)
    return selected


def _max_drawdown(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    if drawdown.empty:
        return None
    return float(drawdown.min())


def _annualization_factor(rebalance_days: int) -> float:
    return max(252.0 / float(max(rebalance_days, 1)), 1e-12)


def _simulate_strategy(
    frame: pd.DataFrame,
    *,
    score_col: str,
    fwd_ret_col: str,
    top_k: int,
    rebalance_days: int,
    cost_bps: float,
    min_score: float | None,
) -> pd.DataFrame:
    unique_timestamps = sorted(pd.unique(frame["timestamp"]))
    rebalance_timestamps = _choose_rebalance_timestamps(unique_timestamps, rebalance_days)
    if not rebalance_timestamps:
        raise RuntimeError("No rebalance timestamps generated. Check timestamp coverage.")

    cost_rate = float(cost_bps) / 10_000.0
    rows: list[dict[str, Any]] = []
    for ts in rebalance_timestamps:
        snapshot = frame.loc[frame["timestamp"] == ts].copy()
        if snapshot.empty:
            continue

        baseline_return = float(snapshot[fwd_ret_col].mean())
        eligible = snapshot
        if min_score is not None:
            eligible = eligible.loc[eligible[score_col] >= float(min_score)].copy()
        eligible = eligible.sort_values(score_col, ascending=False)
        selected = eligible.head(int(max(1, top_k))).copy()
        selected_count = int(selected.shape[0])

        gross_return = float(selected[fwd_ret_col].mean()) if selected_count > 0 else 0.0
        net_return = gross_return - (cost_rate if selected_count > 0 else 0.0)
        rows.append(
            {
                "timestamp": ts,
                "period_return_gross": gross_return,
                "period_return": net_return,
                "baseline_period_return": baseline_return,
                "selected_count": selected_count,
                "universe_count": int(snapshot.shape[0]),
            }
        )

    periods = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if periods.empty:
        raise RuntimeError("No strategy periods available after rebalance simulation.")

    periods["equity"] = (1.0 + periods["period_return"]).cumprod()
    periods["baseline_equity"] = (1.0 + periods["baseline_period_return"]).cumprod()
    return periods


def _compute_summary_metrics(periods: pd.DataFrame, *, top_k: int, rebalance_days: int) -> dict[str, Any]:
    returns = pd.to_numeric(periods["period_return"], errors="coerce").fillna(0.0)
    baseline_returns = pd.to_numeric(periods["baseline_period_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(periods["equity"], errors="coerce").ffill().fillna(1.0)
    baseline_equity = pd.to_numeric(periods["baseline_equity"], errors="coerce").ffill().fillna(1.0)

    periods_count = int(len(periods))
    annual_factor = _annualization_factor(rebalance_days)
    total_return = float(equity.iloc[-1] - 1.0)
    baseline_total_return = float(baseline_equity.iloc[-1] - 1.0)

    cagr = None
    baseline_cagr = None
    if periods_count > 0 and float(equity.iloc[-1]) > 0:
        cagr = float(math.pow(float(equity.iloc[-1]), annual_factor / periods_count) - 1.0)
    if periods_count > 0 and float(baseline_equity.iloc[-1]) > 0:
        baseline_cagr = float(
            math.pow(float(baseline_equity.iloc[-1]), annual_factor / periods_count) - 1.0
        )

    std = float(returns.std(ddof=0))
    vol_annual = float(std * math.sqrt(annual_factor))
    sharpe = float(returns.mean() / std * math.sqrt(annual_factor)) if std > 0 else None
    max_drawdown = _max_drawdown(equity)
    win_rate = float((returns > 0).mean()) if periods_count > 0 else None

    return {
        "periods": periods_count,
        "start": periods["timestamp"].min().isoformat() if periods_count > 0 else None,
        "end": periods["timestamp"].max().isoformat() if periods_count > 0 else None,
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_period_return": float(returns.mean()) if periods_count > 0 else None,
        "median_period_return": float(returns.median()) if periods_count > 0 else None,
        "turnover_proxy": int(periods_count * int(top_k)),
        "baseline_total_return": baseline_total_return,
        "baseline_cagr": baseline_cagr,
        "baseline_avg_period_return": float(baseline_returns.mean()) if periods_count > 0 else None,
        "alpha_total_return": float(total_return - baseline_total_return),
    }


def _run_param_sweep(
    frame: pd.DataFrame,
    *,
    score_col: str,
    fwd_ret_col: str,
    rebalance_days: int,
    sweep_topk: list[int],
    sweep_min_scores: list[float | None],
    sweep_cost_bps: list[float],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    LOG.info(
        "[INFO] STRATEGY_SWEEP_START topk=%s min_score=%s cost_bps=%s",
        ",".join(str(v) for v in sweep_topk),
        ",".join("none" if v is None else f"{v:g}" for v in sweep_min_scores),
        ",".join(f"{v:g}" for v in sweep_cost_bps),
    )
    rows: list[dict[str, Any]] = []
    for top_k in sweep_topk:
        for min_score in sweep_min_scores:
            for cost_bps in sweep_cost_bps:
                try:
                    periods = _simulate_strategy(
                        frame,
                        score_col=score_col,
                        fwd_ret_col=fwd_ret_col,
                        top_k=int(top_k),
                        rebalance_days=int(rebalance_days),
                        cost_bps=float(cost_bps),
                        min_score=min_score,
                    )
                    metrics = _compute_summary_metrics(
                        periods,
                        top_k=int(top_k),
                        rebalance_days=int(rebalance_days),
                    )
                    rows.append(
                        {
                            "top_k": int(top_k),
                            "min_score": min_score,
                            "cost_bps": float(cost_bps),
                            **metrics,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "top_k": int(top_k),
                            "min_score": min_score,
                            "cost_bps": float(cost_bps),
                            "error": str(exc),
                        }
                    )

    sweep = pd.DataFrame(rows)
    best_row: dict[str, Any] | None = None
    if not sweep.empty and "cagr" in sweep.columns:
        ranked = sweep.copy()
        ranked["cagr_rank"] = pd.to_numeric(ranked.get("cagr"), errors="coerce")
        ranked["sharpe_rank"] = pd.to_numeric(ranked.get("sharpe"), errors="coerce")
        ranked = ranked.sort_values(["cagr_rank", "sharpe_rank"], ascending=[False, False])
        if not ranked.empty and pd.notna(ranked.iloc[0].get("cagr_rank")):
            best_row = ranked.iloc[0].to_dict()

    best_key = "none"
    best_metric = None
    if best_row is not None:
        best_row = _json_safe(best_row)
        best_min_score = best_row.get("min_score")
        if best_min_score is None:
            best_min_score = "none"
        best_key = (
            f"top_k={int(best_row.get('top_k'))},"
            f"min_score={best_min_score},"
            f"cost_bps={best_row.get('cost_bps')}"
        )
        try:
            best_metric = float(best_row.get("cagr"))
        except (TypeError, ValueError):
            best_metric = None
    LOG.info(
        "[INFO] STRATEGY_SWEEP_END rows=%d best_key=%s best_metric=%s output=param_sweep.csv",
        int(sweep.shape[0]),
        best_key,
        best_metric,
    )
    return sweep, best_row


def run_strategy_eval(args: StrategyArgs) -> dict[str, Any]:
    horizon_days = int(args.horizon_days or 0)
    if horizon_days <= 0:
        horizon_days = _infer_label_horizon_days(args.target)
    if horizon_days <= 0:
        raise RuntimeError(
            "Unable to infer horizon days from --target. "
            "Provide --horizon-days explicitly."
        )

    rebalance_days = int(args.rebalance_days or horizon_days)
    if rebalance_days < 1:
        raise RuntimeError("--rebalance-days must be >= 1")

    LOG.info(
        "[INFO] STRATEGY_EVAL_START target=%s score_col=%s top_k=%d horizon_days=%d rebalance_days=%d cost_bps=%s",
        args.target,
        args.score_col,
        int(args.top_k),
        int(horizon_days),
        int(rebalance_days),
        args.cost_bps,
    )

    raw, data_source, artifact_run_date = _load_oos_inputs(args)
    score_col_used = _resolve_score_column(raw, args.score_col)
    fwd_ret_col = _resolve_fwd_return_column(raw, horizon_days)
    frame = _prepare_frame(raw, score_col=score_col_used, fwd_ret_col=fwd_ret_col)
    frame, rows_in, rows_out = _apply_date_filter(
        frame,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    LOG.info(
        "[INFO] STRATEGY_EVAL_DATE_FILTER start=%s end=%s rows_in=%d rows_out=%d",
        args.start_date.isoformat() if args.start_date else None,
        args.end_date.isoformat() if args.end_date else None,
        rows_in,
        rows_out,
    )
    if frame.empty:
        raise RuntimeError("OOS dataset has no usable rows after normalization.")
    fwd_ret_sanity = log_forward_return_sanity(
        frame[fwd_ret_col],
        column_name=fwd_ret_col,
        logger=LOG,
    )
    clipped_rows = 0
    if float(args.max_abs_fwd_ret or 0.0) > 0:
        frame, clipped_rows = clip_forward_returns(
            frame,
            column_name=fwd_ret_col,
            max_abs=float(args.max_abs_fwd_ret),
            logger=LOG,
        )

    periods = _simulate_strategy(
        frame,
        score_col=score_col_used,
        fwd_ret_col=fwd_ret_col,
        top_k=int(args.top_k),
        rebalance_days=int(rebalance_days),
        cost_bps=float(args.cost_bps),
        min_score=args.min_score,
    )
    metrics = _compute_summary_metrics(periods, top_k=int(args.top_k), rebalance_days=int(rebalance_days))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    equity_path = output_dir / "equity_curve.csv"
    sweep_path = output_dir / "param_sweep.csv"

    equity_out = periods.copy()
    equity_out["date"] = pd.to_datetime(equity_out["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    equity_out = equity_out[
        [
            "date",
            "equity",
            "period_return",
            "baseline_equity",
            "baseline_period_return",
            "selected_count",
            "universe_count",
        ]
    ]
    equity_out.to_csv(equity_path, index=False)

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": str(args.run_date) if args.run_date else None,
        "data_source": data_source,
        "start_date": args.start_date.isoformat() if args.start_date else None,
        "end_date": args.end_date.isoformat() if args.end_date else None,
        "date_filter_rows_in": int(rows_in),
        "date_filter_rows_out": int(rows_out),
        "target": args.target,
        "horizon_days": int(horizon_days),
        "score_col": args.score_col,
        "score_col_used": score_col_used,
        "fwd_return_column": fwd_ret_col,
        "fwd_return_sanity": fwd_ret_sanity or {},
        "max_abs_fwd_ret": float(args.max_abs_fwd_ret or 0.0),
        "fwd_ret_clipped_rows": int(clipped_rows),
        "top_k": int(args.top_k),
        "rebalance_days": int(rebalance_days),
        "cost_bps": float(args.cost_bps),
        "cost_model": "entry_only_bps",
        "min_score": args.min_score,
        "population_size": int(frame.shape[0]),
        "metrics": metrics,
    }

    sweep_out = pd.DataFrame()
    best_sweep_row: dict[str, Any] | None = None
    if args.sweep:
        sweep_topk = _parse_csv_ints(args.sweep_topk)
        sweep_min_scores = _parse_sweep_min_scores(args.sweep_min_score)
        sweep_cost = _parse_csv_floats(args.sweep_cost_bps)
        if not sweep_topk or not sweep_min_scores or not sweep_cost:
            raise RuntimeError("Sweep enabled but one or more sweep parameter lists are empty.")
        sweep_out, best_sweep_row = _run_param_sweep(
            frame,
            score_col=score_col_used,
            fwd_ret_col=fwd_ret_col,
            rebalance_days=int(rebalance_days),
            sweep_topk=sweep_topk,
            sweep_min_scores=sweep_min_scores,
            sweep_cost_bps=sweep_cost,
        )
        sweep_out.to_csv(sweep_path, index=False)
        payload["sweep_rows"] = int(sweep_out.shape[0])
        payload["sweep_best"] = best_sweep_row

    payload = _json_safe(payload)

    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    run_date_for_artifact = artifact_run_date or args.run_date or datetime.now(timezone.utc).date()
    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_strategy_eval",
            run_date_for_artifact,
            payload=payload,
            rows_count=int(metrics.get("periods", 0) or 0),
            source="ranker_strategy_eval",
            file_name=latest_path.name,
        ):
            LOG.info(
                "[INFO] STRATEGY_EVAL_DB_WRITTEN artifact_type=ranker_strategy_eval run_date=%s",
                run_date_for_artifact,
            )
        if db.upsert_ml_artifact_frame(
            "ranker_strategy_equity",
            run_date_for_artifact,
            equity_out,
            source="ranker_strategy_eval",
            file_name=equity_path.name,
        ):
            LOG.info(
                "[INFO] STRATEGY_EVAL_DB_WRITTEN artifact_type=ranker_strategy_equity run_date=%s",
                run_date_for_artifact,
            )
        if args.sweep and not sweep_out.empty:
            if db.upsert_ml_artifact_frame(
                "ranker_strategy_sweep",
                run_date_for_artifact,
                sweep_out,
                source="ranker_strategy_eval",
                file_name=sweep_path.name,
            ):
                LOG.info(
                    "[INFO] STRATEGY_EVAL_DB_WRITTEN artifact_type=ranker_strategy_sweep run_date=%s",
                    run_date_for_artifact,
                )
    else:
        LOG.warning("[WARN] DB_DISABLED strategy_eval_using_fs_fallback=true")

    LOG.info(
        "[INFO] STRATEGY_EVAL_END periods=%s total_return=%s cagr=%s sharpe=%s max_dd=%s win_rate=%s output=%s",
        metrics.get("periods"),
        metrics.get("total_return"),
        metrics.get("cagr"),
        metrics.get("sharpe"),
        metrics.get("max_drawdown"),
        metrics.get("win_rate"),
        latest_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> StrategyArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Label target used to infer horizon (default: label_5d_pos_300bp).",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default=DEFAULT_SCORE_COL,
        help="Score column used for ranking (default: score_5d).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of symbols selected each rebalance (default: 25).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Optional horizon days override; otherwise inferred from --target.",
    )
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=None,
        help="Rebalance cadence in days (default: inferred horizon days).",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost in basis points per rebalance entry (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum score threshold before top-k selection.",
    )
    parser.add_argument(
        "--max-abs-fwd-ret",
        type=float,
        default=0.0,
        help=(
            "Optional forward-return clipping threshold for research runs only. "
            "0 disables clipping (default: 0)."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep and write data/ranker_strategy_eval/param_sweep.csv.",
    )
    parser.add_argument(
        "--sweep-topk",
        type=str,
        default="5,10,15,20,25,30,40",
        help="Comma-separated top-k values for sweep.",
    )
    parser.add_argument(
        "--sweep-min-score",
        type=str,
        default="none,0.55,0.60",
        help="Comma-separated min-score values for sweep (use 'none' for no threshold).",
    )
    parser.add_argument(
        "--sweep-cost-bps",
        type=str,
        default="0,5,10",
        help="Comma-separated cost-bps values for sweep.",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional artifact run date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD) for OOS rows.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD) for OOS rows.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional filesystem input CSV path (fallback mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_strategy_eval",
        help="Output directory for latest.json/equity_curve.csv/param_sweep.csv.",
    )
    parsed = parser.parse_args(argv)

    if parsed.top_k < 1:
        parser.error("--top-k must be >= 1")
    if parsed.horizon_days is not None and parsed.horizon_days < 1:
        parser.error("--horizon-days must be >= 1")
    if parsed.rebalance_days is not None and parsed.rebalance_days < 1:
        parser.error("--rebalance-days must be >= 1")
    if parsed.cost_bps < 0:
        parser.error("--cost-bps must be >= 0")
    if parsed.max_abs_fwd_ret < 0:
        parser.error("--max-abs-fwd-ret must be >= 0")

    try:
        run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits
    try:
        start_date = _parse_optional_date(parsed.start_date, label="--start-date")
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits
    try:
        end_date = _parse_optional_date(parsed.end_date, label="--end-date")
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits
    if start_date is not None and end_date is not None and start_date > end_date:
        parser.error("--start-date must be <= --end-date")

    return StrategyArgs(
        target=str(parsed.target),
        score_col=str(parsed.score_col),
        top_k=int(parsed.top_k),
        horizon_days=parsed.horizon_days,
        rebalance_days=parsed.rebalance_days,
        cost_bps=float(parsed.cost_bps),
        min_score=parsed.min_score,
        max_abs_fwd_ret=float(parsed.max_abs_fwd_ret),
        sweep=bool(parsed.sweep),
        sweep_topk=str(parsed.sweep_topk),
        sweep_min_score=str(parsed.sweep_min_score),
        sweep_cost_bps=str(parsed.sweep_cost_bps),
        run_date=run_date,
        start_date=start_date,
        end_date=end_date,
        input_path=parsed.input_path,
        output_dir=Path(parsed.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_strategy_eval(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("STRATEGY_EVAL_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
