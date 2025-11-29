"""Evaluate ranker decile performance from historical predictions and trades.

This module ingests prediction snapshots and realised trade outcomes to
compute forward returns by decile. The resulting summary is written to
``data/ranker_eval/latest.json`` for dashboard consumption, with an optional
append-only history CSV for longitudinal tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils import write_csv_atomic  # noqa: E402
from utils.env import load_env  # noqa: E402


load_env()

LOG = logging.getLogger("ranker_eval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class EvalConfig:
    label_horizon: int = 5
    predictions_dir: Path = BASE_DIR / "data" / "predictions"
    executed_trades: Path = BASE_DIR / "data" / "executed_trades.csv"
    output_json: Path = BASE_DIR / "data" / "ranker_eval" / "latest.json"
    output_history: Path | None = BASE_DIR / "data" / "ranker_eval" / "history.csv"
    min_samples: int = 50
    min_per_decile: int = 5


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _load_predictions(predictions_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not predictions_dir.exists():
        LOG.warning("Predictions directory missing at %s", predictions_dir)
        return pd.DataFrame()
    for path in sorted(predictions_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive I/O
            LOG.warning("Failed to read predictions %s: %s", path, exc)
            continue
        if df.empty:
            continue
        rename_map = {}
        if "as_of" in df.columns and "run_date" not in df.columns:
            rename_map["as_of"] = "run_date"
        if "Score" in df.columns and "score" not in df.columns:
            rename_map["Score"] = "score"
        if rename_map:
            df = df.rename(columns=rename_map)
        required = {"symbol", "score", "run_date"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            LOG.warning("Predictions %s missing columns: %s", path, sorted(missing))
            continue
        df = df.loc[:, ["symbol", "score", "run_date"]].copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["run_date"] = _coerce_datetime(df["run_date"])
        df = df.dropna(subset=["symbol", "score", "run_date"])
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["symbol", "score", "run_date"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("run_date")
    return combined.reset_index(drop=True)


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOG.warning("Executed trades file missing at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O
        LOG.warning("Failed to read executed trades %s: %s", path, exc)
        return pd.DataFrame()
    if df.empty:
        return df
    rename_map = {}
    if "filled_qty" in df.columns and "qty" not in df.columns:
        rename_map["filled_qty"] = "qty"
    df = df.rename(columns=rename_map)
    for col in ("qty", "entry_price", "exit_price", "pnl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = _coerce_datetime(df[col])
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _compute_forward_return(row: Mapping[str, Any]) -> float | None:
    qty = abs(float(row.get("qty") or 0))
    entry_price = float(row.get("entry_price") or 0)
    pnl = row.get("pnl")
    side = str(row.get("side") or "").lower()
    exit_price = row.get("exit_price")
    if entry_price <= 0 or qty <= 0:
        return None
    if pd.notna(pnl):
        return float(pnl) / (entry_price * qty) if entry_price else None
    if pd.notna(exit_price):
        sign = 1 if side != "sell" else -1
        try:
            return sign * (float(exit_price) - entry_price) / entry_price
        except Exception:
            return None
    return None


def _attach_labels(preds: pd.DataFrame, trades: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if preds.empty or trades.empty:
        return pd.DataFrame(columns=["symbol", "score", "run_date", "forward_return"])

    trades = trades.dropna(subset=["symbol", "entry_time"])
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "score", "run_date", "forward_return"])
    trades = trades.sort_values("entry_time")

    preds = preds.copy()
    preds["symbol"] = preds["symbol"].astype(str).str.upper()
    preds = preds.sort_values("run_date")

    joined = pd.merge_asof(
        preds,
        trades,
        left_on="run_date",
        right_on="entry_time",
        by="symbol",
        direction="forward",
        allow_exact_matches=True,
    )

    cutoff = joined["run_date"] + pd.to_timedelta(horizon_days, unit="D")
    joined = joined.loc[(joined["entry_time"] <= cutoff)]

    joined["forward_return"] = joined.apply(_compute_forward_return, axis=1)
    joined = joined.dropna(subset=["forward_return", "score", "run_date"])
    return joined.loc[:, ["symbol", "score", "run_date", "forward_return"]]


def _compute_deciles(labeled: pd.DataFrame, cfg: EvalConfig) -> tuple[list[dict[str, Any]], str | None]:
    if labeled.empty:
        return [], "no labeled samples"

    labeled = labeled.sort_values("score", ascending=False)
    if len(labeled) < cfg.min_samples:
        return [], f"insufficient samples (<{cfg.min_samples})"

    try:
        bins = pd.qcut(labeled["score"], 10, labels=False, duplicates="drop")
    except ValueError as exc:
        return [], f"decile split failed: {exc}"

    if bins.isna().all():
        return [], "decile split failed: insufficient unique scores"

    max_bin = bins.max()
    labeled["decile"] = (max_bin - bins) + 1
    labeled["decile"] = labeled["decile"].astype(int)

    results: list[dict[str, Any]] = []
    for decile in range(1, 11):
        subset = labeled.loc[labeled["decile"] == decile]
        if subset.empty or len(subset) < cfg.min_per_decile:
            results.append(
                {
                    "decile": decile,
                    "count": int(len(subset)),
                    "avg_return": None,
                    "median_return": None,
                    "hit_rate": None,
                }
            )
            continue
        returns = subset["forward_return"].astype(float)
        hit_rate = float((returns > 0).mean())
        results.append(
            {
                "decile": decile,
                "count": int(len(subset)),
                "avg_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "hit_rate": hit_rate,
            }
        )
    return results, None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(serialised, encoding="utf-8")


def _append_history(path: Path, run_date: datetime, cfg: EvalConfig, deciles: Iterable[Mapping[str, Any]]) -> None:
    if path is None:
        return
    rows = []
    for dec in deciles:
        rows.append(
            {
                "run_date": run_date.date().isoformat(),
                "label_horizon_days": cfg.label_horizon,
                "decile": dec.get("decile"),
                "count": dec.get("count"),
                "avg_return": dec.get("avg_return"),
                "median_return": dec.get("median_return"),
                "hit_rate": dec.get("hit_rate"),
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:  # pragma: no cover - defensive read
            existing = pd.DataFrame()
        if not existing.empty:
            frame = pd.concat([existing, frame], ignore_index=True)
    write_csv_atomic(str(path), frame)


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    preds = _load_predictions(cfg.predictions_dir)
    trades = _load_trades(cfg.executed_trades)
    labeled = _attach_labels(preds, trades, cfg.label_horizon)
    deciles, reason = _compute_deciles(labeled, cfg)

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "label_horizon_days": cfg.label_horizon,
        "sample_size": int(len(labeled)),
        "deciles": deciles,
    }
    if reason:
        payload["reason"] = reason
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranker decile performance")
    parser.add_argument("--label-horizon", type=int, default=5, help="Forward return horizon in days")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=BASE_DIR / "data" / "predictions",
        help="Directory containing prediction CSV snapshots",
    )
    parser.add_argument(
        "--executed-trades",
        type=Path,
        default=BASE_DIR / "data" / "executed_trades.csv",
        help="CSV of executed trades with entry/exit info",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "data" / "ranker_eval" / "latest.json",
        help="Path to write JSON summary",
    )
    parser.add_argument(
        "--output-history",
        type=Path,
        default=BASE_DIR / "data" / "ranker_eval" / "history.csv",
        help="Path to append decile history (CSV)",
    )
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum rows required to compute deciles")
    parser.add_argument(
        "--min-per-decile",
        type=int,
        default=5,
        help="Minimum rows required per decile; deciles below this threshold emit nulls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = EvalConfig(
        label_horizon=int(args.label_horizon),
        predictions_dir=Path(args.predictions_dir),
        executed_trades=Path(args.executed_trades),
        output_json=Path(args.output_json),
        output_history=Path(args.output_history) if args.output_history else None,
        min_samples=int(args.min_samples),
        min_per_decile=int(args.min_per_decile),
    )

    LOG.info(
        "[INFO] RANKER_EVAL start horizon=%s preds=%s trades=%s",
        cfg.label_horizon,
        cfg.predictions_dir,
        cfg.executed_trades,
    )
    payload = evaluate(cfg)
    try:
        _write_json(cfg.output_json, payload)
        LOG.info(
            "[INFO] RANKER_EVAL samples=%d deciles=%d output=%s",
            payload.get("sample_size", 0),
            len(payload.get("deciles") or []),
            cfg.output_json,
        )
    except Exception as exc:  # pragma: no cover - defensive write
        LOG.exception("Failed to write ranker_eval JSON: %s", exc)
        return 1

    try:
        _append_history(cfg.output_history, datetime.now(timezone.utc), cfg, payload.get("deciles") or [])
    except Exception:  # pragma: no cover - defensive history write
        LOG.warning("Failed to append ranker_eval history at %s", cfg.output_history, exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
