from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.io_utils import atomic_write_bytes

try:  # pragma: no cover - PyYAML optional
    import yaml
except Exception:  # pragma: no cover - allow running without yaml
    yaml = None

PredictionFile = Tuple[date, Path]


@dataclass(frozen=True)
class EvaluationConfig:
    days: int = 60
    label_horizon: int = 3
    hit_threshold: float = 0.04
    drawdown_threshold: Optional[float] = 0.03
    predictions_dir: Path = Path("data") / "predictions"
    prices_path: Path = Path("data") / "daily_prices.csv"
    output_dir: Path = Path("data") / "ranker_eval"
    as_of: Optional[date] = None

    @property
    def resolved_drawdown(self) -> float:
        return self.drawdown_threshold if self.drawdown_threshold is not None else self.hit_threshold


DATE_FORMAT = "%Y-%m-%d"
PREDICTION_SUFFIX = ".csv"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ranker.yml"


def _load_eval_defaults() -> Dict[str, object]:
    if yaml is None:
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # pragma: no cover - config missing or invalid
        return {}
    eval_cfg = data.get("eval") if isinstance(data, dict) else {}
    return eval_cfg if isinstance(eval_cfg, dict) else {}


EVAL_DEFAULTS = _load_eval_defaults()


def _parse_date_from_name(path: Path) -> Optional[date]:
    try:
        return datetime.strptime(path.stem, DATE_FORMAT).date()
    except ValueError:
        return None


def _list_prediction_files(directory: Path) -> List[PredictionFile]:
    if not directory.exists():
        return []
    candidates: List[PredictionFile] = []
    for child in sorted(directory.glob(f"*{PREDICTION_SUFFIX}")):
        parsed = _parse_date_from_name(child)
        if parsed is None:
            continue
        candidates.append((parsed, child))
    return candidates


def _normalise_price_frame(df: pd.DataFrame, *, symbol: Optional[str] = None) -> pd.DataFrame:
    frame = df.copy()
    if "symbol" not in frame.columns and symbol is not None:
        frame["symbol"] = symbol
    elif "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str)
    else:
        frame["symbol"] = ""

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    elif "timestamp" in frame.columns:
        frame["date"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    else:
        frame["date"] = pd.NaT

    frame.dropna(subset=["date"], inplace=True)
    frame["date"] = frame["date"].dt.date

    def _resolve_col(names: Sequence[str]) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index)

    frame["open"] = _resolve_col(["open", "Open", "OPEN"])
    frame["high"] = _resolve_col(["high", "High", "HIGH"])
    frame["low"] = _resolve_col(["low", "Low", "LOW"])
    frame["close"] = _resolve_col(["close", "Close", "CLOSE", "adj_close", "Adj Close"])

    keep = ["symbol", "date", "open", "high", "low", "close"]
    return frame[keep]


def load_price_history(prices_path: Path) -> pd.DataFrame:
    if prices_path.is_dir():
        frames: List[pd.DataFrame] = []
        for child in sorted(prices_path.glob("*.csv")):
            frames.append(_normalise_price_frame(pd.read_csv(child), symbol=child.stem.upper()))
        if not frames:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close"])
        return pd.concat(frames, ignore_index=True)
    if prices_path.is_file():
        return _normalise_price_frame(pd.read_csv(prices_path))
    return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close"])


def load_prediction_history(cfg: EvaluationConfig) -> pd.DataFrame:
    files = _list_prediction_files(cfg.predictions_dir)
    if not files:
        return pd.DataFrame()

    end_date = cfg.as_of or files[-1][0]
    start_date = end_date - timedelta(days=cfg.days - 1)

    selected: List[PredictionFile] = [
        (file_date, path)
        for file_date, path in files
        if start_date <= file_date <= end_date
    ]
    if not selected:
        return pd.DataFrame()

    frames = []
    for file_date, path in selected:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["as_of"] = file_date
        if "score" in df.columns and "Score" not in df.columns:
            df["Score"] = pd.to_numeric(df["score"], errors="coerce")
        if "passed_gates" in df.columns and "gates_passed" not in df.columns:
            df["gates_passed"] = df["passed_gates"].astype(bool)
        if "score_breakdown_json" in df.columns and "score_breakdown" not in df.columns:
            df["score_breakdown"] = df["score_breakdown_json"]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str)
    return combined


def _series_from_breakdown(values: Iterable[str], components: Sequence[str]) -> pd.DataFrame:
    rows: List[List[float]] = []
    for entry in values:
        try:
            parsed = json.loads(entry) if isinstance(entry, str) else {}
        except json.JSONDecodeError:
            parsed = {}
        row = [float(parsed.get(component, 0.0)) for component in components]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(components))


@dataclass
class LabelResult:
    label: Optional[int]
    max_return: Optional[float]
    min_return: Optional[float]


def _evaluate_price_path(
    price_frame: pd.DataFrame,
    *,
    as_of: date,
    horizon: int,
    hit_threshold: float,
    drawdown_threshold: float,
) -> LabelResult:
    if price_frame.empty:
        return LabelResult(None, None, None)

    frame = price_frame.sort_values("date").reset_index(drop=True)
    base_mask = frame["date"] == as_of
    if not base_mask.any():
        return LabelResult(None, None, None)

    base_close = frame.loc[base_mask, "close"].iloc[-1]
    if not math.isfinite(base_close) or base_close <= 0:
        return LabelResult(None, None, None)

    future = frame.loc[frame["date"] > as_of].head(max(1, horizon))
    if future.empty:
        return LabelResult(0, None, None)

    max_ret = float("-inf")
    min_ret = float("inf")
    hit_target = base_close * (1.0 + hit_threshold)
    stop_target = base_close * (1.0 - drawdown_threshold)

    for _, row in future.iterrows():
        high = row.get("high", np.nan)
        low = row.get("low", np.nan)
        close = row.get("close", np.nan)

        if not math.isfinite(high):
            high = close
        if not math.isfinite(low):
            low = close
        if not math.isfinite(high) or not math.isfinite(low):
            continue

        max_ret = max(max_ret, (high / base_close) - 1.0)
        min_ret = min(min_ret, (low / base_close) - 1.0)

        if high >= hit_target:
            return LabelResult(1, max_ret, min_ret)
        if low <= stop_target:
            return LabelResult(0, max_ret, min_ret)

    return LabelResult(0, max_ret if max_ret != float("-inf") else None, min_ret if min_ret != float("inf") else None)


def label_predictions(
    predictions: pd.DataFrame,
    price_history: pd.DataFrame,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    price_history = price_history.copy()
    price_history["symbol"] = price_history["symbol"].astype(str).str.upper()
    grouped = {sym: frame for sym, frame in price_history.groupby("symbol")}

    labels: List[int] = []
    max_returns: List[Optional[float]] = []
    min_returns: List[Optional[float]] = []

    for _, row in predictions.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        as_of = row.get("as_of")
        if isinstance(as_of, pd.Timestamp):
            as_of_date = as_of.date()
        elif isinstance(as_of, datetime):
            as_of_date = as_of.date()
        elif isinstance(as_of, str):
            try:
                as_of_date = datetime.strptime(as_of, DATE_FORMAT).date()
            except ValueError:
                labels.append(None)
                max_returns.append(None)
                min_returns.append(None)
                continue
        elif isinstance(as_of, date):
            as_of_date = as_of
        else:
            labels.append(None)
            max_returns.append(None)
            min_returns.append(None)
            continue

        frame = grouped.get(symbol)
        if frame is None:
            labels.append(None)
            max_returns.append(None)
            min_returns.append(None)
            continue

        result = _evaluate_price_path(
            frame,
            as_of=as_of_date,
            horizon=cfg.label_horizon,
            hit_threshold=cfg.hit_threshold,
            drawdown_threshold=cfg.resolved_drawdown,
        )
        labels.append(result.label)
        max_returns.append(result.max_return)
        min_returns.append(result.min_return)

    labelled = predictions.copy()
    labelled["label"] = labels
    labelled["max_return"] = max_returns
    labelled["min_return"] = min_returns
    return labelled


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    tpr = cum_pos / pos
    fpr = cum_neg / neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos_total = (y_true == 1).sum()
    if pos_total == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1)
    recall = cum_pos / pos_total

    ap = 0.0
    prev_recall = 0.0
    for prec, rec, label in zip(precision, recall, y_sorted):
        if label <= 0:
            continue
        ap += prec * (rec - prev_recall)
        prev_recall = rec
    return float(ap)


def compute_decile_lifts(y_true: np.ndarray, y_score: np.ndarray, deciles: int = 10) -> Dict[str, Dict[str, float]]:
    if len(y_true) == 0:
        return {}
    frame = pd.DataFrame({"label": y_true, "score": y_score})
    frame.sort_values("score", ascending=False, inplace=True)
    base_rate = frame["label"].mean()
    results: Dict[str, Dict[str, float]] = {}
    total = len(frame)
    start = 0
    for idx in range(deciles):
        remaining = deciles - idx
        chunk_size = max((total - start + remaining - 1) // remaining, 0)
        end = start + chunk_size
        subset = frame.iloc[start:end]
        if subset.empty:
            avg = float("nan")
            lift = float("nan")
        else:
            avg = float(subset["label"].mean())
            lift = float("nan") if base_rate == 0 else float(avg / base_rate)
        results[str(idx + 1)] = {"avg": avg, "lift": lift, "count": float(len(subset))}
        start = end
    return results


def compile_metrics(labelled: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, object]:
    usable = labelled.dropna(subset=["label"]).copy()
    usable["label"] = usable["label"].astype(int)
    y_true = usable["label"].to_numpy(dtype=float)
    score_column = "Score" if "Score" in usable.columns else "score"
    if score_column not in usable.columns:
        scores = np.zeros(len(usable), dtype=float)
    else:
        scores = usable[score_column].to_numpy(dtype=float)

    auc = roc_auc_score(y_true, scores)
    pr = average_precision(y_true, scores)
    deciles = compute_decile_lifts(y_true, scores)

    gate_rate = None
    if "gates_passed" in usable.columns:
        gate_rate = float(usable["gates_passed"].mean()) if len(usable) else None

    metrics: Dict[str, object] = {
        "as_of": (cfg.as_of or date.today()).isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "samples": int(len(usable)),
        "positives": int((usable["label"] == 1).sum()),
        "positive_rate": float(usable["label"].mean()) if len(usable) else 0.0,
        "auc": auc,
        "average_precision": pr,
        "deciles": deciles,
        "hit_threshold": float(cfg.hit_threshold),
        "drawdown_threshold": float(cfg.resolved_drawdown),
        "label_horizon": int(cfg.label_horizon),
        "window_days": int(cfg.days),
    }
    if gate_rate is not None:
        metrics["gate_pass_rate"] = gate_rate
    return metrics


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    serialisable = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, serialisable)


def evaluate(cfg: EvaluationConfig) -> Dict[str, object]:
    predictions = load_prediction_history(cfg)
    prices = load_price_history(cfg.prices_path)
    labelled = label_predictions(predictions, prices, cfg)
    metrics = compile_metrics(labelled, cfg)
    return metrics


def _parse_args(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate nightly ranker predictions")
    defaults = {
        "days": EVAL_DEFAULTS.get("days", EvaluationConfig.days),
        "label_horizon": EVAL_DEFAULTS.get("label_horizon", EvaluationConfig.label_horizon),
        "hit_threshold": EVAL_DEFAULTS.get("hit_threshold", EvaluationConfig.hit_threshold),
        "max_drawdown": EVAL_DEFAULTS.get("max_drawdown", EVAL_DEFAULTS.get("drawdown_threshold", EvaluationConfig.drawdown_threshold)),
    }
    try:
        defaults["days"] = int(defaults["days"])
    except (TypeError, ValueError):
        defaults["days"] = EvaluationConfig.days
    try:
        defaults["label_horizon"] = int(defaults["label_horizon"])
    except (TypeError, ValueError):
        defaults["label_horizon"] = EvaluationConfig.label_horizon
    try:
        defaults["hit_threshold"] = float(defaults["hit_threshold"])
    except (TypeError, ValueError):
        defaults["hit_threshold"] = EvaluationConfig.hit_threshold
    try:
        drawdown_default = (
            float(defaults["max_drawdown"])
            if defaults["max_drawdown"] is not None
            else None
        )
    except (TypeError, ValueError):
        drawdown_default = EvaluationConfig.drawdown_threshold

    parser.add_argument(
        "--days",
        type=int,
        default=defaults["days"],
        help="Number of daily prediction files to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=defaults["label_horizon"],
        help="Days to look ahead when computing labels (default: %(default)s)",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=defaults["hit_threshold"],
        help="Return threshold to consider a hit (default: %(default)s)",
    )
    parser.add_argument(
        "--max-dd",
        "--drawdown-threshold",
        dest="drawdown_threshold",
        type=float,
        default=drawdown_default,
        help="Maximum tolerated drawdown before marking a miss",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=EvaluationConfig.predictions_dir,
        help="Directory containing daily prediction CSV files",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=EvaluationConfig.prices_path,
        help="CSV file or directory providing historical prices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EvaluationConfig.output_dir,
        help="Directory where evaluation JSON should be written",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Override the evaluation date (YYYY-MM-DD). Defaults to the most recent predictions file.",
    )
    args = parser.parse_args(argv)

    as_of = datetime.strptime(args.as_of, DATE_FORMAT).date() if args.as_of else None

    return EvaluationConfig(
        days=max(1, args.days),
        label_horizon=max(1, args.label_horizon),
        hit_threshold=float(args.hit_threshold),
        drawdown_threshold=args.drawdown_threshold,
        predictions_dir=args.predictions_dir,
        prices_path=args.prices,
        output_dir=args.output_dir,
        as_of=as_of,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    metrics = evaluate(cfg)
    if not metrics:
        raise SystemExit("No metrics computed")

    target_date = cfg.as_of or datetime.now(timezone.utc).date()
    output_path = cfg.output_dir / f"{target_date.isoformat()}.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, metrics)
    latest_path = cfg.output_dir / "latest.json"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(output_path.name)
    except OSError:
        _write_json(latest_path, metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
