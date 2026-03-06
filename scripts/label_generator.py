"""Utility to generate forward return labels from daily bars data.

This script ingests a CSV containing daily bars for multiple symbols and
produces forward returns and binary labels for configurable horizons.

Usage example (from repo root):
    python scripts/label_generator.py --bars-path data/bars/daily_bars.csv

Another example with custom thresholds and horizons:
    python scripts/label_generator.py \
        --bars-path data/bars/daily_bars.csv \
        --horizons 3 7 14 \
        --threshold-percent 2.5
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import db
from scripts.utils.fwd_return_sanity import log_forward_return_sanity
from scripts.utils.split_adjust import SPLIT_MODES, adjust_for_splits

REQUIRED_COLUMNS = {"symbol", "timestamp", "close"}
RUN_TZ = ZoneInfo("America/New_York")
SPLIT_ADJUST_ENV = "JBR_SPLIT_ADJUST"
STRICT_FWD_RET_ENV = "JBR_STRICT_FWD_RET"
BARS_ADJUSTMENT_ENV = "JBR_BARS_ADJUSTMENT"
DEFAULT_BARS_ADJUSTMENT = "raw"
ALPACA_BARS_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
OUTLIER_SUGGESTION = "set JBR_BARS_ADJUSTMENT=split or enable --split-adjust auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate forward return labels for multiple symbols from a daily bars CSV.")
    )
    parser.add_argument(
        "--bars-path",
        required=True,
        type=Path,
        help="Path to the input CSV containing daily bars with symbol, timestamp, and close columns.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 10],
        help=(
            "Forward return horizons in trading days. Provide one or more integers. "
            "Defaults to 5 and 10."
        ),
    )
    parser.add_argument(
        "--threshold-percent",
        dest="threshold_pct",
        type=float,
        default=3.0,
        help=(
            "Positive return threshold in percent used to generate binary labels. "
            "Defaults to 3.0 (i.e., 300 basis points)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory where the output labels CSV will be written.",
    )
    parser.add_argument(
        "--split-adjust",
        choices=sorted(SPLIT_MODES),
        default=None,
        help=(
            "Split-adjust mode for forward-return generation "
            "(off|auto|force). Precedence: CLI > JBR_SPLIT_ADJUST > off."
        ),
    )
    parser.add_argument(
        "--strict-fwd-ret",
        action="store_true",
        default=None,
        help=(
            "Fail labels generation with rc=2 when severe forward-return outliers "
            "are detected. Precedence: CLI > JBR_STRICT_FWD_RET > false."
        ),
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Input data is missing required columns: {missing_list}. "
            "Required columns are: symbol, timestamp, close."
        )


def _format_threshold_label(threshold_percent: float) -> str:
    basis_points = int(round(threshold_percent * 100))
    return f"pos_{basis_points}bp"


def compute_forward_returns(
    df: pd.DataFrame, horizons: Iterable[int], *, price_col: str = "close"
) -> pd.DataFrame:
    result = df.copy()
    sorted_horizons = sorted(set(horizons))
    if price_col not in result.columns:
        raise KeyError(f"Missing price column '{price_col}' for forward-return computation.")
    prices = pd.to_numeric(result[price_col], errors="coerce")

    for horizon in sorted_horizons:
        fwd_ret_col = f"fwd_ret_{horizon}d"
        result[fwd_ret_col] = prices.groupby(result["symbol"]).shift(-horizon) / prices - 1
    return result


def add_labels(df: pd.DataFrame, horizons: Iterable[int], threshold_percent: float) -> pd.DataFrame:
    labeled = df.copy()
    threshold_decimal = threshold_percent / 100.0
    threshold_label = _format_threshold_label(threshold_percent)

    for horizon in sorted(set(horizons)):
        fwd_ret_col = f"fwd_ret_{horizon}d"
        if fwd_ret_col not in labeled.columns:
            raise KeyError(
                f"Missing forward return column {fwd_ret_col}. Ensure compute_forward_returns was run first."
            )
        label_col = f"label_{horizon}d_{threshold_label}"
        labeled[label_col] = (labeled[fwd_ret_col] >= threshold_decimal).astype(int)
    return labeled


def load_bars(path: Path) -> pd.DataFrame:
    if db.db_enabled():
        bars_df = db.load_ml_artifact_csv("daily_bars")
        if bars_df.empty:
            raise FileNotFoundError("Bars data not found in DB (ml_artifacts: daily_bars).")
        if "timestamp" in bars_df.columns:
            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True, errors="coerce")
        return bars_df
    if not path.exists():
        raise FileNotFoundError(f"Bars file not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _current_run_date() -> datetime.date:
    try:
        return datetime.now(RUN_TZ).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def _latest_timestamp_date(df: pd.DataFrame) -> datetime.date | None:
    if "timestamp" not in df.columns or df.empty:
        return None

    latest_timestamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
    if pd.isna(latest_timestamp):
        return None

    try:
        return latest_timestamp.tz_convert(RUN_TZ).date()
    except Exception:
        return latest_timestamp.date()


def _resolve_split_adjust(cli_value: str | None) -> tuple[str, str]:
    if cli_value:
        value = str(cli_value).strip().lower()
        if value in SPLIT_MODES:
            return value, "cli"
    env_value = (os.getenv(SPLIT_ADJUST_ENV) or "").strip().lower()
    if env_value in SPLIT_MODES:
        return env_value, "env"
    return "off", "default"


def _as_bool(value: str | bool | None, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_strict_fwd_ret(cli_value: bool | None) -> tuple[bool, str]:
    if cli_value is True:
        return True, "cli"
    env_value = os.getenv(STRICT_FWD_RET_ENV)
    if env_value is not None:
        return _as_bool(env_value, False), "env"
    return False, "default"


def _resolve_bars_adjustment() -> str:
    env_value = (os.getenv(BARS_ADJUSTMENT_ENV) or "").strip().lower()
    if env_value in ALPACA_BARS_ADJUSTMENTS:
        return env_value
    return DEFAULT_BARS_ADJUSTMENT


def _sanitize_split_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(meta or {})
    # Keep payload JSON compact and deterministic.
    payload.pop("events_sample", None)
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    logger = logging.getLogger(__name__)

    bars_df = load_bars(args.bars_path)
    validate_columns(bars_df)

    # Sort to enforce deterministic forward calculations within each symbol.
    bars_df = bars_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    split_adjust_mode, split_adjust_source = _resolve_split_adjust(args.split_adjust)
    strict_fwd_ret, strict_fwd_ret_source = _resolve_strict_fwd_ret(args.strict_fwd_ret)
    bars_adjustment = _resolve_bars_adjustment()
    logger.info(
        "[INFO] SPLIT_ADJUST value=%s source=%s",
        split_adjust_mode,
        split_adjust_source,
    )
    logger.info(
        "[INFO] STRICT_FWD_RET value=%s source=%s",
        str(strict_fwd_ret).lower(),
        strict_fwd_ret_source,
    )

    split_meta: dict[str, Any] = {
        "split_adjust_mode": split_adjust_mode,
        "split_adjust_applied": False,
        "split_events": 0,
        "split_symbols": 0,
        "rows_affected": 0,
    }
    if split_adjust_mode != "off":
        bars_df, split_meta = adjust_for_splits(
            bars_df,
            mode=split_adjust_mode,
            price_col="close",
            volume_col="volume",
            group_col="symbol",
            time_col="timestamp",
        )
        if split_meta.get("split_adjust_applied"):
            logger.warning(
                "[WARN] SPLIT_ADJUST_APPLIED symbols=%d events=%d rows_affected=%d",
                int(split_meta.get("split_symbols") or 0),
                int(split_meta.get("split_events") or 0),
                int(split_meta.get("rows_affected") or 0),
            )

    price_col_for_returns = (
        "close_adj"
        if split_meta.get("split_adjust_applied") and "close_adj" in bars_df.columns
        else "close"
    )

    with_returns = compute_forward_returns(bars_df, args.horizons, price_col=price_col_for_returns)
    labeled = add_labels(with_returns, args.horizons, args.threshold_pct)
    fwd_ret_sanity: dict[str, Any] = {}
    severe_outlier = False
    for horizon in sorted(set(args.horizons)):
        fwd_ret_col = f"fwd_ret_{horizon}d"
        if fwd_ret_col not in labeled.columns:
            continue
        summary = log_forward_return_sanity(
            labeled[fwd_ret_col],
            column_name=fwd_ret_col,
            logger=logger,
            suggestion=OUTLIER_SUGGESTION,
        )
        fwd_ret_sanity[fwd_ret_col] = summary
        if bool(summary.get("outlier_suspected")):
            severe_outlier = True

    if strict_fwd_ret and severe_outlier:
        logger.error(
            "[ERROR] STRICT_FWD_RET_TRIGGERED rc=2 suggestion=%s",
            OUTLIER_SUGGESTION,
        )
        return 2

    run_date = _current_run_date()
    latest_bar_date = _latest_timestamp_date(labeled)
    if latest_bar_date and latest_bar_date < run_date:
        logging.warning(
            "[WARN] LABELS_INPUT_STALE latest_bar_date=%s run_date=%s bars_path=%s",
            latest_bar_date,
            run_date,
            args.bars_path,
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labels_{run_date.strftime('%Y%m%d')}.csv"

    labeled.to_csv(output_path, index=False)
    logging.info("[INFO] LABELS_WRITTEN path=%s rows=%d", output_path, int(labeled.shape[0]))
    if db.db_enabled():
        labels_payload = {
            "bars_adjustment": bars_adjustment,
            "split_adjust_mode": split_adjust_mode,
            "split_adjust_applied": bool(split_meta.get("split_adjust_applied")),
            "split_adjust_counts": {
                "symbols": int(split_meta.get("split_symbols") or 0),
                "events": int(split_meta.get("split_events") or 0),
                "rows_affected": int(split_meta.get("rows_affected") or 0),
            },
            "split_adjust_price_col": price_col_for_returns,
            "strict_fwd_ret": bool(strict_fwd_ret),
            "fwd_return_sanity": fwd_ret_sanity,
            "horizons": [int(value) for value in sorted(set(args.horizons))],
            "threshold_percent": float(args.threshold_pct),
            "split_adjust_meta": _sanitize_split_meta(split_meta),
        }
        ok = db.upsert_ml_artifact_frame(
            "labels",
            run_date,
            labeled,
            payload=labels_payload,
            source="label_generator",
            file_name=output_path.name,
        )
        if ok:
            logging.info(
                "[INFO] LABELS_DB_WRITTEN run_date=%s rows=%d", run_date, int(labeled.shape[0])
            )
        else:
            logging.warning("[WARN] LABELS_DB_WRITE_FAILED run_date=%s", run_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
