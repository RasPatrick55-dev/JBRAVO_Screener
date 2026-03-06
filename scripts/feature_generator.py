"""Generate ML features by combining nightly bars with label artifacts.

Usage examples (from repo root with environment activated):
    python scripts/feature_generator.py
    python scripts/feature_generator.py --labels-path data/labels/labels_20240131.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts import db
from scripts.utils.feature_schema import (
    compute_feature_signature,
    infer_feature_columns_for_ml,
    per_file_meta_path,
)

BAR_COLUMNS = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
LABEL_COLUMNS = [
    "symbol",
    "timestamp",
    "label_5d_pos_300bp",
    "label_10d_pos_300bp",
]
FEATURE_SET_ENV = "JBR_ML_FEATURE_SET"
SUPPORTED_FEATURE_SETS = {"v1", "v2"}
RUN_TZ = ZoneInfo("America/New_York")
SPLIT_ADJUST_ENV = "JBR_SPLIT_ADJUST"
DEFAULT_SPLIT_ADJUST = "off"
BARS_ADJUSTMENT_ENV = "JBR_BARS_ADJUSTMENT"
DEFAULT_BARS_ADJUSTMENT = "raw"
ALPACA_BARS_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
SPLIT_ADJUST_MODES = {"off", "auto", "force"}
FEATURE_META_NAME = "latest_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ML features from nightly daily bars and label artifacts. "
            "Features are merged with labels on symbol and timestamp."
        )
    )
    parser.add_argument(
        "--bars-path",
        type=Path,
        default=Path("data/daily_bars.csv"),
        help="Path to the daily bars CSV (expects symbol, timestamp, OHLC, and volume columns).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to labels CSV. If omitted, the newest labels_*.csv "
            "under data/labels is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory where the generated features CSV will be written.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with NaNs in the output instead of dropping them.",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(SUPPORTED_FEATURE_SETS),
        default=None,
        help=("Feature set version. If omitted, reads from JBR_ML_FEATURE_SET (default: v1)."),
    )
    return parser.parse_args()


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


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        required_list = ", ".join(sorted(required))
        raise ValueError(
            f"Input data is missing required columns: {missing_list}. "
            f"Required columns are: {required_list}."
        )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _latest_labels_path(labels_dir: Path) -> Path:
    candidates = sorted(labels_dir.glob("labels_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No labels files found under {labels_dir}")
    return candidates[-1]


def _prepare_bars(bars_path: Path) -> pd.DataFrame:
    if db.db_enabled():
        bars = db.load_ml_artifact_csv("daily_bars")
        if bars.empty:
            raise FileNotFoundError("Bars data not found in DB (ml_artifacts: daily_bars).")
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    else:
        bars = _load_csv(bars_path)
    _require_columns(bars, BAR_COLUMNS)

    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    return bars


def _load_labels(labels_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload: dict[str, Any] = {}
    if db.db_enabled():
        labels = db.load_ml_artifact_csv("labels")
        if labels.empty:
            raise FileNotFoundError("Labels data not found in DB (ml_artifacts: labels).")
        payload = db.load_ml_artifact_payload("labels")
    else:
        labels = _load_csv(labels_path)
    _require_columns(labels, LABEL_COLUMNS)

    labels = labels.copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    return labels.sort_values(["symbol", "timestamp"]).reset_index(drop=True), payload


def _resolve_bars_adjustment(value: Any | None) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in ALPACA_BARS_ADJUSTMENTS:
        return candidate
    env_value = str(os.getenv(BARS_ADJUSTMENT_ENV, "")).strip().lower()
    if env_value in ALPACA_BARS_ADJUSTMENTS:
        return env_value
    return DEFAULT_BARS_ADJUSTMENT


def _resolve_split_adjust_mode(value: Any | None) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in SPLIT_ADJUST_MODES:
        return candidate
    env_value = str(os.getenv(SPLIT_ADJUST_ENV, "")).strip().lower()
    if env_value in SPLIT_ADJUST_MODES:
        return env_value
    return DEFAULT_SPLIT_ADJUST


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _write_feature_meta(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception:
        logging.exception("FEATURES_META_WRITE_FAILED path=%s", path)
        return False


def _attach_adjusted_close(
    bars: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, str, bool]:
    out = bars.copy()
    if "close_adj" in out.columns:
        out["close_adj"] = pd.to_numeric(out["close_adj"], errors="coerce")
        out["close_adj"] = out["close_adj"].fillna(pd.to_numeric(out["close"], errors="coerce"))
        applied = (
            "close" in out.columns
            and (out["close_adj"] - pd.to_numeric(out["close"], errors="coerce"))
            .abs()
            .gt(1e-12)
            .any()
        )
        return out, "close_adj", bool(applied)

    if "close_adj" not in labels.columns:
        return out, "close", False

    labels_adj = labels[["symbol", "timestamp", "close_adj"]].copy()
    labels_adj["close_adj"] = pd.to_numeric(labels_adj["close_adj"], errors="coerce")
    out = out.merge(labels_adj, on=["symbol", "timestamp"], how="left")
    out["close_adj"] = out["close_adj"].fillna(pd.to_numeric(out["close"], errors="coerce"))
    applied = (
        "close" in out.columns
        and (out["close_adj"] - pd.to_numeric(out["close"], errors="coerce")).abs().gt(1e-12).any()
    )
    return out, "close_adj", bool(applied)


def _resolve_feature_set(cli_feature_set: str | None) -> str:
    requested = (
        str(cli_feature_set).strip().lower()
        if cli_feature_set is not None
        else str(os.getenv(FEATURE_SET_ENV, "v1")).strip().lower()
    )
    if requested not in SUPPORTED_FEATURE_SETS:
        logging.warning(
            "[WARN] FEATURES_SET_INVALID requested=%s default=v1 env_var=%s",
            requested,
            FEATURE_SET_ENV,
        )
        return "v1"
    return requested


def _compute_features_v1(bars: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    grouped_close = bars.groupby("symbol")[price_col]
    grouped_volume = bars.groupby("symbol")["volume"]

    features = bars.copy()
    features["ret_1d"] = grouped_close.transform(lambda s: s / s.shift(1) - 1)
    features["mom_5d"] = grouped_close.transform(lambda s: s / s.shift(5) - 1)
    features["mom_10d"] = grouped_close.transform(lambda s: s / s.shift(10) - 1)

    features["vol_10d"] = features.groupby("symbol")["ret_1d"].transform(
        lambda s: s.rolling(window=10, min_periods=5).std()
    )

    features["vol_raw"] = features["volume"]
    features["vol_avg_10d"] = grouped_volume.transform(
        lambda s: s.rolling(window=10, min_periods=5).mean()
    )
    features["vol_rvol_10d"] = features["vol_raw"] / features["vol_avg_10d"]

    selected = [
        "symbol",
        "timestamp",
        "close",
        "mom_5d",
        "mom_10d",
        "vol_10d",
        "vol_raw",
        "vol_avg_10d",
        "vol_rvol_10d",
    ]
    return features[selected]


def _compute_features_v2(bars: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    features = bars.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped_close = features.groupby("symbol")[price_col]
    grouped_volume = features.groupby("symbol")["volume"]
    close_basis = pd.to_numeric(features[price_col], errors="coerce")

    if price_col == "close_adj" and "close" in features.columns:
        raw_close = pd.to_numeric(features["close"], errors="coerce").replace(0.0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            adjust_factor = close_basis / raw_close
        adjust_factor = pd.to_numeric(adjust_factor, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        adjust_factor = adjust_factor.fillna(1.0)
    else:
        adjust_factor = pd.Series(1.0, index=features.index, dtype="float64")

    open_basis = pd.to_numeric(features["open"], errors="coerce") * adjust_factor
    high_basis = pd.to_numeric(features["high"], errors="coerce") * adjust_factor
    low_basis = pd.to_numeric(features["low"], errors="coerce") * adjust_factor

    # Returns / momentum
    features["ret_1d"] = grouped_close.transform(lambda s: s.pct_change(1))
    features["ret_5d"] = grouped_close.transform(lambda s: s.pct_change(5))
    features["ret_10d"] = grouped_close.transform(lambda s: s.pct_change(10))
    features["logret_1d"] = np.log(close_basis / grouped_close.shift(1))
    features["mom_5d"] = grouped_close.transform(lambda s: s / s.shift(5) - 1)
    features["mom_10d"] = grouped_close.transform(lambda s: s / s.shift(10) - 1)

    # Volatility
    features["vol_10d"] = features.groupby("symbol")["ret_1d"].transform(
        lambda s: s.rolling(window=10, min_periods=5).std()
    )
    features["volatility_10d"] = features.groupby("symbol")["logret_1d"].transform(
        lambda s: s.rolling(window=10, min_periods=1).std()
    )
    prev_close = grouped_close.shift(1)
    true_range = pd.concat(
        [
            high_basis - low_basis,
            (high_basis - prev_close).abs(),
            (low_basis - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["tr"] = true_range
    features["atr14"] = features.groupby("symbol")["tr"].transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    )
    features["atr_pct"] = features["atr14"] / close_basis

    # Trend / moving averages
    features["sma_10"] = grouped_close.transform(lambda s: s.rolling(10, min_periods=1).mean())
    features["sma_20"] = grouped_close.transform(lambda s: s.rolling(20, min_periods=1).mean())
    features["sma_50"] = grouped_close.transform(lambda s: s.rolling(50, min_periods=1).mean())
    features["ema_20"] = grouped_close.transform(
        lambda s: s.ewm(span=20, adjust=False, min_periods=1).mean()
    )
    features["dist_sma20"] = (close_basis / features["sma_20"]) - 1.0
    features["dist_ema20"] = (close_basis / features["ema_20"]) - 1.0

    # RSI / MACD / Bollinger
    delta = grouped_close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.groupby(features["symbol"]).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    )
    avg_loss = losses.groupby(features["symbol"]).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    ema_12 = grouped_close.transform(lambda s: s.ewm(span=12, adjust=False, min_periods=1).mean())
    ema_26 = grouped_close.transform(lambda s: s.ewm(span=26, adjust=False, min_periods=1).mean())
    features["macd_line"] = ema_12 - ema_26
    features["macd_signal"] = features.groupby("symbol")["macd_line"].transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=1).mean()
    )
    features["macd_hist"] = features["macd_line"] - features["macd_signal"]
    features["bb_std_20"] = grouped_close.transform(
        lambda s: s.rolling(window=20, min_periods=1).std()
    )
    features["bb_upper"] = features["sma_20"] + (2.0 * features["bb_std_20"])
    features["bb_lower"] = features["sma_20"] - (2.0 * features["bb_std_20"])
    features["bb_bandwidth"] = (features["bb_upper"] - features["bb_lower"]) / features[
        "sma_20"
    ].replace(0, np.nan)

    # Volume / flow
    features["vol_raw"] = features["volume"]
    features["vol_avg_10d"] = grouped_volume.transform(
        lambda s: s.rolling(window=10, min_periods=5).mean()
    )
    features["vol_rvol_10d"] = features["vol_raw"] / features["vol_avg_10d"]
    features["vol_ma30"] = grouped_volume.transform(lambda s: s.rolling(30, min_periods=1).mean())
    features["rel_volume"] = features["volume"] / features["vol_ma30"]
    close_diff_sign = np.sign(grouped_close.diff().fillna(0.0))
    obv_step = close_diff_sign * features["volume"].fillna(0.0)
    features["obv"] = obv_step.groupby(features["symbol"]).cumsum()
    features["obv_delta"] = features.groupby("symbol")["obv"].diff()

    # Candlestick geometry + pattern-lite flags
    bar_range = (high_basis - low_basis).replace(0, np.nan)
    features["candle_body_pct"] = (close_basis - open_basis).abs() / bar_range
    features["candle_upper_wick_pct"] = (
        high_basis - pd.concat([open_basis, close_basis], axis=1).max(axis=1)
    ) / bar_range
    features["candle_lower_wick_pct"] = (
        pd.concat([open_basis, close_basis], axis=1).min(axis=1) - low_basis
    ) / bar_range
    features["candle_doji"] = (features["candle_body_pct"] < 0.10).astype(int)
    features["candle_hammer"] = (
        (features["candle_lower_wick_pct"] >= 0.60)
        & (features["candle_body_pct"] <= 0.30)
        & (features["candle_upper_wick_pct"] <= 0.20)
    ).astype(int)

    open_series = open_basis
    close_series = close_basis
    prev_open = open_series.groupby(features["symbol"]).shift(1)
    prev_close_bar = close_series.groupby(features["symbol"]).shift(1)
    prev_body_low = pd.concat([prev_open, prev_close_bar], axis=1).min(axis=1)
    prev_body_high = pd.concat([prev_open, prev_close_bar], axis=1).max(axis=1)
    curr_body_low = pd.concat([open_series, close_series], axis=1).min(axis=1)
    curr_body_high = pd.concat([open_series, close_series], axis=1).max(axis=1)
    prev_bear = prev_close_bar < prev_open
    prev_bull = prev_close_bar > prev_open
    curr_bull = close_series > open_series
    curr_bear = close_series < open_series
    features["candle_engulfing_bull"] = (
        prev_bear
        & curr_bull
        & (curr_body_low <= prev_body_low)
        & (curr_body_high >= prev_body_high)
    ).astype(int)
    features["candle_engulfing_bear"] = (
        prev_bull
        & curr_bear
        & (curr_body_low <= prev_body_low)
        & (curr_body_high >= prev_body_high)
    ).astype(int)

    # Neutral defaults for oscillators at series start.
    features["rsi14"] = features["rsi14"].fillna(50.0)
    features["macd_hist"] = features["macd_hist"].fillna(0.0)

    base_columns = [
        "symbol",
        "timestamp",
        "close",
        "mom_5d",
        "mom_10d",
        "vol_10d",
        "vol_raw",
        "vol_avg_10d",
        "vol_rvol_10d",
    ]
    v2_columns = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "logret_1d",
        "volatility_10d",
        "atr14",
        "atr_pct",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "dist_sma20",
        "dist_ema20",
        "rsi14",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",
        "vol_ma30",
        "rel_volume",
        "obv",
        "obv_delta",
        "candle_body_pct",
        "candle_upper_wick_pct",
        "candle_lower_wick_pct",
        "candle_doji",
        "candle_hammer",
        "candle_engulfing_bull",
        "candle_engulfing_bear",
    ]

    result = features[base_columns + v2_columns].copy()
    feature_columns = [col for col in result.columns if col not in {"symbol", "timestamp"}]
    result[feature_columns] = result[feature_columns].replace([np.inf, -np.inf], np.nan)
    result[feature_columns] = result[feature_columns].fillna(0.0)
    for col in ("candle_doji", "candle_hammer", "candle_engulfing_bull", "candle_engulfing_bear"):
        result[col] = result[col].astype(int)
    return result


def build_feature_set(
    bars_path: Path, labels_path: Path, keep_na: bool, feature_set: str
) -> Tuple[pd.DataFrame, datetime.date | None, datetime.date | None, dict[str, Any]]:
    bars = _prepare_bars(bars_path)
    labels, labels_payload = _load_labels(labels_path)

    latest_bars_date = _latest_timestamp_date(bars)
    latest_labels_date = _latest_timestamp_date(labels)
    bars, price_col, split_applied_from_join = _attach_adjusted_close(bars, labels)
    bars_adjustment = _resolve_bars_adjustment(labels_payload.get("bars_adjustment"))
    split_adjust_mode = _resolve_split_adjust_mode(labels_payload.get("split_adjust_mode"))
    split_adjust_applied = _truthy(labels_payload.get("split_adjust_applied")) or bool(
        split_applied_from_join
    )

    label_subset = labels[LABEL_COLUMNS].copy()
    if feature_set == "v2":
        features = _compute_features_v2(bars, price_col=price_col)
    else:
        features = _compute_features_v1(bars, price_col=price_col)
    merged = features.merge(label_subset, on=["symbol", "timestamp"], how="inner")

    if merged.empty:
        raise ValueError(
            "Merged feature set is empty; ensure bars and labels share symbol/timestamp keys."
        )

    base_columns = [
        "symbol",
        "timestamp",
        "close",
        "mom_5d",
        "mom_10d",
        "vol_10d",
        "vol_raw",
        "vol_avg_10d",
        "vol_rvol_10d",
    ]
    v2_columns = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "logret_1d",
        "volatility_10d",
        "atr14",
        "atr_pct",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "dist_sma20",
        "dist_ema20",
        "rsi14",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",
        "vol_ma30",
        "rel_volume",
        "obv",
        "obv_delta",
        "candle_body_pct",
        "candle_upper_wick_pct",
        "candle_lower_wick_pct",
        "candle_doji",
        "candle_hammer",
        "candle_engulfing_bull",
        "candle_engulfing_bear",
    ]
    label_columns = [
        "label_5d_pos_300bp",
        "label_10d_pos_300bp",
    ]
    columns = base_columns + (v2_columns if feature_set == "v2" else []) + label_columns

    if not keep_na:
        merged = merged.dropna(subset=columns)

    metadata: dict[str, Any] = {
        "price_col_used": price_col,
        "bars_adjustment": bars_adjustment,
        "split_adjust_mode": split_adjust_mode,
        "split_adjust_applied": bool(split_adjust_applied),
        "labels_payload_present": bool(labels_payload),
    }
    return merged[columns], latest_bars_date, latest_labels_date, metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    feature_set = _resolve_feature_set(args.feature_set)
    logging.info("[INFO] FEATURES_START feature_set=%s", feature_set)

    try:
        labels_path = args.labels_path
        if not db.db_enabled():
            if labels_path is None:
                labels_path = _latest_labels_path(Path("data/labels"))
                logging.info("Using latest labels file: %s", labels_path)

        merged, latest_bars_date, latest_labels_date, feature_meta = build_feature_set(
            args.bars_path, labels_path, keep_na=args.keep_na, feature_set=feature_set
        )
        logging.info(
            "[INFO] FEATURES_PRICE_COL price_col=%s", feature_meta.get("price_col_used", "close")
        )

        run_date = _current_run_date()
        if latest_bars_date and latest_bars_date < run_date:
            logging.warning(
                "[WARN] FEATURES_BARS_STALE latest_bar_date=%s run_date=%s bars_path=%s",
                latest_bars_date,
                run_date,
                args.bars_path,
            )
        if latest_labels_date and latest_labels_date < run_date:
            logging.warning(
                "[WARN] FEATURES_LABELS_STALE latest_label_date=%s run_date=%s labels_path=%s",
                latest_labels_date,
                run_date,
                labels_path,
            )

        as_of = run_date.strftime("%Y-%m-%d")
        output_dir: Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"features_{as_of}.csv"

        merged.to_csv(output_path, index=False)
        logging.info("Features written to %s", output_path)
        ml_feature_columns = infer_feature_columns_for_ml(merged)
        feature_signature = compute_feature_signature(ml_feature_columns)
        features_payload = {
            "feature_set": feature_set,
            "feature_signature": feature_signature,
            "feature_columns": ml_feature_columns,
            "feature_count": int(len(ml_feature_columns)),
            "price_col_used": feature_meta.get("price_col_used", "close"),
            "bars_adjustment": feature_meta.get("bars_adjustment", DEFAULT_BARS_ADJUSTMENT),
            "split_adjust_mode": feature_meta.get("split_adjust_mode", DEFAULT_SPLIT_ADJUST),
            "split_adjust_applied": bool(feature_meta.get("split_adjust_applied")),
            "rows": int(merged.shape[0]),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output_path),
        }
        per_file_meta = per_file_meta_path(output_path)
        latest_meta_path = output_dir / FEATURE_META_NAME
        if _write_feature_meta(per_file_meta, features_payload):
            logging.info(
                "[INFO] FEATURES_META_WRITTEN source=fs feature_set=%s feature_signature=%s rows=%d path=%s feature_count=%d",
                feature_set,
                feature_signature,
                int(merged.shape[0]),
                per_file_meta,
                int(len(ml_feature_columns)),
            )
        _write_feature_meta(latest_meta_path, features_payload)
        if db.db_enabled():
            ok = db.upsert_ml_artifact_frame(
                "features",
                run_date,
                merged,
                payload=features_payload,
                source="feature_generator",
                file_name=output_path.name,
            )
            if ok:
                logging.info(
                    "[INFO] FEATURES_DB_WRITTEN run_date=%s rows=%d",
                    run_date,
                    int(merged.shape[0]),
                )
                logging.info(
                    "[INFO] FEATURES_META_WRITTEN source=db feature_set=%s feature_signature=%s rows=%d path=%s feature_count=%d",
                    feature_set,
                    feature_signature,
                    int(merged.shape[0]),
                    output_path,
                    int(len(ml_feature_columns)),
                )
            else:
                logging.warning("[WARN] FEATURES_DB_WRITE_FAILED run_date=%s", run_date)
        logging.info(
            "[INFO] FEATURES_END rows=%d cols=%d feature_set=%s",
            int(merged.shape[0]),
            int(merged.shape[1]),
            feature_set,
        )
    except (FileNotFoundError, ValueError) as err:
        logging.error(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
