"""Train an ML rank model from screener outcomes."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from utils.env import load_env  # noqa: E402

try:  # pragma: no cover - environment dependent
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:  # pragma: no cover - environment dependent
    LogisticRegression = None
    accuracy_score = None
    roc_auc_score = None
    StandardScaler = None
    joblib = None


LOGGER = logging.getLogger("train_rank_model")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_DIR = BASE_DIR / "data" / "models"
DEFAULT_TRAIN_WINDOW_DAYS = 60
MIN_TRAIN_SAMPLES = 30
MIN_SAMPLES_10D = 60

FEATURE_COLUMNS = [
    "score",
    "sma9",
    "ema20",
    "sma180",
    "rsi14",
    "adv20",
    "atrp",
    "ma_spread_9_20",
    "ma_spread_20_180",
]


def _series_by_alias(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return pd.to_numeric(frame[alias], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["score"] = _series_by_alias(frame, "score", "Score")
    features["sma9"] = _series_by_alias(frame, "sma9", "SMA9")
    features["ema20"] = _series_by_alias(frame, "ema20", "EMA20")
    features["sma180"] = _series_by_alias(frame, "sma180", "SMA180")
    features["rsi14"] = _series_by_alias(frame, "rsi14", "RSI14")
    features["adv20"] = _series_by_alias(frame, "adv20", "ADV20")
    features["atrp"] = _series_by_alias(frame, "atrp", "ATR_pct", "ATR%")
    features["ma_spread_9_20"] = features["sma9"] - features["ema20"]
    features["ma_spread_20_180"] = features["ema20"] - features["sma180"]
    return features


def _load_training_data(conn, limit: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT
            o.run_date,
            o.symbol,
            o.ret_5d,
            c.score,
            c.sma9,
            c.ema20,
            c.sma180,
            c.rsi14,
            c.adv20,
            c.atrp
        FROM screener_outcomes_app o
        JOIN screener_candidates c
          ON c.run_date = o.run_date
         AND c.symbol = o.symbol
        WHERE o.ret_5d IS NOT NULL
        ORDER BY o.run_date DESC
    """
    params: list[object] = []
    if limit is not None and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["ret_5d"] = pd.to_numeric(frame["ret_5d"], errors="coerce")
    frame["run_date"] = pd.to_datetime(frame["run_date"], errors="coerce").dt.date
    return frame


def _select_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if window_days <= 0 or df.empty:
        return df
    dates = pd.to_datetime(df["run_date"], errors="coerce")
    unique_dates = pd.Series(dates.dropna().unique()).sort_values().tolist()
    if not unique_dates:
        return df.iloc[0:0].copy()
    window_dates = set(unique_dates[-window_days:])
    return df.loc[dates.isin(window_dates)].copy()


def _train_model(
    df: pd.DataFrame,
) -> tuple[object, object, pd.DataFrame, pd.Series, float | None, float | None]:
    if LogisticRegression is None or StandardScaler is None or joblib is None:
        raise RuntimeError("scikit-learn/joblib not available; install requirements.txt")

    features = _build_features(df)
    target = (pd.to_numeric(df["ret_5d"], errors="coerce") > 0).astype(int)
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    if features.empty:
        raise RuntimeError("No training rows after filtering missing features.")

    dates = pd.to_datetime(df.loc[features.index, "run_date"], errors="coerce")
    order = dates.sort_values().index if dates.notna().any() else features.index
    features = features.loc[order]
    target = target.loc[order]
    split_idx = max(1, int(len(features) * 0.8))
    if split_idx >= len(features):
        split_idx = max(len(features) - 1, 1)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_val = target.iloc[split_idx:]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, y_train)
    val_auc = None
    val_acc = None
    if not X_val.empty and accuracy_score is not None:
        X_val_scaled = scaler.transform(X_val)
        preds = model.predict(X_val_scaled)
        val_acc = float(accuracy_score(y_val, preds))
        if roc_auc_score is not None:
            try:
                probas = model.predict_proba(X_val_scaled)[:, 1]
                val_auc = float(roc_auc_score(y_val, probas))
            except Exception:
                val_auc = None
    return model, scaler, features, target, val_auc, val_acc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write model artifact (default: data/models/rank_model_<YYYYMMDD>.joblib)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of training rows (most recent first)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Rolling training window (trading days). Default 60.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv or sys.argv[1:])
    env_window = os.getenv("TRAIN_WINDOW_DAYS")
    try:
        window_days = (
            int(args.window_days) if args.window_days is not None else int(env_window or 0)
        )
    except (TypeError, ValueError):
        window_days = 0
    if window_days <= 0:
        window_days = DEFAULT_TRAIN_WINDOW_DAYS

    conn = db.get_db_conn()
    if conn is None:
        LOGGER.error("DB unavailable; cannot train rank model")
        return 1

    try:
        training_df = _load_training_data(conn, limit=args.limit)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if training_df.empty:
        LOGGER.error("No training data found in screener_outcomes_app")
        return 1

    training_df = _select_window(training_df, window_days)
    if training_df.empty:
        LOGGER.warning("No training rows after applying %d-day window; skipping.", window_days)
        return 0

    recent = _select_window(training_df, 10)
    samples_10d = int(recent.shape[0])
    if samples_10d < MIN_SAMPLES_10D:
        LOGGER.warning(
            "Insufficient 10-day samples (%d < %d); skipping training.",
            samples_10d,
            MIN_SAMPLES_10D,
        )
        return 0

    if len(training_df) < MIN_TRAIN_SAMPLES:
        LOGGER.warning(
            "Insufficient samples (%d < %d); skipping training.",
            len(training_df),
            MIN_TRAIN_SAMPLES,
        )
        return 0

    try:
        model, scaler, features, target, val_auc, val_acc = _train_model(training_df)
    except Exception as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1

    coeffs = None
    if hasattr(model, "coef_"):
        coeffs = {name: float(value) for name, value in zip(FEATURE_COLUMNS, model.coef_[0])}
        coeffs = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))

    trained_at = datetime.now(timezone.utc)
    model_date = trained_at.strftime("%Y%m%d")
    model_version = f"rank_model_{model_date}"
    output_path = args.output or (DEFAULT_MODEL_DIR / f"{model_version}.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "trained_at": trained_at.isoformat(),
        "model_version": model_version,
        "window_days": int(window_days),
        "rows": int(features.shape[0]),
        "pos_rate": float(target.mean()),
        "val_auc": val_auc,
        "val_accuracy": val_acc,
        "samples_used": int(features.shape[0]),
        "features_used": FEATURE_COLUMNS,
    }
    joblib.dump(payload, output_path)
    LOGGER.info(
        "Model saved: %s (samples_used=%d pos_rate=%.3f window_days=%d model_version=%s)",
        output_path,
        int(features.shape[0]),
        float(target.mean()),
        int(window_days),
        model_version,
    )
    LOGGER.info(
        "Validation: auc=%s accuracy=%s",
        f"{val_auc:.4f}" if isinstance(val_auc, float) else "n/a",
        f"{val_acc:.4f}" if isinstance(val_acc, float) else "n/a",
    )
    if coeffs:
        LOGGER.info("Feature coefficients: %s", coeffs)
    metadata_path = DEFAULT_MODEL_DIR / f"{model_version}.json"
    try:
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        metadata_payload = {
            "model_version": model_version,
            "trained_at": trained_at.isoformat(),
            "samples_used": int(features.shape[0]),
            "samples_10d": samples_10d,
            "features_used": FEATURE_COLUMNS,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "window_days": int(window_days),
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - non-fatal
        LOGGER.warning("Failed to write model metadata %s: %s", metadata_path, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
