"""Train an ML rank model from screener outcomes."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from utils.env import load_env  # noqa: E402

try:  # pragma: no cover - environment dependent
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:  # pragma: no cover - environment dependent
    LogisticRegression = None
    StandardScaler = None
    joblib = None


LOGGER = logging.getLogger("train_rank_model")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_PATH = BASE_DIR / "data" / "models" / "rank_model.joblib"

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
    return frame


def _train_model(df: pd.DataFrame) -> tuple[object, object, pd.DataFrame, pd.Series]:
    if LogisticRegression is None or StandardScaler is None or joblib is None:
        raise RuntimeError("scikit-learn/joblib not available; install requirements.txt")

    features = _build_features(df)
    target = (pd.to_numeric(df["ret_5d"], errors="coerce") > 0).astype(int)
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    if features.empty:
        raise RuntimeError("No training rows after filtering missing features.")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, target)
    return model, scaler, features, target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to write model artifact",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of training rows (most recent first)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv or sys.argv[1:])

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

    try:
        model, scaler, features, target = _train_model(training_df)
    except Exception as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(features.shape[0]),
        "pos_rate": float(target.mean()),
    }
    joblib.dump(payload, output_path)
    LOGGER.info(
        "Model saved: %s (rows=%d pos_rate=%.3f)",
        output_path,
        int(features.shape[0]),
        float(target.mean()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
