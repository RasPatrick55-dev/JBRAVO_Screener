from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

CANON = {
    "score breakdown": "score_breakdown",
    "universe count": "universe_count",
    "Score Breakdown": "score_breakdown",
    "Universe Count": "universe_count",
}

REQUIRED = [
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
]

OPTIONAL = ["entry_price", "adv20", "atrp"]


def normalize_candidate_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()
    df = df.copy()
    rename_map: dict[str, str] = {}
    for column in df.columns:
        key = column.strip()
        target = CANON.get(key)
        if target is None:
            target = CANON.get(key.lower())
        rename_map[column] = target or key
    if rename_map:
        df = df.rename(columns=rename_map)
    df.columns = [col.strip() for col in df.columns]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T

    for col in REQUIRED:
        if col not in df.columns:
            if col in ("universe_count", "volume"):
                df[col] = 0
            else:
                df[col] = pd.NA

    if "entry_price" not in df.columns:
        df["entry_price"] = df.get("close")

    numeric_cols = [col for col in ("score", "close", "volume", "universe_count") if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "entry_price" in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        if "close" in df.columns:
            df["entry_price"] = df["entry_price"].fillna(df["close"])

    essential = [col for col in ("symbol", "score", "close") if col in df.columns]
    if essential:
        drop_targets = [col for col in essential if col != "symbol"]
        if drop_targets:
            df = df.dropna(subset=drop_targets)
        if "symbol" in essential:
            df["symbol"] = df["symbol"].astype("string").str.strip()
            df = df[df["symbol"] != ""]

    if "timestamp" in df.columns:
        now_iso = pd.Timestamp.utcnow().isoformat()
        ts_series = df["timestamp"].astype("string")
        ts_series = ts_series.fillna(now_iso).str.strip()
        ts_series = ts_series.replace({"": now_iso, "NaT": now_iso, "NaN": now_iso})
        df["timestamp"] = ts_series

    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    return df.reset_index(drop=True)


def prepare_latest_candidates(
    df: pd.DataFrame,
    source: Optional[str],
    *,
    canonicalize: bool,
) -> pd.DataFrame:
    prepared = normalize_candidate_df(df) if canonicalize else df.copy()
    if "entry_price" not in prepared.columns:
        prepared["entry_price"] = prepared.get("close")

    if source is not None:
        prepared["source"] = source
    elif "source" in prepared.columns:
        prepared["source"] = (
            prepared["source"].astype("string").fillna("screener")
        )
    else:
        prepared["source"] = "screener"

    optional_present = [col for col in OPTIONAL if col in prepared.columns and col != "entry_price"]
    keep_order = REQUIRED + ["entry_price"] + optional_present + ["source"]
    for column in keep_order:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[keep_order]


def _write_latest(
    df: pd.DataFrame,
    latest: Path,
    reason: str,
    *,
    source: str,
    canonicalize: bool,
) -> tuple[int, str]:
    prepared = prepare_latest_candidates(df, source, canonicalize=canonicalize)
    prepared.to_csv(latest, index=False)
    return len(prepared), reason


def ensure_min_candidates(
    base_dir: Path,
    min_rows: int = 1,
    *,
    canonicalize: bool = False,
) -> Tuple[int, str]:
    """Ensure ``data/latest_candidates.csv`` contains at least ``min_rows`` rows."""

    data_dir = base_dir / "data"
    latest = data_dir / "latest_candidates.csv"
    scored = data_dir / "scored_candidates.csv"
    top = data_dir / "top_candidates.csv"

    if latest.exists():
        try:
            existing = pd.read_csv(latest)
            if canonicalize:
                prepared = prepare_latest_candidates(existing, None, canonicalize=True)
                prepared.to_csv(latest, index=False)
                rows = len(prepared)
            else:
                rows = max(len(existing), 0)
            if rows >= min_rows:
                return rows, "already_populated"
        except Exception:
            pass

    if scored.exists():
        df = pd.read_csv(scored)
        if "adv20" in df.columns:
            df = df[df["adv20"] >= 2_000_000]
        df = df[(df["close"] >= 1.0) & (df["close"] <= 60.0)]
        if "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]
        df = df.sort_values("score", ascending=False).head(max(1, min_rows))
        if len(df) > 0:
            return _write_latest(
                df,
                latest,
                "scored_candidates",
                source="screener",
                canonicalize=canonicalize,
            )

    if top.exists():
        df = pd.read_csv(top).head(max(1, min_rows))
        if len(df) > 0:
            return _write_latest(
                df,
                latest,
                "top_candidates",
                source="screener",
                canonicalize=canonicalize,
            )

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "symbol": "AAPL",
                "score": 0.0,
                "exchange": "NASDAQ",
                "close": 1.0,
                "volume": 1,
                "universe_count": 1,
                "score_breakdown": "fallback",
                "entry_price": 1.0,
            }
        ]
    )
    return _write_latest(
        df,
        latest,
        "static_fallback",
        source="fallback",
        canonicalize=canonicalize,
    )
