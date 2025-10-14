from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANON = {
    "score breakdown": "score_breakdown",
    "Score Breakdown": "score_breakdown",
    "universe count": "universe_count",
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_rename(columns: Iterable[object]) -> dict[object, str]:
    mapping: dict[object, str] = {}
    for column in columns:
        key = str(column).strip()
        mapping[column] = CANON.get(key, CANON.get(key.lower(), key))
    return mapping


def normalize_candidate_df(df: Optional[pd.DataFrame], now_ts: Optional[str] = None) -> pd.DataFrame:
    now_value = now_ts or _now_iso()
    frame = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if not frame.empty or isinstance(df, pd.DataFrame):
        frame = frame.rename(columns=_canonical_rename(frame.columns))
        frame.columns = [str(col).strip() for col in frame.columns]
        if frame.columns.duplicated().any():
            frame = frame.loc[:, ~frame.columns.duplicated()]

    if "entry_price" not in frame.columns and "close" in frame.columns:
        frame["entry_price"] = frame["close"]

    for column in REQUIRED + OPTIONAL:
        if column not in frame.columns:
            frame[column] = pd.NA

    for column in ("score", "close", "volume", "universe_count", "entry_price"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "entry_price" in frame.columns and "close" in frame.columns:
        frame["entry_price"] = frame["entry_price"].fillna(frame["close"])

    if "timestamp" not in frame.columns:
        frame["timestamp"] = now_value
    else:
        ts_series = frame["timestamp"].astype("string").fillna(now_value).str.strip()
        frame["timestamp"] = ts_series.replace({"": now_value, "NaT": now_value, "NaN": now_value})

    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype("string").str.strip()

    core = [column for column in ("symbol", "score", "close") if column in frame.columns]
    if core:
        frame = frame.dropna(subset=[column for column in core if column != "symbol"])
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] != ""]

    if "score" in frame.columns:
        frame = frame.sort_values("score", ascending=False)

    if "source" not in frame.columns:
        frame["source"] = "fallback"

    ordered_columns = REQUIRED + ["entry_price"] + [col for col in OPTIONAL if col != "entry_price"] + [
        column
        for column in frame.columns
        if column not in REQUIRED + OPTIONAL + ["source", "entry_price"]
    ]
    ordered_columns.append("source")
    frame = frame.reindex(columns=ordered_columns, fill_value=pd.NA)
    return frame.reset_index(drop=True)


def prepare_latest_candidates(
    df: pd.DataFrame,
    source: Optional[str],
    *,
    canonicalize: bool,
) -> pd.DataFrame:
    prepared = normalize_candidate_df(df, now_ts=_now_iso()) if canonicalize else df.copy()
    if "entry_price" not in prepared.columns and "close" in prepared.columns:
        prepared["entry_price"] = prepared["close"]

    if source is not None:
        prepared["source"] = source
    elif "source" in prepared.columns:
        prepared["source"] = prepared["source"].astype("string").fillna("screener")
    else:
        prepared["source"] = "screener"

    keep_order = REQUIRED + ["entry_price"] + [col for col in OPTIONAL if col != "entry_price"] + [
        column
        for column in prepared.columns
        if column not in REQUIRED + OPTIONAL + ["source", "entry_price"]
    ]
    keep_order.append("source")
    return prepared.reindex(columns=keep_order, fill_value=pd.NA)


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


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        LOGGER.warning("[INFO] FALLBACK_LOAD_FAILED path=%s error=%s", path, exc)
    return pd.DataFrame()


def _previous_prediction_frame(pred_dir: Path, now_ts: str, max_rows: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=36)
    if not pred_dir.exists():
        return pd.DataFrame()
    candidates: list[tuple[float, Path]] = []
    for path in pred_dir.glob("*.csv"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime >= cutoff.timestamp():
            candidates.append((stat.st_mtime, path))
    for _, path in sorted(candidates, reverse=True):
        frame = _load_csv(path)
        if frame.empty:
            continue
        prepared = normalize_candidate_df(frame, now_ts)
        if not prepared.empty:
            return prepared.head(max_rows)
    return pd.DataFrame()


def ensure_min_candidates(
    base_dir: Path,
    min_rows: int = 1,
    *,
    canonicalize: bool = False,
    prefer: str = "top_then_scored",
    max_rows: int = 3,
) -> Tuple[int, str]:
    """Ensure ``data/latest_candidates.csv`` contains at least ``min_rows`` rows."""

    data_dir = base_dir / "data"
    latest = data_dir / "latest_candidates.csv"
    now_ts = _now_iso()

    existing = _load_csv(latest)
    if not existing.empty and len(existing.dropna(how="all")) >= min_rows:
        if canonicalize:
            prepared = prepare_latest_candidates(existing, None, canonicalize=True)
            prepared.to_csv(latest, index=False)
            return len(prepared), "already_populated"
        return len(existing), "already_populated"

    order = ["top", "scored"] if prefer == "top_then_scored" else ["scored", "top"]
    sources: list[tuple[str, Path]] = []
    for label in order:
        if label == "top":
            sources.append(("top", data_dir / "top_candidates.csv"))
        else:
            sources.append(("scored", data_dir / "scored_candidates.csv"))

    for label, path in sources:
        LOGGER.info("[INFO] FALLBACK_START source=%s", label)
        frame = _load_csv(path).head(max_rows)
        if frame.empty:
            LOGGER.info("[INFO] FALLBACK_END rows_out=0 reason=empty_%s", label)
            continue
        prepared = normalize_candidate_df(frame, now_ts) if canonicalize else frame.copy()
        prepared = prepared.head(max_rows)
        rows_written, reason = _write_latest(prepared, latest, f"{label}_candidates", source=label, canonicalize=True)
        LOGGER.info("[INFO] FALLBACK_END rows_out=%s reason=%s", rows_written, reason)
        if rows_written >= min_rows:
            return rows_written, reason

    LOGGER.info("[INFO] FALLBACK_START source=previous_day")
    previous = _previous_prediction_frame(data_dir / "predictions", now_ts, max_rows)
    if not previous.empty:
        rows_written, reason = _write_latest(previous, latest, "previous_predictions", source="previous", canonicalize=True)
        LOGGER.info("[INFO] FALLBACK_END rows_out=%s reason=%s", rows_written, reason)
        if rows_written >= min_rows:
            return rows_written, reason
    else:
        LOGGER.info("[INFO] FALLBACK_END rows_out=0 reason=no_previous")

    filler = normalize_candidate_df(pd.DataFrame([{"symbol": "AAPL", "score": 0.0, "exchange": "NASDAQ", "close": 1.0, "volume": 1, "universe_count": 1, "score_breakdown": "fallback"}]), now_ts)
    rows_written, reason = _write_latest(filler, latest, "static_fallback", source="fallback", canonicalize=True)
    LOGGER.info("[INFO] FALLBACK_END rows_out=%s reason=%s", rows_written, reason)
    return rows_written, reason
