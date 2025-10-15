"""Generate canonical fallback candidates when the screener emits zero rows."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANON = {
    "score breakdown": "score_breakdown",
    "Score Breakdown": "score_breakdown",
    "universe count": "universe_count",
    "Universe Count": "universe_count",
    "adv20": "adv20",
    "ADV20": "adv20",
    "atr%": "atrp",
    "ATR%": "atrp",
}

CANONICAL_COLUMNS: Sequence[str] = (
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
    "entry_price",
    "adv20",
    "atrp",
    "source",
)

_NUMERIC_COLUMNS = ("score", "close", "volume", "universe_count", "entry_price", "adv20", "atrp")
_RENAME_MAP = {
    "Score": "score",
    "Composite Score": "score",
    "symbol": "symbol",
    "Symbol": "symbol",
    "close": "close",
    "Close": "close",
    "price": "close",
    "last": "close",
    "volume": "volume",
    "Volume": "volume",
    "avg_volume": "volume",
    "universe count": "universe_count",
    "Universe Count": "universe_count",
    "universe_count": "universe_count",
    "score breakdown": "score_breakdown",
    "Score Breakdown": "score_breakdown",
    "score_breakdown": "score_breakdown",
    "ADV20": "adv20",
    "adv20": "adv20",
    "adv_20": "adv20",
    "avg_daily_volume": "adv20",
    "ATR%": "atrp",
    "atrp": "atrp",
    "ATR_pct": "atrp",
    "atr_percent": "atrp",
    "exchange": "exchange",
    "primary_exchange": "exchange",
    "Primary Exchange": "exchange",
    "timestamp": "timestamp",
    "run_utc": "timestamp",
    "entry_price": "entry_price",
    "Entry": "entry_price",
    "entry": "entry_price",
    "source": "source",
}

_DEFAULT_ROW = {
    "timestamp": "",
    "symbol": "AAPL",
    "score": 0.0,
    "exchange": "UNKNOWN",
    "close": 0.0,
    "volume": 0,
    "universe_count": 0,
    "score_breakdown": "fallback",
    "entry_price": 0.0,
    "adv20": 0.0,
    "atrp": 0.0,
    "source": "fallback",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        LOGGER.warning("FALLBACK_LOAD_FAILED path=%s error=%s", path, exc)
    return pd.DataFrame()


def _latest_prediction_frame(predictions_dir: Path) -> pd.DataFrame:
    if not predictions_dir.exists():
        return pd.DataFrame()
    candidates: list[Tuple[float, Path]] = []
    for csv_path in predictions_dir.glob("*.csv"):
        try:
            stat = csv_path.stat()
        except OSError:
            continue
        candidates.append((stat.st_mtime, csv_path))
    for _, csv_path in sorted(candidates, reverse=True):
        frame = _safe_read_csv(csv_path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _canonicalize_columns(columns: Iterable[object]) -> dict[object, str]:
    mapping: dict[object, str] = {}
    for column in columns:
        key = str(column).strip()
        mapping[column] = _RENAME_MAP.get(key, _RENAME_MAP.get(key.lower(), key.lower()))
    return mapping


def _canonical_frame(df: Optional[pd.DataFrame], now_ts: Optional[str] = None) -> pd.DataFrame:
    now_value = now_ts or _now_iso()
    frame = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if frame.empty and not isinstance(df, pd.DataFrame):
        frame = pd.DataFrame()

    if not frame.empty:
        frame = frame.rename(columns=_canonicalize_columns(frame.columns))
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        if frame.columns.duplicated().any():
            frame = frame.loc[:, ~frame.columns.duplicated()]

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            default_value = _DEFAULT_ROW[column]
            frame[column] = default_value

    if "timestamp" in frame.columns:
        frame["timestamp"] = (
            frame["timestamp"].astype("string").fillna("").str.strip().replace({"": now_value})
        )
    else:
        frame["timestamp"] = now_value

    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype("string").str.strip()

    if "entry_price" in frame.columns and "close" in frame.columns:
        entry_numeric = pd.to_numeric(frame["entry_price"], errors="coerce")
        close_numeric = pd.to_numeric(frame["close"], errors="coerce")
        frame["entry_price"] = entry_numeric.where(entry_numeric > 0, close_numeric)
        frame["entry_price"] = frame["entry_price"].fillna(close_numeric).fillna(0.0)

    for column in _NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(_DEFAULT_ROW[column])

    frame["score_breakdown"] = frame["score_breakdown"].astype("string").fillna("fallback")
    frame["exchange"] = frame["exchange"].astype("string").fillna("UNKNOWN").replace({"": "UNKNOWN"})
    frame["source"] = "fallback"

    frame = frame[list(CANONICAL_COLUMNS)]
    frame = frame.dropna(subset=["symbol", "close", "score"], how="any")
    frame = frame[frame["symbol"].astype("string").str.len() > 0]
    frame = frame.drop_duplicates(subset=["symbol"])

    if frame.empty:
        fallback = pd.DataFrame([_DEFAULT_ROW.copy()])
        fallback["timestamp"] = now_value
        return fallback

    if (frame["adv20"] > 0).any():
        frame = frame.sort_values(["adv20", "score"], ascending=[False, False])
    elif (frame["score"].notna()).any():
        frame = frame.sort_values(["score", "volume"], ascending=[False, False])
    elif (frame["volume"] > 0).any():
        frame = frame.sort_values("volume", ascending=False)
    else:
        frame = frame.sort_values("symbol", ascending=True)

    return frame.reset_index(drop=True)


def normalize_candidate_df(df: Optional[pd.DataFrame], now_ts: Optional[str] = None) -> pd.DataFrame:
    """Public helper retained for callers that expect the legacy normalizer."""

    return _canonical_frame(df, now_ts)


def _write_candidates(base_dir: Path, prepared: pd.DataFrame) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    top_path = data_dir / "top_candidates.csv"
    latest_path = data_dir / "latest_candidates.csv"
    prepared.to_csv(top_path, index=False)
    prepared.to_csv(latest_path, index=False)


def generate_candidates(base_dir: Path, *, max_rows: int = 3) -> Tuple[pd.DataFrame, str]:
    data_dir = base_dir / "data"
    now_value = _now_iso()

    scored = _safe_read_csv(data_dir / "scored_candidates.csv")
    if not scored.empty:
        prepared = _canonical_frame(scored, now_value).head(max_rows)
        if not prepared.empty:
            return prepared, "scored"

    predictions = _latest_prediction_frame(data_dir / "predictions")
    if not predictions.empty:
        prepared = _canonical_frame(predictions, now_value).head(max_rows)
        if not prepared.empty:
            return prepared, "predictions"

    fallback = _canonical_frame(pd.DataFrame([_DEFAULT_ROW.copy()]), now_value).head(max_rows)
    return fallback, "synthetic"


def ensure_min_candidates(
    base_dir: Path,
    min_rows: int = 1,
    *,
    canonicalize: bool = True,
    prefer: str = "top_then_scored",
    max_rows: int = 3,
) -> Tuple[int, str]:
    """Maintain compatibility with previous callers by writing canonical fallback rows."""

    prepared, source = generate_candidates(base_dir, max_rows=max_rows)
    if canonicalize:
        prepared = _canonical_frame(prepared)
    _write_candidates(base_dir, prepared.head(max_rows))
    return max(len(prepared), min_rows), source


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - fallback - %(message)s")

    prepared, source = generate_candidates(base_dir)
    _write_candidates(base_dir, prepared)
    LOGGER.info("FALLBACK generated rows=%s source=%s", len(prepared), source)
    LOGGER.debug("FALLBACK payload=%s", json.dumps(prepared.to_dict(orient="records"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
