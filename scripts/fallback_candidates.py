"""Generate canonical fallback candidates when the screener emits zero rows."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from utils import write_csv_atomic

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LATEST_CANDIDATES = DATA_DIR / "latest_candidates.csv"
SCORED_CANDIDATES = DATA_DIR / "scored_candidates.csv"
PREDICTIONS_DIR = DATA_DIR / "predictions"

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
    "ATR": "atrp",
    "atr": "atrp",
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
    "symbol": "",
    "score": 0.0,
    "exchange": "",
    "close": 0.0,
    "volume": 0,
    "universe_count": 0,
    "score_breakdown": "fallback",
    "entry_price": 0.0,
    "adv20": 0.0,
    "atrp": 0.0,
    "source": "fallback",
}

_STATIC_FALLBACK_ROW = {
    "timestamp": "",
    "symbol": "AAPL",
    "score": 0.0,
    "exchange": "NASDAQ",
    "close": 175.0,
    "volume": 0,
    "universe_count": 0,
    "score_breakdown": "fallback",
    "entry_price": 175.0,
    "adv20": 5_000_000.0,
    "atrp": 0.02,
    "source": "fallback:static",
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
    latest_path = predictions_dir / "latest.csv"
    frame = _safe_read_csv(latest_path)
    if not frame.empty:
        return frame
    if not predictions_dir.exists():
        return pd.DataFrame()
    candidates: list[Tuple[float, Path]] = []
    for csv_path in predictions_dir.glob("*.csv"):
        if csv_path.name == "latest.csv":
            continue
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
    source_frame = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if source_frame.empty and not isinstance(df, pd.DataFrame):
        source_frame = pd.DataFrame()

    if not source_frame.empty:
        source_frame = source_frame.rename(columns=_canonicalize_columns(source_frame.columns))
        source_frame.columns = [str(col).strip().lower() for col in source_frame.columns]
        if source_frame.columns.duplicated().any():
            source_frame = source_frame.loc[:, ~source_frame.columns.duplicated()]

    out = pd.DataFrame(index=source_frame.index.copy())

    ts_series = source_frame.get("timestamp")
    if ts_series is not None:
        out["timestamp"] = (
            ts_series.astype("string").fillna("").str.strip().replace({"": now_value})
        )
    else:
        out["timestamp"] = now_value

    out["symbol"] = (
        source_frame.get("symbol", pd.Series(index=out.index, dtype="string"))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    exchange_series = source_frame.get("exchange", pd.Series(index=out.index, dtype="string"))
    out["exchange"] = (
        exchange_series.astype("string").fillna("").str.strip().str.upper()
    )

    for column in ("score", "close", "volume", "universe_count", "entry_price", "adv20", "atrp"):
        values = source_frame.get(column)
        out[column] = pd.to_numeric(values, errors="coerce") if values is not None else pd.Series(
            _DEFAULT_ROW[column], index=out.index
        )
        out[column] = out[column].fillna(_DEFAULT_ROW[column])

    out["volume"] = out["volume"].clip(lower=0)
    out["universe_count"] = out["universe_count"].clip(lower=0)

    score_breakdown = source_frame.get("score_breakdown", pd.Series(index=out.index, dtype="string"))
    out["score_breakdown"] = (
        score_breakdown.astype("string").fillna("").str.strip().replace({"": "fallback"})
    )

    out["entry_price"] = out["entry_price"].where(out["entry_price"] > 0, out["close"])
    out["entry_price"] = out["entry_price"].where(out["entry_price"] > 0, _DEFAULT_ROW["entry_price"])

    out["adv20"] = out["adv20"].clip(lower=0)
    out["atrp"] = out["atrp"].clip(lower=0)

    source_series = source_frame.get("source")
    if source_series is not None:
        out["source"] = source_series.astype("string").fillna("fallback")
    else:
        out["source"] = _DEFAULT_ROW["source"]

    out = out[list(CANONICAL_COLUMNS)]
    out["timestamp"] = out["timestamp"].replace({"": now_value})
    out["timestamp"] = out["timestamp"].fillna(now_value)

    mask_symbol = out["symbol"].astype("string").str.len() > 0
    out = out.loc[mask_symbol]
    out = out.reset_index(drop=True)
    return out


def normalize_candidate_df(df: Optional[pd.DataFrame], now_ts: Optional[str] = None) -> pd.DataFrame:
    """Public helper retained for callers that expect the legacy normalizer."""

    return _canonical_frame(df, now_ts)


def _guard_fallback_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    close_numeric = pd.to_numeric(frame["close"], errors="coerce")
    exchange_series = frame["exchange"].astype("string").fillna("").str.strip()
    adv_numeric = pd.to_numeric(frame["adv20"], errors="coerce")
    symbol_series = frame["symbol"].astype("string").fillna("").str.strip()

    adv_mask = adv_numeric.isna() | (adv_numeric <= 0) | (adv_numeric >= 2_000_000)
    mask = (close_numeric > 0) & (exchange_series != "") & adv_mask & (symbol_series != "")

    guarded = frame.loc[mask].copy()
    if guarded.empty:
        return guarded

    guarded["timestamp"] = _now_iso()
    guarded = guarded.drop_duplicates(subset=["symbol"])

    guarded = guarded.sort_values(
        by=["score", "adv20", "volume", "symbol"],
        ascending=[False, False, False, True],
    )
    return guarded.reset_index(drop=True)


def _write_latest(data_dir: Path, frame: pd.DataFrame) -> None:
    latest_path = data_dir / "latest_candidates.csv"
    write_csv_atomic(str(latest_path), frame[list(CANONICAL_COLUMNS)])
    LOGGER.info(
        "[INFO] FALLBACK_OUT rows=%d path=%s",
        len(frame.index),
        os.path.relpath(latest_path, start=PROJECT_ROOT),
    )


def _write_candidates(base_dir: Path, prepared: pd.DataFrame) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    top_path = data_dir / "top_candidates.csv"
    write_csv_atomic(str(top_path), prepared[list(prepared.columns)])


def build_latest_candidates(base_dir: Path | None = None, *, max_rows: int = 1) -> tuple[pd.DataFrame, str]:
    """Construct the latest candidates CSV using canonical fallback ordering."""

    base = base_dir or PROJECT_ROOT
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scored = _safe_read_csv(data_dir / "scored_candidates.csv")
    if not scored.empty:
        canonical = _canonical_frame(scored)
        canonical["source"] = "fallback:scored"
        guarded = _guard_fallback_candidates(canonical)
        if not guarded.empty:
            prepared = guarded.head(max_rows)
            _write_latest(data_dir, prepared)
            return prepared, "scored"

    predictions = _latest_prediction_frame(data_dir / "predictions")
    if not predictions.empty:
        canonical = _canonical_frame(predictions)
        canonical["source"] = "fallback:predictions"
        guarded = _guard_fallback_candidates(canonical)
        if not guarded.empty:
            prepared = guarded.head(max_rows)
            _write_latest(data_dir, prepared)
            return prepared, "predictions"

    fallback = _canonical_frame(pd.DataFrame([_STATIC_FALLBACK_ROW.copy()]))
    fallback["source"] = "fallback:static"
    fallback["timestamp"] = _now_iso()
    prepared = fallback.head(max_rows)
    _write_latest(data_dir, prepared)
    return prepared, "static"


def generate_candidates(base_dir: Path, *, max_rows: int = 3) -> Tuple[pd.DataFrame, str]:
    frame, source = build_latest_candidates(base_dir, max_rows=max_rows)
    return frame.head(max_rows), source


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
    LOGGER.info("FALLBACK produced rows=%d", len(prepared))
    LOGGER.info("FALLBACK source=%s", source)
    LOGGER.debug("FALLBACK payload=%s", json.dumps(prepared.to_dict(orient="records"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
