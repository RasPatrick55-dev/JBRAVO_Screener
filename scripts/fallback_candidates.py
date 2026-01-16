"""Generate canonical fallback candidates when the screener emits zero rows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone, timedelta
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
    "SMA9": "sma9",
    "EMA20": "ema20",
    "SMA180": "sma180",
    "RSI14": "rsi14",
    "gates_passed": "passed_gates",
    "passed_gates": "passed_gates",
    "gate_fail_reason": "gate_fail_reason",
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
    "sma9",
    "ema20",
    "sma180",
    "rsi14",
    "passed_gates",
    "gate_fail_reason",
)

CANONICAL = list(CANONICAL_COLUMNS)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LATEST_CANDIDATES = DATA_DIR / "latest_candidates.csv"
SCORED_CANDIDATES = DATA_DIR / "scored_candidates.csv"
PREDICTIONS_DIR = DATA_DIR / "predictions"

_NUMERIC_COLUMNS = (
    "score",
    "close",
    "volume",
    "universe_count",
    "entry_price",
    "adv20",
    "atrp",
    "sma9",
    "ema20",
    "sma180",
    "rsi14",
)
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
    "SMA9": "sma9",
    "EMA20": "ema20",
    "SMA180": "sma180",
    "RSI14": "rsi14",
    "gates_passed": "passed_gates",
    "passed_gates": "passed_gates",
    "gate_fail_reason": "gate_fail_reason",
}

_DEFAULT_ROW = {
    "timestamp": "",
    "symbol": "",
    "score": 0.0,
    "exchange": "",
    "close": 0.0,
    "volume": 0,
    "universe_count": 0,
    "score_breakdown": "{}",
    "entry_price": 0.0,
    "adv20": 0.0,
    "atrp": 0.0,
    "source": "fallback",
    "sma9": 0.0,
    "ema20": 0.0,
    "sma180": 0.0,
    "rsi14": 0.0,
    "passed_gates": True,
    "gate_fail_reason": "[]",
}

_STATIC_FALLBACK_ROWS = [
    {
        "timestamp": "",
        "symbol": "AAPL",
        "score": 0.0,
        "exchange": "NASDAQ",
        "close": 190.0,
        "volume": 10_000_000,
        "universe_count": 0,
        "score_breakdown": "{}",
        "entry_price": 190.0,
        "adv20": 60_000_000.0,
        "atrp": 0.02,
        "source": "fallback",
        "sma9": 0.0,
        "ema20": 0.0,
        "sma180": 0.0,
        "rsi14": 0.0,
        "passed_gates": True,
        "gate_fail_reason": "[]",
    },
    {
        "timestamp": "",
        "symbol": "SPY",
        "score": 0.0,
        "exchange": "ARCA",
        "close": 430.0,
        "volume": 75_000_000,
        "universe_count": 0,
        "score_breakdown": "{}",
        "entry_price": 430.0,
        "adv20": 90_000_000.0,
        "atrp": 0.015,
        "source": "fallback",
        "sma9": 0.0,
        "ema20": 0.0,
        "sma180": 0.0,
        "rsi14": 0.0,
        "passed_gates": True,
        "gate_fail_reason": "[]",
    },
    {
        "timestamp": "",
        "symbol": "QQQ",
        "score": 0.0,
        "exchange": "NASDAQ",
        "close": 360.0,
        "volume": 50_000_000,
        "universe_count": 0,
        "score_breakdown": "{}",
        "entry_price": 360.0,
        "adv20": 45_000_000.0,
        "atrp": 0.018,
        "source": "fallback",
        "sma9": 0.0,
        "ema20": 0.0,
        "sma180": 0.0,
        "rsi14": 0.0,
        "passed_gates": True,
        "gate_fail_reason": "[]",
    },
]

SCORED_STALE_MINUTES = 30


def _write_static_three(path):
    # three liquid tickers as a safety net
    now = datetime.utcnow().isoformat()
    rows = [
      [now,"AAPL",0.0,"NASDAQ",190.00,0,0,"{}",190.00,0.0,0.0,"fallback:static",0.0,0.0,0.0,0.0,True,"[]"],
      [now,"SPY", 0.0,"NYSEARCA",520.00,0,0,"{}",520.00,0.0,0.0,"fallback:static",0.0,0.0,0.0,0.0,True,"[]"],
      [now,"QQQ", 0.0,"NASDAQ",460.00,0,0,"{}",460.00,0.0,0.0,"fallback:static",0.0,0.0,0.0,0.0,True,"[]"],
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(CANONICAL); w.writerows(rows)
    print("[INFO] FALLBACK_CHECK rows_out=3 source=fallback:static")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        LOGGER.warning("FALLBACK_LOAD_FAILED path=%s error=%s", path, exc)
    return pd.DataFrame()


def _is_stale(path: Path, *, max_age_minutes: int = SCORED_STALE_MINUTES) -> bool:
    if not path.exists():
        return True
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return True
    age = datetime.now(timezone.utc) - modified
    return age > timedelta(minutes=max_age_minutes)


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

    for column in ("sma9", "ema20", "sma180", "rsi14"):
        values = source_frame.get(column)
        out[column] = pd.to_numeric(values, errors="coerce") if values is not None else pd.Series(
            _DEFAULT_ROW[column], index=out.index
        )
        out[column] = out[column].fillna(_DEFAULT_ROW[column])

    out["volume"] = out["volume"].clip(lower=0)
    out["universe_count"] = out["universe_count"].clip(lower=0)

    score_breakdown = source_frame.get("score_breakdown", pd.Series(index=out.index, dtype="string"))
    out["score_breakdown"] = (
        score_breakdown.astype("string")
        .fillna("")
        .str.strip()
        .replace({"": "{}", "fallback": "{}"})
    )

    out["entry_price"] = out["entry_price"].where(out["entry_price"] > 0, out["close"])
    out["entry_price"] = out["entry_price"].where(out["entry_price"] > 0, _DEFAULT_ROW["entry_price"])

    out["adv20"] = out["adv20"].clip(lower=0)
    out["atrp"] = out["atrp"].clip(lower=0)

    passed_series = source_frame.get("passed_gates")
    if passed_series is None:
        passed_series = source_frame.get("gates_passed")
    if passed_series is not None:
        out["passed_gates"] = passed_series
    else:
        out["passed_gates"] = _DEFAULT_ROW["passed_gates"]

    fail_series = source_frame.get("gate_fail_reason")
    if fail_series is not None:
        out["gate_fail_reason"] = fail_series
    else:
        out["gate_fail_reason"] = _DEFAULT_ROW["gate_fail_reason"]

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
    source_frame = source_frame.loc[mask_symbol] if not source_frame.empty else source_frame

    out = out.reset_index(drop=True)
    source_frame = source_frame.reset_index(drop=True)

    out = out[list(CANONICAL_COLUMNS)]

    extra_columns = [column for column in source_frame.columns if column not in out.columns]
    for column in extra_columns:
        try:
            out[column] = source_frame[column]
        except Exception:
            out[column] = source_frame.get(column)

    return out


def normalize_candidate_df(df: Optional[pd.DataFrame], now_ts: Optional[str] = None) -> pd.DataFrame:
    """Public helper retained for callers that expect the legacy normalizer."""

    return _canonical_frame(df, now_ts)


def _guard_fallback_candidates(
    frame: pd.DataFrame,
    *,
    min_adv_usd: float,
    min_price: float,
    max_price: float,
    allowed_exchanges: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame

    price_series = frame.get("entry_price")
    if price_series is None:
        price_series = frame.get("close")
    price_numeric = pd.to_numeric(price_series, errors="coerce")
    fallback_close = pd.to_numeric(frame.get("close"), errors="coerce")
    price_numeric = price_numeric.fillna(fallback_close)

    exchange_series = frame.get("exchange", pd.Series(dtype="string")).astype("string").str.upper()
    adv_numeric = pd.to_numeric(frame.get("adv20"), errors="coerce")
    score_numeric = pd.to_numeric(frame.get("score"), errors="coerce").fillna(0.0)
    volume_numeric = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    allowed = {str(exc).upper() for exc in allowed_exchanges}

    mask = (
        price_numeric.between(float(min_price), float(max_price), inclusive="both")
        & adv_numeric.ge(float(min_adv_usd))
        & exchange_series.isin(allowed)
        & price_numeric.notna()
        & adv_numeric.notna()
    )

    guarded = frame.loc[mask].copy()
    if guarded.empty:
        return guarded

    now_value = _now_iso()
    guarded["timestamp"] = now_value
    guarded["entry_price"] = price_numeric.loc[guarded.index].fillna(guarded["close"])
    guarded["score"] = score_numeric.loc[guarded.index]
    guarded["volume"] = volume_numeric.loc[guarded.index].clip(lower=0)
    guarded["universe_count"] = pd.to_numeric(
        guarded.get("universe_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    if "score_breakdown" in guarded.columns:
        sb_series = (
            guarded["score_breakdown"].astype("string").fillna("").replace({"": "{}", "fallback": "{}"})
        )
        guarded["score_breakdown"] = sb_series
    else:
        guarded["score_breakdown"] = pd.Series(
            ["{}"] * len(guarded.index), index=guarded.index, dtype="string"
        )
    guarded["source"] = "fallback"
    guarded = guarded.drop_duplicates(subset=["symbol"], keep="first")

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
    write_csv_atomic(str(top_path), prepared[list(CANONICAL_COLUMNS)])


def build_latest_candidates(
    base_dir: Path | None = None,
    *,
    max_rows: int = 1,
    min_adv_usd: float = 2_000_000.0,
    min_price: float = 2.0,
    max_price: float = 150.0,
    exchanges: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Construct the latest candidates CSV using canonical fallback ordering."""

    base = base_dir or PROJECT_ROOT
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    allowed_exchanges = tuple(exchanges) if exchanges else ("NASDAQ", "NYSE", "ARCA")
    scored_path = data_dir / "scored_candidates.csv"
    scored = _safe_read_csv(scored_path)
    scored_stale = _is_stale(scored_path)
    scored_age_minutes: float | None = None
    if scored_path.exists():
        try:
            modified = datetime.fromtimestamp(scored_path.stat().st_mtime, tz=timezone.utc)
            scored_age_minutes = (datetime.now(timezone.utc) - modified).total_seconds() / 60.0
        except OSError:
            scored_age_minutes = None
    prepared_frames: list[pd.DataFrame] = []

    if not scored.empty and not scored_stale:
        canonical = _canonical_frame(scored)
        guarded = _guard_fallback_candidates(
            canonical,
            min_adv_usd=min_adv_usd,
            min_price=min_price,
            max_price=max_price,
            allowed_exchanges=allowed_exchanges,
        )
        if not guarded.empty:
            guarded["_source_tag"] = "scored"
            prepared_frames.append(guarded)
    elif scored_stale:
        LOGGER.info(
            "[INFO] FALLBACK_SCORED_STALE minutes=%.1f path=%s",
            scored_age_minutes if scored_age_minutes is not None else -1.0,
            scored_path,
        )

    predictions = _latest_prediction_frame(data_dir / "predictions")
    if not predictions.empty:
        canonical = _canonical_frame(predictions)
        guarded = _guard_fallback_candidates(
            canonical,
            min_adv_usd=min_adv_usd,
            min_price=min_price,
            max_price=max_price,
            allowed_exchanges=allowed_exchanges,
        )
        if not guarded.empty:
            guarded["_source_tag"] = "predictions"
            prepared_frames.append(guarded)

    static_frame = _canonical_frame(pd.DataFrame(_STATIC_FALLBACK_ROWS))
    static_guarded = _guard_fallback_candidates(
        static_frame,
        min_adv_usd=min_adv_usd,
        min_price=min_price,
        max_price=max_price,
        allowed_exchanges=allowed_exchanges,
    )
    if static_guarded.empty:
        static_guarded = _canonical_frame(pd.DataFrame(_STATIC_FALLBACK_ROWS))
        static_guarded["timestamp"] = _now_iso()
    static_guarded["_source_tag"] = "static"
    prepared_frames.append(static_guarded)

    combined = pd.concat(prepared_frames, ignore_index=True, sort=False)
    combined["_source_tag"] = pd.Categorical(
        combined["_source_tag"],
        categories=["scored", "predictions", "static"],
        ordered=True,
    )
    combined = combined.sort_values(
        by=["_source_tag", "score", "adv20", "volume", "symbol"],
        ascending=[True, False, False, False, True],
    )
    combined = combined.drop_duplicates(subset=["symbol"], keep="first")
    now_value = _now_iso()
    combined["timestamp"] = now_value
    if "score_breakdown" in combined.columns:
        combined["score_breakdown"] = (
            combined["score_breakdown"].astype("string").fillna("fallback")
        )
    else:
        combined["score_breakdown"] = "fallback"
    combined["universe_count"] = pd.to_numeric(
        combined.get("universe_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    combined["source"] = "fallback"

    tag_series = combined.get("_source_tag")
    fallback_only = True
    if tag_series is not None and not tag_series.empty:
        unique_tags = {
            str(tag).strip().lower()
            for tag in pd.Series(tag_series).dropna().unique().tolist()
        }
        fallback_only = not any(tag in {"scored", "predictions"} for tag in unique_tags)

    if fallback_only:
        static_rows = combined.loc[combined.get("_source_tag") == "static"].copy()
        if static_rows.empty:
            static_rows = combined.copy()
        symbol_order = {
            str(row.get("symbol")): idx for idx, row in enumerate(_STATIC_FALLBACK_ROWS)
        }
        static_rows["_fallback_order"] = static_rows.get("symbol").astype("string").map(
            symbol_order
        ).fillna(len(symbol_order)).astype(int)
        static_rows = static_rows.sort_values("_fallback_order", kind="stable")
        selected = static_rows.head(max(3, max_rows)).copy()
        selected = selected.drop(columns=["_fallback_order"], errors="ignore")
    else:
        selected = combined.head(max_rows).copy()

    if selected.empty:
        selected = combined.head(max(3, max_rows)).copy()

    selected_tags = selected.get("_source_tag", pd.Series(["static"] * len(selected))).tolist()
    selected = selected.drop(columns=["_source_tag"], errors="ignore")

    source_labels: list[str] = []
    normalized_tags: list[str] = []
    for tag in selected_tags:
        tag_str = str(tag).strip().lower() if str(tag).strip() else "static"
        normalized_tags.append(tag_str)
        if tag_str == "scored":
            source_labels.append("fallback:scored")
        elif tag_str == "predictions":
            source_labels.append("fallback:predictions")
        else:
            source_labels.append("fallback:static")
    if not source_labels:
        source_labels = ["fallback:static"] * len(selected.index)

    selected["source"] = source_labels
    prepared = selected[list(CANONICAL_COLUMNS)]
    _write_latest(data_dir, prepared)

    latest_path = data_dir / "latest_candidates.csv"
    written = pd.read_csv(latest_path)
    if written.shape[0] < 1:
        raise RuntimeError("Fallback candidates write produced no rows")

    primary_source = "static"
    for tag_str in normalized_tags:
        if tag_str in {"scored", "predictions"}:
            primary_source = tag_str
            break

    source_token = source_labels[0] if source_labels else "fallback:static"
    LOGGER.info("[INFO] FALLBACK_CHECK rows_out=%d source=%s", len(prepared), source_token)
    return prepared, primary_source


def generate_candidates(base_dir: Path, *, max_rows: int = 3, **kwargs: object) -> Tuple[pd.DataFrame, str]:
    frame, source = build_latest_candidates(base_dir, max_rows=max_rows, **kwargs)
    return frame.head(max_rows), source


def ensure_min_candidates(
    base_dir: Path,
    min_rows: int = 1,
    *,
    canonicalize: bool = True,
    prefer: str = "top_then_scored",
    max_rows: int = 3,
    **kwargs: object,
) -> Tuple[int, str]:
    """Maintain compatibility with previous callers by writing canonical fallback rows."""

    prepared, source = generate_candidates(base_dir, max_rows=max_rows, **kwargs)
    if canonicalize:
        prepared = _canonical_frame(prepared)
    _write_candidates(base_dir, prepared.head(max_rows))
    return max(len(prepared), min_rows), source


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fallback candidate list")
    parser.add_argument("--top-n", type=int, default=3, help="Number of fallback symbols to emit")
    parser.add_argument(
        "--min-adv20-usd",
        type=float,
        default=2_000_000.0,
        help="Minimum 20-day average dollar volume threshold",
    )
    parser.add_argument(
        "--min-order-usd",
        type=float,
        default=None,
        help=(
            "Optional compatibility flag; ignored because order sizing is handled "
            "by scripts.execute_trades."
        ),
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=2.0,
        help="Minimum allowed price for fallback symbols",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=150.0,
        help="Maximum allowed price for fallback symbols",
    )
    parser.add_argument(
        "--exchanges",
        type=str,
        default="NASDAQ,NYSE,ARCA",
        help="Comma separated list of allowed exchanges",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root for locating data directory",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - fallback - %(message)s")

    base_dir = Path(args.base_dir)
    if args.min_order_usd is not None:
        LOGGER.info(
            "fallback_candidates: ignoring --min-order-usd=%s (compat)",
            args.min_order_usd,
        )
    scored_path = Path(SCORED_CANDIDATES)
    latest_path = Path(LATEST_CANDIDATES)
    top_path = latest_path.parent / "top_candidates.csv"
    if not scored_path.exists() or scored_path.stat().st_size == 0:
        _write_static_three(latest_path)
        _write_static_three(top_path)

    exchanges = [exc.strip() for exc in str(args.exchanges).split(",") if exc.strip()]
    prepared, source = generate_candidates(
        base_dir,
        max_rows=args.top_n,
        min_adv_usd=args.min_adv20_usd,
        min_price=args.min_price,
        max_price=args.max_price,
        exchanges=exchanges,
    )
    if prepared.empty:
        _write_static_three(latest_path)
        _write_static_three(top_path)
        try:
            prepared = pd.read_csv(latest_path)
        except Exception:
            prepared = pd.DataFrame(columns=CANONICAL)
        source = "fallback:static"
    _write_candidates(base_dir, prepared)
    LOGGER.info("FALLBACK produced rows=%d", len(prepared))
    LOGGER.info("FALLBACK source=%s", source)
    LOGGER.debug("FALLBACK payload=%s", json.dumps(prepared.to_dict(orient="records"), default=str))

    sm_path = DATA_DIR / "screener_metrics.json"
    existing: dict[str, object]
    if sm_path.exists():
        try:
            existing = json.loads(sm_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    else:
        existing = {}
    payload = {
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_in": existing.get("symbols_in"),
        "symbols_with_bars": existing.get("symbols_with_bars"),
        "bars_rows_total": existing.get("bars_rows_total"),
        "rows": existing.get("rows"),
    }
    try:
        sm_path.parent.mkdir(parents=True, exist_ok=True)
        sm_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to update screener_metrics.json from fallback", exc_info=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
