from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

from .utils.models import BarData, classify_exchange
from utils.io_utils import atomic_write_bytes

try:  # pragma: no cover - compatibility across pydantic versions
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - older pydantic exposes it elsewhere
    from pydantic.error_wrappers import ValidationError  # type: ignore


LOGGER = logging.getLogger(__name__)

INPUT_COLUMNS = [
    "symbol",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

SCORED_COLUMNS = [
    "symbol",
    "exchange",
    "timestamp",
    "score",
    "close",
    "volume",
    "score_breakdown",
    "rsi",
    "macd",
    "macd_hist",
    "adx",
    "aroon_up",
    "aroon_down",
    "sma9",
    "ema20",
    "sma180",
    "atr",
]

TOP_COLUMNS = [
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
    "rsi",
    "macd",
    "macd_hist",
    "adx",
    "aroon_up",
    "aroon_down",
    "sma9",
    "ema20",
    "sma180",
    "atr",
]

SKIP_KEYS = [
    "UNKNOWN_EXCHANGE",
    "NON_EQUITY",
    "VALIDATION_ERROR",
    "NAN_DATA",
    "INSUFFICIENT_HISTORY",
]

DEFAULT_TOP_N = 15
DEFAULT_MIN_HISTORY = 30
ASSET_CACHE_RELATIVE = Path("data") / "reference" / "assets_cache.json"
ASSET_HTTP_TIMEOUT = 15


def _ensure_logger() -> None:
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _assets_cache_path(base_dir: Optional[Path] = None) -> Path:
    base = base_dir or Path(__file__).resolve().parents[1]
    return base / ASSET_CACHE_RELATIVE


def _read_assets_cache(cache_path: Path) -> List[str]:
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - cache corruption is unexpected
        LOGGER.warning("Failed to read assets cache %s: %s", cache_path, exc)
        return []
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        return []
    cleaned: List[str] = []
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if symbol:
            cleaned.append(symbol)
    return cleaned


def _write_assets_cache(cache_path: Path, symbols: List[str]) -> None:
    if not symbols:
        return
    payload = {
        "cached_utc": _format_timestamp(datetime.now(timezone.utc)),
        "symbols": sorted({str(sym).strip().upper() for sym in symbols if sym}),
    }
    atomic_write_bytes(cache_path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _determine_alpaca_base_url() -> str:
    base_url = os.environ.get("APCA_API_BASE_URL")
    if base_url:
        return base_url.rstrip("/")
    env = os.environ.get("APCA_API_ENV", "").lower()
    if env == "live":
        return "https://api.alpaca.markets"
    return "https://paper-api.alpaca.markets"


def _fetch_assets_via_http() -> List[str]:
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        LOGGER.warning("Alpaca credentials missing; cannot fetch assets via HTTP fallback.")
        return []

    base_url = _determine_alpaca_base_url()
    url = f"{base_url}/v2/assets"
    try:
        response = requests.get(
            url,
            params={"status": "active", "asset_class": "us_equity"},
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=ASSET_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        LOGGER.warning("Raw Alpaca asset fetch failed: %s", exc)
        return []

    if not isinstance(payload, list):
        LOGGER.warning("Unexpected payload from Alpaca assets endpoint: %s", type(payload))
        return []

    symbols: set[str] = set()
    for item in payload:
        if isinstance(item, dict):
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol:
                symbols.add(symbol)
    return sorted(symbols)


def fetch_active_equity_symbols(trading_client, *, base_dir: Optional[Path] = None) -> List[str]:
    """Fetch active US equity symbols from Alpaca with resilient fallbacks."""

    cache_path = _assets_cache_path(base_dir)
    try:
        assets = trading_client.get_all_assets(status="active", asset_class="us_equity")
    except Exception as exc:  # pragma: no cover - requires Alpaca SDK error
        LOGGER.warning("Alpaca assets validation failed; falling back to raw HTTP: %s", exc)
        symbols = _fetch_assets_via_http()
        if symbols:
            _write_assets_cache(cache_path, symbols)
            return symbols

        cached = _read_assets_cache(cache_path)
        if cached:
            LOGGER.warning(
                "Using cached Alpaca assets (%d symbols) after validation failure.",
                len(cached),
            )
            return cached

        LOGGER.error("Failed to fetch Alpaca assets; continuing with empty universe.")
        return []

    symbols: List[str] = []
    for asset in assets:
        symbol = getattr(asset, "symbol", None)
        if not symbol:
            continue
        cleaned = str(symbol).strip().upper()
        if cleaned:
            symbols.append(cleaned)

    if symbols:
        _write_assets_cache(cache_path, symbols)

    return sorted(set(symbols))


def _prepare_input_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)
    prepared = df.copy()
    for column in INPUT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[INPUT_COLUMNS]


def _safe_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_timestamp(ts: datetime) -> str:
    return ts.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    data = df.to_csv(index=False).encode("utf-8")
    atomic_write_bytes(path, data)


def _write_json_atomic(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _score_symbol(df: pd.DataFrame) -> Tuple[float, str]:
    returns = df["close"].pct_change().dropna()
    recent_return = returns.tail(5).mean() if not returns.empty else 0.0
    baseline_window = min(len(df), 30)
    baseline = df["close"].rolling(baseline_window).mean().iloc[-1]
    latest_close = df["close"].iloc[-1]
    if pd.isna(baseline) or baseline == 0:
        trend = 0.0
    else:
        trend = (latest_close / baseline) - 1
    volatility = returns.tail(20).std() if len(returns) >= 2 else 0.0
    score = (np.nan_to_num(recent_return) + np.nan_to_num(trend)) * 100 - np.nan_to_num(volatility) * 10
    breakdown = f"recent_return={recent_return:.4f}; trend={trend:.4f}; vol={volatility:.4f}"
    return round(float(score), 2), breakdown


def run_screener(
    df: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    min_history: int = DEFAULT_MIN_HISTORY,
    now: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """Run the screener on ``df`` returning (top, scored, stats, skips)."""

    now = now or datetime.now(timezone.utc)
    prepared = _prepare_input_frame(df)
    stats = {"symbols_in": 0, "candidates_out": 0}
    skip_reasons = {key: 0 for key in SKIP_KEYS}
    scored_records: list[dict[str, object]] = []

    if prepared.empty:
        LOGGER.info("No input rows supplied to screener; outputs will be empty.")
    else:
        for symbol, group in prepared.groupby("symbol"):
            stats["symbols_in"] += 1
            if group.empty:
                skip_reasons["NAN_DATA"] += 1
                LOGGER.info("[SKIP] %s has no rows", symbol or "<UNKNOWN>")
                continue

            bars: list[BarData] = []
            skip_symbol = False
            for row in group.to_dict("records"):
                try:
                    bar = BarData(**row)
                except ValidationError as exc:
                    LOGGER.warning("[SKIP] %s ValidationError: %s", row.get("symbol") or "<UNKNOWN>", exc)
                    skip_reasons["VALIDATION_ERROR"] += 1
                    skip_symbol = True
                    break
                kind = classify_exchange(bar.exchange)
                if kind != "EQUITY":
                    LOGGER.info(
                        "[SKIP] %s exchange=%s kind=%s",
                        bar.symbol or "<UNKNOWN>",
                        bar.exchange or "",
                        kind,
                    )
                    key = "UNKNOWN_EXCHANGE" if kind == "OTHER" else "NON_EQUITY"
                    skip_reasons[key] += 1
                    skip_symbol = True
                    break
                bars.append(bar)

            if skip_symbol or not bars:
                continue

            bars_df = pd.DataFrame([bar.to_dict() for bar in bars])
            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
            bars_df.sort_values("timestamp", inplace=True)

            numeric_cols = ["open", "high", "low", "close", "volume"]
            clean_df = bars_df.dropna(subset=numeric_cols)
            if clean_df.empty:
                skip_reasons["NAN_DATA"] += 1
                LOGGER.info("[SKIP] %s dropped due to NaN data", symbol or "<UNKNOWN>")
                continue
            if len(clean_df) < min_history:
                skip_reasons["INSUFFICIENT_HISTORY"] += 1
                LOGGER.info(
                    "[SKIP] %s insufficient history (%d < %d)",
                    symbol or "<UNKNOWN>",
                    len(clean_df),
                    min_history,
                )
                continue

            clean_df.set_index("timestamp", inplace=True)
            score, breakdown = _score_symbol(clean_df)
            latest = clean_df.iloc[-1]
            record = {
                "symbol": symbol,
                "exchange": latest.get("exchange", ""),
                "timestamp": _format_timestamp(now),
                "score": score,
                "close": _safe_float(latest.get("close")),
                "volume": _safe_float(latest.get("volume")),
                "score_breakdown": breakdown,
                "rsi": None,
                "macd": None,
                "macd_hist": None,
                "adx": None,
                "aroon_up": None,
                "aroon_down": None,
                "sma9": None,
                "ema20": None,
                "sma180": None,
                "atr": None,
            }
            scored_records.append(record)

    scored_df = pd.DataFrame(scored_records)
    if not scored_df.empty:
        scored_df.sort_values("score", ascending=False, inplace=True)
    scored_df = scored_df.reindex(columns=SCORED_COLUMNS)
    stats["candidates_out"] = int(scored_df.shape[0])

    top_df = scored_df.head(top_n).copy()
    top_df["universe_count"] = stats["symbols_in"]
    top_df = top_df.reindex(columns=TOP_COLUMNS)

    return top_df, scored_df, stats, skip_reasons


def _load_source_dataframe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - file corruption is rare
            LOGGER.error("Failed to read screener source %s: %s", path, exc)
            return pd.DataFrame(columns=INPUT_COLUMNS)
    LOGGER.warning("Screener source %s missing; proceeding with empty frame.", path)
    return pd.DataFrame(columns=INPUT_COLUMNS)


def write_outputs(
    base_dir: Path,
    top_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    stats: dict[str, int],
    skip_reasons: dict[str, int],
    *,
    status: str = "ok",
    now: Optional[datetime] = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    top_path = data_dir / "top_candidates.csv"
    scored_path = data_dir / "scored_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"

    _write_csv_atomic(top_path, top_df)
    _write_csv_atomic(scored_path, scored_df)

    metrics = {
        "last_run_utc": _format_timestamp(now),
        "rows": int(scored_df.shape[0]),
        "symbols_in": int(stats.get("symbols_in", 0)),
        "candidates_out": int(stats.get("candidates_out", 0)),
        "skips": {key: int(skip_reasons.get(key, 0)) for key in SKIP_KEYS},
        "status": status,
    }
    _write_json_atomic(metrics_path, metrics)
    return metrics_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly screener")
    parser.add_argument(
        "--source",
        help="Path to input OHLC CSV (defaults to data/screener_source.csv)",
        default=os.environ.get("SCREENER_SOURCE"),
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write outputs (defaults to repo root)",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    input_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> int:
    _ensure_logger()
    args = parse_args(argv)
    base_dir = Path(output_dir or args.output_dir or Path(__file__).resolve().parents[1])
    source_path = Path(args.source) if args.source else base_dir / "data" / "screener_source.csv"
    now = datetime.now(timezone.utc)

    frame = input_df if input_df is not None else _load_source_dataframe(source_path)
    top_df, scored_df, stats, skip_reasons = run_screener(
        frame,
        top_n=args.top_n,
        min_history=args.min_history,
        now=now,
    )
    write_outputs(base_dir, top_df, scored_df, stats, skip_reasons, status="ok", now=now)
    LOGGER.info(
        "Screener complete: %d symbols examined, %d candidates.",
        stats["symbols_in"],
        stats["candidates_out"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
