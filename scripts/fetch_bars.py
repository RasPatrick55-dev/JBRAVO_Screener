"""Fetch daily bars from Alpaca and store them in the daily_bars table."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
from psycopg2 import extras

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from scripts import db
from scripts.utils import get_last_trading_day_end
from utils import logger_utils
from utils.env import load_env, get_alpaca_creds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_env()

logger = logger_utils.init_logging(__name__, "fetch_bars.log")
logger.info("Script started")

DEFAULT_DAYS = 750
MIN_WINDOW_DAYS = 800
TRADING_DAYS_RATIO = 0.66
ALIAS_MAP = {
    "BRK.A": ["BRK-A", "BRKA"],
    "BRK.B": ["BRK-B", "BRKB"],
}


def _init_data_client() -> Optional[StockHistoricalDataClient]:
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        logger.error("Missing Alpaca credentials; data client unavailable.")
        return None
    try:
        return StockHistoricalDataClient(api_key, api_secret)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("Alpaca client unavailable: %s", exc)
        return None


def _ensure_daily_bars_table(conn) -> None:
    create_stmt = """
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
        );
    """
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(create_stmt)


def _ensure_daily_bars_view(conn, min_bars: int) -> str:
    threshold = max(int(min_bars), 1)
    view_name = f"daily_bars_full_{threshold}"
    view_stmt = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT db.*
        FROM daily_bars db
        WHERE db.symbol IN (
            SELECT symbol
            FROM daily_bars
            GROUP BY symbol
            HAVING COUNT(*) >= {threshold}
        );
    """
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(view_stmt)
    logger.info("BARS_VIEW_READY name=%s min_bars=%d", view_name, threshold)
    return view_name


def _fetch_symbols_from_db(conn) -> List[str]:
    if conn is None:
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT symbol
                FROM screener_candidates
                WHERE run_date = (SELECT MAX(run_date) FROM screener_candidates)
                ORDER BY score DESC NULLS LAST
                """
            )
            rows = cursor.fetchall()
        seen = set()
        symbols: List[str] = []
        for row in rows:
            symbol = (row[0] or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
        logger.info("Fetched %d symbols from latest screener_candidates", len(symbols))
        return symbols
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to load symbols from DB: %s", exc)
        return []


def _resolve_date_range(days: int) -> tuple[datetime, datetime]:
    end_dt = get_last_trading_day_end()
    window_days = max(MIN_WINDOW_DAYS, int(days / TRADING_DAYS_RATIO))
    start_dt = end_dt - timedelta(days=window_days)
    return start_dt, end_dt


def _candidate_symbols(symbol: str) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(sym: str) -> None:
        if sym and sym not in seen:
            seen.add(sym)
            candidates.append(sym)

    _add(symbol)
    for alias in ALIAS_MAP.get(symbol, []):
        _add(alias)

    if "." in symbol:
        _add(symbol.replace(".", "-"))
        _add(symbol.replace(".", ""))
    if "-" in symbol:
        _add(symbol.replace("-", "."))
        _add(symbol.replace("-", ""))
    if "/" in symbol:
        _add(symbol.replace("/", "-"))
        _add(symbol.replace("/", ""))

    return candidates


def _request_bars(
    symbol: str,
    data_client: StockHistoricalDataClient,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed="iex",
        )
        return data_client.get_stock_bars(request).df
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to fetch bars for %s: %s", symbol, exc)
        return pd.DataFrame()


def _fetch_symbol_bars(
    symbol: str,
    data_client: StockHistoricalDataClient,
    start_dt: datetime,
    end_dt: datetime,
    limit_days: int,
) -> pd.DataFrame:
    bars = pd.DataFrame()
    used_symbol: Optional[str] = None
    candidates = _candidate_symbols(symbol)
    for candidate in candidates:
        bars = _request_bars(candidate, data_client, start_dt, end_dt)
        if not bars.empty:
            used_symbol = candidate
            break

    if bars.empty:
        logger.warning("No bars returned for %s (aliases tried: %s)", symbol, ",".join(candidates))
        return pd.DataFrame()
    if used_symbol and used_symbol != symbol:
        logger.info("BARS_ALIAS symbol=%s alias=%s", symbol, used_symbol)

    if isinstance(bars.index, pd.MultiIndex):
        if "symbol" in bars.index.names:
            bars = bars.droplevel("symbol")
        else:
            bars = bars.droplevel(0)

    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.sort_index()
    if limit_days:
        bars = bars.tail(int(limit_days))

    frame = bars.reset_index()
    if "timestamp" in frame.columns:
        ts_col = "timestamp"
    elif "time" in frame.columns:
        ts_col = "time"
    else:
        ts_col = frame.columns[0]

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        logger.error("Missing bar columns for %s: %s", symbol, ",".join(missing))
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce").dt.date
    frame["symbol"] = symbol

    frame = frame[["symbol", "date", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["symbol", "date"], keep="last")
    return frame


def _insert_daily_bars(conn, bars_df: pd.DataFrame) -> int:
    if bars_df.empty:
        return 0

    rows = bars_df.to_dict("records")
    insert_stmt = """
        INSERT INTO daily_bars (
            symbol, date, open, high, low, close, volume
        )
        VALUES (
            %(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s
        )
        ON CONFLICT (symbol, date) DO NOTHING
    """
    with conn:
        with conn.cursor() as cursor:
            extras.execute_batch(cursor, insert_stmt, rows, page_size=500)
    return len(rows)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily bars into daily_bars table")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of most recent bars to keep per symbol (default: 750)",
    )
    parser.add_argument(
        "--symbols-from-db",
        action="store_true",
        help="Load symbols from latest screener_candidates in Postgres",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Fetch bars for a specific symbol instead of DB candidates",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between symbol fetches (default: 1.0)",
    )
    return parser.parse_args(argv if argv is not None else None)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else None)
    if args.days <= 0:
        logger.error("Days must be positive, got %s", args.days)
        return 1

    data_client = _init_data_client()
    if data_client is None:
        return 1

    conn = db.get_db_conn() if db.db_enabled() else None
    if conn is None:
        logger.error("DB connection unavailable; aborting.")
        return 1

    try:
        _ensure_daily_bars_table(conn)
        _ensure_daily_bars_view(conn, args.days)

        symbols: List[str] = []
        if args.symbol:
            symbols = [args.symbol.strip().upper()]
        elif args.symbols_from_db:
            symbols = _fetch_symbols_from_db(conn)
        else:
            logger.error("No symbols provided; use --symbol or --symbols-from-db.")
            return 1

        if not symbols:
            logger.warning("No symbols to process.")
            return 0

        start_dt, end_dt = _resolve_date_range(args.days)
        logger.info(
            "Fetching bars window start=%s end=%s target_days=%s symbols=%d",
            start_dt.date(),
            end_dt.date(),
            args.days,
            len(symbols),
        )

        total_rows = 0
        for symbol in symbols:
            bars_df = _fetch_symbol_bars(symbol, data_client, start_dt, end_dt, args.days)
            if bars_df.empty:
                logger.warning("No usable bars for %s", symbol)
            elif len(bars_df) < args.days:
                logger.warning(
                    "INSUFFICIENT_BARS symbol=%s rows=%d min=%d action=skip",
                    symbol,
                    len(bars_df),
                    args.days,
                )
            else:
                inserted = _insert_daily_bars(conn, bars_df)
                total_rows += inserted
                logger.info("BARS_INSERTED symbol=%s rows=%d", symbol, inserted)

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

        logger.info("BARS_DONE symbols=%d rows=%d", len(symbols), total_rows)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
