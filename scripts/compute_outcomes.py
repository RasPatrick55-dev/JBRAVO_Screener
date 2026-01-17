"""Compute forward outcomes for screener candidates and persist to Postgres."""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
from psycopg2 import extras

from scripts import db
from scripts.utils.http_alpaca import fetch_bars_http
from utils.env import get_alpaca_creds, load_env


LOGGER = logging.getLogger(__name__)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _load_missing_candidates(
    conn,
    *,
    run_date: Optional[date] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    base_sql = """
        SELECT
            c.run_date,
            c.symbol,
            c.score,
            c.close,
            c.passed_gates,
            c.gate_fail_reason
        FROM screener_candidates c
        LEFT JOIN screener_outcomes_app o
          ON o.run_date = c.run_date
         AND o.symbol = c.symbol
        WHERE o.symbol IS NULL
    """
    params: list[object] = []
    if run_date is not None:
        base_sql += " AND c.run_date = %s"
        params.append(run_date)
    base_sql += " ORDER BY c.run_date DESC, c.score DESC NULLS LAST"
    if limit is not None and limit > 0:
        base_sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(base_sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    frame["run_date"] = pd.to_datetime(frame["run_date"], errors="coerce").dt.date
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    if "passed_gates" in frame.columns:
        frame["passed_gates"] = frame["passed_gates"].astype("boolean")
    frame["rank"] = (
        frame.groupby("run_date")["score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return frame


def _fetch_bars(
    symbols: Iterable[str],
    *,
    run_date: date,
    feed: str,
    window_days: int,
) -> pd.DataFrame:
    start = datetime.combine(run_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=window_days)
    rows, _stats = fetch_bars_http(
        list(symbols),
        start=_iso(start),
        end=_iso(end),
        feed=feed,
    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    frame["date"] = frame["timestamp"].dt.date
    for col in ("close", "high", "low"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _calc_ret(prices: pd.Series, offset: int, entry_close: float) -> Optional[float]:
    if entry_close is None or pd.isna(entry_close) or entry_close <= 0:
        return None
    if len(prices) <= offset:
        return None
    value = prices.iloc[offset]
    if value is None or pd.isna(value):
        return None
    return float(value / entry_close - 1.0)


def _drawdown_runup(window: pd.DataFrame, entry_close: float) -> tuple[Optional[float], Optional[float]]:
    if entry_close is None or pd.isna(entry_close) or entry_close <= 0:
        return None, None
    if window.empty:
        return None, None
    low_series = pd.to_numeric(window.get("low"), errors="coerce")
    high_series = pd.to_numeric(window.get("high"), errors="coerce")
    close_series = pd.to_numeric(window.get("close"), errors="coerce")
    if low_series.notna().any():
        max_drawdown = (low_series / entry_close - 1.0).min()
    else:
        max_drawdown = (close_series / entry_close - 1.0).min()
    if high_series.notna().any():
        max_runup = (high_series / entry_close - 1.0).max()
    else:
        max_runup = (close_series / entry_close - 1.0).max()
    return (
        float(max_drawdown) if pd.notna(max_drawdown) else None,
        float(max_runup) if pd.notna(max_runup) else None,
    )


def _compute_outcomes(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    run_date: date,
) -> list[tuple]:
    results: list[tuple] = []
    if candidates.empty:
        return results
    bars_by_symbol = {
        sym: df.sort_values("date")
        for sym, df in bars.groupby("symbol")
    } if not bars.empty else {}

    for row in candidates.itertuples(index=False):
        symbol = str(getattr(row, "symbol", "")).strip().upper()
        if not symbol:
            continue
        score = getattr(row, "score", None)
        rank = getattr(row, "rank", None)
        entry_close = getattr(row, "close", None)
        if entry_close is not None and pd.isna(entry_close):
            entry_close = None
        passed_gates = getattr(row, "passed_gates", None)
        gate_fail_reason = getattr(row, "gate_fail_reason", None)
        if isinstance(gate_fail_reason, float) and pd.isna(gate_fail_reason):
            gate_fail_reason = None

        series = bars_by_symbol.get(symbol)
        if series is not None and not series.empty:
            future = series.loc[series["date"] >= run_date].copy()
        else:
            future = pd.DataFrame()

        ret_1d = ret_5d = ret_10d = None
        max_drawdown = max_runup = None
        if not future.empty:
            close_series = pd.to_numeric(future.get("close"), errors="coerce").reset_index(drop=True)
            if (entry_close is None or pd.isna(entry_close)) and not close_series.empty:
                entry_close = close_series.iloc[0]
            ret_1d = _calc_ret(close_series, 1, entry_close)
            ret_5d = _calc_ret(close_series, 5, entry_close)
            ret_10d = _calc_ret(close_series, 10, entry_close)
            window = future.iloc[:11].copy()
            max_drawdown, max_runup = _drawdown_runup(window, entry_close)

        results.append(
            (
                run_date,
                symbol,
                int(rank) if rank is not None and not pd.isna(rank) else None,
                float(score) if score is not None and not pd.isna(score) else None,
                float(entry_close) if entry_close is not None and not pd.isna(entry_close) else None,
                ret_1d,
                ret_5d,
                ret_10d,
                max_drawdown,
                max_runup,
                bool(passed_gates) if passed_gates is not None else None,
                str(gate_fail_reason) if gate_fail_reason is not None else None,
            )
        )
    return results


def _insert_outcomes(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    insert_sql = """
        INSERT INTO screener_outcomes_app (
            run_date,
            symbol,
            rank,
            score,
            close_at_entry,
            ret_1d,
            ret_5d,
            ret_10d,
            max_drawdown_10d,
            max_runup_10d,
            passed_gates,
            gate_fail_reason
        ) VALUES %s
        ON CONFLICT (run_date, symbol) DO NOTHING
    """
    with conn:
        with conn.cursor() as cur:
            extras.execute_values(cur, insert_sql, rows)
    return len(rows)


def _ensure_outcomes_table(conn) -> None:
    ddl = """
        CREATE TABLE IF NOT EXISTS screener_outcomes_app (
            run_date DATE NOT NULL,
            symbol TEXT NOT NULL,
            rank INTEGER,
            score NUMERIC,
            close_at_entry NUMERIC,
            ret_1d NUMERIC,
            ret_5d NUMERIC,
            ret_10d NUMERIC,
            max_drawdown_10d NUMERIC,
            max_runup_10d NUMERIC,
            passed_gates BOOLEAN,
            gate_fail_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (run_date, symbol)
        );
    """
    alters = [
        "ALTER TABLE screener_outcomes_app ADD COLUMN IF NOT EXISTS passed_gates BOOLEAN;",
        "ALTER TABLE screener_outcomes_app ADD COLUMN IF NOT EXISTS gate_fail_reason TEXT;",
    ]
    with conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            for statement in alters:
                cur.execute(statement)


def _backfill_gate_telemetry(conn) -> int:
    update_sql = """
        UPDATE screener_outcomes_app o
        SET
            passed_gates = c.passed_gates,
            gate_fail_reason = c.gate_fail_reason
        FROM screener_candidates c
        WHERE o.run_date = c.run_date
          AND o.symbol = c.symbol
          AND o.passed_gates IS NULL
          AND c.passed_gates IS NOT NULL
    """
    with conn:
        with conn.cursor() as cur:
            cur.execute(update_sql)
            updated = cur.rowcount or 0
    return int(updated)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute forward outcomes for screener candidates.")
    parser.add_argument("--run-date", type=str, default=None, help="Run date (YYYY-MM-DD) to process.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates processed.")
    parser.add_argument("--window-days", type=int, default=30, help="Calendar days to fetch for outcomes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env()

    conn = db.get_db_conn()
    if conn is None:
        raise RuntimeError("DB connection unavailable")
    _ensure_outcomes_table(conn)

    run_date = _parse_date(args.run_date)
    window_days = max(int(args.window_days), 12)
    candidates = _load_missing_candidates(conn, run_date=run_date, limit=args.limit)
    if candidates.empty:
        LOGGER.info("[INFO] Outcomes up to date; no missing candidates.")
        return 0

    _, _, _, feed = get_alpaca_creds()
    feed = (feed or "iex").strip().lower() or "iex"

    total_inserted = 0
    for run_dt, group in candidates.groupby("run_date"):
        if pd.isna(run_dt):
            continue
        symbols = group["symbol"].dropna().astype("string").str.upper().unique().tolist()
        if not symbols:
            continue
        bars = _fetch_bars(symbols, run_date=run_dt, feed=feed, window_days=window_days)
        outcomes = _compute_outcomes(group, bars, run_date=run_dt)
        inserted = _insert_outcomes(conn, outcomes)
        total_inserted += inserted
        LOGGER.info(
            "[INFO] Outcomes inserted run_date=%s rows=%d symbols=%d",
            run_dt,
            inserted,
            len(symbols),
        )

    updated = _backfill_gate_telemetry(conn)
    if updated:
        LOGGER.info("[INFO] Outcomes gate backfill updated=%d", updated)

    LOGGER.info("[INFO] Outcomes complete inserted=%d", total_inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
