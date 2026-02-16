"""Verify end-to-end screener run scoping and export freshness."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from scripts import db
from scripts.utils.env import load_env


def _print_check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{status} {name}{suffix}")


def _tail_lines(path: Path, max_bytes: int = 200_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    read_size = min(size, max_bytes)
    with path.open("rb") as handle:
        if read_size > 0:
            handle.seek(-read_size, os.SEEK_END)
        data = handle.read()
    return data.decode("utf-8", errors="replace").splitlines()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--execute-log", type=Path, default=Path("logs") / "execute_trades.log")
    parser.add_argument("--latest-csv", type=Path, default=Path("data") / "latest_candidates.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv or sys.argv[1:])

    all_ok = True
    conn = db.get_db_conn()
    if conn is None:
        _print_check("db_connect", False, "connection unavailable")
        return 1

    run_date = args.run_date
    latest_run_ts = None
    latest_count = 0
    max_run_count = 0
    recent_null_count = 0
    try:
        with conn.cursor() as cur:
            if run_date is None:
                cur.execute("SELECT max(run_date) FROM screener_candidates")
                row = cur.fetchone()
                run_date = row[0].isoformat() if row and row[0] else None

            if run_date is not None:
                cur.execute(
                    """
                    SELECT COALESCE(max(run_ts_utc), max(created_at)) AS latest_run_ts
                    FROM screener_candidates
                    WHERE run_date = %s
                    """,
                    (run_date,),
                )
                row = cur.fetchone()
                latest_run_ts = row[0] if row else None

                cur.execute(
                    """
                    SELECT count(*) AS row_count
                    FROM screener_candidates
                    WHERE run_date = %s
                      AND created_at > now() - interval '7 days'
                      AND run_ts_utc IS NULL
                    """,
                    (run_date,),
                )
                row = cur.fetchone()
                recent_null_count = int(row[0] or 0) if row else 0

                if latest_run_ts is not None:
                    cur.execute(
                        """
                        SELECT count(*) AS row_count
                        FROM screener_candidates
                        WHERE run_date = %s
                          AND run_ts_utc = %s
                        """,
                        (run_date, latest_run_ts),
                    )
                    row = cur.fetchone()
                    latest_count = int(row[0] or 0) if row else 0

                cur.execute(
                    """
                    WITH latest_ts AS (
                        SELECT COALESCE(max(run_ts_utc), max(created_at)) AS latest_run_ts
                        FROM screener_candidates
                        WHERE run_date = %s
                    )
                    SELECT count(*) AS row_count
                    FROM screener_candidates c
                    JOIN latest_ts t ON c.run_ts_utc = t.latest_run_ts
                    WHERE c.run_date = %s
                    """,
                    (run_date, run_date),
                )
                row = cur.fetchone()
                max_run_count = int(row[0] or 0) if row else 0
    finally:
        conn.close()

    has_run = run_date is not None
    _print_check("run_date", has_run, str(run_date))
    all_ok &= has_run

    _print_check("run_ts_populated_recent", recent_null_count == 0, f"null_count={recent_null_count}")
    all_ok &= recent_null_count == 0

    scoped_ok = latest_count == max_run_count
    _print_check(
        "latest_run_scoping",
        scoped_ok,
        f"latest_run_ts={latest_run_ts} latest_count={latest_count} scoped_count={max_run_count}",
    )
    all_ok &= scoped_ok

    header_expected = "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source"
    csv_exists = args.latest_csv.exists()
    csv_header_ok = False
    csv_fresh_ok = False
    if csv_exists:
        lines = args.latest_csv.read_text(encoding="utf-8").splitlines()
        csv_header_ok = bool(lines and lines[0].strip() == header_expected)
        mtime = datetime.fromtimestamp(args.latest_csv.stat().st_mtime, timezone.utc)
        csv_fresh_ok = mtime >= datetime.now(timezone.utc) - timedelta(hours=24)
    _print_check("latest_csv_exists", csv_exists, str(args.latest_csv))
    _print_check("latest_csv_header", csv_header_ok)
    _print_check("latest_csv_fresh", csv_fresh_ok)
    all_ok &= csv_exists and csv_header_ok and csv_fresh_ok

    execute_lines = _tail_lines(args.execute_log)
    market_closed_lines = [ln for ln in execute_lines if "MARKET_DAY_CLOSED" in ln]
    skip_summary_lines = [ln for ln in execute_lines if "EXECUTE_SUMMARY" in ln and "skips.MARKET_CLOSED=1" in ln]
    no_error_metric_on_close = not any(
        "METRICS_UPDATED" in ln and "status=error" in ln for ln in execute_lines if "MARKET_DAY_CLOSED" in "\n".join(execute_lines)
    )
    _print_check("market_closed_token", bool(market_closed_lines))
    _print_check("market_closed_skip_summary", bool(skip_summary_lines))
    _print_check("market_closed_status_not_error", no_error_metric_on_close)
    all_ok &= bool(market_closed_lines) and bool(skip_summary_lines) and no_error_metric_on_close

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
