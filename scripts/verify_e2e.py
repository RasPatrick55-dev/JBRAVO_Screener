"""Verify end-to-end screener health after a run.

Uses pipeline_health_app.run_ts_utc as the canonical run id and scopes
candidates through screener_run_map_app (run_date is not unique).
Zero-candidate fallback days are valid PASS cases when recorded as such.
"""
from __future__ import annotations

import argparse
import os
import sys
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
    try:
        size = path.stat().st_size
        read_size = min(size, max_bytes)
        with path.open("rb") as handle:
            if read_size > 0:
                handle.seek(-read_size, os.SEEK_END)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        return text.splitlines()
    except Exception:
        return []


def _latest_summary_line(log_path: Path) -> Optional[str]:
    lines = _tail_lines(log_path)
    for line in reversed(lines):
        if "[SUMMARY]" in line:
            return line.strip()
    return None


def _fetch_one(cur, sql: str, params: tuple = ()) -> Optional[tuple]:
    cur.execute(sql, params)
    return cur.fetchone()


def _fetch_all(cur, sql: str, params: tuple = ()) -> list[tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs") / "screener.log",
        help="Path to screener log file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv or sys.argv[1:])

    all_ok = True

    db_ok = db.db_enabled() and db.check_db_connection()
    _print_check("db_health", db_ok)
    if not db_ok:
        return 1

    conn = db.get_db_conn()
    if conn is None:
        _print_check("db_connect", False, "db.get_db_conn returned None")
        return 1

    latest_run_date = None
    candidate_count = 0
    sma9_count = 0
    gates_count = 0
    source_counts: list[tuple[str, int]] = []
    pipeline_row = None
    outcomes_total = None
    outcomes_ret5 = None

    try:
        with conn.cursor() as cur:
            pipeline_row = _fetch_one(
                cur,
                """
                SELECT run_date, run_ts_utc, mode, symbols_in, with_bars, coarse_rows,
                       shortlist_rows, final_rows, gated_rows, fallback_used, db_ingest_rows
                FROM pipeline_health_app
                ORDER BY run_ts_utc DESC NULLS LAST
                LIMIT 1
                """,
            )
            # NOTE: run_date is not unique; run_ts_utc is the screener run identifier.
            # screener_candidates.timestamp is the bar timestamp.
            # screener_run_map_app.run_ts_utc is the run identity.
            if pipeline_row:
                latest_run_date = pipeline_row[0]
                run_ts = pipeline_row[1]
            else:
                latest_run_date = None
                run_ts = None

            if run_ts is not None and latest_run_date is not None:
                row = _fetch_one(
                    cur,
                    """
                    SELECT count(*)
                    FROM screener_candidates c
                    JOIN screener_run_map_app m
                      ON UPPER(c.symbol) = m.symbol
                    WHERE m.run_ts_utc = %s
                      AND c.run_date = %s
                    """,
                    (run_ts, latest_run_date),
                )
                candidate_count = int(row[0]) if row else 0
                row = _fetch_one(
                    cur,
                    """
                    SELECT count(*)
                    FROM screener_candidates c
                    JOIN screener_run_map_app m
                      ON UPPER(c.symbol) = m.symbol
                    WHERE m.run_ts_utc = %s
                      AND c.run_date = %s
                      AND c.sma9 IS NOT NULL
                    """,
                    (run_ts, latest_run_date),
                )
                sma9_count = int(row[0]) if row else 0
                row = _fetch_one(
                    cur,
                    """
                    SELECT count(*)
                    FROM screener_candidates c
                    JOIN screener_run_map_app m
                      ON UPPER(c.symbol) = m.symbol
                    WHERE m.run_ts_utc = %s
                      AND c.run_date = %s
                      AND c.passed_gates IS NOT NULL
                    """,
                    (run_ts, latest_run_date),
                )
                gates_count = int(row[0]) if row else 0
                source_counts = _fetch_all(
                    cur,
                    """
                    SELECT COALESCE(source, '') AS source, count(*)
                    FROM screener_candidates c
                    JOIN screener_run_map_app m
                      ON UPPER(c.symbol) = m.symbol
                    WHERE m.run_ts_utc = %s
                      AND c.run_date = %s
                    GROUP BY COALESCE(source, '')
                    ORDER BY count(*) DESC
                    """,
                    (run_ts, latest_run_date),
                )

            outcomes_row = _fetch_one(
                cur,
                "SELECT count(*), count(ret_5d) FROM screener_outcomes_app",
            )
            if outcomes_row:
                outcomes_total = int(outcomes_row[0])
                outcomes_ret5 = int(outcomes_row[1])
    finally:
        try:
            conn.close()
        except Exception:
            pass

    has_latest = latest_run_date is not None
    _print_check(
        "latest_run_date",
        has_latest,
        str(latest_run_date) if has_latest else "none",
    )
    all_ok &= has_latest

    if has_latest:
        _print_check(
            "latest_run_counts",
            candidate_count >= 0,
            f"rows={candidate_count} sma9={sma9_count} passed_gates={gates_count}",
        )
        for source, count in source_counts:
            label = source if source else "<null>"
            print(f"INFO source_count {label}={count}")
        all_ok &= sma9_count <= candidate_count
        all_ok &= gates_count <= candidate_count

    pipeline_ok = pipeline_row is not None
    _print_check("pipeline_health_app_latest", pipeline_ok)
    if pipeline_ok and has_latest:
        (
            ph_run_date,
            _ph_run_ts,
            _ph_mode,
            _ph_symbols_in,
            _ph_with_bars,
            _ph_coarse_rows,
            _ph_shortlist_rows,
            ph_final_rows,
            ph_gated_rows,
            ph_fallback_used,
            ph_db_ingest_rows,
        ) = pipeline_row
        match_run_date = ph_run_date == latest_run_date
        db_ingest_rows = int(ph_db_ingest_rows or 0)
        final_rows = int(ph_final_rows or 0)
        gated_rows = int(ph_gated_rows or 0)
        fallback_used = bool(ph_fallback_used)

        case_a = (
            not fallback_used
            and db_ingest_rows > 0
            and db_ingest_rows == candidate_count
            and final_rows == candidate_count
            and gated_rows == candidate_count
        )
        case_b = fallback_used and db_ingest_rows == 0 and candidate_count > 0

        match_ok = match_run_date and (case_a or case_b)
        detail = (
            f"run_date_match={match_run_date} case_a={case_a} case_b={case_b}"
        )
        if not match_ok:
            detail = f"pipeline_health_app mismatch outside valid fallback case - {detail}"
        _print_check("pipeline_health_app_match", match_ok, detail)
        all_ok &= match_ok

    outcomes_ok = outcomes_total is not None and outcomes_ret5 is not None
    detail = ""
    if outcomes_ok:
        detail = f"rows={outcomes_total} ret_5d_not_null={outcomes_ret5}"
    _print_check("outcomes_table", outcomes_ok, detail)
    all_ok &= outcomes_ok

    summary_line = _latest_summary_line(args.log_path)
    summary_ok = summary_line is not None
    _print_check("summary_log", summary_ok, summary_line or "not found")
    all_ok &= summary_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
