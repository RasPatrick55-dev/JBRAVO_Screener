from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

# Allow running as `python scripts/audit_dashboard_trades_api.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import db
from scripts.utils.env import load_env

RANGE_ORDER = ["d", "w", "m", "y", "all"]
RANGE_LABELS = {
    "d": "DAILY",
    "w": "WEEKLY",
    "m": "MONTHLY",
    "y": "YEARLY",
    "all": "ALL",
}
RANGE_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


@dataclass
class AuditContext:
    trades_frame: pd.DataFrame
    latest_metrics: dict[str, Any]
    open_count: int
    open_realized_pnl: float
    account_snapshot_db: dict[str, Any]
    open_positions_expected: dict[str, dict[str, float]]


@dataclass
class AuditReport:
    generated_at_utc: str
    endpoint_checks: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add_issue(self, message: str) -> None:
        self.issues.append(message)

    def add_check(self, *, endpoint: str, ok: bool, detail: str) -> None:
        self.endpoint_checks.append({"endpoint": endpoint, "ok": bool(ok), "detail": detail})


def _serialize_record(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _coerce_iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value in (None, ""):
        return ""
    return str(value)


def _float_close(left: Any, right: Any, *, abs_tol: float = 1e-6) -> bool:
    return math.isclose(_to_number(left), _to_number(right), rel_tol=0.0, abs_tol=abs_tol)


def _normalize_trades_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "status",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "realized_pnl",
                "exit_reason",
                "sort_ts",
            ]
        )

    work = frame.copy()
    work["symbol"] = work.get("symbol", "").astype(str).str.upper().str.strip()
    work["qty"] = pd.to_numeric(work.get("qty"), errors="coerce")
    work["status"] = work.get("status", "").astype(str).str.upper().str.strip()
    work["entry_time"] = pd.to_datetime(work.get("entry_time"), utc=True, errors="coerce")
    work["entry_price"] = pd.to_numeric(work.get("entry_price"), errors="coerce")
    work["exit_time"] = pd.to_datetime(work.get("exit_time"), utc=True, errors="coerce")
    work["exit_price"] = pd.to_numeric(work.get("exit_price"), errors="coerce")
    work["realized_pnl"] = pd.to_numeric(work.get("realized_pnl"), errors="coerce")
    work["exit_reason"] = work.get("exit_reason", "").astype(str).replace({"nan": ""})

    computed_pnl = (work["exit_price"] - work["entry_price"]) * work["qty"]
    work["realized_pnl"] = work["realized_pnl"].fillna(computed_pnl).fillna(0.0)

    is_closed = work["exit_time"].notna() | work["status"].eq("CLOSED")
    work = work[is_closed].copy()
    work = work[work["symbol"].str.len() > 0]
    work["sort_ts"] = work["exit_time"].fillna(work["entry_time"])
    work.sort_values("sort_ts", ascending=False, na_position="last", inplace=True)
    work.reset_index(drop=True, inplace=True)
    return work


def _filter_by_range(frame: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if frame.empty or range_key == "all":
        return frame
    days = RANGE_DAYS.get(range_key)
    if days is None:
        return frame
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return frame[(frame["sort_ts"].notna()) & (frame["sort_ts"] >= cutoff)]


def _expected_stats_rows(frame: pd.DataFrame, requested_range: str) -> list[dict[str, Any]]:
    keys = RANGE_ORDER if requested_range == "all" else [requested_range]
    rows: list[dict[str, Any]] = []
    for key in keys:
        scoped = _filter_by_range(frame, key)
        if scoped.empty:
            rows.append(
                {
                    "key": key,
                    "label": RANGE_LABELS[key],
                    "winRatePct": 0.0,
                    "totalPL": 0.0,
                    "topTrade": {"symbol": "--", "pl": 0.0},
                    "worstLoss": {"symbol": "--", "pl": 0.0},
                    "tradesCount": 0,
                }
            )
            continue

        total_trades = int(len(scoped))
        wins = int((scoped["realized_pnl"] > 0).sum())
        win_rate_pct = (wins / total_trades) * 100 if total_trades > 0 else 0.0
        total_pl = float(scoped["realized_pnl"].sum())
        top_row = scoped.loc[scoped["realized_pnl"].idxmax()]
        worst_row = scoped.loc[scoped["realized_pnl"].idxmin()]
        rows.append(
            {
                "key": key,
                "label": RANGE_LABELS[key],
                "winRatePct": float(win_rate_pct),
                "totalPL": float(total_pl),
                "topTrade": {
                    "symbol": str(top_row.get("symbol") or "--"),
                    "pl": float(top_row.get("realized_pnl") or 0.0),
                },
                "worstLoss": {
                    "symbol": str(worst_row.get("symbol") or "--"),
                    "pl": float(worst_row.get("realized_pnl") or 0.0),
                },
                "tradesCount": int(total_trades),
            }
        )
    return rows


def _expected_leaderboard_rows(
    frame: pd.DataFrame, *, range_key: str, mode: str, limit: int
) -> list[dict[str, Any]]:
    scoped = _filter_by_range(frame, range_key)
    if scoped.empty:
        return []

    grouped = (
        scoped.groupby("symbol", dropna=False)["realized_pnl"]
        .sum()
        .reset_index()
        .rename(columns={"realized_pnl": "pl"})
    )

    if mode == "losers":
        grouped = grouped[grouped["pl"] < 0].sort_values("pl", ascending=True)
    else:
        grouped = grouped[grouped["pl"] > 0].sort_values("pl", ascending=False)

    grouped = grouped.head(limit).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i, row in grouped.iterrows():
        rows.append({"rank": i + 1, "symbol": str(row.get("symbol") or "--"), "pl": float(row.get("pl") or 0.0)})
    return rows


def _expected_latest_rows(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    scoped = frame.sort_values("sort_ts", ascending=False, na_position="last").head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in scoped.iterrows():
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time")
        hold_days = 0
        if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            hold_days = max(0, int((exit_time - entry_time).total_seconds() // 86400))

        rows.append(
            {
                "symbol": str(row.get("symbol") or "--"),
                "buyDate": _serialize_record(entry_time),
                "sellDate": _serialize_record(exit_time),
                "totalDays": hold_days,
                "totalShares": _to_int(row.get("qty")),
                "avgEntryPrice": float(row.get("entry_price") or 0.0),
                "priceSold": float(row.get("exit_price") or 0.0),
                "totalPL": float(row.get("realized_pnl") or 0.0),
            }
        )
    return rows


def _db_query_frame(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    conn = db.get_db_conn()
    if conn is None:
        raise RuntimeError("DB connection unavailable")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _load_account_snapshot_db() -> dict[str, Any]:
    sql_candidates = [
        """
        SELECT taken_at, account_id, status, equity, cash, buying_power, portfolio_value,
               'v_account_latest'::text AS source
        FROM v_account_latest
        LIMIT 1
        """,
        """
        SELECT taken_at, account_id, status, equity, cash, buying_power,
               'v_account_latest'::text AS source
        FROM v_account_latest
        LIMIT 1
        """,
        """
        SELECT taken_at, account_id, status, equity, cash, buying_power, portfolio_value,
               'alpaca_account_snapshots'::text AS source
        FROM alpaca_account_snapshots
        ORDER BY taken_at DESC
        LIMIT 1
        """,
        """
        SELECT taken_at, equity, cash, buying_power, 'alpaca_account_snapshots'::text AS source
        FROM alpaca_account_snapshots
        ORDER BY taken_at DESC
        LIMIT 1
        """,
    ]
    for sql in sql_candidates:
        try:
            frame = _db_query_frame(sql)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        return {key: _serialize_record(value) for key, value in row.items()}
    return {}


def _load_open_positions_expected() -> dict[str, dict[str, float]]:
    frame = _db_query_frame(
        """
        SELECT symbol, qty, entry_price
        FROM trades
        WHERE status='OPEN'
        """
    )
    if frame.empty:
        return {}

    expected: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        qty = _to_number(row.get("qty"))
        entry_price = row.get("entry_price")
        if not symbol or qty <= 0:
            continue
        try:
            entry_price_num = float(entry_price)
        except (TypeError, ValueError):
            continue
        bucket = expected.setdefault(symbol, {"qty": 0.0, "entry_value": 0.0})
        bucket["qty"] += qty
        bucket["entry_value"] += qty * entry_price_num

    normalized: dict[str, dict[str, float]] = {}
    for symbol, bucket in expected.items():
        qty = float(bucket.get("qty") or 0.0)
        if qty <= 0:
            continue
        normalized[symbol] = {
            "qty": qty,
            "entryPrice": float(bucket.get("entry_value") or 0.0) / qty,
        }
    return normalized


def _build_audit_context() -> AuditContext:
    trades_df = _db_query_frame(
        """
        SELECT
            trade_id,
            symbol,
            qty,
            status,
            entry_time,
            entry_price,
            exit_time,
            exit_price,
            realized_pnl,
            exit_reason
        FROM trades
        ORDER BY COALESCE(exit_time, entry_time) DESC NULLS LAST
        """
    )
    normalized = _normalize_trades_frame(trades_df)

    metrics_df = _db_query_frame(
        """
        SELECT run_date, total_trades, win_rate, net_pnl, profit_factor
        FROM metrics_daily
        ORDER BY run_date DESC
        LIMIT 1
        """
    )
    latest_metrics: dict[str, Any] = {}
    if not metrics_df.empty:
        row = metrics_df.iloc[0].to_dict()
        latest_metrics = {
            "last_run_utc": _serialize_record(row.get("run_date")),
            "total_trades": _serialize_record(row.get("total_trades")),
            "win_rate": _serialize_record(row.get("win_rate")),
            "net_pnl": _serialize_record(row.get("net_pnl")),
            "profit_factor": _serialize_record(row.get("profit_factor")),
        }

    open_df = _db_query_frame("SELECT realized_pnl FROM trades WHERE status='OPEN'")
    open_count = int(len(open_df))
    open_realized_pnl = float(pd.to_numeric(open_df.get("realized_pnl"), errors="coerce").fillna(0).sum())
    account_snapshot_db = _load_account_snapshot_db()
    open_positions_expected = _load_open_positions_expected()

    return AuditContext(
        trades_frame=normalized,
        latest_metrics=latest_metrics,
        open_count=open_count,
        open_realized_pnl=open_realized_pnl,
        account_snapshot_db=account_snapshot_db,
        open_positions_expected=open_positions_expected,
    )


def _request_json(path: str, *, base_url: str | None = None, flask_client: Any = None) -> tuple[int, dict[str, Any]]:
    if flask_client is not None:
        response = flask_client.get(path)
        payload = response.get_json(silent=True) or {}
        return int(response.status_code), dict(payload)

    if not base_url:
        raise RuntimeError("No request method configured")
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    return int(response.status_code), dict(payload) if isinstance(payload, dict) else {}


def _row_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text


def _audit_stats(report: AuditReport, ctx: AuditContext, *, requester: Any, base_url: str | None) -> None:
    endpoint = "/api/trades/stats?range=all"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)
    expected = _expected_stats_rows(ctx.trades_frame, "all")

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []
    if payload.get("source") != "postgres":
        ok = False
        details.append(f"source={payload.get('source')}")
    if payload.get("db_source_of_truth") is not True:
        ok = False
        details.append("db_source_of_truth flag missing")

    actual_rows = payload.get("rows") or []
    if len(actual_rows) != len(expected):
        ok = False
        details.append(f"row_count actual={len(actual_rows)} expected={len(expected)}")

    expected_map = {row["key"]: row for row in expected}
    actual_map = {}
    for row in actual_rows:
        key = _row_label((row or {}).get("key"))
        if key:
            actual_map[key] = row

    for key in RANGE_ORDER:
        exp = expected_map.get(key)
        got = actual_map.get(key)
        if exp is None or got is None:
            ok = False
            details.append(f"range={key} missing")
            continue
        if not _float_close(got.get("winRatePct"), exp.get("winRatePct"), abs_tol=1e-6):
            ok = False
            details.append(f"{key}.winRatePct actual={got.get('winRatePct')} expected={exp.get('winRatePct')}")
        if not _float_close(got.get("totalPL"), exp.get("totalPL"), abs_tol=1e-6):
            ok = False
            details.append(f"{key}.totalPL actual={got.get('totalPL')} expected={exp.get('totalPL')}")
        if _to_int(got.get("tradesCount")) != _to_int(exp.get("tradesCount")):
            ok = False
            details.append(f"{key}.tradesCount actual={got.get('tradesCount')} expected={exp.get('tradesCount')}")

        exp_top = exp.get("topTrade") or {}
        got_top = (got or {}).get("topTrade") or {}
        if str(got_top.get("symbol") or "--") != str(exp_top.get("symbol") or "--"):
            ok = False
            details.append(f"{key}.topTrade.symbol actual={got_top.get('symbol')} expected={exp_top.get('symbol')}")
        if not _float_close(got_top.get("pl"), exp_top.get("pl"), abs_tol=1e-6):
            ok = False
            details.append(f"{key}.topTrade.pl actual={got_top.get('pl')} expected={exp_top.get('pl')}")

        exp_worst = exp.get("worstLoss") or {}
        got_worst = (got or {}).get("worstLoss") or {}
        if str(got_worst.get("symbol") or "--") != str(exp_worst.get("symbol") or "--"):
            ok = False
            details.append(
                f"{key}.worstLoss.symbol actual={got_worst.get('symbol')} expected={exp_worst.get('symbol')}"
            )
        if not _float_close(got_worst.get("pl"), exp_worst.get("pl"), abs_tol=1e-6):
            ok = False
            details.append(f"{key}.worstLoss.pl actual={got_worst.get('pl')} expected={exp_worst.get('pl')}")

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details[:12]))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:4]))


def _leaderboard_signature(rows: Iterable[dict[str, Any]]) -> list[tuple[int, str, float]]:
    signature: list[tuple[int, str, float]] = []
    for i, row in enumerate(rows):
        rank = _to_int((row or {}).get("rank") or (i + 1))
        symbol = str((row or {}).get("symbol") or "--")
        pl = round(_to_number((row or {}).get("pl")), 6)
        signature.append((rank, symbol, pl))
    return signature


def _audit_leaderboard(report: AuditReport, ctx: AuditContext, *, requester: Any, base_url: str | None) -> None:
    for range_key in RANGE_ORDER:
        for mode in ("winners", "losers"):
            endpoint = f"/api/trades/leaderboard?range={range_key}&mode={mode}&limit=10"
            status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)
            expected = _expected_leaderboard_rows(ctx.trades_frame, range_key=range_key, mode=mode, limit=10)

            if status != 200:
                report.add_issue(f"{endpoint}: unexpected HTTP {status}")
                report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
                continue

            ok = True
            details: list[str] = []
            if payload.get("source") != "postgres":
                ok = False
                details.append(f"source={payload.get('source')}")
            if payload.get("db_source_of_truth") is not True:
                ok = False
                details.append("db_source_of_truth flag missing")

            actual_rows = payload.get("rows") or []
            expected_sig = _leaderboard_signature(expected)
            actual_sig = _leaderboard_signature(actual_rows)
            if actual_sig != expected_sig:
                ok = False
                details.append(f"rows_mismatch actual={actual_sig[:5]} expected={expected_sig[:5]}")

            if not ok:
                report.add_issue(f"{endpoint}: " + "; ".join(details[:6]))
            report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:3]))


def _latest_signature(rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for row in rows:
        record = row or {}
        signature.append(
            (
                str(record.get("symbol") or "--"),
                _coerce_iso(record.get("buyDate")),
                _coerce_iso(record.get("sellDate")),
                _to_int(record.get("totalDays")),
                _to_int(record.get("totalShares")),
                round(_to_number(record.get("avgEntryPrice")), 6),
                round(_to_number(record.get("priceSold")), 6),
                round(_to_number(record.get("totalPL")), 6),
            )
        )
    return signature


def _audit_latest(report: AuditReport, ctx: AuditContext, *, requester: Any, base_url: str | None) -> None:
    endpoint = "/api/trades/latest?limit=25"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)
    expected = _expected_latest_rows(ctx.trades_frame, limit=25)

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []
    if payload.get("source") != "postgres":
        ok = False
        details.append(f"source={payload.get('source')}")
    if payload.get("db_source_of_truth") is not True:
        ok = False
        details.append("db_source_of_truth flag missing")

    actual_rows = payload.get("rows") or []
    if _latest_signature(actual_rows) != _latest_signature(expected):
        ok = False
        details.append("rows_mismatch")

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:3]))


def _audit_overview(report: AuditReport, ctx: AuditContext, *, requester: Any, base_url: str | None) -> None:
    endpoint = "/api/trades/overview"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []

    if payload.get("db_source_of_truth") is not True:
        ok = False
        details.append("db_source_of_truth flag missing")

    open_positions = payload.get("open_positions") or {}
    if _to_int(open_positions.get("count")) != int(ctx.open_count):
        ok = False
        details.append(f"open_count actual={open_positions.get('count')} expected={ctx.open_count}")
    if not _float_close(open_positions.get("realized_pnl"), ctx.open_realized_pnl, abs_tol=1e-6):
        ok = False
        details.append(
            f"open_realized_pnl actual={open_positions.get('realized_pnl')} expected={ctx.open_realized_pnl}"
        )

    got_metrics = payload.get("metrics") or {}
    for key, exp_value in ctx.latest_metrics.items():
        got_value = got_metrics.get(key)
        if isinstance(exp_value, (int, float, Decimal)):
            if not _float_close(got_value, exp_value, abs_tol=1e-6):
                ok = False
                details.append(f"metrics.{key} actual={got_value} expected={exp_value}")
        else:
            if str(got_value) != str(exp_value):
                ok = False
                details.append(f"metrics.{key} actual={got_value} expected={exp_value}")

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details[:8]))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:4]))


def _parse_ts_utc(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    return parsed


def _positions_summary_from_payload(positions: list[dict[str, Any]]) -> dict[str, float]:
    if not positions:
        return {
            "totalShares": 0.0,
            "totalOpenPL": 0.0,
            "avgDaysHeld": 0.0,
            "totalCapturedPL": 0.0,
        }

    total_shares = 0.0
    total_open_pl = 0.0
    total_days = 0.0
    total_captured_pl = 0.0

    for row in positions:
        qty = _to_number((row or {}).get("qty"))
        dollar_pl = _to_number((row or {}).get("dollarPL"))
        days = _to_number((row or {}).get("daysHeld"))
        captured_pl = _to_number((row or {}).get("capturedPL"))
        total_shares += qty
        total_open_pl += dollar_pl
        total_days += days
        total_captured_pl += captured_pl

    avg_days = total_days / len(positions) if positions else 0.0
    return {
        "totalShares": total_shares,
        "totalOpenPL": total_open_pl,
        "avgDaysHeld": avg_days,
        "totalCapturedPL": total_captured_pl,
    }


def _audit_account_overview(report: AuditReport, ctx: AuditContext, *, requester: Any, base_url: str | None) -> None:
    endpoint = "/api/account/overview"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)
    expected = ctx.account_snapshot_db or {}

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        ok = False
        details.append("snapshot missing")
        snapshot = {}

    if payload.get("db_source_of_truth") is not True:
        ok = False
        details.append("db_source_of_truth flag missing")

    if expected:
        if payload.get("source") != "postgres":
            ok = False
            details.append(f"source={payload.get('source')}")
        if payload.get("db_ready") is not True:
            ok = False
            details.append("db_ready expected true")
        if payload.get("ok") is not True:
            ok = False
            details.append("ok expected true")

        for key in ("account_id", "status", "source"):
            if key in expected and str(snapshot.get(key)) != str(expected.get(key)):
                ok = False
                details.append(f"{key} actual={snapshot.get(key)} expected={expected.get(key)}")

        for key in ("equity", "cash", "buying_power", "portfolio_value"):
            if key in expected and expected.get(key) not in (None, ""):
                if not _float_close(snapshot.get(key), expected.get(key), abs_tol=1e-6):
                    ok = False
                    details.append(f"{key} actual={snapshot.get(key)} expected={expected.get(key)}")

        if "taken_at" in expected:
            got_ts = _parse_ts_utc(snapshot.get("taken_at"))
            exp_ts = _parse_ts_utc(expected.get("taken_at"))
            if got_ts is None or exp_ts is None:
                ok = False
                details.append(f"taken_at parse failed actual={snapshot.get('taken_at')} expected={expected.get('taken_at')}")
            else:
                diff = abs((got_ts - exp_ts).total_seconds())
                if diff > 1.0:
                    ok = False
                    details.append(f"taken_at delta={diff:.2f}s")
    else:
        if payload.get("db_ready") is True:
            ok = False
            details.append("db_ready true but no expected DB snapshot")
        if payload.get("source") != "postgres":
            ok = False
            details.append(f"source={payload.get('source')}")

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details[:8]))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:4]))


def _audit_positions_monitoring(
    report: AuditReport,
    ctx: AuditContext,
    *,
    requester: Any,
    base_url: str | None,
) -> None:
    endpoint = "/api/positions/monitoring"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)
    expected_positions = ctx.open_positions_expected or {}

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []

    if payload.get("db_source_of_truth") is not True:
        ok = False
        details.append("db_source_of_truth flag missing")
    if payload.get("calculationSource") != "postgres":
        ok = False
        details.append(f"calculationSource={payload.get('calculationSource')}")
    if str(payload.get("source") or "") not in {"db", "db+alpaca"}:
        ok = False
        details.append(f"source={payload.get('source')}")

    positions = payload.get("positions")
    if not isinstance(positions, list):
        ok = False
        details.append("positions missing")
        positions = []

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        ok = False
        details.append("summary missing")
        summary = {}

    api_by_symbol: dict[str, dict[str, Any]] = {}
    for row in positions:
        if not isinstance(row, dict):
            ok = False
            details.append("position row invalid")
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            ok = False
            details.append("position symbol missing")
            continue
        api_by_symbol[symbol] = row

        for required in ("qty", "entryPrice", "currentPrice", "dollarPL", "percentPL", "capturedPL"):
            if required not in row:
                ok = False
                details.append(f"{symbol}.{required} missing")

    expected_symbols = set(expected_positions.keys())
    api_symbols = set(api_by_symbol.keys())
    if api_symbols != expected_symbols:
        ok = False
        details.append(f"symbol_set actual={sorted(api_symbols)} expected={sorted(expected_symbols)}")

    for symbol, expected_row in expected_positions.items():
        got = api_by_symbol.get(symbol)
        if not got:
            continue
        if not _float_close(got.get("qty"), expected_row.get("qty"), abs_tol=1e-6):
            ok = False
            details.append(f"{symbol}.qty actual={got.get('qty')} expected={expected_row.get('qty')}")
        if not _float_close(got.get("entryPrice"), expected_row.get("entryPrice"), abs_tol=1e-4):
            ok = False
            details.append(
                f"{symbol}.entryPrice actual={got.get('entryPrice')} expected={expected_row.get('entryPrice')}"
            )

    expected_summary = _positions_summary_from_payload([row for row in api_by_symbol.values()])
    for key in ("totalShares", "totalOpenPL", "avgDaysHeld", "totalCapturedPL"):
        if not _float_close(summary.get(key), expected_summary.get(key), abs_tol=1e-6):
            ok = False
            details.append(f"summary.{key} actual={summary.get(key)} expected={expected_summary.get(key)}")

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details[:10]))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:5]))


def _audit_positions_logs(report: AuditReport, *, requester: Any, base_url: str | None) -> None:
    endpoint = "/api/positions/logs?limit=20"
    status, payload = _request_json(endpoint, base_url=base_url, flask_client=requester)

    if status != 200:
        report.add_issue(f"{endpoint}: unexpected HTTP {status}")
        report.add_check(endpoint=endpoint, ok=False, detail=f"http_status={status}")
        return

    ok = True
    details: list[str] = []

    source = str(payload.get("source") or "")
    allowed_sources = {
        "pythonanywhere+alpaca",
        "pythonanywhere",
        "alpaca",
        "local-fallback+alpaca",
        "local-fallback",
        "none",
    }
    if source not in allowed_sources:
        ok = False
        details.append(f"source={source}")

    logs = payload.get("logs")
    if not isinstance(logs, list):
        ok = False
        details.append("logs missing")
        logs = []

    if len(logs) > 20:
        ok = False
        details.append(f"log_count={len(logs)} exceeds limit=20")

    for row in logs:
        if not isinstance(row, dict):
            ok = False
            details.append("log row invalid")
            continue
        if "timestamp" not in row or "type" not in row or "message" not in row:
            ok = False
            details.append("log row missing fields")
            break

    if not ok:
        report.add_issue(f"{endpoint}: " + "; ".join(details[:6]))
    report.add_check(endpoint=endpoint, ok=ok, detail="ok" if ok else "; ".join(details[:3]))


def run_audit(*, base_url: str | None = None) -> AuditReport:
    report = AuditReport(generated_at_utc=datetime.now(timezone.utc).isoformat())
    ctx = _build_audit_context()

    requester = None
    if not base_url:
        from dashboards.dashboard_app import server  # imported lazily to keep CLI startup light

        requester = server.test_client()

    _audit_stats(report, ctx, requester=requester, base_url=base_url)
    _audit_leaderboard(report, ctx, requester=requester, base_url=base_url)
    _audit_latest(report, ctx, requester=requester, base_url=base_url)
    _audit_overview(report, ctx, requester=requester, base_url=base_url)
    _audit_account_overview(report, ctx, requester=requester, base_url=base_url)
    _audit_positions_monitoring(report, ctx, requester=requester, base_url=base_url)
    _audit_positions_logs(report, requester=requester, base_url=base_url)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dashboard trades/account/positions APIs against DB source-of-truth."
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional dashboard base URL (e.g. http://127.0.0.1:8050). If omitted, uses in-process Flask test client.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 if audit reports mismatches.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the JSON report (e.g. reports/dashboard_api_audit.json).",
    )
    args = parser.parse_args()

    load_env(required_keys=())
    base_url = args.base_url.strip() or None

    report = run_audit(base_url=base_url)
    payload = {
        "generated_at_utc": report.generated_at_utc,
        "ok": report.ok,
        "checks_total": len(report.endpoint_checks),
        "checks_passed": sum(1 for check in report.endpoint_checks if check.get("ok")),
        "checks_failed": sum(1 for check in report.endpoint_checks if not check.get("ok")),
        "issues": report.issues,
        "endpoint_checks": report.endpoint_checks,
    }
    serialized = json.dumps(payload, indent=2, default=str)
    print(serialized)

    output_path = args.output.strip()
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")

    if args.fail_on_mismatch and not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
