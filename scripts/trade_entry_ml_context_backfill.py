"""Backfill entry-time ML score context for closed-trade attribution.

DB-first evaluator utility. This script is paper-mode safe and does not alter
trade execution semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from psycopg2 import extras

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("trade_entry_ml_context_backfill")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MAX_LAG_HOURS = 24


@dataclass
class BackfillArgs:
    lookback_days: int
    max_lag_hours: int
    dry_run: bool
    limit: int | None
    output_dir: Path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):  # type: ignore[arg-type]
            return None
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.isoformat()
    if isinstance(value, (float,)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def _window_bounds(lookback_days: int) -> tuple[datetime, datetime]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=max(int(lookback_days), 1))
    return start_dt, end_dt


def _ensure_output_dir(path: Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_closed_trades(
    conn: Any,
    *,
    start_dt: datetime,
    end_dt: datetime,
    limit: int | None,
) -> pd.DataFrame:
    db.trade_policy(conn, scope="VALIDATED_PERFORMANCE")
    predicate = db.disposition.eligible_sql("t", scope="VALIDATED_PERFORMANCE")
    limit_sql = ""
    params: dict[str, Any] = {"start_dt": start_dt, "end_dt": end_dt}
    if limit is not None and int(limit) > 0:
        params["limit"] = int(limit)
        limit_sql = " LIMIT %(limit)s"
    query = (
        f"""
        SELECT
            t.trade_id,
            UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol,
            t.entry_order_id,
            t.entry_time,
            t.exit_time,
            t.entry_price,
            t.exit_price,
            t.realized_pnl,
            t.exit_reason,
            t.status,
            CASE WHEN c.order_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_context
        FROM public.trades t
        LEFT JOIN public.trade_entry_ml_context_app c
          ON c.order_id = t.entry_order_id
        WHERE UPPER(COALESCE(t.status, '')) = 'CLOSED'
          AND {predicate}
          AND COALESCE(t.exit_time, t.entry_time) >= %(start_dt)s
          AND COALESCE(t.exit_time, t.entry_time) <= %(end_dt)s
        ORDER BY COALESCE(t.exit_time, t.entry_time) DESC NULLS LAST, t.trade_id DESC
        """
        + limit_sql
    )
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    frame = pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()
    if frame.empty:
        return frame
    for column in ("entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["entry_order_id"] = frame["entry_order_id"].astype("string")
    frame["has_context"] = frame["has_context"].fillna(False).astype(bool)
    return frame


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _extract_ml_context_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    direct = payload.get("ml_context")
    if isinstance(direct, Mapping):
        return dict(direct)
    nested_raw = payload.get("raw")
    if isinstance(nested_raw, Mapping):
        nested_ctx = nested_raw.get("ml_context")
        if isinstance(nested_ctx, Mapping):
            return dict(nested_ctx)
    return {}


def _raw_context_candidates(conn: Any, trades_missing: pd.DataFrame) -> pd.DataFrame:
    db.trade_policy(conn, scope="VALIDATED_PERFORMANCE")
    predicate = db.disposition.eligible_sql("t", scope="VALIDATED_PERFORMANCE")
    if trades_missing.empty:
        return pd.DataFrame()
    order_ids = [
        str(x).strip()
        for x in trades_missing["entry_order_id"].astype("string").dropna().tolist()
        if str(x).strip()
    ]
    if not order_ids:
        return pd.DataFrame()

    trade_meta: dict[str, dict[str, Any]] = {}
    for _, row in trades_missing.iterrows():
        order_id = str(row.get("entry_order_id") or "").strip()
        if not order_id or order_id in trade_meta:
            continue
        trade_meta[order_id] = {
            "trade_id": row.get("trade_id"),
            "symbol": str(row.get("symbol") or "").strip().upper(),
            "entry_time": row.get("entry_time"),
        }

    trade_ids = [int(x) for x in trades_missing["trade_id"].tolist()]
    db.ml_trade_order_owners(conn, list(zip(trade_ids, trades_missing["entry_order_id"].astype("string").tolist())))
    query = f"""
        SELECT e.order_id, e.raw
        FROM public.executed_trades e
        JOIN public.trades t ON t.entry_order_id=e.order_id
        WHERE e.order_id = ANY(%(order_ids)s) AND t.trade_id=ANY(%(trade_ids)s)
          AND {predicate} AND pg_catalog.upper(COALESCE(t.status,''))='CLOSED'
        ORDER BY e.entry_time DESC NULLS LAST
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, {"order_ids": order_ids, "trade_ids": trade_ids})
        rows = cursor.fetchall()

    candidates_by_order: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        order_id = str(row.get("order_id") or "").strip()
        if not order_id or order_id in candidates_by_order:
            continue
        payload = _coerce_payload(row.get("raw"))
        context = _extract_ml_context_from_payload(payload)
        if not context:
            continue
        model_score = _safe_float(context.get("model_score"))
        model_score_5d = _safe_float(context.get("model_score_5d"))
        if model_score is None and model_score_5d is None:
            continue
        trade_info = trade_meta.get(order_id, {})
        screener_run_ts = db.normalize_ts(
            context.get("screener_run_ts_utc"), field="screener_run_ts_utc"
        )
        entry_time = db.normalize_ts(trade_info.get("entry_time"), field="entry_time")
        lag_hours = None
        if entry_time is not None and screener_run_ts is not None:
            lag_hours = _safe_float((entry_time - screener_run_ts).total_seconds() / 3600.0)
        candidates_by_order[order_id] = {
            "order_id": order_id,
            "trade_id": trade_info.get("trade_id"),
            "symbol": trade_info.get("symbol"),
            "entry_time": entry_time,
            "screener_run_ts_utc": screener_run_ts,
            "model_score": model_score if model_score is not None else model_score_5d,
            "model_score_5d": model_score_5d if model_score_5d is not None else model_score,
            "score_col": str(context.get("score_col") or "model_score"),
            "score_source": "backfill.raw",
            "source_kind": "raw",
            "lag_hours": lag_hours,
        }
    return pd.DataFrame(list(candidates_by_order.values()))


def _scores_asof_candidates(
    conn: Any,
    trades_missing: pd.DataFrame,
    *,
    max_lag_hours: int,
) -> pd.DataFrame:
    db.trade_policy(conn, scope="VALIDATED_PERFORMANCE")
    predicate = db.disposition.eligible_sql("t", scope="VALIDATED_PERFORMANCE")
    if trades_missing.empty:
        return pd.DataFrame()
    order_ids = [
        str(x).strip()
        for x in trades_missing["entry_order_id"].astype("string").dropna().tolist()
        if str(x).strip()
    ]
    if not order_ids:
        return pd.DataFrame()
    trade_ids = [int(x) for x in trades_missing["trade_id"].tolist()]
    db.ml_trade_order_owners(conn, list(zip(trade_ids, trades_missing["entry_order_id"].astype("string").tolist())))
    query = f"""
        WITH missing AS (
            SELECT DISTINCT
                t.trade_id,
                t.entry_order_id AS order_id,
                UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol_norm,
                UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol,
                t.entry_time
            FROM public.trades t
            WHERE t.entry_order_id = ANY(%(order_ids)s)
              AND t.trade_id=ANY(%(trade_ids)s) AND {predicate}
              AND pg_catalog.upper(COALESCE(t.status,''))='CLOSED'
              AND t.entry_order_id IS NOT NULL
              AND t.entry_time IS NOT NULL
        )
        SELECT
            m.trade_id,
            m.order_id,
            m.symbol,
            m.entry_time,
            s.run_ts_utc AS screener_run_ts_utc,
            s.model_score_5d AS model_score_5d,
            EXTRACT(EPOCH FROM (m.entry_time - s.run_ts_utc)) / 3600.0 AS lag_hours
        FROM missing m
        LEFT JOIN LATERAL (
            SELECT run_ts_utc, model_score_5d
            FROM public.screener_ranker_scores_app s
            WHERE UPPER(TRIM(COALESCE(s.symbol, ''))) = m.symbol_norm
              AND date_trunc('second', s.run_ts_utc) <= date_trunc('second', m.entry_time)
            ORDER BY s.run_ts_utc DESC
            LIMIT 1
        ) s ON TRUE
        WHERE s.run_ts_utc IS NOT NULL
          AND (m.entry_time - s.run_ts_utc) <= (%(max_lag_hours)s * INTERVAL '1 hour')
        ORDER BY m.entry_time DESC NULLS LAST
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(
            query,
            {"order_ids": order_ids, "trade_ids": trade_ids, "max_lag_hours": max(int(max_lag_hours), 0)},
        )
        rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["screener_run_ts_utc"] = pd.to_datetime(
        frame["screener_run_ts_utc"], utc=True, errors="coerce"
    )
    frame["model_score_5d"] = pd.to_numeric(frame["model_score_5d"], errors="coerce")
    frame["model_score"] = frame["model_score_5d"]
    frame["score_col"] = "model_score"
    frame["score_source"] = "backfill.scores_direct"
    frame["source_kind"] = "scores_asof"
    frame["lag_hours"] = pd.to_numeric(frame.get("lag_hours"), errors="coerce")
    return frame


def _write_fs_outputs(output_dir: Path, summary: Mapping[str, Any], rows: pd.DataFrame) -> None:
    raise db.disposition.TradePolicyError("ML_TRADE_PUBLICATION_UNVALIDATED")


def run_backfill(args: BackfillArgs) -> dict[str, Any]:
    start_dt, end_dt = _window_bounds(args.lookback_days)
    output_dir = Path(args.output_dir)
    run_date = datetime.now(timezone.utc).date()

    LOG.info(
        "[INFO] ENTRY_CONTEXT_BACKFILL_START lookback_days=%d max_lag_hours=%d dry_run=%s",
        int(args.lookback_days),
        int(args.max_lag_hours),
        str(bool(args.dry_run)).lower(),
    )

    default_summary: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": str(run_date),
        "status": "ok",
        "lookback_days": int(args.lookback_days),
        "max_lag_hours": int(args.max_lag_hours),
        "dry_run": bool(args.dry_run),
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "trades_total": 0,
        "missing_context": 0,
        "candidates_from_raw": 0,
        "candidates_from_scores": 0,
        "rows_candidate_total": 0,
        "rows_upserted": 0,
        "rows_skipped": 0,
        "reasons": {},
        "output_files": {
            "latest_json": str(output_dir / "latest.json"),
            "backfilled_rows_csv": str(output_dir / "backfilled_rows.csv"),
        },
    }

    if not db.db_enabled():
        raise db.disposition.TradePolicyError("TRADE_POLICY_DB_UNAVAILABLE")
    conn = db.get_db_conn()
    if conn is None:
        raise db.disposition.TradePolicyError("TRADE_POLICY_DB_UNAVAILABLE")
    try:
        with db.ml_trade_read_transaction(conn) as policy:
            records = db.ml_trade_input(conn, policy, start_dt=start_dt, end_dt=end_dt)
            db.ml_trade_order_owners(conn, [(row["trade_id"], row.get("entry_order_id")) for row in records])
            match_inputs = db.ml_trade_match_inputs(conn, records, oos_run_date=None)
            evidence = db.ml_trade_evidence(policy, records, start_dt=start_dt, end_dt=end_dt, match_inputs=match_inputs)
            trades = _load_closed_trades(conn, start_dt=start_dt, end_dt=end_dt, limit=args.limit)
            trades_total = int(trades.shape[0])
            if trades.empty:
                combined = pd.DataFrame()
                summary = dict(default_summary, status="no_data", selection_limit=args.limit)
            else:
                missing = trades.loc[~trades["has_context"]].copy()
                missing_context = int(missing.shape[0])
                missing_order_id = int(
                    missing["entry_order_id"].astype("string").fillna("").str.strip().eq("").sum()
                )
                missing_entry_time = int(missing["entry_time"].isna().sum())
                eligible = missing.loc[
                    missing["entry_order_id"].astype("string").fillna("").str.strip().ne("")
                    & missing["entry_time"].notna()
                ].copy()

                raw_candidates = _raw_context_candidates(conn, eligible)
                raw_order_ids = set(
                    str(x).strip()
                    for x in raw_candidates.get("order_id", pd.Series(dtype="string")).dropna().tolist()
                    if str(x).strip()
                )
                remaining = eligible.loc[
                    ~eligible["entry_order_id"].astype("string").isin(list(raw_order_ids))
                ].copy()
                score_candidates = _scores_asof_candidates(
                    conn, remaining, max_lag_hours=args.max_lag_hours
                )
                score_order_ids = set(
                    str(x).strip()
                    for x in score_candidates.get("order_id", pd.Series(dtype="string")).dropna().tolist()
                    if str(x).strip()
                )
                no_score_within_lag = max(
                    int(remaining.shape[0]) - len(score_order_ids),
                    0,
                )

                raw_candidates["source_priority"] = 0
                score_candidates["source_priority"] = 1
                combined = pd.concat([raw_candidates, score_candidates], ignore_index=True, sort=False)
                if not combined.empty:
                    combined.sort_values(
                        by=["source_priority", "lag_hours", "order_id"],
                        ascending=[True, True, True],
                        inplace=True,
                    )
                    combined = combined.drop_duplicates(subset=["order_id"], keep="first")
                    combined.reset_index(drop=True, inplace=True)
                rows_candidate_total = int(combined.shape[0]) if not combined.empty else 0

                LOG.info(
                    "[INFO] ENTRY_CONTEXT_BACKFILL_DIAG trades_total=%d missing_context=%d candidates_from_raw=%d candidates_from_scores=%d",
                    trades_total,
                    missing_context,
                    int(raw_candidates.shape[0]) if not raw_candidates.empty else 0,
                    int(score_candidates.shape[0]) if not score_candidates.empty else 0,
                )

            db._ml_trade_current(conn, evidence)

        context_rows = []
        if not args.dry_run and not combined.empty:
            for _, row in combined.iterrows():
                payload = {
                    "trade_id": int(row.get("trade_id")),
                    "order_id": str(row.get("order_id") or "").strip(),
                    "symbol": str(row.get("symbol") or "").strip().upper(),
                    "entry_time": row.get("entry_time"),
                    "screener_run_ts_utc": row.get("screener_run_ts_utc"),
                    "model_score": _safe_float(row.get("model_score")),
                    "model_score_5d": _safe_float(row.get("model_score_5d")),
                    "score_col": str(row.get("score_col") or "model_score"),
                    "score_source": str(row.get("score_source") or "backfill"),
                    "raw": {
                        "source_kind": str(row.get("source_kind") or ""),
                        "lag_hours": _safe_float(row.get("lag_hours")),
                        "trade_id": row.get("trade_id"),
                        "backfill_run_utc": datetime.now(timezone.utc).isoformat(),
                    },
                }
                context_rows.append(_json_safe(payload))
        if not trades.empty:
            summary = dict(default_summary, status="dry_run" if args.dry_run else "ok",
                trades_total=trades_total, missing_context=missing_context,
                rows_candidate_total=rows_candidate_total, selection_limit=args.limit,
                candidates_from_raw=int(raw_candidates.shape[0]), candidates_from_scores=int(score_candidates.shape[0]),
                reasons={"missing_order_id":missing_order_id,"missing_entry_time":missing_entry_time,
                    "no_score_within_lag":no_score_within_lag})
        rows_out = combined.copy()
        rows_csv = rows_out.to_csv(index=False)
        summary = db.publish_ml_trade_artifacts(evidence, kind="trade_entry_ml_context_backfill",
            run_date=run_date, payload=_json_safe(summary), rows_csv=rows_csv,
            rows_count=int(rows_out.shape[0]), context_rows=context_rows)
        _ensure_output_dir(output_dir)
        (output_dir / "latest.json").write_text(db.disposition.canonical(summary), encoding="utf-8")
        (output_dir / "backfilled_rows.csv").write_text(rows_csv, encoding="utf-8")
        return summary
    finally:
        try:
            conn.close()
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> BackfillArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--max-lag-hours", type=int, default=DEFAULT_MAX_LAG_HOURS)
    parser.add_argument(
        "--dry-run",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Compute summary but skip DB upserts (true/false).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max trades to inspect.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "trade_entry_ml_context_backfill",
    )
    parsed = parser.parse_args(argv)
    if int(parsed.lookback_days) < 1:
        parser.error("--lookback-days must be >= 1")
    if int(parsed.max_lag_hours) < 0:
        parser.error("--max-lag-hours must be >= 0")
    if parsed.limit is not None and int(parsed.limit) < 1:
        parser.error("--limit must be >= 1")
    return BackfillArgs(
        lookback_days=int(parsed.lookback_days),
        max_lag_hours=int(parsed.max_lag_hours),
        dry_run=bool(parsed.dry_run),
        limit=(int(parsed.limit) if parsed.limit is not None else None),
        output_dir=Path(parsed.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_backfill(args)
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.error("ENTRY_CONTEXT_BACKFILL_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
