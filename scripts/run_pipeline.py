"""Daily pipeline runner for JBRAVO (production-ready post Phase V).

The pipeline relies on the screener summary line for observability and uses
pipeline_health_app.run_ts_utc plus screener_run_map_app for run scoping.
The database is the source of truth; CSV fallback is disabled.
"""
import argparse
import csv
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from types import SimpleNamespace
import zoneinfo

import pandas as pd
import requests

from scripts import db
from scripts import db_migrate
from scripts.db_queries import get_latest_screener_candidates
from scripts.export_latest_candidates import export_latest_candidates
from scripts.fallback_candidates import CANONICAL_COLUMNS, normalize_candidate_df
from scripts.log_rotate import rotate_if_needed
from scripts.screener import write_universe_prefix_counts
from scripts.utils.env import load_env, market_data_base_url, trading_base_url
from utils import write_csv_atomic, atomic_write_bytes
from utils.screener_metrics import ensure_canonical_metrics, write_screener_metrics_json
from utils.alerts import send_alert
from utils.env import get_alpaca_creds
from utils.telemetry import emit_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_SNIPS_DIR = REPORTS_DIR / "log_snips"
PIPELINE_SUMMARY_PATH = REPORTS_DIR / "pipeline_summary.json"
LOG = logging.getLogger("pipeline")
logger = LOG

_SPLIT_FLAG_MAP = {
    "--screener-args-split": "screener_args_split",
    "--backtest-args-split": "backtest_args_split",
    "--metrics-args-split": "metrics_args_split",
    "--ranker-eval-args-split": "ranker_eval_args_split",
}

_SUMMARY_RE = re.compile(
    r"PIPELINE_SUMMARY.*?symbols_in=(?P<symbols_in>\d+).*?"
    r"with_bars=(?P<symbols_with_bars>\d+).*?"
    r"(?:with_bars_any=(?P<with_bars_any>\d+).*?)?"
    r"(?:with_bars_required=(?P<with_bars_required>\d+).*?)?"
    r"rows=(?P<rows>\d+)"
    r"(?:.*?(?:bars?_rows(?:_total)?)=(?P<bars_rows_total>\d+))?"
)


def write_error_report(
    *, step: str | None = None, detail: str | None = None, base_dir: Path = PROJECT_ROOT
) -> Path:
    """Persist a minimal failure artifact for dashboard consumption."""

    reports_dir = Path(base_dir) / "reports" / "log_snips"
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = f"{(step or 'pipeline').lower()}.err.txt"
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [f"timestamp={timestamp}", f"step={step or 'pipeline'}"]
    if detail:
        lines.append(f"detail={detail}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    target = reports_dir / name
    try:
        atomic_write_bytes(target, payload)
    except Exception:
        LOG.exception("PIPELINE_ERROR_REPORT_WRITE_FAILED path=%s", target)
    return target


def safe_write_pipeline_summary(rc: int, *, base_dir: Path = PROJECT_ROOT) -> Path:
    """Ensure a lightweight pipeline summary exists even on failure."""

    reports_dir = Path(base_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rc": int(rc),
    }
    target = reports_dir / "pipeline_summary.json"
    try:
        serialised = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(target, serialised)
    except Exception:
        LOG.exception("PIPELINE_SUMMARY_WRITE_FAILED path=%s", target)
    return target


def _refresh_logger() -> None:
    global logger
    logger = LOG
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
SCREENER_METRICS_PATH = DATA_DIR / "screener_metrics.json"
LATEST_CANDIDATES = DATA_DIR / "latest_candidates.csv"
TOP_CANDIDATES = DATA_DIR / "top_candidates.csv"
BACKTEST_RESULTS = DATA_DIR / "backtest_results.csv"
METRICS_SUMMARY = DATA_DIR / "metrics_summary.csv"
DEFAULT_WSGI_PATH = Path("/var/www/raspatrick_pythonanywhere_com_wsgi.py")
BASE_DIR = PROJECT_ROOT
copyfile = shutil.copyfile
emit = emit_event
LATEST_COLUMNS = list(CANONICAL_COLUMNS)
LATEST_HEADER = ",".join(LATEST_COLUMNS) + "\n"
DEFAULT_LABELS_BARS_PATH = Path("data") / "daily_bars.csv"
DEFAULT_RANKER_SCORE_COLUMN = "score_5d"
DEFAULT_RANKER_TARGET_COLUMN = "model_score_5d"
DEFAULT_ALLOC_WEIGHT_TOP_K = 4
_PIPELINE_SUMMARY_FOR_DB: dict[str, Any] | None = None
_PIPELINE_RC: int | None = None
_PIPELINE_STARTED_AT: datetime | None = None
_PIPELINE_ENDED_AT: datetime | None = None


def _record_health(stage: str) -> dict[str, Any]:  # pragma: no cover - legacy hook
    return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialised = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(target, serialised)
    except Exception:
        LOG.exception("PIPELINE_JSON_WRITE_FAILED path=%s", target)


def _latest_by_glob(directory: Path, pattern: str) -> Path | None:
    try:
        candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    except FileNotFoundError:
        return None
    except Exception:
        LOG.exception("ARTIFACT_GLOB_FAILED dir=%s pattern=%s", directory, pattern)
        return None
    return candidates[-1] if candidates else None


def _artifact_payload(path: Path | None) -> dict[str, object]:
    payload: dict[str, object] = {"path": None, "modified": None}
    if path is None:
        return payload

    payload["path"] = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return payload
    except Exception:
        LOG.exception("ARTIFACT_STAT_FAILED path=%s", path)
        return payload

    payload["modified"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return payload


def _artifact_payload_db(artifact_type: str) -> dict[str, object]:
    payload: dict[str, object] = {"path": None, "modified": None}
    if not db.db_enabled():
        return payload
    record = db.fetch_latest_ml_artifact(artifact_type)
    if not record:
        return payload
    created_at = record.get("created_at")
    if isinstance(created_at, datetime):
        modified = created_at.isoformat()
    else:
        modified = None
    payload["path"] = f"db://ml_artifacts/{artifact_type}"
    payload["modified"] = modified
    if record.get("run_date"):
        payload["run_date"] = str(record.get("run_date"))
    if record.get("rows_count") is not None:
        try:
            payload["rows"] = int(record.get("rows_count") or 0)
        except Exception:
            payload["rows"] = record.get("rows_count")
    return payload


def _write_nightly_ml_status(base_dir: Path) -> None:
    data_dir = Path(base_dir) / "data"
    if db.db_enabled():
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "bars": _artifact_payload_db("daily_bars"),
            "labels": _artifact_payload_db("labels"),
            "features": _artifact_payload_db("features"),
            "model": _artifact_payload(
                _latest_by_glob(data_dir / "models", "ranker_*.pkl")
            ),
            "predictions": _artifact_payload_db("predictions"),
            "eval": _artifact_payload_db("ranker_eval"),
        }
    else:
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "bars": _artifact_payload(data_dir / "daily_bars.csv"),
            "labels": _artifact_payload(
                _latest_by_glob(data_dir / "labels", "labels_*.csv")
            ),
            "features": _artifact_payload(
                _latest_by_glob(data_dir / "features", "features_*.csv")
            ),
            "model": _artifact_payload(
                _latest_by_glob(data_dir / "models", "ranker_*.pkl")
            ),
            "predictions": _artifact_payload(
                _latest_by_glob(data_dir / "predictions", "predictions_*.csv")
            ),
            "eval": _artifact_payload(
                _latest_by_glob(data_dir / "ranker_eval", "*.json")
            ),
        }

    try:
        _write_json(data_dir / "nightly_ml_status.json", payload)
        LOG.info(
            "[INFO] NIGHTLY_ML_STATUS_WRITTEN path=%s", data_dir / "nightly_ml_status.json"
        )
    except Exception:
        LOG.exception("NIGHTLY_ML_STATUS_WRITE_FAILED")


_PROBE_SYMBOLS = ("SPY", "AAPL")


def _alpaca_headers() -> dict[str, str]:
    key, secret, _, _ = get_alpaca_creds()
    headers: dict[str, str] = {}
    if isinstance(key, str):
        key = key.strip()
        if key:
            headers["APCA-API-KEY-ID"] = key
    if isinstance(secret, str):
        secret = secret.strip()
        if secret:
            headers["APCA-API-SECRET-KEY"] = secret
    return headers


def _probe_trading_endpoint(session: requests.Session, headers: Mapping[str, str]) -> dict[str, Any]:
    url = f"{trading_base_url().rstrip('/')}/v2/account"
    if "APCA-API-KEY-ID" not in headers or "APCA-API-SECRET-KEY" not in headers:
        LOG.warning("CONNECTION_PROBE trading skipped reason=missing_credentials")
        return {"ok": False, "status": 0}
    try:
        response = session.get(url, headers=headers, timeout=10)
    except Exception as exc:  # pragma: no cover - network failures
        LOG.warning("CONNECTION_PROBE trading error=%s", exc)
        return {"ok": False, "status": 0}
    ok = response.status_code == 200
    if ok:
        try:
            payload = response.json()
        except Exception:  # pragma: no cover - defensive parse
            payload = None
        if isinstance(payload, Mapping):
            if payload.get("trading_blocked"):
                ok = False
    return {"ok": ok, "status": int(response.status_code)}


def _probe_data_endpoint(
    session: requests.Session, headers: Mapping[str, str], feed: str
) -> dict[str, Any]:
    url = f"{market_data_base_url().rstrip('/')}/v2/stocks/bars"
    params = {
        "symbols": ",".join(_PROBE_SYMBOLS),
        "timeframe": "1Day",
        "limit": 1,
        "feed": feed or "iex",
    }
    try:
        response = session.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:  # pragma: no cover - network failures
        LOG.warning("CONNECTION_PROBE data error=%s", exc)
        return {"ok": False, "status": 0}
    ok = response.status_code == 200
    return {"ok": ok, "status": int(response.status_code)}


def _collect_connection_health_snapshot(
    *, fallback_trading_ok: bool, fallback_data_ok: bool
) -> dict[str, Any]:
    feed = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    payload = {
        "trading_ok": bool(fallback_trading_ok),
        "data_ok": bool(fallback_data_ok),
        "trading_status": 200 if fallback_trading_ok else 503,
        "data_status": 200 if fallback_data_ok else 204,
        "feed": feed,
        "timestamp": _now_iso(),
    }
    headers = _alpaca_headers()
    session = requests.Session()
    try:
        trading = _probe_trading_endpoint(session, headers)
        data = _probe_data_endpoint(session, headers, feed)
    except Exception:  # pragma: no cover - unexpected
        LOG.exception("CONNECTION_PROBE_FAILED")
    else:
        payload.update(
            {
                "trading_ok": bool(trading.get("ok")),
                "trading_status": int(trading.get("status", payload["trading_status"])),
                "data_ok": bool(data.get("ok")),
                "data_status": int(data.get("status", payload["data_status"])),
                "timestamp": _now_iso(),
            }
        )
    finally:
        session.close()
    return payload


def _parse_summary_line(line: str) -> Optional[SimpleNamespace]:
    match = _SUMMARY_RE.search(line)
    if not match:
        return None
    try:
        payload = {
            "symbols_in": int(match.group("symbols_in")),
            "with_bars": int(match.group("symbols_with_bars")),
            "rows": int(match.group("rows")),
            "bars_rows_total": (
                int(match.group("bars_rows_total")) if match.group("bars_rows_total") else None
            ),
        }
    except (TypeError, ValueError):
        return None
    payload["raw_line"] = line.strip()
    return SimpleNamespace(**payload)


def _latest_summary_for_date(base_dir: Path, day: date) -> Optional[SimpleNamespace]:
    log_path = Path(base_dir) / "logs" / "pipeline.log"
    if not log_path.exists():
        return None
    date_token = day.strftime("%Y-%m-%d")
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-100000:]
    except Exception:
        return None
    for line in reversed(tail.splitlines()):
        if "PIPELINE_SUMMARY" not in line:
            continue
        if date_token not in line:
            continue
        parsed = _parse_summary_line(line)
        if parsed is not None:
            return parsed
    return None


def _write_status_json(base_dir: Path, payload: Mapping[str, Any]) -> None:
    base_dir = Path(base_dir)
    status_path = base_dir / "data" / "pipeline_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialised = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(status_path, serialised)
    except Exception:
        LOG.exception("PIPELINE_STATUS_WRITE_FAILED path=%s", status_path)


REQUIRED_CAND_COLS = list(CANONICAL_COLUMNS)


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Attempt to coerce ``value`` into an ``int`` while tolerating garbage."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


DEFAULT_REQUIRED_BARS = 250

def _coerce_symbol(value: object) -> str:
    """Return a safe uppercase symbol string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().upper()
    return str(value).strip().upper()


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _should_write_candidate_csvs() -> bool:
    override = os.getenv("JBR_WRITE_CANDIDATE_CSVS")
    if override is not None:
        return _as_bool(override, False)
    return not db.db_enabled()



def compose_metrics_from_artifacts(
    base_dir: Path,
    *,
    symbols_in: int | None = None,
    fallback_symbols_with_bars: int | None = None,
    fallback_bars_rows_total: int | None = None,
    latest_source: str | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    data_dir = base / "data"
    metrics_path = data_dir / "screener_metrics.json"
    existing_metrics = ensure_canonical_metrics(_read_json(metrics_path))
    fetch_stats = _read_json(data_dir / "screener_stage_fetch.json")
    post_stats = _read_json(data_dir / "screener_stage_post.json")
    rows_final = 0
    latest_rows = 0
    scored_rows = 0
    db_rows: int | None = None
    if db.db_enabled():
        db_rows, _ = db.fetch_latest_screener_candidate_count()
        rows_final = int(db_rows or 0)
    else:
        rows_final = _count_rows(data_dir / "top_candidates.csv")
        latest_rows = _count_rows(data_dir / "latest_candidates.csv")
        if latest_rows:
            rows_final = max(rows_final, latest_rows)
    metrics_rows = _coerce_optional_int(existing_metrics.get("rows")) or _coerce_optional_int(
        existing_metrics.get("rows_out")
    )
    if metrics_rows is not None:
        rows_final = max(rows_final, metrics_rows)
    if rows_final == 0 and post_stats:
        hinted = _coerce_optional_int(post_stats.get("candidates_final"))
        if hinted:
            rows_final = hinted
    if not db.db_enabled():
        scored_candidates = data_dir / "scored_candidates.csv"
        scored_rows = _count_csv_lines(scored_candidates)
        if scored_rows <= 0 and scored_candidates.exists():
            scored_rows = 0
    symbols_in_hints = [
        _coerce_optional_int(existing_metrics.get("symbols_in")),
        _coerce_optional_int(symbols_in),
        _coerce_optional_int(fetch_stats.get("symbols_in")) if fetch_stats else None,
        _coerce_optional_int(post_stats.get("symbols_in")) if post_stats else None,
        scored_rows if scored_rows else None,
        db_rows if db_rows else None,
    ]
    symbols_in_effective = 0
    for hint in symbols_in_hints:
        if isinstance(hint, int) and hint > 0:
            symbols_in_effective = hint
            break
    required_bars_value = (
        _coerce_optional_int(existing_metrics.get("required_bars"))
        or _coerce_optional_int(fetch_stats.get("required_bars")) if fetch_stats else None
        or DEFAULT_REQUIRED_BARS
    )
    fetch_symbols_required = (
        _coerce_optional_int(fetch_stats.get("symbols_with_required_bars_fetch"))
        if fetch_stats
        else None
    )
    fetch_symbols_any = (
        _coerce_optional_int(fetch_stats.get("symbols_with_any_bars_fetch")) if fetch_stats else None
    )
    fetch_symbols_legacy = (
        _coerce_optional_int(fetch_stats.get("symbols_with_bars_fetch")) if fetch_stats else None
    )
    fallback_symbols = _coerce_optional_int(fallback_symbols_with_bars)
    with_bars_required = (
        _coerce_optional_int(existing_metrics.get("symbols_with_required_bars"))
        or fetch_symbols_required
        or fetch_symbols_legacy
        or fallback_symbols
        or 0
    )
    with_bars_any = (
        _coerce_optional_int(existing_metrics.get("symbols_with_any_bars"))
        or fetch_symbols_any
        or _coerce_optional_int(existing_metrics.get("symbols_with_bars_any"))
        or with_bars_required
        or 0
    )
    bars_fetch = _coerce_optional_int(fetch_stats.get("bars_rows_total_fetch")) if fetch_stats else None
    fallback_bars = _coerce_optional_int(fallback_bars_rows_total)
    bars_effective = (
        _coerce_optional_int(existing_metrics.get("bars_rows_total"))
        or bars_fetch
        or fallback_bars
        or rows_final
    )
    fetch_any_attempted = fetch_symbols_any or fetch_symbols_legacy
    if symbols_in_effective <= 0 and with_bars_any > 0:
        LOG.warning(
            "[WARN] METRICS_SYMBOLS_IN_RECOVERED from_with_bars=%s",
            with_bars_required,
        )
        symbols_in_effective = with_bars_any
    with_bars_required = min(with_bars_required, max(symbols_in_effective, 0))
    with_bars_any = min(max(with_bars_any, with_bars_required), max(symbols_in_effective, 0))

    bars_effective = max(int(bars_effective or 0), int(rows_final))
    payload = {
        "last_run_utc": _now_iso(),
        "symbols_in": int(symbols_in_effective or 0),
        "required_bars": int(required_bars_value or 0),
        "symbols_with_bars": int(with_bars_required),
        "symbols_with_bars_raw": int(with_bars_required),
        "symbols_with_bars_fetch": None,
        "symbols_attempted_fetch": None,
        "symbols_with_required_bars": int(with_bars_required),
        "symbols_with_any_bars": int(with_bars_any),
        "symbols_with_bars_required": int(with_bars_required),
        "symbols_with_bars_any": int(with_bars_any),
        "symbols_with_bars_post": _coerce_optional_int(post_stats.get("symbols_with_bars_post")) if post_stats else None,
        "bars_rows_total": int(bars_effective or 0),
        "bars_rows_total_fetch": bars_fetch,
        "bars_rows_total_post": _coerce_optional_int(post_stats.get("bars_rows_total_post")) if post_stats else None,
        "rows": int(rows_final),
        "rows_premetrics": int(rows_final),
        "latest_source": latest_source or "unknown",
        "metrics_version": 2,
    }
    if fetch_any_attempted is not None:
        if with_bars_any and fetch_any_attempted == with_bars_any:
            payload["symbols_with_bars_fetch"] = fetch_any_attempted
        else:
            payload["symbols_attempted_fetch"] = fetch_any_attempted
    elif fetch_symbols_required is not None:
        payload["symbols_attempted_fetch"] = fetch_symbols_required
    if post_stats and "candidates_final" in post_stats:
        payload["candidates_final"] = _coerce_optional_int(post_stats.get("candidates_final")) or int(rows_final)
    return payload


def _backfill_metrics_from_summary(
    metrics_path: Path, metrics: Mapping[str, Any] | None, summary: SimpleNamespace
) -> dict[str, Any]:
    """Backfill numeric KPIs in ``metrics`` using a fresh ``PIPELINE_SUMMARY``."""

    payload = dict(metrics or {})
    updated_fields: list[str] = []
    summary_map = {
        "symbols_in": getattr(summary, "symbols_in", None),
        "symbols_with_bars": getattr(summary, "with_bars", None),
        "symbols_with_any_bars": getattr(summary, "with_bars_any", None),
        "rows": getattr(summary, "rows", None),
        "bars_rows_total": getattr(summary, "bars_rows_total", None),
    }
    for key, summary_value in summary_map.items():
        summary_int = _coerce_optional_int(summary_value)
        if summary_int is None:
            continue
        existing_int = _coerce_optional_int(payload.get(key))
        if existing_int is None:
            payload[key] = summary_int
            updated_fields.append(key)
    if updated_fields:
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            LOG.info(
                "SCREENER_METRICS_BACKFILL source=pipeline_summary fields=%s",",".join(sorted(updated_fields))
            )
        except Exception:
            LOG.exception("SCREENER_METRICS_BACKFILL_FAILED path=%s", metrics_path)
    return payload


def _coerce_canonical(df_scored: Optional[pd.DataFrame], df_top: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a canonical candidates DataFrame derived from pipeline outputs."""

    if df_top is None or df_top.empty:
        return pd.DataFrame(columns=REQUIRED_CAND_COLS)

    base = df_top.copy()
    base.columns = [str(col).strip().lower() for col in base.columns]
    base = base.loc[:, ~base.columns.duplicated()]

    if "gates_passed" in base.columns and "passed_gates" not in base.columns:
        base["passed_gates"] = base["gates_passed"]

    if "symbol" not in base.columns or "score" not in base.columns:
        return pd.DataFrame(columns=REQUIRED_CAND_COLS)

    base["symbol"] = (
        base["symbol"].astype("string").fillna("").str.strip().str.upper()
    )
    base = base.loc[base["symbol"].str.len() > 0]

    if base.empty:
        return pd.DataFrame(columns=REQUIRED_CAND_COLS)

    if "timestamp" not in base.columns:
        base["timestamp"] = datetime.now(timezone.utc).isoformat()
    else:
        ts_series = base["timestamp"].astype("string").fillna("")
        now_value = datetime.now(timezone.utc).isoformat()
        base["timestamp"] = ts_series.replace({"": now_value})

    numeric_columns = ["score", "close", "volume", "universe_count", "entry_price", "adv20", "atrp"]
    for column in numeric_columns:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    if df_scored is not None and not df_scored.empty:
        scored = df_scored.copy()
        scored.columns = [str(col).strip().lower() for col in scored.columns]
        scored = scored.loc[:, ~scored.columns.duplicated()]
        if "symbol" in scored.columns:
            scored["symbol"] = (
                scored["symbol"].astype("string").fillna("").str.strip().str.upper()
            )
            merge_cols = [
                col
                for col in (
                    "symbol",
                    "close",
                    "exchange",
                    "volume",
                    "adv20",
                    "sma9",
                    "ema20",
                    "sma180",
                    "rsi14",
                    "passed_gates",
                    "gates_passed",
                    "gate_fail_reason",
                )
                if col in scored.columns
            ]
            if len(merge_cols) > 1:
                joinable = scored[merge_cols].drop_duplicates("symbol")
                base = base.merge(joinable, on="symbol", how="left", suffixes=("", "_scored"))
                for col in merge_cols:
                    if col == "symbol":
                        continue
                    scored_col = f"{col}_scored"
                    if scored_col not in base.columns:
                        continue
                    if col == "exchange":
                        existing = (
                            base[col].astype("string").fillna("")
                            if col in base.columns
                            else pd.Series("", index=base.index)
                        )
                        fallback = base[scored_col].astype("string").fillna("")
                        base[col] = existing.where(existing.str.strip() != "", fallback)
                    else:
                        if col not in base.columns:
                            base[col] = pd.NA
                        series = base[col]
                        mask_missing = series.isna()
                        if col in {"volume", "adv20", "sma9", "ema20", "sma180", "rsi14"}:
                            numeric_series = pd.to_numeric(series, errors="coerce")
                            mask_missing = numeric_series.isna()
                            base[col] = numeric_series
                        if col == "gate_fail_reason":
                            text_series = series.astype("string").fillna("")
                            mask_missing |= text_series.str.strip() == ""
                        base.loc[mask_missing, col] = base.loc[mask_missing, scored_col]
                    base.drop(columns=[scored_col], inplace=True)

    if "close" not in base.columns:
        base["close"] = pd.NA
    if "entry_price" not in base.columns:
        base["entry_price"] = pd.NA

    close_series = pd.to_numeric(base["close"], errors="coerce")
    entry_series = pd.to_numeric(base["entry_price"], errors="coerce")
    close_missing = close_series.isna()
    entry_missing = entry_series.isna()
    entry_series.loc[entry_missing] = close_series.loc[entry_missing]
    close_series.loc[close_missing] = entry_series.loc[close_missing]
    base["close"] = close_series
    base["entry_price"] = entry_series

    if "score_breakdown" in base.columns:
        base["score_breakdown"] = (
            base["score_breakdown"].astype("string").fillna("").replace({"": "{}", "fallback": "{}"})
        )
    else:
        base["score_breakdown"] = "{}"

    if "source" in base.columns:
        source_series = base["source"].astype("string").fillna("")
        base["source"] = source_series.replace({"": "screener"})
    else:
        base["source"] = "screener"

    if "universe_count" in base.columns:
        base["universe_count"] = (
            pd.to_numeric(base["universe_count"], errors="coerce").fillna(0).astype(int)
        )
    else:
        base["universe_count"] = 0

    if "atrp" in base.columns:
        base["atrp"] = pd.to_numeric(base["atrp"], errors="coerce").fillna(0.0)
    else:
        base["atrp"] = 0.0

    if "volume" in base.columns:
        base["volume"] = pd.to_numeric(base["volume"], errors="coerce").fillna(0)
    else:
        base["volume"] = 0

    if "adv20" in base.columns:
        base["adv20"] = pd.to_numeric(base["adv20"], errors="coerce").fillna(0.0)
    else:
        base["adv20"] = 0.0

    for column in REQUIRED_CAND_COLS:
        if column not in base.columns:
            base[column] = pd.NA

    ordered = REQUIRED_CAND_COLS + [col for col in base.columns if col not in REQUIRED_CAND_COLS]
    canonical = base[ordered]
    canonical = canonical.loc[canonical["symbol"].astype("string").str.len() > 0]
    return canonical[REQUIRED_CAND_COLS]


def write_latest_candidates_canonical(base_dir: Path) -> pd.DataFrame:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    top_path = data_dir / "top_candidates.csv"
    scored_path = data_dir / "scored_candidates.csv"
    latest_path = data_dir / "latest_candidates.csv"

    df_top = None
    if top_path.exists() and top_path.stat().st_size > 0:
        try:
            df_top = pd.read_csv(top_path)
        except Exception:
            logger.exception("Failed to read top_candidates.csv for canonicalization")
            df_top = None

    df_scored = None
    if scored_path.exists() and scored_path.stat().st_size > 0:
        try:
            df_scored = pd.read_csv(scored_path)
        except Exception:
            logger.exception("Failed to read scored_candidates.csv for canonicalization")
            df_scored = None

    canonical = _coerce_canonical(df_scored, df_top)
    if not canonical.empty and "close" in canonical.columns and canonical["close"].notna().any():
        canonical = canonical[REQUIRED_CAND_COLS]
        if top_path.exists():
            try:
                copyfile(str(top_path), str(latest_path))
            except Exception:
                logger.debug("LATEST_CANDIDATES pre-copy failed", exc_info=True)
        canonical.to_csv(latest_path, index=False)
        logger.info(
            "LATEST_CANDIDATES canonicalized from top_candidates rows=%d",
            len(canonical.index),
        )
    else:
        logger.info("LATEST_CANDIDATES not overwritten (top_candidates lacks canonical fields)")
    return canonical


def _resolve_base_dir(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    cwd = Path.cwd()
    if cwd != PROJECT_ROOT and (cwd / "data").exists():
        return cwd
    return BASE_DIR


def _resolve_labels_bars_path(raw_path: str | Path | None, base_dir: Path) -> Path:
    path = Path(raw_path or DEFAULT_LABELS_BARS_PATH)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _ingest_daily_bars_to_db(base_dir: Path, bars_path: Path, run_date: date) -> None:
    if not db.db_enabled():
        return
    if not bars_path.exists() or bars_path.stat().st_size <= 0:
        LOG.warning(
            "[WARN] DAILY_BARS_DB_SKIPPED reason=missing_file path=%s",
            bars_path,
        )
        return
    try:
        bars_df = pd.read_csv(bars_path)
    except Exception as exc:
        LOG.warning("[WARN] DAILY_BARS_DB_SKIPPED reason=read_error err=%s", exc)
        return
    if bars_df.empty:
        LOG.warning("[WARN] DAILY_BARS_DB_SKIPPED reason=empty_file path=%s", bars_path)
        return
    ok = db.upsert_ml_artifact_frame(
        "daily_bars",
        run_date,
        bars_df,
        source="run_pipeline",
        file_name=bars_path.name,
    )
    if ok:
        LOG.info(
            "[INFO] DAILY_BARS_DB_WRITTEN run_date=%s rows=%d",
            run_date,
            int(bars_df.shape[0]),
        )
    else:
        LOG.warning("[WARN] DAILY_BARS_DB_WRITE_FAILED run_date=%s", run_date)


def _snapshot_label_files(output_dir: Path) -> dict[Path, int]:
    if not output_dir.exists():
        return {}
    snapshot: dict[Path, int] = {}
    for path in output_dir.glob("labels_*.csv"):
        try:
            snapshot[path] = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
    return snapshot


def _detect_new_labels_file(
    output_dir: Path, before: Mapping[Path, int]
) -> Path | None:
    if not output_dir.exists():
        return None
    updated: list[tuple[int, Path]] = []
    for path in output_dir.glob("labels_*.csv"):
        try:
            current_mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        previous_mtime = before.get(path)
        if previous_mtime is None or current_mtime > previous_mtime:
            updated.append((current_mtime, path))
    if not updated:
        return None
    updated.sort()
    return updated[-1][1]


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    except FileNotFoundError:
        return 0
    except Exception:
        logger.exception("LATEST_SYNC_COUNT_ERROR path=%s", path)
        return 0


def _count_csv_lines(path: Path | None) -> int:
    """Return the number of data rows in a CSV, tolerating ``None`` or missing files."""

    if path is None:
        return 0
    return _count_csv_rows(path)


def _sync_top_candidates_to_latest(base_dir: Path | None = None) -> None:
    if db.db_enabled():
        logger.info("LATEST_SYNC skipped: DB is source of truth.")
        return
    base = _resolve_base_dir(base_dir)
    data_dir = base / "data"
    top = data_dir / "top_candidates.csv"
    latest = data_dir / "latest_candidates.csv"
    try:
        canonical = write_latest_candidates_canonical(base)
        if not canonical.empty and canonical["close"].notna().any():
            logger.info("LATEST_SYNC source=top_candidates rows=%s", len(canonical.index))
        else:
            rows_latest = _count_csv_rows(latest)
            logger.info("LATEST_SYNC source=fallback_or_screener rows=%s", rows_latest)
    except Exception:
        logger.exception("LATEST_SYNC_ERROR path_top=%s path_latest=%s", top, latest)


def refresh_latest_candidates(base_dir: Path | None = None, run_date: date | None = None) -> pd.DataFrame:
    if not db.db_enabled():
        logger.error("LATEST_CANDIDATES refresh skipped: DB required.")
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    target_run_date = run_date or db.fetch_latest_run_date("screener_candidates")
    if target_run_date is None:
        logger.warning("LATEST_CANDIDATES DB empty; no candidates to refresh.")
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    frame, _ = get_latest_screener_candidates(target_run_date)
    if frame.empty:
        logger.warning("LATEST_CANDIDATES DB empty; no candidates to refresh.")
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    logger.info("LATEST_CANDIDATES refreshed from DB rows=%d", len(frame.index))
    return frame


def _load_latest_candidates_frame(base_dir: Path | None = None) -> pd.DataFrame:
    base = _resolve_base_dir(base_dir)
    if db.db_enabled():
        frame = refresh_latest_candidates(base)
        if frame.empty:
            LOG.warning("[WARN] CANDIDATES_LOAD_SKIPPED reason=db_empty")
        return frame
    candidates_path = base / "data" / "latest_candidates.csv"
    if not candidates_path.exists() or candidates_path.stat().st_size <= 0:
        LOG.warning(
            "[WARN] CANDIDATES_LOAD_SKIPPED reason=missing_file path=%s",
            candidates_path,
        )
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    try:
        return pd.read_csv(candidates_path)
    except Exception:
        LOG.exception("CANDIDATES_READ_FAILED path=%s", candidates_path)
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))


def _maybe_fallback(base_dir: Path | None = None) -> int:
    LOG.warning("FALLBACK disabled: DB is source of truth.")
    frame = refresh_latest_candidates(base_dir)
    return int(len(frame.index))


def run_cmd(cmd: Sequence[str], name: str) -> int:
    try:
        subprocess.check_call(list(cmd), cwd=PROJECT_ROOT)
        return 0
    except subprocess.CalledProcessError as exc:  # pragma: no cover - legacy shim
        return exc.returncode


def _should_enrich_candidates(args: argparse.Namespace, steps: Sequence[str]) -> bool:
    return bool(getattr(args, "enrich_candidates_with_ranker", False))


def _find_latest_predictions_path(base_dir: Path) -> Path | None:
    predictions_dir = base_dir / "data" / "predictions"
    if not predictions_dir.exists():
        return None
    try:
        candidates = sorted(predictions_dir.glob("predictions_*.csv"))
    except Exception:
        LOG.exception("PREDICTIONS_GLOB_FAILED dir=%s", predictions_dir)
        return None
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except Exception:
        LOG.exception("PREDICTIONS_STAT_FAILED dir=%s", predictions_dir)
        return None


def _normalize_predictions_frame(
    df: pd.DataFrame, score_column: str = DEFAULT_RANKER_SCORE_COLUMN
) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns or score_column not in df.columns:
        return pd.DataFrame(columns=["symbol", score_column])
    working = df.copy()
    if "timestamp" in working.columns:
        working["__pred_ts__"] = pd.to_datetime(working["timestamp"], errors="coerce")
        working.sort_values(by="__pred_ts__", inplace=True)
    working.drop(columns=["__pred_ts__"], errors="ignore", inplace=True)
    working = working.drop_duplicates(subset=["symbol"], keep="last")
    return working[["symbol", score_column]]


def _prepare_predictions_frame(
    predictions_path: Path, score_column: str = DEFAULT_RANKER_SCORE_COLUMN
) -> pd.DataFrame:
    try:
        df = pd.read_csv(predictions_path)
    except Exception:
        LOG.exception("PREDICTIONS_READ_FAILED path=%s", predictions_path)
        return pd.DataFrame(columns=["symbol", score_column])
    return _normalize_predictions_frame(df, score_column)


def _load_latest_predictions_frame(
    base_dir: Path, score_column: str = DEFAULT_RANKER_SCORE_COLUMN
) -> tuple[pd.DataFrame, str]:
    if db.db_enabled():
        predictions = db.load_ml_artifact_csv("predictions")
        if predictions.empty:
            return pd.DataFrame(columns=["symbol", score_column]), "db:missing"
        return _normalize_predictions_frame(predictions, score_column), "db"
    predictions_path = _find_latest_predictions_path(base_dir)
    if predictions_path is None:
        return pd.DataFrame(columns=["symbol", score_column]), "file:missing"
    return _prepare_predictions_frame(predictions_path, score_column), str(predictions_path)


def enrich_candidates_with_predictions(
    base_dir: Path | None = None,
    *,
    score_column: str = DEFAULT_RANKER_SCORE_COLUMN,
    target_column: str = DEFAULT_RANKER_TARGET_COLUMN,
) -> Tuple[int, int] | None:
    if not _should_write_candidate_csvs():
        LOG.info("[INFO] CANDIDATE_ENRICHMENT_SKIPPED reason=db_enabled")
        return None
    base = _resolve_base_dir(base_dir)
    latest_path = base / "data" / "latest_candidates.csv"
    candidates = _load_latest_candidates_frame(base)
    if candidates.empty:
        LOG.warning("[WARN] CANDIDATE_ENRICHMENT_SKIPPED reason=no_candidates")
        return None
    predictions, predictions_source = _load_latest_predictions_frame(base, score_column)
    if predictions.empty:
        LOG.warning("[WARN] CANDIDATE_ENRICHMENT_SKIPPED reason=predictions_empty")
        return None
    renamed = predictions.rename(columns={score_column: target_column})
    merged = candidates.merge(renamed[["symbol", target_column]], on="symbol", how="left")
    matched = int(merged[target_column].notna().sum()) if target_column in merged.columns else 0
    ordered: list[str] = list(CANONICAL_COLUMNS)
    ordered.extend(col for col in candidates.columns if col not in ordered)
    ordered = [col for col in ordered if col in merged.columns and col != target_column]
    ordered.append(target_column)
    merged = merged[ordered]
    write_csv_atomic(str(latest_path), merged)
    LOG.info(
        "[INFO] CANDIDATES_ENRICHED model_score_column=%s rows=%s matched=%s predictions_source=%s",
        target_column,
        len(merged.index),
        matched,
        predictions_source,
    )


def _enrich_candidates_with_ranker(
    base_dir: Path | None = None,
    *,
    score_column: str = DEFAULT_RANKER_SCORE_COLUMN,
    target_column: str = DEFAULT_RANKER_TARGET_COLUMN,
) -> pd.DataFrame | None:
    base = _resolve_base_dir(base_dir)
    candidates_path = base / "data" / "latest_candidates.csv"
    predictions, predictions_source = _load_latest_predictions_frame(base, score_column)
    if predictions.empty:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_SKIPPED reason=predictions_missing candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
        )
        return

    candidates = _load_latest_candidates_frame(base)
    if candidates.empty:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_SKIPPED reason=candidates_empty candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
        )
        return

    try:
        renamed = predictions.rename(columns={score_column: target_column})
        merged = candidates.merge(renamed[["symbol", target_column]], on="symbol", how="left")
        matched = int(merged[target_column].notna().sum()) if target_column in merged.columns else 0
        ordered: list[str] = list(CANONICAL_COLUMNS)
        ordered.extend(col for col in candidates.columns if col not in ordered)
        ordered = [col for col in ordered if col in merged.columns and col != target_column]
        ordered.append(target_column)
        merged = merged[ordered]
    except Exception:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_FAILED reason=merge_error candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
            exc_info=True,
        )
        return

    try:
        write_csv_atomic(str(candidates_path), merged)
    except Exception:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_FAILED reason=write_error candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
            exc_info=True,
        )
        return

    LOG.info(
        "[INFO] CANDIDATES_ENRICHED model_score_column=%s rows=%s matched=%s",
        target_column,
        len(merged.index),
        matched,
    )
    return merged


def _apply_allocation_weights(
    frame: pd.DataFrame,
    *,
    key: str = "model_score_5d",
    top_k: int = DEFAULT_ALLOC_WEIGHT_TOP_K,
    weight_column: str = "alloc_weight",
) -> pd.DataFrame:
    """
    Append per-candidate allocation weights that sum to 1 across the top N rows.

    Existing weight columns are dropped to avoid stale values. Returns the updated
    frame regardless of whether weighting was possible.
    """

    working = frame.copy()
    working = working.drop(columns=[weight_column], errors="ignore")

    if key not in working.columns:
        LOG.warning(
            "[WARN] CANDIDATES_WEIGHTED_SKIPPED reason=missing_key key=%s", key
        )
        return working

    scores = pd.to_numeric(working[key], errors="coerce")
    working[key] = scores

    if working.empty:
        LOG.warning("[WARN] CANDIDATES_WEIGHTED_SKIPPED reason=empty_frame")
        return working

    top_scores = scores.head(max(0, int(top_k))).clip(lower=0.0)
    total = float(top_scores.sum())
    if not math.isfinite(total) or total <= 0:
        LOG.warning(
            "[WARN] CANDIDATES_WEIGHTED_SKIPPED reason=zero_total key=%s top_k=%s",
            key,
            top_k,
        )
        return working

    weights = pd.Series(0.0, index=working.index, dtype="float64")
    weights.iloc[: len(top_scores)] = (top_scores / total).fillna(0.0)
    working[weight_column] = weights
    LOG.info(
        "[INFO] CANDIDATES_WEIGHTED key=%s top_k=%s sum_weights=%.4f",
        key,
        top_k,
        float(weights.sum()),
    )
    return working


def _rerank_latest_candidates(
    base_dir: Path,
    frame: pd.DataFrame | None = None,
    *,
    primary: str = "model_score_5d",
    secondary: str = "score",
) -> bool:
    """
    Load data/latest_candidates.csv, verify primary column exists,
    sort descending by primary then secondary, write back safely.
    Returns True if reranked, False if skipped.
    """

    if not _should_write_candidate_csvs():
        LOG.info("[INFO] CANDIDATES_RERANK_SKIPPED reason=db_enabled")
        return False

    candidates_path = Path(base_dir) / "data" / "latest_candidates.csv"
    try:
        if frame is None:
            if db.db_enabled():
                frame = refresh_latest_candidates(base_dir)
            else:
                frame = pd.read_csv(candidates_path)
        if frame.empty:
            LOG.warning(
                "[WARN] CANDIDATES_RERANK_SKIPPED reason=empty_frame candidates_path=%s",
                candidates_path,
            )
            return False
        if primary not in frame.columns:
            LOG.warning(
                "[WARN] CANDIDATES_RERANK_SKIPPED reason=missing_model_score candidates_path=%s",
                candidates_path,
            )
            return False

        primary_numeric = pd.to_numeric(frame[primary], errors="coerce")
        nan_scores = int(primary_numeric.isna().sum())
        frame[primary] = primary_numeric

        try:
            sorted_frame = frame.sort_values(
                by=[primary, secondary],
                ascending=[False, False],
                na_position="last",
                kind="mergesort",
            )
            weighted_frame = _apply_allocation_weights(
                sorted_frame,
                key=primary,
                top_k=DEFAULT_ALLOC_WEIGHT_TOP_K,
            )
            write_csv_atomic(str(candidates_path), weighted_frame)
        except Exception as exc:  # pragma: no cover - defensive sort/write
            LOG.warning(
                "[WARN] CANDIDATES_RERANK_FAILED error=%s candidates_path=%s",
                exc,
                candidates_path,
                exc_info=True,
            )
            return False

        LOG.info(
            "[INFO] CANDIDATES_RERANKED primary=%s secondary=%s nan_scores=%s rows=%s",
            primary,
            secondary,
            nan_scores,
            len(sorted_frame.index),
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive read
        LOG.warning(
            "[WARN] CANDIDATES_RERANK_FAILED error=%s candidates_path=%s",
            exc,
            candidates_path,
            exc_info=True,
        )
        return False
    finally:
        try:
            _write_nightly_ml_status(base_dir)
        except Exception:
            LOG.exception("NIGHTLY_ML_STATUS_SNAPSHOT_FAILED base_dir=%s", base_dir)


def emit_metric(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - legacy hook
    return None


def write_metrics_summary(**kwargs: Any) -> None:  # pragma: no cover - legacy hook
    return None
REQUIRED_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)


def configure_logging() -> None:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - pipeline - %(message)s")

    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    LOG.addHandler(file_handler)
    LOG.propagate = False


def _split_args(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        return shlex.split(raw)
    except ValueError as exc:
        LOG.error("PIPELINE_ARG_PARSE_FAILED raw=%s error=%s", raw, exc)
        return []


def _merge_split_args(raw: str, split: Optional[Sequence[str]]) -> list[str]:
    merged: list[str] = []
    if raw:
        merged.extend(_split_args(raw))
    if split:
        merged.extend(token for token in split if token != "--")
    return merged


def _strip_labels_args(tokens: Sequence[str]) -> list[str]:
    """Remove any ``--labels-bars-path`` arguments from a token list."""

    cleaned: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--labels-bars-path":
            skip_next = True
            continue
        if token.startswith("--labels-bars-path="):
            continue
        cleaned.append(token)
    return cleaned


def _extract_split_tokens(
    argv: list[str], option_strings: set[str]
) -> tuple[list[str], dict[str, list[str]]]:
    filtered: list[str] = []
    collected: dict[str, list[str]] = {
        dest: [] for dest in _SPLIT_FLAG_MAP.values()
    }
    current_flag: Optional[str] = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if current_flag:
            if token == "--":
                current_flag = None
                index += 1
                continue
            if token in _SPLIT_FLAG_MAP:
                current_flag = token
                index += 1
                continue
            if token.startswith("--") and token in option_strings:
                current_flag = None
                continue
            collected[_SPLIT_FLAG_MAP[current_flag]].append(token)
            index += 1
            continue
        if token in _SPLIT_FLAG_MAP:
            current_flag = token
            index += 1
            continue
        filtered.append(token)
        index += 1
    return filtered, collected


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JBRAVO daily pipeline")
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated list of steps to run (default: screener,backtest,metrics,ranker_eval). "
            "Include 'labels' to generate label CSVs."
        ),
    )
    parser.add_argument(
        "--reload-web",
        choices=("true", "false"),
        default="true",
        help="Reload the hosted web application on completion",
    )
    parser.add_argument(
        "--screener-args",
        default=os.getenv("JBR_SCREENER_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.screener",
    )
    parser.add_argument(
        "--screener-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.screener provided as separate tokens",
    )
    parser.add_argument(
        "--backtest-args",
        default=os.getenv("JBR_BACKTEST_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.backtest",
    )
    parser.add_argument(
        "--backtest-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.backtest provided as separate tokens",
    )
    parser.add_argument(
        "--backtest-quick",
        choices=("true", "false"),
        default=None,
        help="Enable quick backtest mode by forwarding --quick to scripts.backtest",
    )
    parser.add_argument(
        "--metrics-args",
        default=os.getenv("JBR_METRICS_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.metrics",
    )
    parser.add_argument(
        "--metrics-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.metrics provided as separate tokens",
    )
    parser.add_argument(
        "--labels-bars-path",
        default=None,
        help=(
            "Path to a daily bars CSV used by the optional labels step "
            "(default: data/daily_bars.csv)"
        ),
    )
    parser.add_argument(
        "--ranker-eval-args",
        default=os.getenv("JBR_RANKER_EVAL_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_eval",
    )
    parser.add_argument(
        "--ranker-eval-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_eval provided as separate tokens",
    )
    parser.add_argument(
        "--enrich-candidates-with-ranker",
        action="store_true",
        help=(
            "Append model scores from the latest predictions file onto latest_candidates.csv. "
            "Runs automatically when ranker_eval is in --steps."
        ),
    )
    parser.add_argument(
        "--export-daily-bars-path",
        default=str(DEFAULT_LABELS_BARS_PATH),
        help="Optional path to write fetched daily bars as CSV (default: data/daily_bars.csv)",
    )
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    option_strings = set(parser._option_string_actions.keys())
    filtered_args, collected = _extract_split_tokens(raw_args, option_strings)
    namespace = parser.parse_args(filtered_args)
    for dest in _SPLIT_FLAG_MAP.values():
        tokens = collected.get(dest)
        setattr(namespace, dest, tokens if tokens else None)
    return namespace


def determine_steps(raw: Optional[str]) -> list[str]:
    default = "screener,backtest,metrics,ranker_eval"
    target = raw or os.environ.get("PIPE_STEPS", default)
    steps = [part.strip().lower() for part in target.split(",") if part.strip()]
    allowed = {"screener", "backtest", "metrics", "labels", "ranker_eval"}
    normalized: list[str] = []
    for step in steps or default.split(","):
        if step in allowed:
            normalized.append(step)
    return normalized


def run_step(
    name: str,
    cmd: Sequence[str],
    *,
    timeout: Optional[float] = None,
    tee_to_stdout: bool = False,
    tee_to_logger: bool = False,
) -> tuple[int, float]:
    started = time.time()
    LOG.info("[INFO] START %s cmd=%s", name, shlex.join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("APCA_DATA_API_BASE_URL", "https://data.alpaca.markets")
    env.setdefault("ALPACA_DATA_FEED", "iex")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"step.{name}.out"
    heartbeat_interval = 15.0
    last_heartbeat = started
    rc = -1
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now(timezone.utc).isoformat()}] START {name}\n")
        log_file.flush()
        try:
            LOG.info(
                "[INFO] CHILD_ENV_KEYS has_database_url=%s keys_count=%d",
                bool(env.get("DATABASE_URL")),
                len(env),
            )
            if tee_to_stdout or tee_to_logger:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    env=env,
                    text=True,
                    bufsize=1,
                )
            else:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    env=env,
                    text=True,
                )
        except Exception as exc:
            LOG.error("Failed to launch step %s: %s", name, exc)
            return 1, 0.0
        stream_thread = None
        if tee_to_stdout or tee_to_logger:
            def _stream_output() -> None:
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    if tee_to_stdout:
                        print(line, end="")
                    if tee_to_logger:
                        LOG.info("[STEP:%s] %s", name, line.rstrip("\n"))

            stream_thread = threading.Thread(target=_stream_output, daemon=True)
            stream_thread.start()

        while True:
            try:
                rc = proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass

            now = time.time()
            if timeout and (now - started) > timeout:
                proc.kill()
                rc = proc.wait(timeout=5)
                LOG.error("STEP_TIMEOUT name=%s timeout=%s rc=%s", name, timeout, rc)
                break
            if now - last_heartbeat >= heartbeat_interval:
                elapsed = now - started
                LOG.info("[INFO] %s still running secs=%.1f log=%s", name, elapsed, log_path)
                last_heartbeat = now
        elapsed = time.time() - started
        if stream_thread is not None:
            stream_thread.join(timeout=5)
        log_file.write(f"[{datetime.now(timezone.utc).isoformat()}] END {name} rc={rc} secs={elapsed:.1f}\n")
    LOG.info("END %s rc=%s secs=%.1f log=%s", name, rc, elapsed, log_path)
    return rc, elapsed


def _run_step_metrics(cmd: Sequence[str], *, timeout: Optional[float] = None) -> tuple[int, float]:
    return run_step(
        "metrics",
        cmd,
        timeout=timeout,
        tee_to_stdout=True,
        tee_to_logger=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse guard
        LOG.warning("PIPELINE_JSON_READ_FAILED path=%s error=%s", path, exc)
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        LOG.warning("PIPELINE_COUNT_FAILED path=%s error=%s", path, exc)
        return 0
    return int(len(df.index))


def _derive_universe_prefix_counts(base_dir: Path) -> Dict[str, int]:
    """
    Fallback for metrics['universe_prefix_counts'].

    Derives prefix counts from DB screener_candidates when available.

    "Prefix" = first character of the 'symbol' column, upper-cased.

    Returns {} if no usable data is found or if any error occurs.
    """
    logger = logging.getLogger(__name__)

    if not db.db_enabled():
        logger.warning("prefix_counts: DB disabled")
        return {}

    try:
        df, _ = db.fetch_latest_screener_candidates()
    except Exception as exc:
        logger.warning("prefix_counts: DB read failed (%s)", exc)
        return {}

    if df.empty or "symbol" not in df.columns:
        return {}

    counts: Counter[str] = Counter()
    for sym in df["symbol"].astype(str).tolist():
        sym = sym.strip()
        if not sym:
            continue
        counts[sym[0].upper()] += 1

    if counts:
        return {k: int(counts[k]) for k in sorted(counts)}
    return {}


def write_complete_screener_metrics(base_dir: Path) -> dict[str, Any]:
    """Ensure ``screener_metrics.json`` contains integer KPIs even on fallback nights."""

    base_dir = Path(base_dir)
    logger = LOG
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = data_dir / "screener_metrics.json"
    existing_metrics = _read_json(metrics_path)
    existing_last_run = None
    if isinstance(existing_metrics, Mapping):
        existing_last_run = existing_metrics.get("last_run_utc")
    fallback_hint: dict[str, Any] = {
        "symbols_in": None,
        "symbols_with_bars": None,
        "symbols_with_bars_raw": None,
        "bars_rows_total": None,
        "rows": 0,
        "latest_source": "fallback",
    }

    log_path = base_dir / "logs" / "pipeline.log"
    if log_path.exists():
        try:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-50000:]
        except Exception:
            tail = ""
        if tail:
            for line in reversed(tail.splitlines()):
                if "PIPELINE_SUMMARY" not in line:
                    continue
                match = _SUMMARY_RE.search(line)
                if match:
                    fallback_hint["symbols_in"] = int(match.group("symbols_in"))
                    fallback_hint["symbols_with_bars"] = int(match.group("symbols_with_bars"))
                    fallback_hint["symbols_with_bars_raw"] = fallback_hint["symbols_with_bars"]
                    fallback_hint["rows"] = int(match.group("rows"))
                    bars_total = match.group("bars_rows_total")
                    fallback_hint["bars_rows_total"] = int(bars_total) if bars_total else None
                break

    if db.db_enabled():
        fallback_hint["latest_source"] = "db"
        db_rows, _ = db.fetch_latest_screener_candidate_count()
        if db_rows:
            fallback_hint["rows"] = int(db_rows)
            if fallback_hint["symbols_in"] is None:
                fallback_hint["symbols_in"] = int(db_rows)
    else:
        latest_candidates = data_dir / "latest_candidates.csv"
        top_candidates = data_dir / "top_candidates.csv"
        top_rows = _count_csv_lines(top_candidates)
        latest_rows = _count_csv_lines(latest_candidates)
        if top_rows:
            fallback_hint["rows"] = top_rows
        elif latest_rows and not fallback_hint.get("rows"):
            fallback_hint["rows"] = latest_rows

        scored_candidates = data_dir / "scored_candidates.csv"
        if fallback_hint["symbols_in"] is None:
            rows = _count_csv_lines(scored_candidates)
            if rows or scored_candidates.exists():
                fallback_hint["symbols_in"] = rows

    if fallback_hint["symbols_with_bars"] is None and fallback_hint["symbols_in"] is not None:
        fallback_hint["symbols_with_bars"] = fallback_hint["symbols_in"]
    if fallback_hint["symbols_with_bars_raw"] is None:
        fallback_hint["symbols_with_bars_raw"] = fallback_hint["symbols_with_bars"]

    if fallback_hint["bars_rows_total"] is None:
        fallback_hint["bars_rows_total"] = fallback_hint["rows"]

    for key in ("symbols_in", "symbols_with_bars", "symbols_with_bars_raw", "bars_rows_total"):
        if fallback_hint[key] is None:
            fallback_hint[key] = 0

    fallback_hint["symbols_with_bars"] = int(fallback_hint.get("symbols_with_bars_raw", 0))

    metrics = compose_metrics_from_artifacts(
        base_dir,
        symbols_in=fallback_hint.get("symbols_in"),
        fallback_symbols_with_bars=fallback_hint.get("symbols_with_bars"),
        fallback_bars_rows_total=fallback_hint.get("bars_rows_total"),
        latest_source=fallback_hint.get("latest_source"),
    )
    if existing_last_run:
        metrics["last_run_utc"] = existing_last_run
    try:
        metrics = write_universe_prefix_counts(base_dir, metrics)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger = logging.getLogger("run_pipeline")
        logger.warning("Unable to compute universe_prefix_counts: %s", exc)
    # Ensure universe_prefix_counts is populated when possible.
    upc = metrics.get("universe_prefix_counts")
    needs_prefix_counts = not isinstance(upc, dict) or not upc  # None, {}, or missing

    if needs_prefix_counts:
        derived = _derive_universe_prefix_counts(base_dir)
        if derived:
            metrics["universe_prefix_counts"] = derived

    metrics = ensure_canonical_metrics(metrics)
    write_screener_metrics_json(metrics_path, metrics)
    logger.info(
        "Wrote screener_metrics.json: symbols_in=%s symbols_with_bars=%s rows=%s bars_rows_total=%s",
        metrics.get("symbols_in"),
        metrics.get("symbols_with_bars"),
        metrics.get("rows"),
        metrics.get("bars_rows_total"),
    )
    return metrics


def _annotate_screener_metrics(
    base_dir: Path,
    *,
    rc: int,
    steps: Sequence[str],
    step_rcs: Mapping[str, int],
    stage_times: Mapping[str, float],
    latest_source: str | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics_path = Path(base_dir) / "data" / "screener_metrics.json"
    payload: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping):
                payload.update(existing)
        except Exception:
            payload = {}
    payload["status"] = "ok" if rc == 0 else "error"
    payload["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    payload["stage_times"] = {k: float(v) for k, v in stage_times.items()}
    payload["step_rcs"] = {step: int(step_rcs.get(step, 0)) for step in steps}
    if latest_source:
        payload["latest_source"] = latest_source
    if rc == 0:
        payload.pop("error", None)
    else:
        error_block: dict[str, Any] = {
            "message": "pipeline_failed",
            "rc": int(rc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(error, Mapping):
            for key, value in error.items():
                if value is None:
                    continue
                try:
                    json.dumps({key: value})
                    error_block[key] = value
                except TypeError:
                    error_block[key] = str(value)
        payload["error"] = error_block
    payload = ensure_canonical_metrics(payload)
    write_screener_metrics_json(metrics_path, payload)
    return payload


def _ensure_latest_headers() -> None:
    if db.db_enabled():
        return
    if LATEST_CANDIDATES.exists() and LATEST_CANDIDATES.stat().st_size > 0:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    write_csv_atomic(str(LATEST_CANDIDATES), empty)


def _load_top_candidates() -> pd.DataFrame:
    if not TOP_CANDIDATES.exists() or TOP_CANDIDATES.stat().st_size == 0:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    try:
        return pd.read_csv(TOP_CANDIDATES)
    except Exception as exc:  # pragma: no cover - defensive read guard
        LOG.warning("PIPELINE_TOP_READ_FAILED path=%s error=%s", TOP_CANDIDATES, exc)
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))


def _write_latest_from_frame(frame: pd.DataFrame, *, source: str = "screener") -> int:
    normalized = normalize_candidate_df(frame)
    if "source" in normalized.columns:
        source_series = normalized["source"].astype("string").fillna("")
    else:
        source_series = pd.Series("", index=normalized.index, dtype="string")
    normalized["source"] = source_series
    normalized.loc[normalized["source"].str.strip() == "", "source"] = source
    normalized = normalized[list(CANONICAL_COLUMNS)]
    write_csv_atomic(str(LATEST_CANDIDATES), normalized)
    return int(len(normalized.index))


def _inject_export_daily_bars_arg(args: list[str], export_path: str | Path | None) -> list[str]:
    if not export_path:
        return args

    flag = "--export-daily-bars-path"
    for token in args:
        if token == flag:
            return args
        if token.startswith(f"{flag}="):
            return args
    args = list(args)
    args.extend([flag, str(export_path)])
    return args


def _previous_trading_day(value: date) -> date:
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def _resolve_pipeline_run_date(now: datetime | None = None) -> date:
    tz = zoneinfo.ZoneInfo("America/New_York")
    local_now = now.astimezone(tz) if now else datetime.now(tz)
    close_time = local_now.replace(hour=16, minute=0, second=0, microsecond=0)
    local_date = local_now.date()
    if local_now <= close_time:
        local_date -= timedelta(days=1)
    return _previous_trading_day(local_date)


def _inject_run_date_arg(args: list[str], run_date: date | None) -> list[str]:
    if run_date is None:
        return args
    cleaned: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--run-date":
            skip_next = True
            continue
        if token.startswith("--run-date="):
            continue
        cleaned.append(token)
    cleaned.extend(["--run-date", run_date.isoformat()])
    return cleaned


def _write_refresh_metrics(metrics_path: Path) -> None:
    payload: dict[str, Any] = {
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "http": {"429": 0, "404": 0, "empty_pages": 0},
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "universe_prefix_counts": {},
        "auth_missing": [],
        "timings": {},
    }
    try:
        payload.update(_record_health("refresh"))
    except Exception:  # pragma: no cover - defensive guard
        LOG.debug("_record_health refresh failed", exc_info=True)
    payload = ensure_canonical_metrics(payload)
    write_screener_metrics_json(metrics_path, payload)


def ensure_candidates(min_rows: int = 1) -> int:
    if not db.db_enabled():
        LOG.error("[ERROR] CANDIDATES_DB_REQUIRED")
        return 0
    count, _ = db.fetch_latest_screener_candidate_count()
    count = int(count or 0)
    if count < min_rows:
        LOG.warning(
            "[WARN] CANDIDATES_DB_INSUFFICIENT count=%s required=%s",
            count,
            min_rows,
        )
    return count


def _extract_timing(metrics: Mapping[str, Any], key: str) -> float:
    if not metrics:
        return 0.0
    value: Any = metrics.get(key)
    timings = metrics.get("timings") if isinstance(metrics.get("timings"), Mapping) else {}
    if value in (None, "") and isinstance(timings, Mapping):
        value = timings.get(key)
        if value in (None, ""):
            alias_candidates: list[str] = []
            if "_" in key:
                parts = key.split("_", 1)
                alias_candidates.append(f"{parts[0]}s_{parts[1]}")
            if key.endswith("s"):
                alias_candidates.append(key[:-1])
            else:
                alias_candidates.append(f"{key}s")
            for alias in alias_candidates:
                if alias in timings:
                    value = timings.get(alias)
                    break
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _reload_dashboard(enabled: bool) -> None:
    if not enabled:
        return
    path = DEFAULT_WSGI_PATH
    try:
        path.touch()
        LOG.info("[INFO] DASH RELOAD method=touch rc=0 path=%s", path)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.warning(
            "[WARN] DASH RELOAD failed method=touch path=%s detail=%s",
            path,
            exc,
        )


def ingest_artifacts_to_db(run_date: date) -> None:
    """Ingest CSV artifacts into the database for durability."""

    def _log_ok(table: str, rows: int) -> None:
        logger.info("[INFO] DB_INGEST_OK table=%s rows=%s", table, rows)

    def _log_fail(table: str, err: Exception | str) -> None:
        logger.warning("[WARN] DB_INGEST_FAILED table=%s err=%s", table, err)

    if db.db_enabled():
        logger.info("CSV ingest disabled: DB is source of truth.")
        return

    if not db.db_enabled():
        _log_fail("all", "db_disabled")
        return

    run_date_value: str
    run_date_value = run_date.isoformat() if isinstance(run_date, date) else str(run_date)
    score_breakdown_raw: list[Any] = []

    try:
        candidates_df = pd.read_csv(LATEST_CANDIDATES)
    except FileNotFoundError:
        candidates_df = pd.DataFrame()
    except Exception as exc:
        _log_fail("screener_candidates", exc)
        candidates_df = pd.DataFrame()

    if not candidates_df.empty:
        for _, row in candidates_df.iterrows():
            record = row.to_dict() if isinstance(row, Mapping) else dict(row)
            symbol = _coerce_symbol(record.get("symbol"))
            score_breakdown_raw_value = record.get("score_breakdown")
            normalized_score_breakdown = db.normalize_score_breakdown(score_breakdown_raw_value, symbol=symbol)
            if normalized_score_breakdown is None:
                try:
                    is_empty_score = score_breakdown_raw_value in (None, "") or pd.isna(score_breakdown_raw_value)
                except Exception:
                    is_empty_score = score_breakdown_raw_value in (None, "")
                if not is_empty_score:
                    score_breakdown_raw.append(score_breakdown_raw_value)
        try:
            run_ts_utc = summary_payload.get("run_ts_utc") if summary_payload else None
            if run_ts_utc is None:
                run_ts_utc = db.get_latest_pipeline_health_run_ts()
            db.insert_screener_candidates(
                run_date_value,
                candidates_df,
                run_ts_utc=run_ts_utc,
            )
            _log_ok("screener_candidates", int(candidates_df.shape[0]))
        except Exception as exc:
            _log_fail("screener_candidates", exc)

    try:
        backtest_df = pd.read_csv(BACKTEST_RESULTS)
    except FileNotFoundError:
        backtest_df = pd.DataFrame()
    except Exception as exc:
        _log_fail("backtest_results", exc)
        backtest_df = pd.DataFrame()

    if not backtest_df.empty:
        try:
            ok = db.insert_backtest_results(run_date_value, backtest_df)
            if ok:
                _log_ok("backtest_results", int(backtest_df.shape[0]))
            else:
                _log_fail("backtest_results", "write_failed")
        except Exception as exc:
            _log_fail("backtest_results", exc)

    try:
        metrics_df = pd.read_csv(METRICS_SUMMARY)
    except FileNotFoundError:
        metrics_df = pd.DataFrame()
    except Exception as exc:
        _log_fail("metrics_daily", exc)
        metrics_df = pd.DataFrame()

    if not metrics_df.empty:
        latest_row = metrics_df.tail(1).to_dict(orient="records")[0]
        payload_metrics = {
            "run_date": run_date_value,
            "total_trades": latest_row.get("total_trades"),
            "win_rate": latest_row.get("win_rate"),
            "net_pnl": latest_row.get("net_pnl"),
            "expectancy": latest_row.get("expectancy"),
            "profit_factor": latest_row.get("profit_factor"),
            "max_drawdown": latest_row.get("max_drawdown"),
            "sharpe": latest_row.get("sharpe"),
            "sortino": latest_row.get("sortino"),
        }
        try:
            db.upsert_metrics_daily(run_date_value, payload_metrics)
            _log_ok("metrics_daily", 1)
        except Exception as exc:
            _log_fail("metrics_daily", exc)

    summary_payload = _PIPELINE_SUMMARY_FOR_DB or {}
    if score_breakdown_raw:
        summary_payload = dict(summary_payload)
        summary_payload["score_breakdown_raw"] = score_breakdown_raw

    def _dump_summary(payload: Mapping[str, Any] | None) -> str:
        try:
            return json.dumps(payload or {})
        except Exception:
            try:
                return json.dumps({"raw": str(payload)})
            except Exception:
                return "{}"

    payload_pipeline = {
        "run_date": run_date_value,
        "started_at": _PIPELINE_STARTED_AT,
        "ended_at": _PIPELINE_ENDED_AT or datetime.now(timezone.utc),
        "rc": int(_PIPELINE_RC or 0),
        "summary": _dump_summary(summary_payload),
    }
    try:
        db.upsert_pipeline_run(
            payload_pipeline["run_date"],
            payload_pipeline["started_at"],
            payload_pipeline["ended_at"],
            payload_pipeline["rc"],
            summary_payload,
        )
        _log_ok("pipeline_runs", 1)
    except Exception as exc:
        _log_fail("pipeline_runs", exc)


def main(argv: Optional[Iterable[str]] = None) -> int:
    _refresh_logger()
    rotate_if_needed("logs/pipeline.log", max_bytes=10_000_000, max_age_days=14, keep=14)
    loaded_files, missing_keys = load_env(REQUIRED_ENV_KEYS)
    configure_logging()
    files_repr = f"[{', '.join(loaded_files)}]" if loaded_files else "[]"
    LOG.info("[INFO] ENV_LOADED files=%s", files_repr)
    if missing_keys:
        LOG.error("[ERROR] ENV_MISSING_KEYS=%s", f"[{', '.join(missing_keys)}]")
        raise SystemExit(2)
    args = parse_args(argv)
    steps = tuple(determine_steps(args.steps))
    if not db.db_enabled():
        LOG.error("[ERROR] DB_REQUIRED: DATABASE_URL/DB_* not configured.")
        write_error_report(step="pipeline", detail="db_required")
        raise SystemExit(2)
    if not db_migrate.ensure_schema():
        LOG.error("[ERROR] DB_MIGRATE_FAILED")
        raise SystemExit(2)
    if "screener" not in steps:
        LOG.error("[ERROR] SCREENER_STEP_REQUIRED steps=%s", ",".join(steps))
        write_error_report(step="screener", detail="step_missing")
        try:
            send_alert(
                "JBRAVO pipeline failed: screener missing",
                {"steps": ",".join(steps), "run_utc": datetime.now(timezone.utc).isoformat()},
            )
        except Exception:
            LOG.debug("ALERT_SCREEENER_MISSING_FAILED", exc_info=True)
        raise SystemExit(2)
    LOG.info("[INFO] PIPELINE_START steps=%s", ",".join(steps))
    pipeline_run_date = _resolve_pipeline_run_date()
    LOG.info(
        "[INFO] PIPELINE_RUN_DATE run_date=%s tz=America/New_York",
        pipeline_run_date.isoformat(),
    )

    _ensure_latest_headers()
    base_dir = _resolve_base_dir()

    extras = {
        "screener": _strip_labels_args(
            _merge_split_args(args.screener_args, args.screener_args_split)
        ),
        "backtest": _merge_split_args(args.backtest_args, args.backtest_args_split),
        "metrics": _merge_split_args(args.metrics_args, args.metrics_args_split),
        "ranker_eval": _merge_split_args(
            getattr(args, "ranker_eval_args", ""), getattr(args, "ranker_eval_args_split", None)
        ),
    }

    if getattr(args, "backtest_quick", None) == "true":
        extras["backtest"].append("--quick")

    export_bars_path = args.export_daily_bars_path
    if "screener" in steps:
        extras["screener"] = _inject_export_daily_bars_arg(
            extras["screener"], export_bars_path
        )
    elif export_bars_path:
        LOG.warning(
            "[WARN] DAILY_BARS_EXPORT_FAILED path=%s error=%s",
            export_bars_path,
            "screener_step_disabled",
        )

    extras["screener"] = _inject_run_date_arg(extras["screener"], pipeline_run_date)
    extras["backtest"] = _inject_run_date_arg(extras["backtest"], pipeline_run_date)
    extras["metrics"] = _inject_run_date_arg(extras["metrics"], pipeline_run_date)

    step_rcs: dict[str, int] = {step: 0 for step in steps}

    LOG.info(
        "[INFO] PIPELINE_ARGS screener=%s backtest=%s metrics=%s ranker_eval=%s",
        extras["screener"],
        extras["backtest"],
        extras["metrics"],
        extras["ranker_eval"],
    )

    started = time.time()
    started_dt = datetime.fromtimestamp(started, timezone.utc)
    global _PIPELINE_STARTED_AT
    _PIPELINE_STARTED_AT = started_dt
    metrics: dict[str, Any] = {}
    symbols_in = 0
    symbols_with_bars = 0
    rows = 0
    stage_times: dict[str, float] = {}
    rc = 0
    current_step: str | None = None

    metrics_rows: int | None = None
    latest_source: str | None = "unknown"
    error_info: dict[str, Any] | None = None
    degraded = False
    zero_candidates_alerted = False
    try:
        if "screener" in steps:
            current_step = "screener"
            cmd = [sys.executable, "-m", "scripts.screener", "--mode", "screener"]
            if extras["screener"]:
                cmd.extend(extras["screener"])
            rc_scr, secs = run_step("screener", cmd, timeout=60 * 20)
            stage_times["screener"] = secs
            step_rcs["screener"] = rc_scr
            if rc_scr:
                rc = rc_scr
                if error_info is None:
                    error_info = {"step": "screener", "rc": int(rc_scr), "message": "step_failed"}
            metrics = _read_json(SCREENER_METRICS_PATH)
            symbols_in = int(metrics.get("symbols_in", 0) or 0)
            symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
            if "rows" in metrics:
                try:
                    metrics_rows = int(metrics.get("rows") or 0)
                except Exception:
                    metrics_rows = 0
            db_rows = 0
            if db.db_enabled():
                db_rows, _ = db.fetch_latest_screener_candidate_count()
            rows = int(db_rows or metrics_rows or 0)
            latest_source = "db" if db.db_enabled() else "screener_metrics"
            LOG.info("[INFO] CANDIDATE_COUNTS rows=%d source=%s", rows, latest_source)
            if (metrics_rows or 0) == 0 and rows == 0 and not zero_candidates_alerted:
                zero_candidates_alerted = True
                LOG.warning("[WARN] ZERO_CANDIDATES symbols_in=%s", symbols_in)
                send_alert(
                    "JBRAVO pipeline: 0 candidates after screener",
                    {
                        "symbols_in": symbols_in,
                        "with_bars": symbols_with_bars,
                        "rows": int(rows or 0),
                        "source": latest_source,
                        "run_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            if db.db_enabled():
                bars_path = _resolve_labels_bars_path(export_bars_path, base_dir)
                _ingest_daily_bars_to_db(base_dir, bars_path, pipeline_run_date)
        else:
            metrics = _read_json(SCREENER_METRICS_PATH)
            symbols_in = int(metrics.get("symbols_in", 0) or 0)
            symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
            if "rows" in metrics:
                try:
                    metrics_rows = int(metrics.get("rows") or 0)
                except Exception:
                    metrics_rows = 0
            rows = ensure_candidates(metrics_rows or 0)
            latest_source = "db"
            LOG.info("[INFO] CANDIDATE_COUNTS rows=%d source=%s", rows, latest_source)

        if "backtest" in steps:
            current_step = "backtest"
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.backtest"]
            if extras["backtest"]:
                cmd.extend(extras["backtest"])
            rc_bt, secs = run_step("backtest", cmd, timeout=60 * 3)
            stage_times["backtest"] = secs
            step_rcs["backtest"] = rc_bt
            if rc_bt and not rc:
                rc = rc_bt
                if error_info is None:
                    error_info = {"step": "backtest", "rc": int(rc_bt), "message": "step_failed"}

        if "metrics" in steps:
            current_step = "metrics"
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.metrics"]
            if extras["metrics"]:
                cmd.extend(extras["metrics"])
            rc_metrics = 0
            secs = 0.0
            try:
                print(">>> [run_pipeline] launching metrics.py subprocess <<<")
                logger.info(">>> [run_pipeline] launching metrics.py subprocess <<<")
                rc_metrics, secs = _run_step_metrics(cmd, timeout=60 * 3)
                print(">>> [run_pipeline] metrics.py completed <<<")
                logger.info(">>> [run_pipeline] metrics.py completed <<<")
            except FileNotFoundError:
                logger.warning("METRICS skipped: dependency missing.")
                rc_metrics, secs = 0, 0.0
            except Exception as e:
                logger.warning("METRICS non-fatal error: %s", e)
                rc_metrics, secs = 0, 0.0
            stage_times["metrics"] = secs
            step_rcs["metrics"] = rc_metrics
            if rc_metrics:
                logger.warning("[WARN] METRICS_STEP rc=%s (continuing)", rc_metrics)
            _sync_top_candidates_to_latest(base_dir)
            try:
                env = os.environ.copy()
                LOG.info(
                    "[INFO] CHILD_ENV_KEYS has_database_url=%s keys_count=%d",
                    bool(env.get("DATABASE_URL")),
                    len(env),
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.trade_performance_refresh",
                        "--lookback-days",
                        "400",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    env=env,
                )
                LOG.info("[INFO] TRADE_PERFORMANCE_CACHE_REFRESHED")
            except Exception:
                LOG.warning("TRADE_PERFORMANCE_CACHE_REFRESH_FAILED", exc_info=True)
        if "labels" in steps:
            current_step = "labels"
            bars_path = _resolve_labels_bars_path(args.labels_bars_path, base_dir)
            if db.db_enabled():
                bars_meta = db.fetch_latest_ml_artifact("daily_bars")
                bars_rows = int(bars_meta.get("rows_count") or 0) if bars_meta else 0
            else:
                bars_rows = _count_csv_lines(bars_path)
            if bars_rows <= 0:
                LOG.warning("[WARN] LABELS_SKIPPED reason=missing_or_empty path=%s", bars_path)
                stage_times["labels"] = 0.0
                step_rcs["labels"] = 0
            else:
                labels_output_dir = base_dir / "data" / "labels"
                labels_before = _snapshot_label_files(labels_output_dir)
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.label_generator",
                    "--bars-path",
                    str(bars_path),
                    "--output-dir",
                    str(labels_output_dir),
                ]
                labels_path: Path | None = None
                labels_rows = 0
                rc_labels = 0
                secs = 0.0
                LOG.info("[INFO] START labels bars_path=%s", bars_path)
                LOG.info("[INFO] LABELS_START bars_path=%s", bars_path)
                try:
                    rc_labels, secs = run_step("labels", cmd, timeout=60 * 5)
                    try:
                        labels_path = _detect_new_labels_file(labels_output_dir, labels_before)
                        labels_rows = _count_csv_lines(labels_path) if labels_path else 0
                    except Exception:
                        LOG.exception(
                            "LABELS_DETECT_FAILED output_dir=%s", labels_output_dir
                        )
                finally:
                    LOG.info(
                        "[INFO] LABELS_END labels_path=%s rows=%s",
                        labels_path or "unknown",
                        labels_rows,
                    )
                stage_times["labels"] = secs
                step_rcs["labels"] = rc_labels
                if rc_labels:
                    LOG.warning("[WARN] LABELS_STEP rc=%s (continuing)", rc_labels)

        if db.db_enabled() and (
            "ranker_eval" in steps or getattr(args, "enrich_candidates_with_ranker", False)
        ):
            missing_artifacts: list[str] = []
            if not db.fetch_latest_ml_artifact("daily_bars"):
                missing_artifacts.append("daily_bars")
            if not db.fetch_latest_ml_artifact("labels"):
                missing_artifacts.append("labels")
            if missing_artifacts:
                LOG.warning(
                    "[WARN] RANKER_PIPELINE_SKIPPED reason=missing_artifacts artifacts=%s",
                    ",".join(missing_artifacts),
                )
            else:
                LOG.info("[INFO] FEATURES_START")
                rc_features, secs = run_step(
                    "features",
                    [sys.executable, "-m", "scripts.feature_generator"],
                    timeout=60 * 5,
                )
                stage_times["features"] = secs
                if rc_features:
                    LOG.warning("[WARN] FEATURES_FAILED rc=%s (continuing)", rc_features)
                else:
                    model_path = _latest_by_glob(
                        base_dir / "data" / "models", "ranker_*.pkl"
                    )
                    if model_path is None:
                        LOG.info("[INFO] RANKER_TRAIN_START reason=model_missing")
                        rc_train, secs = run_step(
                            "ranker_train",
                            [sys.executable, "-m", "scripts.ranker_train"],
                            timeout=60 * 5,
                        )
                        stage_times["ranker_train"] = secs
                        if rc_train:
                            LOG.warning(
                                "[WARN] RANKER_TRAIN_FAILED rc=%s (continuing)",
                                rc_train,
                            )
                        model_path = _latest_by_glob(
                            base_dir / "data" / "models", "ranker_*.pkl"
                        )
                    if model_path is None:
                        LOG.warning("[WARN] PREDICTIONS_SKIPPED reason=model_missing")
                    else:
                        LOG.info("[INFO] PREDICTIONS_START model=%s", model_path.name)
                        rc_predict, secs = run_step(
                            "ranker_predict",
                            [sys.executable, "-m", "scripts.ranker_predict"],
                            timeout=60 * 3,
                        )
                        stage_times["ranker_predict"] = secs
                        if rc_predict:
                            LOG.warning(
                                "[WARN] PREDICTIONS_FAILED rc=%s (continuing)",
                                rc_predict,
                            )
        if "ranker_eval" in steps:
            current_step = "ranker_eval"
            if db.db_enabled():
                missing: list[str] = []
                if not db.fetch_latest_ml_artifact("features"):
                    missing.append("features")
                if not db.fetch_latest_ml_artifact("predictions"):
                    missing.append("predictions")
                if missing:
                    LOG.warning(
                        "[WARN] RANKER_EVAL_SKIPPED reason=missing_artifacts artifacts=%s",
                        ",".join(missing),
                    )
                    stage_times["ranker_eval"] = 0.0
                    step_rcs["ranker_eval"] = 0
                else:
                    cmd = [sys.executable, "-m", "scripts.ranker_eval"]
                    if extras["ranker_eval"]:
                        cmd.extend(extras["ranker_eval"])
                    rc_eval = 0
                    secs = 0.0
                    try:
                        rc_eval, secs = run_step("ranker_eval", cmd, timeout=60 * 3)
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_EVAL error (continuing): %s", exc)
                        rc_eval, secs = 1, 0.0
                    stage_times["ranker_eval"] = secs
                    step_rcs["ranker_eval"] = rc_eval
                    if rc_eval:
                        LOG.warning("[WARN] RANKER_EVAL_FAILED rc=%s (continuing)", rc_eval)
                    else:
                        payload = db.load_ml_artifact_payload("ranker_eval")
                        LOG.info(
                            "[INFO] RANKER_EVAL sample_size=%s deciles=%s",
                            int(payload.get("sample_size", 0) or 0),
                            len(payload.get("deciles") or []),
                        )
            else:
                cmd = [sys.executable, "-m", "scripts.ranker_eval"]
                if extras["ranker_eval"]:
                    cmd.extend(extras["ranker_eval"])
                rc_eval = 0
                secs = 0.0
                try:
                    rc_eval, secs = run_step("ranker_eval", cmd, timeout=60 * 3)
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_EVAL error (continuing): %s", exc)
                    rc_eval, secs = 1, 0.0
                stage_times["ranker_eval"] = secs
                step_rcs["ranker_eval"] = rc_eval
                if rc_eval:
                    LOG.warning("[WARN] RANKER_EVAL_FAILED rc=%s (continuing)", rc_eval)
                else:
                    payload = _read_json(DATA_DIR / "ranker_eval" / "latest.json")
                    LOG.info(
                        "[INFO] RANKER_EVAL sample_size=%s deciles=%s",
                        int(payload.get("sample_size", 0) or 0),
                        len(payload.get("deciles") or []),
                    )

        if getattr(args, "enrich_candidates_with_ranker", False):
            enriched = _enrich_candidates_with_ranker(
                base_dir=base_dir,
                score_column=DEFAULT_RANKER_SCORE_COLUMN,
                target_column=DEFAULT_RANKER_TARGET_COLUMN,
            )
            if enriched is not None:
                _rerank_latest_candidates(base_dir, frame=enriched)
    except Exception as exc:  # pragma: no cover - defensive guard
        rc = 1
        LOG.exception("PIPELINE_FATAL")
        error_info = {
            "step": "pipeline",
            "message": str(exc),
            "exception": exc.__class__.__name__,
        }
        try:
            send_alert(
                "JBRAVO pipeline FAILED",
                {
                    "step": current_step or "pipeline",
                    "exception": repr(exc),
                    "run_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            LOG.debug("ALERT_PIPELINE_FAILED", exc_info=True)
        write_error_report(step="pipeline", detail=str(exc))
    finally:
        rows_out = metrics_rows if metrics_rows is not None else rows
        try:
            ensured = ensure_candidates(max(1, rows_out or 0))
            rows_out = ensured if ensured else max(rows_out or 0, 1)
        except Exception:
            LOG.exception("PIPELINE_ENSURE_CANDIDATES_FAILED")
        final_rows = None
        if db.db_enabled():
            try:
                final_rows, _ = db.fetch_latest_screener_candidate_count()
            except Exception as exc:
                logger.warning(
                    "[WARN] PIPELINE_SUMMARY_ROWS_FALLBACK reason=db_error detail=%s", exc
                )
        if final_rows is not None and int(final_rows or 0) > 0:
            rows_out = int(final_rows)
        screener_rc = step_rcs.get("screener") if "screener" in step_rcs else None
        backtest_rc = step_rcs.get("backtest") if "backtest" in step_rcs else None
        metrics_rc = step_rcs.get("metrics") if "metrics" in step_rcs else None
        degraded = (
            "screener" in steps
            and "backtest" in steps
            and "metrics" in steps
            and screener_rc not in (None, 0)
            and backtest_rc in (0, None)
            and metrics_rc in (0, None)
            and latest_source == "fallback"
        )
        if degraded and rc != 0:
            logger.warning("[WARN] SCREENER_FAILED_FALLBACK_USED rows=%s", rows_out)
            rc = 0
        safe_write_pipeline_summary(rc, base_dir=base_dir)
        fetch_secs = _extract_timing(metrics, "fetch_secs")
        feature_secs = _extract_timing(metrics, "feature_secs")
        rank_secs = _extract_timing(metrics, "rank_secs")
        gate_secs = _extract_timing(metrics, "gate_secs")
        summary_rows = rows_out
        bars_rows_total = None
        if isinstance(metrics, Mapping):
            for key in ("bars_rows_total", "bars_rows", "bar_rows_total", "bar_rows"):
                candidate = _coerce_optional_int(metrics.get(key))
                if candidate is not None:
                    bars_rows_total = candidate
                    break
        summary_source = latest_source
        if screener_rc not in (0, None) and (summary_rows or 0) > 0:
            summary_source = "fallback"
        metrics_path = base_dir / "data" / "screener_metrics.json"
        metrics_payload = compose_metrics_from_artifacts(
            base_dir,
            symbols_in=symbols_in,
            fallback_symbols_with_bars=symbols_with_bars,
            fallback_bars_rows_total=bars_rows_total,
            latest_source=summary_source,
        )
        metrics = dict(metrics or {})
        metrics.update(metrics_payload)
        try:
            write_screener_metrics_json(metrics_path, metrics_payload)
        except Exception:
            LOG.exception("SCREENER_METRICS_WRITE_FAILED path=%s", metrics_path)
        metrics_final = ensure_canonical_metrics(_read_json(metrics_path))
        candidates_final = int(metrics_final.get("rows", 0))
        bars_rows_total_int = int(metrics_final.get("bars_rows_total", 0) or 0)
        with_bars_effective = int(metrics_final.get("symbols_with_required_bars", 0) or 0)
        with_bars_any = int(metrics_final.get("symbols_with_any_bars", with_bars_effective) or 0)
        summary_rows = int(metrics_final.get("rows", summary_rows) or 0)
        summary = SimpleNamespace(
            symbols_in=int(metrics_final.get("symbols_in", symbols_in) or 0),
            with_bars=with_bars_effective,
            with_bars_any=with_bars_any,
            rows=candidates_final,
            bars_rows_total=bars_rows_total_int,
            source=metrics_final.get("latest_source", summary_source),
        )
        t = SimpleNamespace(
            fetch=fetch_secs,
            features=feature_secs,
            rank=rank_secs,
            gates=gate_secs,
        )
        conn_payload = _collect_connection_health_snapshot(
            fallback_trading_ok=rc == 0,
            fallback_data_ok=bool(candidates_final > 0),
        )
        trading_ok = bool(conn_payload.get("trading_ok"))
        data_ok = bool(conn_payload.get("data_ok"))
        trading_status = int(conn_payload.get("trading_status", 0))
        data_status = int(conn_payload.get("data_status", 0))
        health = SimpleNamespace(
            trading_ok=trading_ok,
            data_ok=data_ok,
            trading_status=trading_status,
            data_status=data_status,
        )
        summary_parts = [
            "[INFO] PIPELINE_SUMMARY",
            f"symbols_in={summary.symbols_in}",
            f"with_bars={summary.with_bars}",
            f"with_bars_any={summary.with_bars_any}",
            f"with_bars_required={summary.with_bars}",
            f"rows={summary.rows}",
            f"fetch_secs={t.fetch:.3f}",
            f"feature_secs={t.features:.3f}",
            f"rank_secs={t.rank:.3f}",
            f"gate_secs={t.gates:.3f}",
        ]
        if summary.bars_rows_total is not None:
            summary_parts.append(f"bars_rows_total={summary.bars_rows_total}")
        if isinstance(summary.source, str) and summary.source:
            summary_parts.append(f"source={summary.source}")
        LOG.info(" ".join(summary_parts))
        metrics = _backfill_metrics_from_summary(metrics_path, metrics, summary)
        today = datetime.now(timezone.utc).date()
        todays_summary = _latest_summary_for_date(base_dir, today)
        if todays_summary is None:
            LOG.warning("[WARN] SUMMARY_TODAY_MISSING date=%s", today.isoformat())
        rows_for_stamp = 0
        try:
            if rc != 0:
                kpis = write_complete_screener_metrics(base_dir)
            else:
                kpis = metrics_final
            rows_for_stamp = int(kpis.get("rows") or 0)
            LOG.info(
                "SCREENER_METRICS_SYNC symbols_in=%s with_bars=%s rows=%s bars_rows_total=%s",
                kpis.get("symbols_in"),
                kpis.get("symbols_with_bars"),
                kpis.get("rows"),
                kpis.get("bars_rows_total"),
            )
        except Exception:
            LOG.exception("Failed to backfill screener_metrics.json")
        if rc == 0:
            try:
                ny = zoneinfo.ZoneInfo("America/New_York")
                now_ny = datetime.now(ny)
                fresh_payload = {
                    "ny_date": now_ny.strftime("%Y-%m-%d"),
                    "end_ny": now_ny.isoformat(),
                    "end_utc": datetime.utcnow().isoformat() + "Z",
                    "rc": 0,
                    "rows": rows_for_stamp,
                }
                fresh_path = base_dir / "data" / "pipeline_fresh.json"
                fresh_path.parent.mkdir(parents=True, exist_ok=True)
                fresh_path.write_text(json.dumps(fresh_payload, indent=2))
            except Exception:
                logger.exception("FRESH_STAMP_WRITE_FAILED")
        try:
            annotated = _annotate_screener_metrics(
                base_dir,
                rc=rc,
                steps=steps,
                step_rcs=step_rcs,
                stage_times=stage_times,
                latest_source=latest_source,
                error=error_info,
            )
            if isinstance(annotated, Mapping):
                metrics.update(dict(annotated))
        except Exception:
            LOG.exception("Failed to annotate screener_metrics.json with status")
        status_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rc": int(rc),
            "steps": list(steps),
            "symbols_in": int(summary.symbols_in or 0),
            "symbols_with_bars": int(summary.with_bars or 0),
            "rows": int(summary.rows or 0) if summary.rows is not None else 0,
            "bars_rows_total": int(summary.bars_rows_total or 0)
            if summary.bars_rows_total is not None
            else None,
            "latest_source": summary.source,
            "stage_times": {k: float(v) for k, v in stage_times.items()},
            "trading_ok": trading_ok,
            "data_ok": data_ok,
            "has_today_summary": todays_summary is not None,
            "summary_line": getattr(todays_summary, "raw_line", ""),
        }
        status_payload["degraded"] = bool(degraded)
        status_payload["step_rcs"] = {k: int(v) for k, v in step_rcs.items()}
        if error_info:
            status_payload["error"] = dict(error_info)
        _write_status_json(base_dir, status_payload)
        _write_json(base_dir / "data" / "connection_health.json", conn_payload)
        logger.info(
            "[INFO] HEALTH trading_ok=%s data_ok=%s stage=end trading_status=%s data_status=%s",
            health.trading_ok,
            health.data_ok,
            health.trading_status,
            health.data_status,
        )
        ended_at = datetime.now(timezone.utc)
        summary_payload = {
            "symbols_in": int(summary.symbols_in or 0),
            "with_bars": int(summary.with_bars or 0),
            "with_bars_any": int(summary.with_bars_any or 0),
            "rows": int(summary.rows or 0),
            "bars_rows_total": int(summary.bars_rows_total or 0) if summary.bars_rows_total is not None else None,
            "source": summary.source,
            "fetch_secs": t.fetch,
            "feature_secs": t.features,
            "rank_secs": t.rank,
            "gate_secs": t.gates,
            "stage_times": {k: float(v) for k, v in stage_times.items()},
            "step_rcs": {k: int(v) for k, v in step_rcs.items()},
            "degraded": bool(degraded),
            "latest_source": summary.source,
            "trading_ok": bool(trading_ok),
            "data_ok": bool(data_ok),
        }
        global _PIPELINE_SUMMARY_FOR_DB, _PIPELINE_RC, _PIPELINE_ENDED_AT
        _PIPELINE_SUMMARY_FOR_DB = dict(summary_payload)
        _PIPELINE_RC = int(rc)
        try:
            db.upsert_pipeline_run(today, started_dt, ended_at, int(rc), summary_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("[WARN] DB_WRITE_FAILED table=pipeline_runs err=%s", exc)
        _PIPELINE_ENDED_AT = ended_at
        duration = time.time() - started
        ingest_artifacts_to_db(today)
        if rc == 0 and step_rcs.get("metrics", 0) == 0:
            try:
                export_latest_candidates(str(pipeline_run_date), LATEST_CANDIDATES)
            except Exception as exc:
                LOG.warning("[WARN] LATEST_CANDIDATES_EXPORT_FAILED err=%s", exc)
        logger.info("[INFO] PIPELINE_END rc=%s duration=%.1fs", rc, duration)
        try:
            env = os.environ.copy()
            LOG.info(
                "[INFO] CHILD_ENV_KEYS has_database_url=%s keys_count=%d",
                bool(env.get("DATABASE_URL")),
                len(env),
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.update_account_equity",
                    "--period",
                    "1M",
                    "--timeframe",
                    "1D",
                    "--extended-hours",
                    "true",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                env=env,
            )
            LOG.info("ACCOUNT_EQUITY_REFRESH_OK")
        except Exception:
            LOG.warning("ACCOUNT_EQUITY_REFRESH_FAILED", exc_info=True)
        _reload_dashboard(args.reload_web.lower() == "true")
        should_raise = LOG.name != "pipeline" or os.environ.get("JBR_PIPELINE_RAISE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if should_raise:
            raise SystemExit(rc)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
