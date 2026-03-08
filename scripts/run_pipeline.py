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
from scripts.utils.champion_config import champion_env_overrides, load_latest_champion
from scripts.utils.ml_health_guard import (
    decide_ml_enrichment,
    load_latest_ml_health,
    resolve_ml_health_max_age_days,
)
from scripts.utils.prediction_freshness import evaluate_predictions_freshness
from scripts.utils.feature_schema import compute_feature_signature, load_features_meta_for_path
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
    "--ranker-predict-args-split": "ranker_predict_args_split",
    "--ranker-eval-args-split": "ranker_eval_args_split",
    "--ranker-walkforward-args-split": "ranker_walkforward_args_split",
    "--ranker-strategy-eval-args-split": "ranker_strategy_eval_args_split",
    "--ranker-autotune-args-split": "ranker_autotune_args_split",
    "--ranker-monitor-args-split": "ranker_monitor_args_split",
    "--ranker-recalibrate-args-split": "ranker_recalibrate_args_split",
    "--ranker-autoremediate-args-split": "ranker_autoremediate_args_split",
    "--ranker-trade-attribution-args-split": "ranker_trade_attribution_args_split",
}

_SUMMARY_RE = re.compile(
    r"PIPELINE_SUMMARY.*?symbols_in=(?P<symbols_in>\d+).*?"
    r"with_bars=(?P<symbols_with_bars>\d+).*?"
    r"(?:with_bars_any=(?P<with_bars_any>\d+).*?)?"
    r"(?:with_bars_required=(?P<with_bars_required>\d+).*?)?"
    r"rows=(?P<rows>\d+)"
    r"(?:.*?(?:bars?_rows(?:_total)?)=(?P<bars_rows_total>\d+))?"
)
_RANKER_PREDICT_SCORE_SOURCE_RE = re.compile(
    r"RANKER_PREDICT_SCORE_SOURCE calibrated=(?P<calibrated>true|false) method=(?P<method>[a-zA-Z0-9_\\-]+)"
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
DEFAULT_FEATURES_TIMEOUT_SECS = 900
DEFAULT_RANKER_PREDICT_TIMEOUT_SECS = 900
DEFAULT_RANKER_EVAL_TIMEOUT_SECS = 180
_PIPELINE_SUMMARY_FOR_DB: dict[str, Any] | None = None
_PIPELINE_RC: int | None = None
_PIPELINE_STARTED_AT: datetime | None = None
_PIPELINE_ENDED_AT: datetime | None = None


def _record_health(stage: str) -> dict[str, Any]:  # pragma: no cover - legacy hook
    return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _features_timeout_secs() -> int:
    raw = (os.getenv("JBR_FEATURES_TIMEOUT_SECS") or "").strip()
    if not raw:
        return DEFAULT_FEATURES_TIMEOUT_SECS
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        LOG.warning(
            "[WARN] FEATURES_TIMEOUT_INVALID env=JBR_FEATURES_TIMEOUT_SECS value=%s default=%s",
            raw,
            DEFAULT_FEATURES_TIMEOUT_SECS,
        )
        return DEFAULT_FEATURES_TIMEOUT_SECS
    if value <= 0:
        LOG.warning(
            "[WARN] FEATURES_TIMEOUT_INVALID env=JBR_FEATURES_TIMEOUT_SECS value=%s default=%s",
            value,
            DEFAULT_FEATURES_TIMEOUT_SECS,
        )
        return DEFAULT_FEATURES_TIMEOUT_SECS
    return value


def _ranker_predict_timeout_config() -> tuple[int, str]:
    raw = (os.getenv("JBR_RANKER_PREDICT_TIMEOUT_SECS") or "").strip()
    if not raw:
        return DEFAULT_RANKER_PREDICT_TIMEOUT_SECS, "default"
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        LOG.warning(
            "[WARN] RANKER_PREDICT_TIMEOUT_INVALID env=JBR_RANKER_PREDICT_TIMEOUT_SECS value=%s default=%s",
            raw,
            DEFAULT_RANKER_PREDICT_TIMEOUT_SECS,
        )
        return DEFAULT_RANKER_PREDICT_TIMEOUT_SECS, "default"
    if value <= 0:
        LOG.warning(
            "[WARN] RANKER_PREDICT_TIMEOUT_INVALID env=JBR_RANKER_PREDICT_TIMEOUT_SECS value=%s default=%s",
            value,
            DEFAULT_RANKER_PREDICT_TIMEOUT_SECS,
        )
        return DEFAULT_RANKER_PREDICT_TIMEOUT_SECS, "default"
    return value, "env"


def _ranker_eval_timeout_config() -> tuple[int, str]:
    raw = (os.getenv("JBR_RANKER_EVAL_TIMEOUT_SECS") or "").strip()
    if not raw:
        return DEFAULT_RANKER_EVAL_TIMEOUT_SECS, "default"
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        LOG.warning(
            "[WARN] RANKER_EVAL_TIMEOUT_INVALID env=JBR_RANKER_EVAL_TIMEOUT_SECS value=%s default=%s",
            raw,
            DEFAULT_RANKER_EVAL_TIMEOUT_SECS,
        )
        return DEFAULT_RANKER_EVAL_TIMEOUT_SECS, "default"
    if value <= 0:
        LOG.warning(
            "[WARN] RANKER_EVAL_TIMEOUT_INVALID env=JBR_RANKER_EVAL_TIMEOUT_SECS value=%s default=%s",
            value,
            DEFAULT_RANKER_EVAL_TIMEOUT_SECS,
        )
        return DEFAULT_RANKER_EVAL_TIMEOUT_SECS, "default"
    return value, "env"


def _predictions_source_state(base_dir: Path) -> str:
    if db.db_enabled():
        try:
            present = bool(db.fetch_latest_ml_artifact("predictions"))
        except Exception:
            present = False
        return f"db:{'present' if present else 'missing'}"
    try:
        present = _find_latest_predictions_path(base_dir) is not None
    except Exception:
        present = False
    return f"fs:{'present' if present else 'missing'}"


def _ranker_eval_source_state(base_dir: Path) -> str:
    if db.db_enabled():
        try:
            present = bool(db.fetch_latest_ml_artifact("ranker_eval"))
        except Exception:
            present = False
        return f"db:{'present' if present else 'missing'}"
    try:
        present = (base_dir / "data" / "ranker_eval" / "latest.json").exists()
    except Exception:
        present = False
    return f"fs:{'present' if present else 'missing'}"


def _ranker_predict_score_source_from_log(base_dir: Path) -> tuple[str, str]:
    log_path = Path(base_dir) / "logs" / "step.ranker_predict.out"
    if not log_path.exists():
        return "unknown", "unknown"
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-120000:]
    except Exception:
        return "unknown", "unknown"
    for line in reversed(tail.splitlines()):
        match = _RANKER_PREDICT_SCORE_SOURCE_RE.search(line)
        if match:
            return match.group("calibrated"), match.group("method")
    return "unknown", "unknown"


def _env_truthy(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _resolve_use_champion(args: argparse.Namespace) -> bool:
    if getattr(args, "use_champion", None) is not None:
        return bool(args.use_champion)
    return _env_truthy(os.getenv("JBR_USE_CHAMPION"))


def _resolve_ml_health_guard(args: argparse.Namespace) -> bool:
    if getattr(args, "ml_health_guard", None) is not None:
        return bool(args.ml_health_guard)
    return _env_truthy(os.getenv("JBR_ML_HEALTH_GUARD"))


def _resolve_ml_health_guard_mode(args: argparse.Namespace) -> str:
    cli_value = getattr(args, "ml_health_guard_mode", None)
    if cli_value in {"warn", "block"}:
        return str(cli_value)
    env_value = str(os.getenv("JBR_ML_HEALTH_GUARD_MODE") or "").strip().lower()
    if env_value in {"warn", "block"}:
        return env_value
    return "warn"


def _resolve_auto_refresh_predictions(args: argparse.Namespace) -> bool:
    cli_value = getattr(args, "auto_refresh_predictions", None)
    if cli_value is not None:
        return _as_bool(cli_value, False)
    return _as_bool(os.getenv("JBR_AUTO_REFRESH_PREDICTIONS"), False)


def _resolve_auto_refresh_features(args: argparse.Namespace) -> bool:
    cli_value = getattr(args, "auto_refresh_features", None)
    if cli_value is not None:
        return _as_bool(cli_value, False)
    return _as_bool(os.getenv("JBR_AUTO_REFRESH_FEATURES"), False)


def _resolve_refresh_predictions_for_candidates(args: argparse.Namespace) -> bool:
    cli_value = getattr(args, "refresh_predictions_for_candidates", None)
    if cli_value is not None:
        return _as_bool(cli_value, False)
    return _as_bool(os.getenv("JBR_REFRESH_PREDICTIONS_FOR_CANDIDATES"), False)


def _resolve_strict_auto_refresh_predictions() -> bool:
    return _as_bool(os.getenv("JBR_STRICT_AUTO_REFRESH_PREDICTIONS"), False)


def _resolve_strict_predictions_meta() -> tuple[bool, str]:
    raw = os.getenv("JBR_STRICT_PREDICTIONS_META")
    if raw is None or str(raw).strip() == "":
        return False, "default"
    return _as_bool(raw, False), "env"


def _resolve_allow_no_screener(args: argparse.Namespace) -> bool:
    cli_value = getattr(args, "allow_no_screener", None)
    if cli_value is not None:
        return bool(cli_value)
    return _env_truthy(os.getenv("JBR_ALLOW_NO_SCREENER"))


def _apply_champion_overrides(
    overrides: Mapping[str, str], *, mode: str
) -> tuple[dict[str, str], list[str]]:
    if mode not in {"fill", "force"}:
        mode = "fill"
    resolved: dict[str, str] = {}
    applied: list[str] = []
    for key, value in overrides.items():
        if not key:
            continue
        text = str(value)
        existing = os.environ.get(key)
        if mode == "force" or existing in (None, ""):
            resolved[key] = text
            applied.append(key)
    return resolved, applied


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
            "model": _artifact_payload(_latest_by_glob(data_dir / "models", "ranker_*.pkl")),
            "predictions": _artifact_payload_db("predictions"),
            "eval": _artifact_payload_db("ranker_eval"),
        }
    else:
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "bars": _artifact_payload(data_dir / "daily_bars.csv"),
            "labels": _artifact_payload(_latest_by_glob(data_dir / "labels", "labels_*.csv")),
            "features": _artifact_payload(_latest_by_glob(data_dir / "features", "features_*.csv")),
            "model": _artifact_payload(_latest_by_glob(data_dir / "models", "ranker_*.pkl")),
            "predictions": _artifact_payload(
                _latest_by_glob(data_dir / "predictions", "predictions_*.csv")
            ),
            "eval": _artifact_payload(_latest_by_glob(data_dir / "ranker_eval", "*.json")),
        }

    try:
        _write_json(data_dir / "nightly_ml_status.json", payload)
        LOG.info("[INFO] NIGHTLY_ML_STATUS_WRITTEN path=%s", data_dir / "nightly_ml_status.json")
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


def _probe_trading_endpoint(
    session: requests.Session, headers: Mapping[str, str]
) -> dict[str, Any]:
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
        or _coerce_optional_int(fetch_stats.get("required_bars"))
        if fetch_stats
        else None or DEFAULT_REQUIRED_BARS
    )
    fetch_symbols_required = (
        _coerce_optional_int(fetch_stats.get("symbols_with_required_bars_fetch"))
        if fetch_stats
        else None
    )
    fetch_symbols_any = (
        _coerce_optional_int(fetch_stats.get("symbols_with_any_bars_fetch"))
        if fetch_stats
        else None
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
    bars_fetch = (
        _coerce_optional_int(fetch_stats.get("bars_rows_total_fetch")) if fetch_stats else None
    )
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
    bars_reported = int(bars_effective)
    bars_effective_preserved: int | None = None
    if isinstance(bars_fetch, int) and bars_fetch > 0:
        bars_reported = int(bars_fetch)
        if bars_effective != bars_reported:
            bars_effective_preserved = int(bars_effective)

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
        "symbols_with_bars_post": _coerce_optional_int(post_stats.get("symbols_with_bars_post"))
        if post_stats
        else None,
        "bars_rows_total": int(bars_reported or 0),
        "bars_rows_total_fetch": bars_fetch,
        "bars_rows_total_post": _coerce_optional_int(post_stats.get("bars_rows_total_post"))
        if post_stats
        else None,
        "rows": int(rows_final),
        "rows_premetrics": int(rows_final),
        "latest_source": latest_source or "unknown",
        "metrics_version": 2,
    }
    if bars_effective_preserved is not None:
        payload["bars_rows_total_effective"] = bars_effective_preserved
    if fetch_any_attempted is not None:
        if with_bars_any and fetch_any_attempted == with_bars_any:
            payload["symbols_with_bars_fetch"] = fetch_any_attempted
        else:
            payload["symbols_attempted_fetch"] = fetch_any_attempted
    elif fetch_symbols_required is not None:
        payload["symbols_attempted_fetch"] = fetch_symbols_required
    if post_stats and "candidates_final" in post_stats:
        payload["candidates_final"] = _coerce_optional_int(
            post_stats.get("candidates_final")
        ) or int(rows_final)
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
                "SCREENER_METRICS_BACKFILL source=pipeline_summary fields=%s",
                ",".join(sorted(updated_fields)),
            )
        except Exception:
            LOG.exception("SCREENER_METRICS_BACKFILL_FAILED path=%s", metrics_path)
    return payload


def _coerce_canonical(
    df_scored: Optional[pd.DataFrame], df_top: Optional[pd.DataFrame]
) -> pd.DataFrame:
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

    base["symbol"] = base["symbol"].astype("string").fillna("").str.strip().str.upper()
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
            scored["symbol"] = scored["symbol"].astype("string").fillna("").str.strip().str.upper()
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
            base["score_breakdown"]
            .astype("string")
            .fillna("")
            .replace({"": "{}", "fallback": "{}"})
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


def _detect_new_labels_file(output_dir: Path, before: Mapping[Path, int]) -> Path | None:
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


def refresh_latest_candidates(
    base_dir: Path | None = None, run_date: date | None = None
) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["symbol", score_column, "score_ts"])
    working = df.copy()
    working["symbol"] = working["symbol"].map(_coerce_symbol)
    working = working.loc[working["symbol"].astype(str).str.len() > 0]
    working["score_ts"] = pd.NaT
    if "timestamp" in working.columns:
        working["score_ts"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working.sort_values(by="score_ts", inplace=True)
    if "run_date" in working.columns:
        working["run_date"] = pd.to_datetime(working["run_date"], errors="coerce", utc=True)
    working = working.drop_duplicates(subset=["symbol"], keep="last")
    output_columns = ["symbol", score_column, "score_ts"]
    if "run_date" in working.columns:
        output_columns.append("run_date")
    return working[output_columns]


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
            return pd.DataFrame(columns=["symbol", score_column, "score_ts"]), "db:missing"
        return _normalize_predictions_frame(predictions, score_column), "db"
    predictions_path = _find_latest_predictions_path(base_dir)
    if predictions_path is None:
        return pd.DataFrame(columns=["symbol", score_column, "score_ts"]), "file:missing"
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
    if "model_score" not in merged.columns and target_column in merged.columns:
        merged["model_score"] = pd.to_numeric(merged[target_column], errors="coerce")
    matched = int(merged[target_column].notna().sum()) if target_column in merged.columns else 0
    ordered: list[str] = list(CANONICAL_COLUMNS)
    ordered.extend(col for col in candidates.columns if col not in ordered)
    ordered = [col for col in ordered if col in merged.columns and col != target_column]
    ordered.append(target_column)
    if "model_score" in merged.columns and "model_score" not in ordered:
        ordered.append("model_score")
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
    refresh_predictions_for_candidates: bool = False,
    refresh_predictions_callback: Any = None,
    _refresh_attempted: bool = False,
) -> pd.DataFrame | None:
    def _extract_candidate_run_ts(frame: pd.DataFrame) -> pd.Timestamp | None:
        if frame.empty:
            return None
        if "run_ts_utc" not in frame.columns:
            return None
        parsed = pd.to_datetime(frame["run_ts_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed.dropna().max()
        return None

    def _extract_run_date(frame: pd.DataFrame, run_ts: pd.Timestamp | None) -> date | None:
        if "run_date" in frame.columns:
            parsed = pd.to_datetime(frame["run_date"], errors="coerce", utc=True)
            if parsed.notna().any():
                try:
                    return parsed.dropna().max().date()
                except Exception:
                    pass
        if run_ts is not None:
            try:
                return run_ts.date()
            except Exception:
                return None
        return None

    def _count_rows_for_run_ts(frame: pd.DataFrame, run_ts: pd.Timestamp | None) -> int:
        if frame.empty or run_ts is None or "run_ts_utc" not in frame.columns:
            return 0
        parsed = pd.to_datetime(frame["run_ts_utc"], errors="coerce", utc=True)
        if not parsed.notna().any():
            return 0
        return int((parsed == run_ts).sum())

    def _build_overlap_diag(
        candidates_frame: pd.DataFrame,
        predictions_frame: pd.DataFrame,
        *,
        score_col: str,
        run_ts_utc: pd.Timestamp | None,
        run_date: date | None,
    ) -> dict[str, Any]:
        candidate_symbols: list[str] = []
        if "symbol" in candidates_frame.columns:
            candidate_symbols = [
                _coerce_symbol(value)
                for value in candidates_frame["symbol"].tolist()
                if _coerce_symbol(value)
            ]
        prediction_symbols: list[str] = []
        if "symbol" in predictions_frame.columns:
            prediction_symbols = [
                _coerce_symbol(value)
                for value in predictions_frame["symbol"].tolist()
                if _coerce_symbol(value)
            ]
        candidate_symbol_set = set(candidate_symbols)
        prediction_symbol_set = set(prediction_symbols)
        overlap_symbols = sorted(candidate_symbol_set & prediction_symbol_set)
        overlap_count = int(len(overlap_symbols))
        pred_ts_min = None
        pred_ts_max = None
        pred_date_values: set[date] = set()
        if "run_date" in predictions_frame.columns:
            pred_run_dates = pd.to_datetime(
                predictions_frame["run_date"], errors="coerce", utc=True
            )
            if pred_run_dates.notna().any():
                pred_date_values.update(ts.date() for ts in pred_run_dates.dropna().tolist())
        if "score_ts" in predictions_frame.columns:
            pred_ts = pd.to_datetime(predictions_frame["score_ts"], errors="coerce", utc=True)
            if pred_ts.notna().any():
                pred_ts_min = pred_ts.dropna().min()
                pred_ts_max = pred_ts.dropna().max()
                pred_date_values = {ts.date() for ts in pred_ts.dropna().tolist()}
        prediction_non_null_symbols: set[str] = set()
        if score_col in predictions_frame.columns:
            score_series = pd.to_numeric(predictions_frame[score_col], errors="coerce")
            prediction_non_null_symbols = {
                _coerce_symbol(symbol)
                for symbol in predictions_frame.loc[score_series.notna(), "symbol"].tolist()
                if _coerce_symbol(symbol)
            }
        overlap_non_null = int(len(candidate_symbol_set & prediction_non_null_symbols))

        LOG.info(
            "[INFO] MODEL_SCORE_OVERLAP_DIAG candidates=%s prediction_symbols=%s overlap=%s run_ts_utc=%s run_date=%s score_col=%s pred_ts_min=%s pred_ts_max=%s",
            int(len(candidate_symbol_set)),
            int(len(prediction_symbol_set)),
            overlap_count,
            run_ts_utc.isoformat() if run_ts_utc is not None else None,
            run_date.isoformat() if run_date is not None else None,
            score_col,
            pred_ts_min.isoformat() if isinstance(pred_ts_min, pd.Timestamp) else None,
            pred_ts_max.isoformat() if isinstance(pred_ts_max, pd.Timestamp) else None,
        )

        missing_symbols = sorted(candidate_symbol_set - prediction_symbol_set)
        missing_fraction = (
            float(len(missing_symbols)) / float(len(candidate_symbol_set))
            if candidate_symbol_set
            else 0.0
        )
        should_sample = bool(missing_symbols) and (overlap_count == 0 or missing_fraction >= 0.25)
        sample_missing_symbols: list[str] = []
        overlap_reason = "no_prediction_for_symbol"
        if should_sample:
            sample_missing_symbols = missing_symbols[:10]
            if len(prediction_symbol_set) <= 0:
                overlap_reason = "no_prediction_for_symbol"
            elif run_date is not None and pred_date_values and run_date not in pred_date_values:
                overlap_reason = "date_mismatch"
            elif missing_symbols:
                overlap_reason = "no_prediction_for_symbol"
            else:
                overlap_reason = "run_scope_mismatch"
            LOG.info(
                "[INFO] MODEL_SCORE_OVERLAP_SAMPLE missing_symbols=%s reason=%s",
                sample_missing_symbols,
                overlap_reason,
            )
        return {
            "candidate_symbol_count": int(len(candidate_symbol_set)),
            "prediction_symbol_count": int(len(prediction_symbol_set)),
            "overlap_count": overlap_count,
            "overlap_non_null": overlap_non_null,
            "sample_missing_symbols": sample_missing_symbols,
            "sample_reason": overlap_reason,
            "prediction_dates": sorted(d.isoformat() for d in pred_date_values),
        }

    def _log_model_score_join_diag(
        frame: pd.DataFrame,
        *,
        scores_rows_for_run: int,
        run_ts_utc: pd.Timestamp | None,
        score_col: str,
    ) -> None:
        candidates_count = int(len(frame.index))
        if score_col in frame.columns:
            joined_series = pd.to_numeric(frame[score_col], errors="coerce")
        else:
            joined_series = pd.Series([None] * candidates_count, index=frame.index)
        joined_non_null = int(joined_series.notna().sum())
        joined_null = max(candidates_count - joined_non_null, 0)
        LOG.info(
            "[INFO] MODEL_SCORE_JOIN_DIAG candidates=%s scores_rows_for_run=%s joined_non_null=%s joined_null=%s run_ts_utc=%s score_col=%s matched=%s",
            candidates_count,
            int(max(scores_rows_for_run, 0)),
            joined_non_null,
            joined_null,
            run_ts_utc.isoformat() if run_ts_utc is not None else None,
            score_col,
            joined_non_null,
        )
        if joined_null <= 0:
            return
        if int(scores_rows_for_run or 0) <= 0:
            reason = "missing_scores_for_run"
        elif joined_non_null <= 0:
            reason = "symbol_mismatch_or_join_key_mismatch"
        else:
            reason = "partial_missing_scores"
        sample_symbols: list[str] = []
        if "symbol" in frame.columns:
            sample = frame.loc[joined_series.isna(), "symbol"].head(10)
            sample_symbols = [
                _coerce_symbol(value)
                for value in sample
                if isinstance(value, str) and value.strip()
            ]
        LOG.info(
            "[INFO] MODEL_SCORE_JOIN_SAMPLE_UNMATCHED symbols=%s reason=%s",
            sample_symbols,
            reason,
        )

    def _classify_matched_zero_subreason(
        overlap_diag: Mapping[str, Any], *, run_date: date | None
    ) -> str:
        prediction_symbol_count = int(overlap_diag.get("prediction_symbol_count") or 0)
        overlap_count = int(overlap_diag.get("overlap_count") or 0)
        overlap_non_null = int(overlap_diag.get("overlap_non_null") or 0)
        prediction_dates = overlap_diag.get("prediction_dates") or []
        if prediction_symbol_count <= 0:
            return "matched_zero:no_prediction_symbols"
        if overlap_count <= 0:
            if (
                run_date is not None
                and prediction_dates
                and run_date.isoformat() not in prediction_dates
            ):
                return "matched_zero:date_or_run_scope_mismatch"
            sample_reason = str(overlap_diag.get("sample_reason") or "").strip().lower()
            if sample_reason in {"date_mismatch", "run_scope_mismatch"}:
                return "matched_zero:date_or_run_scope_mismatch"
            return "matched_zero:symbol_join_mismatch"
        if overlap_non_null <= 0:
            return "matched_zero:null_scores_only"
        return "matched_zero:symbol_join_mismatch"

    base = _resolve_base_dir(base_dir)
    candidates_path = base / "data" / "latest_candidates.csv"
    candidates = _load_latest_candidates_frame(base)
    if candidates.empty:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_SKIPPED reason=candidates_empty candidates_path=%s predictions_source=%s",
            candidates_path,
            "db:unknown" if db.db_enabled() else "file:unknown",
        )
        return
    predictions, predictions_source = _load_latest_predictions_frame(base, score_column)
    if predictions.empty:
        if db.db_enabled():
            run_ts_utc = _extract_candidate_run_ts(candidates)
            diag_frame = candidates.copy()
            if target_column not in diag_frame.columns:
                diag_frame[target_column] = pd.NA
            _log_model_score_join_diag(
                diag_frame,
                scores_rows_for_run=0,
                run_ts_utc=run_ts_utc,
                score_col=target_column,
            )
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_SKIPPED reason=predictions_missing candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
        )
        return

    try:
        candidates = candidates.copy()
        predictions = predictions.copy()
        if "symbol" in candidates.columns:
            candidates["symbol"] = candidates["symbol"].map(_coerce_symbol)
        if target_column in candidates.columns:
            candidates = candidates.drop(columns=[target_column], errors="ignore")
        if "symbol" in predictions.columns:
            predictions["symbol"] = predictions["symbol"].map(_coerce_symbol)
        renamed = predictions.rename(columns={score_column: target_column})
        candidate_symbol_set = set(candidates["symbol"].dropna().astype(str))
        run_ts_utc = _extract_candidate_run_ts(candidates)
        run_date = _extract_run_date(candidates, run_ts_utc)
        overlap_diag = _build_overlap_diag(
            candidates,
            renamed,
            score_col=target_column,
            run_ts_utc=run_ts_utc,
            run_date=run_date,
        )
        scores_rows_for_run = int(
            renamed.loc[renamed["symbol"].isin(candidate_symbol_set), target_column].notna().sum()
        )
        merged = candidates.merge(
            renamed[["symbol", target_column, "score_ts"]], on="symbol", how="left"
        )
        if "model_score" not in merged.columns and target_column in merged.columns:
            merged["model_score"] = pd.to_numeric(merged[target_column], errors="coerce")
        matched = int(merged[target_column].notna().sum()) if target_column in merged.columns else 0
        ordered: list[str] = list(CANONICAL_COLUMNS)
        ordered.extend(col for col in candidates.columns if col not in ordered)
        ordered = [col for col in ordered if col in merged.columns and col != target_column]
        ordered.append(target_column)
        if "model_score" in merged.columns and "model_score" not in ordered:
            ordered.append("model_score")
        optional_columns = ["run_date", "run_ts_utc", "created_at"]
        if db.db_enabled():
            optional_columns.append("score_ts")
        ordered_with_optional: list[str] = []
        for column in ordered + [col for col in optional_columns if col in merged.columns]:
            if column not in ordered_with_optional:
                ordered_with_optional.append(column)
        merged = merged[ordered_with_optional]
    except Exception:
        LOG.warning(
            "[WARN] CANDIDATES_ENRICH_FAILED reason=merge_error candidates_path=%s predictions_source=%s",
            candidates_path,
            predictions_source,
            exc_info=True,
        )
        return

    if db.db_enabled():
        if run_ts_utc is None:
            LOG.warning(
                "[WARN] CANDIDATES_ENRICH_SKIPPED reason=run_ts_missing predictions_source=%s",
                predictions_source,
            )
            return
        candidate_rows_for_run = _count_rows_for_run_ts(candidates, run_ts_utc)
        if candidate_rows_for_run <= 0:
            LOG.warning(
                "[WARN] MODEL_SCORE_ENRICH_RUN_TS_MISSING run_ts_utc=%s",
                run_ts_utc.isoformat(),
            )
            return
        _log_model_score_join_diag(
            merged,
            scores_rows_for_run=scores_rows_for_run,
            run_ts_utc=run_ts_utc,
            score_col=target_column,
        )
        if matched <= 0:
            matched_zero_reason = _classify_matched_zero_subreason(overlap_diag, run_date=run_date)
            if (
                refresh_predictions_for_candidates
                and not _refresh_attempted
                and callable(refresh_predictions_callback)
                and candidate_symbol_set
            ):
                LOG.info(
                    "[INFO] AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES enabled=true symbols=%s reason=matched_zero",
                    int(len(candidate_symbol_set)),
                )
                refresh_rc = 1
                try:
                    refresh_rc = int(refresh_predictions_callback(sorted(candidate_symbol_set)))
                except Exception:
                    LOG.warning(
                        "AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES callback failed",
                        exc_info=True,
                    )
                    refresh_rc = 1
                LOG.info(
                    "[INFO] AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_DONE rc=%s predictions_source=%s",
                    refresh_rc,
                    _predictions_source_state(base),
                )
                if refresh_rc == 0:
                    return _enrich_candidates_with_ranker(
                        base,
                        score_column=score_column,
                        target_column=target_column,
                        refresh_predictions_for_candidates=refresh_predictions_for_candidates,
                        refresh_predictions_callback=refresh_predictions_callback,
                        _refresh_attempted=True,
                    )
            LOG.warning(
                "[WARN] CANDIDATES_ENRICH_SKIPPED reason=%s candidates=%s run_ts_utc=%s predictions_source=%s",
                matched_zero_reason,
                int(len(candidates.index)),
                run_ts_utc.isoformat(),
                predictions_source,
            )
            return
        written = db.upsert_screener_ranker_scores_frame(
            run_ts_utc,
            merged,
            run_date=run_date,
            symbol_column="symbol",
            score_column=target_column,
            score_ts_column="score_ts" if "score_ts" in merged.columns else None,
        )
        if written <= 0:
            LOG.warning(
                "[WARN] CANDIDATES_ENRICH_SKIPPED reason=db_write_failed predictions_source=%s",
                predictions_source,
            )
            return
        LOG.info(
            "[INFO] CANDIDATES_ENRICHED destination=db table=screener_ranker_scores_app rows=%s run_ts_utc=%s",
            written,
            run_ts_utc.isoformat(),
        )
        return merged

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


def _model_score_coverage_summary(
    frame: pd.DataFrame | None,
    *,
    source: str,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        payload = {
            "total": 0,
            "non_null": 0,
            "pct": 0.0,
            "source": source,
            "run_ts_utc": None,
        }
        LOG.info(
            "[INFO] MODEL_SCORE_COVERAGE total=%d non_null=%d pct=%.2f source=%s run_ts_utc=%s",
            0,
            0,
            0.0,
            source,
            None,
        )
        return payload

    total = int(len(frame.index))
    model_score = pd.to_numeric(frame.get("model_score"), errors="coerce")
    model_score_5d = pd.to_numeric(frame.get("model_score_5d"), errors="coerce")
    combined = model_score
    if combined is None or combined.empty:
        combined = model_score_5d
    else:
        combined = combined.combine_first(model_score_5d)
    non_null = int(combined.notna().sum()) if combined is not None else 0
    pct = (float(non_null) / float(total) * 100.0) if total > 0 else 0.0

    run_ts_utc = None
    if "run_ts_utc" in frame.columns:
        parsed = pd.to_datetime(frame["run_ts_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            run_ts_utc = parsed.dropna().max().isoformat()

    payload = {
        "total": total,
        "non_null": non_null,
        "pct": pct,
        "source": source,
        "run_ts_utc": run_ts_utc,
    }
    LOG.info(
        "[INFO] MODEL_SCORE_COVERAGE total=%d non_null=%d pct=%.2f source=%s run_ts_utc=%s",
        total,
        non_null,
        pct,
        source,
        run_ts_utc,
    )
    return payload


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
        LOG.warning("[WARN] CANDIDATES_WEIGHTED_SKIPPED reason=missing_key key=%s", key)
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
    collected: dict[str, list[str]] = {dest: [] for dest in _SPLIT_FLAG_MAP.values()}
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
            "Include 'labels' to generate label CSVs, 'ranker_predict' to refresh predictions from "
            "the latest model, 'ranker_walkforward' for optional walk-forward "
            "ML evaluation, 'ranker_strategy_eval' for optional OOS strategy evaluation, "
            "'ranker_monitor' for optional drift/health monitoring, and "
            "'ranker_recalibrate' for optional post-hoc model probability calibration, "
            "'ranker_autotune' for optional OOS autotuning sweeps, "
            "'ranker_autoremediate' for optional health-triggered bounded autotune, and "
            "'ranker_trade_attribution' for optional closed-trade score attribution."
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
        "--ranker-predict-args",
        default=os.getenv("JBR_RANKER_PREDICT_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_predict",
    )
    parser.add_argument(
        "--ranker-predict-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_predict provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-eval-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_eval provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-walkforward-args",
        default=os.getenv("JBR_RANKER_WALKFORWARD_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_walkforward",
    )
    parser.add_argument(
        "--ranker-walkforward-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_walkforward provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-strategy-eval-args",
        default=os.getenv("JBR_RANKER_STRATEGY_EVAL_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_strategy_eval",
    )
    parser.add_argument(
        "--ranker-strategy-eval-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_strategy_eval provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-autotune-args",
        default=os.getenv("JBR_RANKER_AUTOTUNE_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_autotune",
    )
    parser.add_argument(
        "--ranker-autotune-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_autotune provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-monitor-args",
        default=os.getenv("JBR_RANKER_MONITOR_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_monitor",
    )
    parser.add_argument(
        "--ranker-monitor-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_monitor provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-recalibrate-args",
        default=os.getenv("JBR_RANKER_RECALIBRATE_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_recalibrate",
    )
    parser.add_argument(
        "--ranker-recalibrate-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_recalibrate provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-autoremediate-args",
        default=os.getenv("JBR_RANKER_AUTOREMEDIATE_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_autoremediate",
    )
    parser.add_argument(
        "--ranker-autoremediate-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_autoremediate provided as separate tokens",
    )
    parser.add_argument(
        "--ranker-trade-attribution-args",
        default=os.getenv("JBR_RANKER_TRADE_ATTRIBUTION_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.ranker_trade_attribution",
    )
    parser.add_argument(
        "--ranker-trade-attribution-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.ranker_trade_attribution provided as separate tokens",
    )
    parser.add_argument(
        "--use-champion",
        action="store_true",
        default=None,
        help=(
            "Opt-in: load latest ranker_champion and apply champion-derived ML env overrides "
            "to ML analysis steps."
        ),
    )
    parser.add_argument(
        "--champion-mode",
        choices=("fill", "force"),
        default="fill",
        help=(
            "Champion env application mode: fill applies only missing keys; "
            "force overwrites existing env keys."
        ),
    )
    parser.add_argument(
        "--ml-health-guard",
        action="store_true",
        default=None,
        help=(
            "Opt-in: gate candidate ranker enrichment using latest ranker_monitor "
            "recommended_action."
        ),
    )
    parser.add_argument(
        "--ml-health-guard-mode",
        choices=("warn", "block"),
        default=None,
        help=(
            "ML health guard mode. warn logs and proceeds; block skips enrichment "
            "when ranker_monitor recommends action!=none."
        ),
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
        "--allow-no-screener",
        action="store_true",
        default=None,
        help=(
            "Opt-in: allow pipeline runs without the screener step. "
            "Useful for ML-only maintenance/evaluation runs."
        ),
    )
    parser.add_argument(
        "--auto-refresh-predictions",
        nargs="?",
        const="true",
        default=None,
        help=(
            "Opt-in: when predictions are stale vs latest model, rerun ranker_predict "
            "before eval/enrichment/monitor paths that consume predictions."
        ),
    )
    parser.add_argument(
        "--auto-refresh-features",
        nargs="?",
        const="true",
        default=None,
        help=(
            "Opt-in: when predictions are stale and feature/model metadata mismatch is detected, "
            "refresh features (and labels if required) before rerunning ranker_predict."
        ),
    )
    parser.add_argument(
        "--refresh-predictions-for-candidates",
        nargs="?",
        const="true",
        default=None,
        help=(
            "Opt-in: when enrichment overlap is zero for the current candidate universe, "
            "attempt one candidate-scoped ranker_predict refresh before final matched_zero skip."
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
    allowed = {
        "screener",
        "backtest",
        "metrics",
        "labels",
        "ranker_predict",
        "ranker_eval",
        "ranker_walkforward",
        "ranker_strategy_eval",
        "ranker_monitor",
        "ranker_recalibrate",
        "ranker_autotune",
        "ranker_autoremediate",
        "ranker_trade_attribution",
    }
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
    env: Mapping[str, str] | None = None,
) -> tuple[int, float]:
    started = time.time()
    LOG.info("[INFO] START %s cmd=%s", name, shlex.join(cmd))
    child_env = os.environ.copy()
    if env:
        for key, value in env.items():
            if key:
                child_env[str(key)] = str(value)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("APCA_DATA_API_BASE_URL", "https://data.alpaca.markets")
    child_env.setdefault("ALPACA_DATA_FEED", "iex")
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
                bool(child_env.get("DATABASE_URL")),
                len(child_env),
            )
            if tee_to_stdout or tee_to_logger:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    text=True,
                    bufsize=1,
                )
            else:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    env=child_env,
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
        log_file.write(
            f"[{datetime.now(timezone.utc).isoformat()}] END {name} rc={rc} secs={elapsed:.1f}\n"
        )
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
            normalized_score_breakdown = db.normalize_score_breakdown(
                score_breakdown_raw_value, symbol=symbol
            )
            if normalized_score_breakdown is None:
                try:
                    is_empty_score = score_breakdown_raw_value in (None, "") or pd.isna(
                        score_breakdown_raw_value
                    )
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


def _help_requested(argv: Optional[Iterable[str]] = None) -> bool:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    return any(flag in args for flag in ("-h", "--help"))


def main(argv: Optional[Iterable[str]] = None) -> int:
    if _help_requested(argv):
        try:
            parse_args(argv)
        except SystemExit as exc:
            if exc.code in (None, 0):
                return 0
            try:
                return int(exc.code)
            except (TypeError, ValueError):
                return 1
        return 0

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
    allow_no_screener = _resolve_allow_no_screener(args)
    if "screener" not in steps and not allow_no_screener:
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
    if "screener" not in steps and allow_no_screener:
        LOG.info("[INFO] ALLOW_NO_SCREENER enabled=true steps=%s", ",".join(steps))
    LOG.info("[INFO] PIPELINE_START steps=%s", ",".join(steps))
    pipeline_run_date = _resolve_pipeline_run_date()
    LOG.info(
        "[INFO] PIPELINE_RUN_DATE run_date=%s tz=America/New_York",
        pipeline_run_date.isoformat(),
    )

    _ensure_latest_headers()
    base_dir = _resolve_base_dir()
    use_champion = _resolve_use_champion(args)
    champion_mode = getattr(args, "champion_mode", "fill") or "fill"
    ml_health_guard_enabled = _resolve_ml_health_guard(args)
    ml_health_guard_mode = _resolve_ml_health_guard_mode(args)
    auto_refresh_predictions = _resolve_auto_refresh_predictions(args)
    auto_refresh_features = _resolve_auto_refresh_features(args)
    refresh_predictions_for_candidates = _resolve_refresh_predictions_for_candidates(args)
    strict_auto_refresh_predictions = _resolve_strict_auto_refresh_predictions()
    strict_predictions_meta, strict_predictions_meta_source = _resolve_strict_predictions_meta()
    LOG.info(
        "[INFO] STRICT_PREDICTIONS_META enabled=%s source=%s",
        str(bool(strict_predictions_meta)).lower(),
        strict_predictions_meta_source,
    )
    LOG.info(
        "[INFO] REFRESH_PREDICTIONS_FOR_CANDIDATES enabled=%s",
        str(bool(refresh_predictions_for_candidates)).lower(),
    )
    champion_env: dict[str, str] = {}
    if use_champion:
        champion_payload = load_latest_champion(base_dir=base_dir)
        if champion_payload:
            champion_source = str(champion_payload.get("_champion_source") or "unknown")
            champion_run_date = str(champion_payload.get("_champion_run_date") or "unknown")
            LOG.info(
                "[INFO] CHAMPION_LOAD source=%s present=true run_date=%s",
                champion_source,
                champion_run_date,
            )
            overrides = champion_env_overrides(champion_payload)
            champion_env, champion_applied_keys = _apply_champion_overrides(
                overrides, mode=champion_mode
            )
            LOG.info(
                "[INFO] CHAMPION_APPLIED mode=%s keys=%s",
                champion_mode,
                ",".join(champion_applied_keys) if champion_applied_keys else "none",
            )
        else:
            LOG.info("[INFO] CHAMPION_LOAD source=none present=false run_date=unknown")
            LOG.warning("[WARN] CHAMPION_MISSING")

    extras = {
        "screener": _strip_labels_args(
            _merge_split_args(args.screener_args, args.screener_args_split)
        ),
        "backtest": _merge_split_args(args.backtest_args, args.backtest_args_split),
        "metrics": _merge_split_args(args.metrics_args, args.metrics_args_split),
        "ranker_predict": _merge_split_args(
            getattr(args, "ranker_predict_args", ""),
            getattr(args, "ranker_predict_args_split", None),
        ),
        "ranker_eval": _merge_split_args(
            getattr(args, "ranker_eval_args", ""), getattr(args, "ranker_eval_args_split", None)
        ),
        "ranker_walkforward": _merge_split_args(
            getattr(args, "ranker_walkforward_args", ""),
            getattr(args, "ranker_walkforward_args_split", None),
        ),
        "ranker_strategy_eval": _merge_split_args(
            getattr(args, "ranker_strategy_eval_args", ""),
            getattr(args, "ranker_strategy_eval_args_split", None),
        ),
        "ranker_monitor": _merge_split_args(
            getattr(args, "ranker_monitor_args", ""),
            getattr(args, "ranker_monitor_args_split", None),
        ),
        "ranker_recalibrate": _merge_split_args(
            getattr(args, "ranker_recalibrate_args", ""),
            getattr(args, "ranker_recalibrate_args_split", None),
        ),
        "ranker_autotune": _merge_split_args(
            getattr(args, "ranker_autotune_args", ""),
            getattr(args, "ranker_autotune_args_split", None),
        ),
        "ranker_autoremediate": _merge_split_args(
            getattr(args, "ranker_autoremediate_args", ""),
            getattr(args, "ranker_autoremediate_args_split", None),
        ),
        "ranker_trade_attribution": _merge_split_args(
            getattr(args, "ranker_trade_attribution_args", ""),
            getattr(args, "ranker_trade_attribution_args_split", None),
        ),
    }

    if getattr(args, "backtest_quick", None) == "true":
        extras["backtest"].append("--quick")

    export_bars_path = args.export_daily_bars_path
    if "screener" in steps:
        extras["screener"] = _inject_export_daily_bars_arg(extras["screener"], export_bars_path)
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
    champion_ml_steps = {
        "labels",
        "features",
        "ranker_train",
        "ranker_predict",
        "ranker_eval",
        "ranker_walkforward",
        "ranker_strategy_eval",
        "ranker_monitor",
        "ranker_recalibrate",
        "ranker_autoremediate",
        "ranker_trade_attribution",
    }

    def _step_env(step_name: str) -> Mapping[str, str] | None:
        if not use_champion:
            return None
        if not champion_env:
            return None
        if step_name not in champion_ml_steps:
            return None
        return champion_env

    freshness_state: dict[str, Any] = {"refresh_attempted": False}

    def _summary_path_for_model(model_path: Path | None) -> Path | None:
        if model_path is None:
            return None
        match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
        if not match:
            return None
        return model_path.parent / f"ranker_summary_{match.group(1)}.json"

    def _latest_model_meta() -> dict[str, Any]:
        latest_model = _latest_by_glob(base_dir / "data" / "models", "ranker_*.pkl")
        if latest_model is None:
            return {
                "model_path": None,
                "model_mtime_utc": None,
                "feature_set": None,
                "feature_signature": None,
                "feature_count": 0,
            }
        try:
            model_mtime_utc = datetime.fromtimestamp(
                latest_model.stat().st_mtime, timezone.utc
            ).isoformat()
        except Exception:
            model_mtime_utc = None
        feature_set = None
        feature_signature_stored = None
        feature_columns: list[str] = []
        summary_path = _summary_path_for_model(latest_model)
        if summary_path is not None and summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_payload = {}
            if isinstance(summary_payload, Mapping):
                feature_set = str(summary_payload.get("feature_set") or "").strip().lower() or None
                feature_signature_stored = (
                    str(summary_payload.get("feature_signature") or "").strip() or None
                )
                summary_cols = summary_payload.get("feature_columns")
                if isinstance(summary_cols, list):
                    feature_columns = [str(c).strip() for c in summary_cols if str(c).strip()]
        try:
            import joblib  # type: ignore

            payload = joblib.load(latest_model)
            if isinstance(payload, Mapping):
                payload_set = str(payload.get("feature_set") or "").strip().lower() or None
                if payload_set:
                    feature_set = payload_set
                payload_signature = str(payload.get("feature_signature") or "").strip() or None
                if payload_signature:
                    feature_signature_stored = payload_signature
                payload_cols = payload.get("feature_columns")
                if isinstance(payload_cols, list):
                    cleaned = [str(c).strip() for c in payload_cols if str(c).strip()]
                    if cleaned:
                        feature_columns = cleaned
        except Exception:
            pass
        computed_signature = compute_feature_signature(feature_columns) if feature_columns else None
        if (
            feature_signature_stored
            and computed_signature
            and feature_signature_stored != computed_signature
        ):
            LOG.warning(
                "[WARN] MODEL_FEATURE_SIGNATURE_MISMATCH stored=%s computed=%s model_path=%s",
                feature_signature_stored,
                computed_signature,
                latest_model,
            )
        resolved_signature = computed_signature or feature_signature_stored
        return {
            "model_path": str(latest_model),
            "model_mtime_utc": model_mtime_utc,
            "feature_set": feature_set,
            "feature_signature": resolved_signature,
            "feature_count": int(len(feature_columns)),
        }

    def _load_features_meta() -> tuple[dict[str, Any], str]:
        return load_features_meta_for_path(
            None,
            base_dir=base_dir,
            prefer_db=bool(db.db_enabled()),
        )

    def _ensure_features_freshness(context: str) -> dict[str, Any]:
        model_meta = _latest_model_meta()
        features_meta, features_meta_source = _load_features_meta()
        model_path = str(model_meta.get("model_path") or "").strip() or None
        model_feature_set = str(model_meta.get("feature_set") or "").strip().lower() or None
        model_feature_signature = str(model_meta.get("feature_signature") or "").strip() or None
        features_feature_set = str(features_meta.get("feature_set") or "").strip().lower() or None
        meta_feature_signature = str(features_meta.get("feature_signature") or "").strip() or None
        features_columns = features_meta.get("feature_columns")
        computed_features_signature = None
        if isinstance(features_columns, list):
            cleaned_cols = [str(c).strip() for c in features_columns if str(c).strip()]
            if cleaned_cols:
                computed_features_signature = compute_feature_signature(cleaned_cols)
        if (
            meta_feature_signature
            and computed_features_signature
            and meta_feature_signature != computed_features_signature
        ):
            LOG.warning(
                "[WARN] FEATURES_META_SIGNATURE_MISMATCH stored=%s computed=%s source=%s",
                meta_feature_signature,
                computed_features_signature,
                features_meta_source,
            )
        features_feature_signature = computed_features_signature or meta_feature_signature
        stale = False
        reason = "fresh"
        if not features_meta:
            stale = True
            reason = "features_meta_missing"
        elif not model_feature_signature:
            stale = True
            reason = "model_signature_missing"
        elif not features_feature_signature:
            stale = True
            reason = "features_signature_missing"
        elif (
            model_feature_signature
            and features_feature_signature
            and model_feature_signature != features_feature_signature
        ):
            stale = True
            reason = "feature_signature_mismatch"
        elif (
            model_feature_set and features_feature_set and model_feature_set != features_feature_set
        ):
            stale = True
            reason = "feature_set_mismatch"
        LOG.info(
            "[INFO] FEATURES_FRESHNESS stale=%s reason=%s model_feature_set=%s features_feature_set=%s model_feature_signature=%s features_feature_signature=%s",
            str(bool(stale)).lower(),
            reason,
            model_feature_set or None,
            features_feature_set or None,
            model_feature_signature or None,
            features_feature_signature or None,
        )
        return {
            "stale": bool(stale),
            "reason": reason,
            "model_feature_set": model_feature_set,
            "features_feature_set": features_feature_set,
            "model_feature_signature": model_feature_signature,
            "features_feature_signature": features_feature_signature,
            "features_meta_source": features_meta_source,
            "model_path": model_path,
            "context": context,
        }

    def _load_predictions_meta() -> tuple[dict[str, Any], str]:
        if db.db_enabled():
            payload = db.load_ml_artifact_payload("predictions")
            if isinstance(payload, Mapping) and payload:
                return dict(payload), "db"
        meta_path = base_dir / "data" / "predictions" / "latest_meta.json"
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    return dict(payload), "fs"
            except Exception:
                LOG.warning("[WARN] PREDICTIONS_META_READ_FAILED path=%s", meta_path)
        return {}, "missing"

    def _has_cli_flag(tokens: Sequence[str], flag: str) -> bool:
        normalized = str(flag).strip()
        if not normalized:
            return False
        prefix = f"{normalized}="
        for raw in tokens:
            token = str(raw).strip()
            if not token:
                continue
            if token == normalized or token.startswith(prefix):
                return True
        return False

    def _run_labels_and_features_refresh(
        target_feature_set: str,
        *,
        context: str,
        log_prefix: str = "AUTO_REFRESH_FEATURES",
    ) -> int:
        labels_cmd: list[str] | None = None
        if db.db_enabled():
            labels_present = bool(db.fetch_latest_ml_artifact("labels"))
        else:
            labels_present = (
                _latest_by_glob(base_dir / "data" / "labels", "labels_*.csv") is not None
            )
        if not labels_present:
            bars_path = _resolve_labels_bars_path(args.labels_bars_path, base_dir)
            labels_cmd = [
                sys.executable,
                "-m",
                "scripts.label_generator",
                "--bars-path",
                str(bars_path),
                "--output-dir",
                str(base_dir / "data" / "labels"),
            ]
            LOG.info(
                "[INFO] %s enabled=true labels_missing=true -> running labels",
                log_prefix,
            )
            rc_labels = 0
            secs_labels = 0.0
            try:
                rc_labels, secs_labels = run_step(
                    "labels",
                    labels_cmd,
                    timeout=60 * 5,
                    env=_step_env("labels"),
                )
            except Exception as exc:  # pragma: no cover - defensive continue
                LOG.warning("%s labels error: %s", log_prefix, exc)
                rc_labels, secs_labels = 1, 0.0
            stage_times["labels"] = secs_labels
            step_rcs["labels"] = rc_labels
            if rc_labels:
                LOG.warning("[WARN] %s_LABELS_FAILED rc=%s", log_prefix, rc_labels)
        LOG.info(
            "[INFO] %s enabled=true stale=true -> running feature_generator feature_set=%s",
            log_prefix,
            target_feature_set,
        )
        rc_features = 0
        secs_features = 0.0
        features_env = dict(_step_env("features") or {})
        features_env["JBR_ML_FEATURE_SET"] = target_feature_set
        try:
            rc_features, secs_features = run_step(
                "features",
                [sys.executable, "-m", "scripts.feature_generator"],
                timeout=_features_timeout_secs(),
                env=features_env or None,
            )
        except Exception as exc:  # pragma: no cover - defensive continue
            LOG.warning("%s feature_generator error: %s", log_prefix, exc)
            rc_features, secs_features = 1, 0.0
        stage_times["features"] = secs_features
        step_rcs["features"] = rc_features
        LOG.info("[INFO] %s_DONE rc=%s", log_prefix, rc_features)
        _ensure_features_freshness(context)
        return int(rc_features)

    def _ensure_predictions_freshness(context: str) -> dict[str, Any]:
        predict_rc: int | None = None
        model_meta = _latest_model_meta()
        features_meta, _ = _load_features_meta()
        predictions_meta, predictions_meta_source = _load_predictions_meta()
        stale, reason, freshness_details = evaluate_predictions_freshness(
            model_meta,
            features_meta,
            predictions_meta,
            strict_meta=bool(strict_predictions_meta),
        )
        model_path = str(model_meta.get("model_path") or "")
        pred_model_path = str(predictions_meta.get("model_path") or "")
        latest_features_set = str(freshness_details.get("latest_features_feature_set") or "")
        latest_features_signature = str(
            freshness_details.get("latest_features_feature_signature") or ""
        )
        pred_features_set = str(freshness_details.get("predictions_feature_set") or "")
        pred_features_signature = str(freshness_details.get("predictions_feature_signature") or "")
        pred_compatible = freshness_details.get("pred_compatible")
        pred_missing_frac = freshness_details.get("pred_missing_frac")
        pred_compat_reason = str(freshness_details.get("pred_compat_reason") or "")
        LOG.info(
            "[INFO] PREDICTIONS_FRESHNESS stale=%s reason=%s model_path=%s pred_model_path=%s latest_features_set=%s latest_features_signature=%s pred_features_set=%s pred_features_signature=%s pred_compatible=%s pred_missing_frac=%s pred_compat_reason=%s",
            str(bool(stale)).lower(),
            reason,
            model_path or None,
            pred_model_path or None,
            latest_features_set or None,
            latest_features_signature or None,
            pred_features_set or None,
            pred_features_signature or None,
            pred_compatible,
            pred_missing_frac,
            pred_compat_reason or None,
        )
        if stale:
            LOG.warning("[WARN] PREDICTIONS_STALE reason=%s suggestion=run ranker_predict", reason)
            if auto_refresh_predictions and not freshness_state.get("refresh_attempted"):
                freshness_state["refresh_attempted"] = True
                features_freshness = _ensure_features_freshness(context)
                refresh_due_to_prediction_feature_mismatch = any(
                    tag in str(reason or "")
                    for tag in (
                        "pred_feature_set_mismatch",
                        "pred_feature_signature_mismatch",
                        "features_meta_missing",
                    )
                )
                if auto_refresh_features and (
                    bool(features_freshness.get("stale"))
                    or refresh_due_to_prediction_feature_mismatch
                ):
                    target_feature_set = (
                        str(features_freshness.get("model_feature_set") or "").strip().lower()
                    )
                    LOG.info(
                        "[INFO] AUTO_REFRESH_FEATURES_MODEL_CONTEXT model_path=%s model_feature_set=%s model_feature_signature=%s",
                        features_freshness.get("model_path"),
                        target_feature_set or None,
                        features_freshness.get("model_feature_signature"),
                    )
                    if target_feature_set not in {"v1", "v2"}:
                        LOG.warning(
                            "[WARN] AUTO_REFRESH_FEATURES_SKIPPED reason=model_feature_set_missing"
                        )
                    else:
                        _run_labels_and_features_refresh(
                            target_feature_set,
                            context=context,
                        )
                LOG.info(
                    "[INFO] AUTO_REFRESH_PREDICTIONS enabled=true stale=true -> running ranker_predict"
                )
                predict_timeout, _ = _ranker_predict_timeout_config()
                cmd = [sys.executable, "-m", "scripts.ranker_predict"]
                if extras["ranker_predict"]:
                    cmd.extend(extras["ranker_predict"])
                if strict_auto_refresh_predictions:
                    LOG.info(
                        "[INFO] STRICT_AUTO_REFRESH_PREDICTIONS enabled=true max_missing_feature_fraction=0.2"
                    )
                    if not _has_cli_flag(cmd, "--strict-feature-match"):
                        cmd.extend(["--strict-feature-match", "true"])
                    if not _has_cli_flag(cmd, "--max-missing-feature-fraction"):
                        cmd.extend(["--max-missing-feature-fraction", "0.2"])
                rc_predict = 0
                secs = 0.0
                try:
                    rc_predict, secs = run_step(
                        "ranker_predict",
                        cmd,
                        timeout=predict_timeout,
                        env=_step_env("ranker_predict"),
                    )
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("AUTO_REFRESH_PREDICTIONS ranker_predict error: %s", exc)
                    rc_predict, secs = 1, 0.0
                stage_times["ranker_predict"] = secs
                step_rcs["ranker_predict"] = rc_predict
                predict_rc = rc_predict
                calibrated, method = _ranker_predict_score_source_from_log(base_dir)
                LOG.info(
                    "[INFO] RANKER_PREDICT rc=%s calibrated=%s method=%s predictions_source=%s",
                    rc_predict,
                    calibrated,
                    method,
                    _predictions_source_state(base_dir),
                )
                LOG.info("[INFO] AUTO_REFRESH_PREDICTIONS_DONE rc=%s", rc_predict)
                model_meta = _latest_model_meta()
                features_meta, _ = _load_features_meta()
                predictions_meta, predictions_meta_source = _load_predictions_meta()
                stale, reason, freshness_details = evaluate_predictions_freshness(
                    model_meta,
                    features_meta,
                    predictions_meta,
                    strict_meta=bool(strict_predictions_meta),
                )
                model_path = str(model_meta.get("model_path") or "")
                pred_model_path = str(predictions_meta.get("model_path") or "")
                latest_features_set = str(
                    freshness_details.get("latest_features_feature_set") or ""
                )
                latest_features_signature = str(
                    freshness_details.get("latest_features_feature_signature") or ""
                )
                pred_features_set = str(freshness_details.get("predictions_feature_set") or "")
                pred_features_signature = str(
                    freshness_details.get("predictions_feature_signature") or ""
                )
                pred_compatible = freshness_details.get("pred_compatible")
                pred_missing_frac = freshness_details.get("pred_missing_frac")
                pred_compat_reason = str(freshness_details.get("pred_compat_reason") or "")
                LOG.info(
                    "[INFO] PREDICTIONS_FRESHNESS stale=%s reason=%s model_path=%s pred_model_path=%s latest_features_set=%s latest_features_signature=%s pred_features_set=%s pred_features_signature=%s pred_compatible=%s pred_missing_frac=%s pred_compat_reason=%s",
                    str(bool(stale)).lower(),
                    reason,
                    model_path or None,
                    pred_model_path or None,
                    latest_features_set or None,
                    latest_features_signature or None,
                    pred_features_set or None,
                    pred_features_signature or None,
                    pred_compatible,
                    pred_missing_frac,
                    pred_compat_reason or None,
                )
                if stale and any(
                    token in str(reason or "")
                    for token in (
                        "pred_feature_incompatible",
                        "pred_feature_compat_missing",
                        "pred_feature_set_missing",
                        "pred_feature_signature_missing",
                        "pred_model_meta_missing",
                    )
                ):
                    LOG.warning(
                        "[WARN] AUTO_REFRESH_PREDICTIONS_INEFFECTIVE reason=%s suggestion=enable_strict_auto_refresh_or_refresh_features",
                        reason,
                    )
        if predict_rc is None:
            rc_value = step_rcs.get("ranker_predict")
            if isinstance(rc_value, int):
                predict_rc = rc_value
        return {
            "stale": bool(stale),
            "reason": reason,
            "model_path": model_path or None,
            "pred_model_path": pred_model_path or None,
            "predictions_meta_source": predictions_meta_source,
            "latest_features_set": latest_features_set or None,
            "latest_features_signature": latest_features_signature or None,
            "pred_features_set": pred_features_set or None,
            "pred_features_signature": pred_features_signature or None,
            "pred_compatible": pred_compatible,
            "pred_missing_frac": pred_missing_frac,
            "pred_compat_reason": pred_compat_reason or None,
            "predict_rc": predict_rc,
            "context": context,
        }

    def _run_candidate_scoped_prediction_refresh(candidate_symbols: Sequence[str]) -> int:
        normalized_symbols = [
            _coerce_symbol(symbol) for symbol in candidate_symbols if _coerce_symbol(symbol)
        ]
        if not normalized_symbols:
            return 1
        unique_symbols = sorted(set(normalized_symbols))
        features_freshness = _ensure_features_freshness("candidate_scoped_refresh")
        if bool(features_freshness.get("stale")):
            target_feature_set = (
                str(features_freshness.get("model_feature_set") or "").strip().lower()
            )
            LOG.info(
                "[INFO] AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_FEATURES_MODEL_CONTEXT model_path=%s model_feature_set=%s model_feature_signature=%s",
                features_freshness.get("model_path"),
                target_feature_set or None,
                features_freshness.get("model_feature_signature"),
            )
            if target_feature_set in {"v1", "v2"}:
                rc_features = _run_labels_and_features_refresh(
                    target_feature_set,
                    context="candidate_scoped_refresh",
                    log_prefix="AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_FEATURES",
                )
                if rc_features != 0:
                    return int(rc_features)
            else:
                LOG.warning(
                    "[WARN] AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_FEATURES_SKIPPED reason=model_feature_set_missing"
                )
                return 1

        symbols_dir = base_dir / "data" / "tmp"
        symbols_dir.mkdir(parents=True, exist_ok=True)
        symbols_path = symbols_dir / f"candidate_symbols_{int(time.time())}.txt"
        symbols_path.write_text("\n".join(unique_symbols) + "\n", encoding="utf-8")

        predict_timeout, _ = _ranker_predict_timeout_config()
        cmd = [sys.executable, "-m", "scripts.ranker_predict"]
        if extras["ranker_predict"]:
            cmd.extend(extras["ranker_predict"])
        if strict_auto_refresh_predictions:
            LOG.info(
                "[INFO] STRICT_AUTO_REFRESH_PREDICTIONS enabled=true max_missing_feature_fraction=0.2"
            )
            if not _has_cli_flag(cmd, "--strict-feature-match"):
                cmd.extend(["--strict-feature-match", "true"])
            if not _has_cli_flag(cmd, "--max-missing-feature-fraction"):
                cmd.extend(["--max-missing-feature-fraction", "0.2"])

        predict_env = dict(_step_env("ranker_predict") or {})
        predict_env["JBR_PREDICT_SYMBOLS_PATH"] = str(symbols_path)
        rc_predict = 0
        secs_predict = 0.0
        try:
            rc_predict, secs_predict = run_step(
                "ranker_predict",
                cmd,
                timeout=predict_timeout,
                env=predict_env or None,
            )
        except Exception as exc:  # pragma: no cover - defensive continue
            LOG.warning("AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES ranker_predict error: %s", exc)
            rc_predict, secs_predict = 1, 0.0
        finally:
            try:
                symbols_path.unlink(missing_ok=True)
            except Exception:
                LOG.debug("AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES symbol temp cleanup failed")
        stage_times["ranker_predict"] = secs_predict
        step_rcs["ranker_predict"] = rc_predict
        calibrated, method = _ranker_predict_score_source_from_log(base_dir)
        LOG.info(
            "[INFO] RANKER_PREDICT rc=%s calibrated=%s method=%s predictions_source=%s",
            rc_predict,
            calibrated,
            method,
            _predictions_source_state(base_dir),
        )
        return int(rc_predict)

    LOG.info(
        "[INFO] PIPELINE_ARGS screener=%s backtest=%s metrics=%s ranker_predict=%s ranker_eval=%s ranker_walkforward=%s ranker_strategy_eval=%s ranker_monitor=%s ranker_recalibrate=%s ranker_autotune=%s ranker_autoremediate=%s ranker_trade_attribution=%s",
        extras["screener"],
        extras["backtest"],
        extras["metrics"],
        extras["ranker_predict"],
        extras["ranker_eval"],
        extras["ranker_walkforward"],
        extras["ranker_strategy_eval"],
        extras["ranker_monitor"],
        extras["ranker_recalibrate"],
        extras["ranker_autotune"],
        extras["ranker_autoremediate"],
        extras["ranker_trade_attribution"],
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
    ml_health_summary: dict[str, Any] | None = None
    model_score_coverage_summary: dict[str, Any] | None = None

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
            LOG.info("[INFO] START labels bars_path=%s", bars_path)
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
                LOG.info("[INFO] LABELS_START bars_path=%s", bars_path)
                try:
                    rc_labels, secs = run_step(
                        "labels",
                        cmd,
                        timeout=60 * 5,
                        env=_step_env("labels"),
                    )
                    try:
                        labels_path = _detect_new_labels_file(labels_output_dir, labels_before)
                        labels_rows = _count_csv_lines(labels_path) if labels_path else 0
                    except Exception:
                        LOG.exception("LABELS_DETECT_FAILED output_dir=%s", labels_output_dir)
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

        if "ranker_recalibrate" in steps:
            current_step = "ranker_recalibrate"
            cmd = [sys.executable, "-m", "scripts.ranker_recalibrate"]
            if extras["ranker_recalibrate"]:
                cmd.extend(extras["ranker_recalibrate"])
            rc_recalibrate = 0
            secs = 0.0
            try:
                rc_recalibrate, secs = run_step(
                    "ranker_recalibrate",
                    cmd,
                    timeout=60 * 10,
                    env=_step_env("ranker_recalibrate"),
                )
            except Exception as exc:  # pragma: no cover - defensive continue
                LOG.warning("RANKER_RECALIBRATE error (continuing): %s", exc)
                rc_recalibrate, secs = 1, 0.0
            stage_times["ranker_recalibrate"] = secs
            step_rcs["ranker_recalibrate"] = rc_recalibrate
            if rc_recalibrate:
                LOG.warning("[WARN] RANKER_RECALIBRATE_FAILED rc=%s (continuing)", rc_recalibrate)
            else:
                method = "unknown"
                model_path = ""
                if db.db_enabled():
                    recal_payload = db.load_ml_artifact_payload("ranker_recalibrate")
                    method = str(recal_payload.get("method") or "unknown")
                    model_path = str(recal_payload.get("model_path") or "")
                else:
                    summary_path = _latest_by_glob(DATA_DIR / "models", "ranker_summary_*.json")
                    summary_payload = _read_json(summary_path) if summary_path else {}
                    calibration_block = summary_payload.get("calibration")
                    if isinstance(calibration_block, Mapping):
                        method = str(
                            calibration_block.get("method")
                            or calibration_block.get("requested_method")
                            or "unknown"
                        )
                    else:
                        method = str(summary_payload.get("method") or "unknown")
                    model_path = str(summary_payload.get("model_path") or "")
                if not model_path:
                    latest_model = _latest_by_glob(DATA_DIR / "models", "ranker_*.pkl")
                    model_path = str(latest_model) if latest_model else "unknown"
                LOG.info(
                    "[INFO] RANKER_RECALIBRATE rc=%s method=%s model_path=%s",
                    rc_recalibrate,
                    method,
                    model_path,
                )

        if "ranker_predict" in steps:
            current_step = "ranker_predict"
            cmd = [sys.executable, "-m", "scripts.ranker_predict"]
            if extras["ranker_predict"]:
                cmd.extend(extras["ranker_predict"])
            predict_timeout, _ = _ranker_predict_timeout_config()
            rc_predict = 0
            secs = 0.0
            try:
                rc_predict, secs = run_step(
                    "ranker_predict",
                    cmd,
                    timeout=predict_timeout,
                    env=_step_env("ranker_predict"),
                )
            except Exception as exc:  # pragma: no cover - defensive continue
                LOG.warning("RANKER_PREDICT error (continuing): %s", exc)
                rc_predict, secs = 1, 0.0
            stage_times["ranker_predict"] = secs
            step_rcs["ranker_predict"] = rc_predict
            calibrated, method = _ranker_predict_score_source_from_log(base_dir)
            LOG.info(
                "[INFO] RANKER_PREDICT rc=%s calibrated=%s method=%s predictions_source=%s",
                rc_predict,
                calibrated,
                method,
                _predictions_source_state(base_dir),
            )
            if rc_predict:
                LOG.warning(
                    "[WARN] RANKER_PREDICT_FAILED rc=%s timeout_secs=%s predictions_source=%s",
                    rc_predict,
                    predict_timeout,
                    _predictions_source_state(base_dir),
                )

        if db.db_enabled() and (
            "ranker_eval" in steps or getattr(args, "enrich_candidates_with_ranker", False)
        ):
            # Let explicit predict step or freshness manager own prediction refresh.
            skip_internal_predict = (
                "ranker_predict" in steps or auto_refresh_predictions or ml_health_guard_enabled
            )
            if skip_internal_predict:
                if "ranker_predict" in steps:
                    LOG.info("[INFO] RANKER_PIPELINE_SKIPPED reason=explicit_ranker_predict_step")
                elif auto_refresh_predictions:
                    LOG.info("[INFO] RANKER_PIPELINE_SKIPPED reason=freshness_manager_auto_refresh")
                else:
                    LOG.info(
                        "[INFO] RANKER_PIPELINE_SKIPPED reason=guard_managed_prediction_freshness"
                    )
            else:
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
                    features_timeout = _features_timeout_secs()
                    LOG.info("[INFO] FEATURES_TIMEOUT secs=%s", features_timeout)
                    rc_features, secs = run_step(
                        "features",
                        [sys.executable, "-m", "scripts.feature_generator"],
                        timeout=features_timeout,
                        env=_step_env("features"),
                    )
                    stage_times["features"] = secs
                    if rc_features:
                        LOG.warning("[WARN] FEATURES_FAILED rc=%s (continuing)", rc_features)
                    else:
                        model_path = _latest_by_glob(base_dir / "data" / "models", "ranker_*.pkl")
                        if model_path is None:
                            LOG.info("[INFO] RANKER_TRAIN_START reason=model_missing")
                            rc_train, secs = run_step(
                                "ranker_train",
                                [sys.executable, "-m", "scripts.ranker_train"],
                                timeout=60 * 5,
                                env=_step_env("ranker_train"),
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
                            predict_timeout, predict_timeout_source = (
                                _ranker_predict_timeout_config()
                            )
                            LOG.info(
                                "[INFO] RANKER_PREDICT_TIMEOUT secs=%s source=%s",
                                predict_timeout,
                                predict_timeout_source,
                            )
                            LOG.info("[INFO] PREDICTIONS_START model=%s", model_path.name)
                            rc_predict, secs = run_step(
                                "ranker_predict",
                                [sys.executable, "-m", "scripts.ranker_predict"],
                                timeout=predict_timeout,
                                env=_step_env("ranker_predict"),
                            )
                            stage_times["ranker_predict"] = secs
                            if rc_predict:
                                LOG.warning(
                                    "[WARN] RANKER_PREDICT_FAILED rc=%s timeout_secs=%s predictions_source=%s",
                                    rc_predict,
                                    predict_timeout,
                                    _predictions_source_state(base_dir),
                                )
                                LOG.warning(
                                    "[WARN] PREDICTIONS_FAILED rc=%s (continuing)",
                                    rc_predict,
                                )
                            calibrated, method = _ranker_predict_score_source_from_log(base_dir)
                            LOG.info(
                                "[INFO] RANKER_PREDICT rc=%s calibrated=%s method=%s predictions_source=%s",
                                rc_predict,
                                calibrated,
                                method,
                                _predictions_source_state(base_dir),
                            )
        if "ranker_eval" in steps:
            current_step = "ranker_eval"
            eval_timeout, eval_timeout_source = _ranker_eval_timeout_config()
            LOG.info(
                "[INFO] RANKER_EVAL_TIMEOUT secs=%s source=%s",
                eval_timeout,
                eval_timeout_source,
            )
            _ensure_predictions_freshness("ranker_eval")
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
                        rc_eval, secs = run_step(
                            "ranker_eval",
                            cmd,
                            timeout=eval_timeout,
                            env=_step_env("ranker_eval"),
                        )
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_EVAL error (continuing): %s", exc)
                        rc_eval, secs = 1, 0.0
                    stage_times["ranker_eval"] = secs
                    step_rcs["ranker_eval"] = rc_eval
                    if rc_eval:
                        LOG.warning(
                            "[WARN] RANKER_EVAL_FAILED rc=%s timeout_secs=%s eval_source=%s",
                            rc_eval,
                            eval_timeout,
                            _ranker_eval_source_state(base_dir),
                        )
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
                    rc_eval, secs = run_step(
                        "ranker_eval",
                        cmd,
                        timeout=eval_timeout,
                        env=_step_env("ranker_eval"),
                    )
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_EVAL error (continuing): %s", exc)
                    rc_eval, secs = 1, 0.0
                stage_times["ranker_eval"] = secs
                step_rcs["ranker_eval"] = rc_eval
                if rc_eval:
                    LOG.warning(
                        "[WARN] RANKER_EVAL_FAILED rc=%s timeout_secs=%s eval_source=%s",
                        rc_eval,
                        eval_timeout,
                        _ranker_eval_source_state(base_dir),
                    )
                    LOG.warning("[WARN] RANKER_EVAL_FAILED rc=%s (continuing)", rc_eval)
                else:
                    payload = _read_json(DATA_DIR / "ranker_eval" / "latest.json")
                    LOG.info(
                        "[INFO] RANKER_EVAL sample_size=%s deciles=%s",
                        int(payload.get("sample_size", 0) or 0),
                        len(payload.get("deciles") or []),
                    )

        if "ranker_walkforward" in steps:
            current_step = "ranker_walkforward"
            if db.db_enabled():
                missing_wf: list[str] = []
                if not db.fetch_latest_ml_artifact("features"):
                    missing_wf.append("features")
                if not db.fetch_latest_ml_artifact("labels"):
                    missing_wf.append("labels")
                if missing_wf:
                    LOG.warning(
                        "[WARN] RANKER_WALKFORWARD_SKIPPED reason=missing_artifacts artifacts=%s",
                        ",".join(missing_wf),
                    )
                    stage_times["ranker_walkforward"] = 0.0
                    step_rcs["ranker_walkforward"] = 0
                else:
                    cmd = [sys.executable, "-m", "scripts.ranker_walkforward"]
                    if extras["ranker_walkforward"]:
                        cmd.extend(extras["ranker_walkforward"])
                    rc_wf = 0
                    secs = 0.0
                    try:
                        rc_wf, secs = run_step(
                            "ranker_walkforward",
                            cmd,
                            timeout=60 * 5,
                            env=_step_env("ranker_walkforward"),
                        )
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_WALKFORWARD error (continuing): %s", exc)
                        rc_wf, secs = 1, 0.0
                    stage_times["ranker_walkforward"] = secs
                    step_rcs["ranker_walkforward"] = rc_wf
                    if rc_wf:
                        LOG.warning("[WARN] RANKER_WALKFORWARD_FAILED rc=%s (continuing)", rc_wf)
                    else:
                        wf_payload = db.load_ml_artifact_payload("ranker_walkforward")
                        LOG.info(
                            "[INFO] RANKER_WALKFORWARD folds=%s sample_size_total=%s",
                            int(wf_payload.get("folds_count", 0) or 0),
                            int(wf_payload.get("sample_size_total", 0) or 0),
                        )
            else:
                cmd = [sys.executable, "-m", "scripts.ranker_walkforward"]
                if extras["ranker_walkforward"]:
                    cmd.extend(extras["ranker_walkforward"])
                rc_wf = 0
                secs = 0.0
                try:
                    rc_wf, secs = run_step(
                        "ranker_walkforward",
                        cmd,
                        timeout=60 * 5,
                        env=_step_env("ranker_walkforward"),
                    )
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_WALKFORWARD error (continuing): %s", exc)
                    rc_wf, secs = 1, 0.0
                stage_times["ranker_walkforward"] = secs
                step_rcs["ranker_walkforward"] = rc_wf
                if rc_wf:
                    LOG.warning("[WARN] RANKER_WALKFORWARD_FAILED rc=%s (continuing)", rc_wf)
                else:
                    wf_payload = _read_json(DATA_DIR / "ranker_walkforward" / "latest.json")
                    LOG.info(
                        "[INFO] RANKER_WALKFORWARD folds=%s sample_size_total=%s",
                        int(wf_payload.get("folds_count", 0) or 0),
                        int(wf_payload.get("sample_size_total", 0) or 0),
                    )

        if "ranker_strategy_eval" in steps:
            current_step = "ranker_strategy_eval"
            if db.db_enabled():
                if not db.fetch_latest_ml_artifact("ranker_oos_predictions"):
                    LOG.warning(
                        "[WARN] RANKER_STRATEGY_EVAL_SKIPPED reason=missing_artifacts artifacts=ranker_oos_predictions"
                    )
                    stage_times["ranker_strategy_eval"] = 0.0
                    step_rcs["ranker_strategy_eval"] = 0
                else:
                    cmd = [sys.executable, "-m", "scripts.ranker_strategy_eval"]
                    if extras["ranker_strategy_eval"]:
                        cmd.extend(extras["ranker_strategy_eval"])
                    rc_strategy = 0
                    secs = 0.0
                    try:
                        rc_strategy, secs = run_step(
                            "ranker_strategy_eval",
                            cmd,
                            timeout=60 * 5,
                            env=_step_env("ranker_strategy_eval"),
                        )
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_STRATEGY_EVAL error (continuing): %s", exc)
                        rc_strategy, secs = 1, 0.0
                    stage_times["ranker_strategy_eval"] = secs
                    step_rcs["ranker_strategy_eval"] = rc_strategy
                    if rc_strategy:
                        LOG.warning(
                            "[WARN] RANKER_STRATEGY_EVAL_FAILED rc=%s (continuing)",
                            rc_strategy,
                        )
                    else:
                        strategy_payload = db.load_ml_artifact_payload("ranker_strategy_eval")
                        strategy_metrics = strategy_payload.get("metrics") or {}
                        LOG.info(
                            "[INFO] RANKER_STRATEGY_EVAL periods=%s cagr=%s sharpe=%s max_dd=%s",
                            int(strategy_metrics.get("periods", 0) or 0),
                            strategy_metrics.get("cagr"),
                            strategy_metrics.get("sharpe"),
                            strategy_metrics.get("max_drawdown"),
                        )
            else:
                cmd = [sys.executable, "-m", "scripts.ranker_strategy_eval"]
                if extras["ranker_strategy_eval"]:
                    cmd.extend(extras["ranker_strategy_eval"])
                rc_strategy = 0
                secs = 0.0
                try:
                    rc_strategy, secs = run_step(
                        "ranker_strategy_eval",
                        cmd,
                        timeout=60 * 5,
                        env=_step_env("ranker_strategy_eval"),
                    )
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_STRATEGY_EVAL error (continuing): %s", exc)
                    rc_strategy, secs = 1, 0.0
                stage_times["ranker_strategy_eval"] = secs
                step_rcs["ranker_strategy_eval"] = rc_strategy
                if rc_strategy:
                    LOG.warning(
                        "[WARN] RANKER_STRATEGY_EVAL_FAILED rc=%s (continuing)", rc_strategy
                    )
                else:
                    strategy_payload = _read_json(DATA_DIR / "ranker_strategy_eval" / "latest.json")
                    strategy_metrics = strategy_payload.get("metrics") or {}
                    LOG.info(
                        "[INFO] RANKER_STRATEGY_EVAL periods=%s cagr=%s sharpe=%s max_dd=%s",
                        int(strategy_metrics.get("periods", 0) or 0),
                        strategy_metrics.get("cagr"),
                        strategy_metrics.get("sharpe"),
                        strategy_metrics.get("max_drawdown"),
                    )

        if "ranker_monitor" in steps:
            current_step = "ranker_monitor"
            _ensure_predictions_freshness("ranker_monitor")
            if db.db_enabled():
                if not db.fetch_latest_ml_artifact("ranker_oos_predictions"):
                    LOG.warning(
                        "[WARN] RANKER_MONITOR_SKIPPED reason=missing_artifacts artifacts=ranker_oos_predictions"
                    )
                    stage_times["ranker_monitor"] = 0.0
                    step_rcs["ranker_monitor"] = 0
                else:
                    cmd = [sys.executable, "-m", "scripts.ranker_monitor"]
                    if extras["ranker_monitor"]:
                        cmd.extend(extras["ranker_monitor"])
                    rc_monitor = 0
                    secs = 0.0
                    try:
                        rc_monitor, secs = run_step(
                            "ranker_monitor",
                            cmd,
                            timeout=60 * 3,
                            env=_step_env("ranker_monitor"),
                        )
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_MONITOR error (continuing): %s", exc)
                        rc_monitor, secs = 1, 0.0
                    stage_times["ranker_monitor"] = secs
                    step_rcs["ranker_monitor"] = rc_monitor
                    if rc_monitor:
                        LOG.warning("[WARN] RANKER_MONITOR_FAILED rc=%s (continuing)", rc_monitor)
                    else:
                        monitor_payload = db.load_ml_artifact_payload("ranker_monitor")
                        drift = monitor_payload.get("drift") or {}
                        recent_strategy = monitor_payload.get("recent_strategy") or {}
                        LOG.info(
                            "[INFO] RANKER_MONITOR psi_score=%s recent_sharpe=%s recommended_action=%s",
                            drift.get("max_psi"),
                            recent_strategy.get("sharpe"),
                            monitor_payload.get("recommended_action"),
                        )
            else:
                cmd = [sys.executable, "-m", "scripts.ranker_monitor"]
                if extras["ranker_monitor"]:
                    cmd.extend(extras["ranker_monitor"])
                rc_monitor = 0
                secs = 0.0
                try:
                    rc_monitor, secs = run_step(
                        "ranker_monitor",
                        cmd,
                        timeout=60 * 3,
                        env=_step_env("ranker_monitor"),
                    )
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_MONITOR error (continuing): %s", exc)
                    rc_monitor, secs = 1, 0.0
                stage_times["ranker_monitor"] = secs
                step_rcs["ranker_monitor"] = rc_monitor
                if rc_monitor:
                    LOG.warning("[WARN] RANKER_MONITOR_FAILED rc=%s (continuing)", rc_monitor)
                else:
                    monitor_payload = _read_json(DATA_DIR / "ranker_monitor" / "latest.json")
                    drift = monitor_payload.get("drift") or {}
                    recent_strategy = monitor_payload.get("recent_strategy") or {}
                    LOG.info(
                        "[INFO] RANKER_MONITOR psi_score=%s recent_sharpe=%s recommended_action=%s",
                        drift.get("max_psi"),
                        recent_strategy.get("sharpe"),
                        monitor_payload.get("recommended_action"),
                    )

        if "ranker_autoremediate" in steps:
            current_step = "ranker_autoremediate"
            cmd = [sys.executable, "-m", "scripts.ranker_autoremediate"]
            if extras["ranker_autoremediate"]:
                cmd.extend(extras["ranker_autoremediate"])
            rc_autoremediate = 0
            secs = 0.0
            try:
                rc_autoremediate, secs = run_step(
                    "ranker_autoremediate",
                    cmd,
                    timeout=60 * 20,
                    env=_step_env("ranker_autoremediate"),
                )
            except Exception as exc:  # pragma: no cover - defensive continue
                LOG.warning("RANKER_AUTOREMEDIATE error (continuing): %s", exc)
                rc_autoremediate, secs = 1, 0.0
            stage_times["ranker_autoremediate"] = secs
            step_rcs["ranker_autoremediate"] = rc_autoremediate
            if rc_autoremediate:
                LOG.warning(
                    "[WARN] RANKER_AUTOREMEDIATE_FAILED rc=%s (continuing)",
                    rc_autoremediate,
                )
            else:
                if db.db_enabled():
                    autoremediate_payload = db.load_ml_artifact_payload("ranker_autoremediate")
                else:
                    autoremediate_payload = _read_json(
                        DATA_DIR / "ranker_autoremediate" / "latest.json"
                    )
                decision_block = autoremediate_payload.get("decision") or {}
                champion_block = autoremediate_payload.get("champion") or {}
                LOG.info(
                    "[INFO] RANKER_AUTOREMEDIATE decision=%s executed=%s champion_status=%s",
                    decision_block.get("decision"),
                    autoremediate_payload.get("executed"),
                    champion_block.get("champion_status"),
                )

        if "ranker_trade_attribution" in steps:
            current_step = "ranker_trade_attribution"
            cmd = [sys.executable, "-m", "scripts.ranker_trade_attribution"]
            if extras["ranker_trade_attribution"]:
                cmd.extend(extras["ranker_trade_attribution"])
            rc_attr = 0
            secs = 0.0
            try:
                rc_attr, secs = run_step(
                    "ranker_trade_attribution",
                    cmd,
                    timeout=60 * 5,
                    env=_step_env("ranker_trade_attribution"),
                )
            except Exception as exc:  # pragma: no cover - defensive continue
                LOG.warning("RANKER_TRADE_ATTRIBUTION error (continuing): %s", exc)
                rc_attr, secs = 1, 0.0
            stage_times["ranker_trade_attribution"] = secs
            step_rcs["ranker_trade_attribution"] = rc_attr
            if rc_attr:
                LOG.warning("[WARN] RANKER_TRADE_ATTRIBUTION_FAILED rc=%s (continuing)", rc_attr)
            else:
                if db.db_enabled():
                    attribution_payload = db.load_ml_artifact_payload("ranker_trade_attribution")
                else:
                    attribution_payload = _read_json(
                        DATA_DIR / "ranker_trade_attribution" / "latest.json"
                    )
                summary = attribution_payload.get("summary") or {}
                LOG.info(
                    "[INFO] RANKER_TRADE_ATTRIBUTION trades_scored=%s win_rate=%s brier=%s unmatched=%s",
                    summary.get("trades_scored"),
                    summary.get("win_rate_scored"),
                    summary.get("brier"),
                    summary.get("trades_unmatched"),
                )

        if "ranker_autotune" in steps:
            current_step = "ranker_autotune"
            if db.db_enabled():
                if not db.fetch_latest_ml_artifact("daily_bars"):
                    LOG.warning(
                        "[WARN] RANKER_AUTOTUNE_SKIPPED reason=missing_artifacts artifacts=daily_bars"
                    )
                    stage_times["ranker_autotune"] = 0.0
                    step_rcs["ranker_autotune"] = 0
                else:
                    cmd = [sys.executable, "-m", "scripts.ranker_autotune"]
                    if extras["ranker_autotune"]:
                        cmd.extend(extras["ranker_autotune"])
                    rc_autotune = 0
                    secs = 0.0
                    try:
                        rc_autotune, secs = run_step("ranker_autotune", cmd, timeout=60 * 20)
                    except Exception as exc:  # pragma: no cover - defensive continue
                        LOG.warning("RANKER_AUTOTUNE error (continuing): %s", exc)
                        rc_autotune, secs = 1, 0.0
                    stage_times["ranker_autotune"] = secs
                    step_rcs["ranker_autotune"] = rc_autotune
                    if rc_autotune:
                        LOG.warning("[WARN] RANKER_AUTOTUNE_FAILED rc=%s (continuing)", rc_autotune)
                    else:
                        autotune_payload = db.load_ml_artifact_payload("ranker_autotune")
                        best = autotune_payload.get("best_config") or {}
                        LOG.info(
                            "[INFO] RANKER_AUTOTUNE best_sharpe=%s best_cagr=%s best_top_k=%s best_cost_bps=%s",
                            best.get("sharpe"),
                            best.get("cagr"),
                            best.get("top_k"),
                            best.get("cost_bps"),
                        )
            else:
                cmd = [sys.executable, "-m", "scripts.ranker_autotune"]
                if extras["ranker_autotune"]:
                    cmd.extend(extras["ranker_autotune"])
                rc_autotune = 0
                secs = 0.0
                try:
                    rc_autotune, secs = run_step("ranker_autotune", cmd, timeout=60 * 20)
                except Exception as exc:  # pragma: no cover - defensive continue
                    LOG.warning("RANKER_AUTOTUNE error (continuing): %s", exc)
                    rc_autotune, secs = 1, 0.0
                stage_times["ranker_autotune"] = secs
                step_rcs["ranker_autotune"] = rc_autotune
                if rc_autotune:
                    LOG.warning("[WARN] RANKER_AUTOTUNE_FAILED rc=%s (continuing)", rc_autotune)
                else:
                    autotune_payload = _read_json(DATA_DIR / "ranker_autotune" / "latest.json")
                    best = autotune_payload.get("best_config") or {}
                    LOG.info(
                        "[INFO] RANKER_AUTOTUNE best_sharpe=%s best_cagr=%s best_top_k=%s best_cost_bps=%s",
                        best.get("sharpe"),
                        best.get("cagr"),
                        best.get("top_k"),
                        best.get("cost_bps"),
                    )

        if getattr(args, "enrich_candidates_with_ranker", False):
            enrichment_freshness = _ensure_predictions_freshness("enrichment")
            LOG.info(
                "[INFO] ML_HEALTH_GUARD enabled=%s mode=%s",
                "true" if ml_health_guard_enabled else "false",
                ml_health_guard_mode,
            )
            enrichment_blocked = False
            if ml_health_guard_enabled:
                health_status = load_latest_ml_health(
                    base_dir=base_dir,
                    logger=LOG,
                )
                decision = decide_ml_enrichment(
                    health_status,
                    mode=ml_health_guard_mode,
                    max_age_days=resolve_ml_health_max_age_days(),
                    pipeline_run_date=pipeline_run_date,
                    predictions_stale=bool(enrichment_freshness.get("stale")),
                    predictions_stale_reason=str(enrichment_freshness.get("reason") or ""),
                )
                reason_text = ",".join(decision.get("reasons") or []) or "none"
                LOG.info(
                    "[INFO] ML_ENRICHMENT_DECISION decision=%s mode=%s action=%s reason=%s psi_score=%s recent_sharpe=%s monitor_run_date=%s predictions_reason=%s",
                    decision.get("decision"),
                    decision.get("mode"),
                    decision.get("recommended_action"),
                    reason_text,
                    decision.get("psi_score"),
                    decision.get("recent_sharpe"),
                    decision.get("monitor_run_date"),
                    decision.get("predictions_stale_reason"),
                )
                ml_health_summary = {
                    "decision": decision.get("decision"),
                    "mode": decision.get("mode"),
                    "reasons": list(decision.get("reasons") or []),
                    "monitor_run_date": decision.get("monitor_run_date"),
                    "psi_score": decision.get("psi_score"),
                    "recent_sharpe": decision.get("recent_sharpe"),
                    "source": decision.get("source"),
                    "predictions_stale": bool(enrichment_freshness.get("stale")),
                    "predictions_stale_reason": decision.get("predictions_stale_reason"),
                }
                if decision.get("decision") == "block":
                    enrichment_blocked = True
                    LOG.warning("[WARN] ML_ENRICHMENT_BLOCKED reason=%s", reason_text)
                elif decision.get("decision") == "warn":
                    LOG.warning("[WARN] ML_ENRICHMENT_WARN reason=%s", reason_text)
                if _env_truthy(os.getenv("JBR_ML_HEALTH_ALERTS")) and decision.get("decision") in {
                    "warn",
                    "block",
                }:
                    try:
                        send_alert(
                            "JBRAVO ML health guard action",
                            {
                                "decision": decision.get("decision"),
                                "action": decision.get("recommended_action"),
                                "reason": reason_text,
                                "psi_score": decision.get("psi_score"),
                                "recent_sharpe": decision.get("recent_sharpe"),
                                "monitor_run_date": decision.get("monitor_run_date"),
                                "source": decision.get("source"),
                                "run_utc": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception as exc:  # pragma: no cover - alert best-effort
                        LOG.warning("[WARN] ML_HEALTH_ALERT_FAILED err=%s", exc)
            else:
                ml_health_summary = {
                    "decision": "allow",
                    "mode": ml_health_guard_mode,
                    "reasons": ["guard_disabled"],
                    "monitor_run_date": None,
                    "psi_score": None,
                    "recent_sharpe": None,
                    "source": "guard_disabled",
                    "predictions_stale": bool(enrichment_freshness.get("stale")),
                    "predictions_stale_reason": (
                        str(enrichment_freshness.get("reason") or "")
                        if enrichment_freshness.get("stale")
                        else None
                    ),
                }
                LOG.info(
                    "[INFO] ML_ENRICHMENT_DECISION decision=allow mode=%s action=none reason=guard_disabled psi_score=%s recent_sharpe=%s monitor_run_date=%s predictions_reason=%s",
                    ml_health_guard_mode,
                    None,
                    None,
                    None,
                    (
                        str(enrichment_freshness.get("reason") or "")
                        if enrichment_freshness.get("stale")
                        else None
                    ),
                )
            if not enrichment_blocked:
                pred_compatible_raw = enrichment_freshness.get("pred_compatible")
                if isinstance(pred_compatible_raw, bool):
                    pred_compatible = pred_compatible_raw
                elif isinstance(pred_compatible_raw, str):
                    lowered = pred_compatible_raw.strip().lower()
                    if lowered in {"true", "false"}:
                        pred_compatible = lowered == "true"
                    else:
                        pred_compatible = None
                else:
                    pred_compatible = None
                predict_rc_raw = enrichment_freshness.get("predict_rc")
                try:
                    predict_rc = int(predict_rc_raw) if predict_rc_raw is not None else None
                except Exception:
                    predict_rc = None
                freshness_stale = bool(enrichment_freshness.get("stale"))
                if freshness_stale or pred_compatible is False or (predict_rc not in (None, 0)):
                    enrichment_blocked = True
                    LOG.warning(
                        "[WARN] CANDIDATES_ENRICH_SKIPPED reason=predictions_stale_or_incompatible stale=%s pred_compatible=%s predict_rc=%s",
                        str(freshness_stale).lower(),
                        (
                            str(pred_compatible).lower()
                            if isinstance(pred_compatible, bool)
                            else "unknown"
                        ),
                        predict_rc,
                    )
            if not enrichment_blocked:
                enriched = _enrich_candidates_with_ranker(
                    base_dir=base_dir,
                    score_column=DEFAULT_RANKER_SCORE_COLUMN,
                    target_column=DEFAULT_RANKER_TARGET_COLUMN,
                    refresh_predictions_for_candidates=refresh_predictions_for_candidates,
                    refresh_predictions_callback=_run_candidate_scoped_prediction_refresh,
                )
                coverage_source = "db" if db.db_enabled() else "csv"
                coverage_frame = enriched
                if coverage_frame is None:
                    try:
                        coverage_frame = _load_latest_candidates_frame(base_dir)
                    except Exception:
                        LOG.warning(
                            "[WARN] MODEL_SCORE_COVERAGE_FALLBACK_FAILED reason=load_latest_candidates_error",
                            exc_info=True,
                        )
                        coverage_frame = pd.DataFrame()
                model_score_coverage_summary = _model_score_coverage_summary(
                    coverage_frame,
                    source=coverage_source,
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
        if isinstance(ml_health_summary, Mapping):
            metrics_payload["ml_health"] = dict(ml_health_summary)
        if isinstance(model_score_coverage_summary, Mapping):
            metrics_payload["model_score_coverage"] = dict(model_score_coverage_summary)
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
                    "end_utc": datetime.now(timezone.utc).isoformat(),
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
        if isinstance(ml_health_summary, Mapping):
            status_payload["ml_health"] = dict(ml_health_summary)
        if isinstance(model_score_coverage_summary, Mapping):
            status_payload["model_score_coverage"] = dict(model_score_coverage_summary)
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
            "bars_rows_total": int(summary.bars_rows_total or 0)
            if summary.bars_rows_total is not None
            else None,
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
        if isinstance(ml_health_summary, Mapping):
            summary_payload["ml_health"] = dict(ml_health_summary)
        if isinstance(model_score_coverage_summary, Mapping):
            summary_payload["model_score_coverage"] = dict(model_score_coverage_summary)
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
        should_raise = LOG.name != "pipeline" or os.environ.get(
            "JBR_PIPELINE_RAISE", ""
        ).lower() in {
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
