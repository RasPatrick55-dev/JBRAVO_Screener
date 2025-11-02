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
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Optional, Sequence
from types import SimpleNamespace

import pandas as pd

from scripts.fallback_candidates import CANONICAL_COLUMNS, build_latest_candidates, normalize_candidate_df
from scripts.utils.env import load_env
from utils import write_csv_atomic, atomic_write_bytes
from utils.telemetry import emit_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG = logging.getLogger("pipeline")
logger = LOG

_SPLIT_FLAG_MAP = {
    "--screener-args-split": "screener_args_split",
    "--backtest-args-split": "backtest_args_split",
    "--metrics-args-split": "metrics_args_split",
    "--exec-args-split": "exec_args_split",
}

_SUMMARY_RE = re.compile(
    r"PIPELINE_SUMMARY.*?symbols_in=(?P<symbols_in>\d+).*?"
    r"with_bars=(?P<symbols_with_bars>\d+).*?"
    r"rows=(?P<rows>\d+)"
    r"(?:.*?(?:bars?_rows(?:_total)?)=(?P<bars_rows_total>\d+))?"
)


def _refresh_logger() -> None:
    global logger
    logger = LOG
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
SCREENER_METRICS_PATH = DATA_DIR / "screener_metrics.json"
LATEST_CANDIDATES = DATA_DIR / "latest_candidates.csv"
TOP_CANDIDATES = DATA_DIR / "top_candidates.csv"
DEFAULT_WSGI_PATH = Path("/var/www/raspatrick_pythonanywhere_com_wsgi.py")
BASE_DIR = PROJECT_ROOT
copyfile = shutil.copyfile
emit = emit_event
LATEST_COLUMNS = list(CANONICAL_COLUMNS)
LATEST_HEADER = ",".join(LATEST_COLUMNS) + "\n"


def _record_health(stage: str) -> dict[str, Any]:  # pragma: no cover - legacy hook
    return {}


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


def _backfill_metrics_from_summary(
    metrics_path: Path, metrics: Mapping[str, Any] | None, summary: SimpleNamespace
) -> dict[str, Any]:
    """Backfill numeric KPIs in ``metrics`` using a fresh ``PIPELINE_SUMMARY``."""

    payload = dict(metrics or {})
    updated_fields: list[str] = []
    summary_map = {
        "symbols_in": getattr(summary, "symbols_in", None),
        "symbols_with_bars": getattr(summary, "with_bars", None),
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

    if df_scored is not None and not df_scored.empty and "symbol" in df_scored.columns:
        join_cols = [
            col
            for col in ("symbol", "close", "exchange", "volume", "adv20")
            if col in df_scored.columns
        ]
        if join_cols:
            joinable = df_scored[join_cols].drop_duplicates("symbol")
            base = base.merge(joinable, on="symbol", how="left", suffixes=("", "_scored"))
            for col in ("close", "exchange", "volume", "adv20"):
                scored_col = f"{col}_scored"
                if scored_col not in base.columns:
                    continue
                if col == "exchange":
                    existing = (
                        base[col].astype("string").fillna("") if col in base.columns else pd.Series("", index=base.index)
                    )
                    fallback = base[scored_col].astype("string").fillna("")
                    base[col] = existing.where(existing.str.strip() != "", fallback)
                else:
                    if col not in base.columns:
                        base[col] = pd.NA
                    series = base[col]
                    mask_missing = series.isna()
                    if col in {"volume", "adv20"}:
                        numeric_series = pd.to_numeric(series, errors="coerce")
                        mask_missing = numeric_series.isna()
                        base[col] = numeric_series
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


def _sync_top_candidates_to_latest(base_dir: Path | None = None) -> None:
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


def refresh_latest_candidates(base_dir: Path | None = None) -> pd.DataFrame:
    base = _resolve_base_dir(base_dir)
    data_dir = base / "data"
    top_path = data_dir / "top_candidates.csv"
    latest_path = data_dir / "latest_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    canonical = write_latest_candidates_canonical(base)
    if not canonical.empty and canonical["close"].notna().any():
        _write_refresh_metrics(metrics_path)
        return canonical
    if latest_path.exists() and latest_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(latest_path)
            if not existing.empty:
                _write_refresh_metrics(metrics_path)
                return existing
        except Exception:
            logger.exception("Failed reading existing latest_candidates.csv during refresh")
    frame, _ = build_latest_candidates(base)
    _write_refresh_metrics(metrics_path)
    return frame


def _maybe_fallback(base_dir: Path | None = None) -> int:
    base = _resolve_base_dir(base_dir)
    try:
        subprocess.check_call([sys.executable, "-m", "scripts.fallback_candidates"], cwd=base)
    except subprocess.CalledProcessError:  # pragma: no cover - legacy shim
        LOG.warning("FALLBACK invocation failed", exc_info=True)
    frame = refresh_latest_candidates(base)
    return int(len(frame.index))


def run_cmd(cmd: Sequence[str], name: str) -> int:
    try:
        subprocess.check_call(list(cmd), cwd=PROJECT_ROOT)
        return 0
    except subprocess.CalledProcessError as exc:  # pragma: no cover - legacy shim
        return exc.returncode


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
    if LOG.handlers:
        return

    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - pipeline - %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH)
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
        help="Comma-separated list of steps to run (default: screener,backtest,metrics)",
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
        "--exec-args",
        default=os.getenv("JBR_EXEC_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.execute_trades",
    )
    parser.add_argument(
        "--exec-args-split",
        nargs="*",
        default=None,
        help="Extra CLI arguments for scripts.execute_trades provided as separate tokens",
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
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    option_strings = set(parser._option_string_actions.keys())
    filtered_args, collected = _extract_split_tokens(raw_args, option_strings)
    namespace = parser.parse_args(filtered_args)
    for dest in _SPLIT_FLAG_MAP.values():
        tokens = collected.get(dest)
        setattr(namespace, dest, tokens if tokens else None)
    return namespace


def determine_steps(raw: Optional[str]) -> list[str]:
    default = "screener,backtest,metrics"
    target = raw or os.environ.get("PIPE_STEPS", default)
    steps = [part.strip().lower() for part in target.split(",") if part.strip()]
    normalized: list[str] = []
    for step in steps or default.split(","):
        if step == "exec":
            normalized.append("execute")
        else:
            normalized.append(step)
    return normalized


def run_step(name: str, cmd: Sequence[str], *, timeout: Optional[float] = None) -> tuple[int, float]:
    started = time.time()
    LOG.info("[INFO] START %s cmd=%s", name, shlex.join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("APCA_DATA_API_BASE_URL", "https://data.alpaca.markets")
    env.setdefault("ALPACA_DATA_FEED", "iex")
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
        text=False,
    )
    out: bytes = b""
    err: bytes = b""
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        LOG.error("STEP_TIMEOUT name=%s timeout=%s rc=%s", name, timeout, proc.returncode)
    if out:
        LOG.info("%s stdout:\n%s", name.upper(), out.decode(errors="replace")[-8000:])
    if err:
        LOG.info("%s stderr:\n%s", name.upper(), err.decode(errors="replace")[-8000:])
    elapsed = time.time() - started
    LOG.info("END %s rc=%s secs=%.1f", name, proc.returncode, elapsed)
    return proc.returncode, elapsed


def _run_step_metrics(cmd: Sequence[str], *, timeout: Optional[float] = None) -> tuple[int, float]:
    return run_step("metrics", cmd, timeout=timeout)


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


def _count_csv_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle)
    except Exception:
        return 0
    return max(total - 1, 0)


def write_complete_screener_metrics(base_dir: Path) -> dict[str, Any]:
    """Ensure ``screener_metrics.json`` contains integer KPIs even on fallback nights."""

    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_in": None,
        "symbols_with_bars": None,
        "bars_rows_total": None,
        "rows": 0,
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
                    payload["symbols_in"] = int(match.group("symbols_in"))
                    payload["symbols_with_bars"] = int(match.group("symbols_with_bars"))
                    payload["rows"] = int(match.group("rows"))
                    bars_total = match.group("bars_rows_total")
                    payload["bars_rows_total"] = int(bars_total) if bars_total else None
                break

    latest_candidates = data_dir / "latest_candidates.csv"
    top_candidates = data_dir / "top_candidates.csv"
    if payload.get("rows", 0) == 0:
        for candidate_path in (latest_candidates, top_candidates):
            rows = _count_csv_lines(candidate_path)
            if rows:
                payload["rows"] = rows
                break

    scored_candidates = data_dir / "scored_candidates.csv"
    if payload["symbols_in"] is None:
        rows = _count_csv_lines(scored_candidates)
        if rows or scored_candidates.exists():
            payload["symbols_in"] = rows

    if payload["symbols_with_bars"] is None and payload["symbols_in"] is not None:
        payload["symbols_with_bars"] = payload["symbols_in"]

    if payload["bars_rows_total"] is None:
        payload["bars_rows_total"] = payload["rows"]

    for key in ("symbols_in", "symbols_with_bars", "bars_rows_total"):
        if payload[key] is None:
            payload[key] = 0

    metrics_path = data_dir / "screener_metrics.json"
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(previous, Mapping) and previous.get("last_run_utc"):
                payload["last_run_utc"] = previous["last_run_utc"]
        except Exception:
            pass

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote screener_metrics.json: symbols_in=%s symbols_with_bars=%s rows=%s bars_rows_total=%s",
        payload["symbols_in"],
        payload["symbols_with_bars"],
        payload["rows"],
        payload["bars_rows_total"],
    )
    return payload


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
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _ensure_latest_headers() -> None:
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
    normalized["source"] = normalized.get("source", "").astype("string").fillna("")
    normalized.loc[normalized["source"].str.strip() == "", "source"] = source
    normalized = normalized[list(CANONICAL_COLUMNS)]
    write_csv_atomic(str(LATEST_CANDIDATES), normalized)
    return int(len(normalized.index))


def _write_refresh_metrics(metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
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
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")


def ensure_candidates(min_rows: int = 1) -> int:
    current = _count_rows(LATEST_CANDIDATES)
    if current >= min_rows:
        return current
    frame, _ = build_latest_candidates(PROJECT_ROOT, max_rows=max(1, min_rows))
    write_csv_atomic(str(TOP_CANDIDATES), frame)
    return int(len(frame.index))


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


def _split_or_string(exec_args: str, split_list: Optional[Sequence[str]]) -> list[str]:
    if split_list is not None and len(split_list) > 0:
        return [token for token in split_list if token != "--"]
    return _split_args(exec_args)


def sync_execute_metrics(base_dir: Path) -> dict[str, Any]:
    base = Path(base_dir)
    metrics_path = base / "data" / "execute_metrics.json"
    payload = _read_json(metrics_path)
    if not payload:
        logger.info("EXECUTE_METRICS_SYNC empty path=%s", metrics_path)
        return {}
    submitted = payload.get("orders_submitted")
    filled = payload.get("orders_filled")
    skips = payload.get("skips") if isinstance(payload.get("skips"), Mapping) else {}
    logger.info(
        "EXECUTE_METRICS_SYNC orders_submitted=%s orders_filled=%s skips=%s",
        submitted,
        filled,
        skips,
    )
    return payload


def main(argv: Optional[Iterable[str]] = None) -> int:
    _refresh_logger()
    loaded_files, missing_keys = load_env(REQUIRED_ENV_KEYS)
    configure_logging()
    files_repr = f"[{', '.join(loaded_files)}]" if loaded_files else "[]"
    LOG.info("[INFO] ENV_LOADED files=%s", files_repr)
    if missing_keys:
        LOG.error("[ERROR] ENV_MISSING_KEYS=%s", f"[{', '.join(missing_keys)}]")
        raise SystemExit(2)
    args = parse_args(argv)
    steps = determine_steps(args.steps)
    LOG.info("[INFO] PIPELINE_START steps=%s", ",".join(steps))

    _ensure_latest_headers()
    base_dir = _resolve_base_dir()

    extras = {
        "screener": _merge_split_args(args.screener_args, args.screener_args_split),
        "execute": _split_or_string(args.exec_args, args.exec_args_split),
        "backtest": _merge_split_args(args.backtest_args, args.backtest_args_split),
        "metrics": _merge_split_args(args.metrics_args, args.metrics_args_split),
    }

    step_rcs: dict[str, int] = {step: 0 for step in steps}

    LOG.info(
        "[INFO] PIPELINE_ARGS screener=%s execute=%s backtest=%s metrics=%s",
        extras["screener"],
        extras["execute"],
        extras["backtest"],
        extras["metrics"],
    )

    started = time.time()
    metrics: dict[str, Any] = {}
    symbols_in = 0
    symbols_with_bars = 0
    rows = 0
    stage_times: dict[str, float] = {}
    rc = 0

    metrics_rows: int | None = None
    latest_source: str | None = "unknown"
    error_info: dict[str, Any] | None = None
    try:
        if "screener" in steps:
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
            top_frame = _load_top_candidates()
            latest_source = "screener"
            latest_rows = 0
            if top_frame.empty:
                LOG.info("[INFO] FALLBACK_CHECK start origin=screener")
                frame, fallback_source = build_latest_candidates(PROJECT_ROOT, max_rows=1)
                write_csv_atomic(str(TOP_CANDIDATES), frame)
                fallback_rows = int(len(frame.index))
                rows = fallback_rows
                if metrics_rows is not None:
                    rows = metrics_rows
                else:
                    metrics_rows = fallback_rows
                try:
                    refreshed = refresh_latest_candidates()
                    if isinstance(refreshed, pd.DataFrame):
                        latest_rows = int(len(refreshed.index))
                    elif isinstance(refreshed, Mapping):
                        metrics = dict(refreshed)
                    base = _resolve_base_dir()
                    local_metrics_path = base / "data" / "screener_metrics.json"
                    metrics = _read_json(local_metrics_path) or metrics
                    if "rows" in metrics:
                        metrics_rows = int(metrics.get("rows") or 0)
                        rows = metrics_rows
                        symbols_in = int(metrics.get("symbols_in", symbols_in) or symbols_in)
                        symbols_with_bars = int(
                            metrics.get("symbols_with_bars", symbols_with_bars) or symbols_with_bars
                        )
                except Exception:  # pragma: no cover - defensive fallback refresh
                    LOG.debug("refresh_latest_candidates failed", exc_info=True)
                latest_source = "fallback"
                if str(fallback_source) in {"scored", "predictions"}:
                    latest_source = f"fallback:{fallback_source}"
                if not latest_rows:
                    latest_rows = fallback_rows
            else:
                rows = _write_latest_from_frame(top_frame, source="screener")
                if metrics_rows is None:
                    metrics_rows = rows
                latest_rows = rows
                latest_source = "screener"
            LOG.info(
                "[INFO] FALLBACK_CHECK rows_out=%d source=%s",
                latest_rows,
                latest_source,
            )
            _sync_top_candidates_to_latest(base_dir)
        else:
            LOG.info("[INFO] FALLBACK_CHECK start origin=latest")
            metrics = _read_json(SCREENER_METRICS_PATH)
            symbols_in = int(metrics.get("symbols_in", 0) or 0)
            symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
            if "rows" in metrics:
                try:
                    metrics_rows = int(metrics.get("rows") or 0)
                except Exception:
                    metrics_rows = 0
            rows = ensure_candidates(metrics_rows or 0)
            latest_source = "fallback"
            try:
                latest_frame = pd.read_csv(LATEST_CANDIDATES)
                if not latest_frame.empty and "source" in latest_frame.columns:
                    value = str(latest_frame.iloc[0]["source"] or "").strip()
                    if value:
                        latest_source = value
            except Exception:  # pragma: no cover - defensive read
                latest_source = "fallback"
            LOG.info("[INFO] FALLBACK_CHECK rows_out=%d source=%s", rows, latest_source)

        if "backtest" in steps:
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
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.metrics"]
            if extras["metrics"]:
                cmd.extend(extras["metrics"])
            rc_metrics = 0
            secs = 0.0
            try:
                rc_metrics, secs = _run_step_metrics(cmd, timeout=60 * 3)
            except FileNotFoundError:
                logger.warning("No trades_log.csv; metrics will write zero summary and continue.")
                rc_metrics, secs = 0, 0.0
            except Exception as e:
                logger.warning("METRICS non-fatal error: %s", e)
                rc_metrics, secs = 0, 0.0
            stage_times["metrics"] = secs
            step_rcs["metrics"] = rc_metrics
            if rc_metrics:
                logger.warning("[WARN] METRICS_STEP rc=%s (continuing)", rc_metrics)
            _sync_top_candidates_to_latest(base_dir)
        if "execute" in steps:
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.execute_trades"]
            exec_args = extras["execute"]
            if not any(arg.startswith("--time-window") for arg in exec_args):
                cmd.extend(["--time-window", "auto"])
            if exec_args:
                cmd.extend(exec_args)
            rc_exec, secs = run_step("execute", cmd, timeout=60 * 10)
            stage_times["execute"] = secs
            step_rcs["execute"] = rc_exec
            if rc_exec and not rc:
                rc = rc_exec
                if error_info is None:
                    error_info = {"step": "execute", "rc": int(rc_exec), "message": "step_failed"}
            if rc_exec == 0:
                try:
                    sync_execute_metrics(base_dir)
                except Exception:
                    logger.exception("EXEC_METRICS_SYNC failed (non-fatal)")
            else:
                logger.error("Execute step failed rc=%s", rc_exec)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("PIPELINE_FATAL: %s", exc)
        rc = 1
        error_info = {
            "step": "pipeline",
            "message": str(exc),
            "exception": exc.__class__.__name__,
        }
    finally:
        fetch_secs = _extract_timing(metrics, "fetch_secs")
        feature_secs = _extract_timing(metrics, "feature_secs")
        rank_secs = _extract_timing(metrics, "rank_secs")
        gate_secs = _extract_timing(metrics, "gate_secs")
        summary_rows = metrics_rows if metrics_rows is not None else rows
        bars_rows_total = None
        if isinstance(metrics, Mapping):
            for key in ("bars_rows_total", "bars_rows", "bar_rows_total", "bar_rows"):
                candidate = _coerce_optional_int(metrics.get(key))
                if candidate is not None:
                    bars_rows_total = candidate
                    break
        summary = SimpleNamespace(
            symbols_in=symbols_in,
            with_bars=symbols_with_bars,
            rows=summary_rows,
            bars_rows_total=bars_rows_total,
            source=latest_source,
        )
        t = SimpleNamespace(
            fetch=fetch_secs,
            features=feature_secs,
            rank=rank_secs,
            gates=gate_secs,
        )
        data_ok = bool((summary.rows or 0) > 0)
        trading_ok = rc == 0
        trading_status = 200 if trading_ok else 503
        data_status = 200 if data_ok else 204
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
        metrics_path = base_dir / "data" / "screener_metrics.json"
        metrics = _backfill_metrics_from_summary(metrics_path, metrics, summary)
        today = datetime.now(timezone.utc).date()
        todays_summary = _latest_summary_for_date(base_dir, today)
        if todays_summary is None:
            LOG.warning("[WARN] SUMMARY_TODAY_MISSING date=%s", today.isoformat())
        try:
            kpis = write_complete_screener_metrics(base_dir)
            LOG.info(
                "SCREENER_METRICS_SYNC symbols_in=%s with_bars=%s rows=%s bars_rows_total=%s",
                kpis.get("symbols_in"),
                kpis.get("symbols_with_bars"),
                kpis.get("rows"),
                kpis.get("bars_rows_total"),
            )
        except Exception:
            LOG.exception("Failed to backfill screener_metrics.json")
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
        status_payload["step_rcs"] = {k: int(v) for k, v in step_rcs.items()}
        if error_info:
            status_payload["error"] = dict(error_info)
        _write_status_json(base_dir, status_payload)
        logger.info(
            "[INFO] HEALTH trading_ok=%s data_ok=%s stage=end trading_status=%s data_status=%s",
            health.trading_ok,
            health.data_ok,
            health.trading_status,
            health.data_status,
        )
        duration = time.time() - started
        logger.info("[INFO] PIPELINE_END rc=%s duration=%.1fs", rc, duration)
        try:
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
