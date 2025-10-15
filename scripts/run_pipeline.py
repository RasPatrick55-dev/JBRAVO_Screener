import argparse
import csv
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from shutil import copyfile

import shlex

import pandas as pd

from scripts.fallback_candidates import (
    CANONICAL_COLUMNS,
    generate_candidates,
    normalize_candidate_df,
)
from scripts.health_check import run_health_check
from utils.env import (
    AlpacaCredentialsError,
    assert_alpaca_creds,
    load_env,
    write_auth_error_artifacts,
    write_metrics_summary_row,
)


LOG = logging.getLogger("pipeline")
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = pathlib.Path("logs") / "pipeline.log"
EVENTS_PATH = pathlib.Path("logs") / "execute_events.jsonl"
EXECUTE_METRICS_PATH = pathlib.Path("data") / "execute_metrics.json"
PIPELINE_METRICS_PATH = pathlib.Path("data") / "pipeline_metrics.json"

LATEST_COLUMNS = list(CANONICAL_COLUMNS)
LATEST_HEADER = ",".join(LATEST_COLUMNS) + "\n"


def configure_logging() -> None:
    """Configure console and file logging for the pipeline."""
    if getattr(configure_logging, "_configured", False):  # pragma: no cover - defensive
        return

    fmt = "%(asctime)s - pipeline - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(logging.Formatter(fmt))
    LOG.addHandler(file_handler)
    LOG.propagate = True

    configure_logging._configured = True  # type: ignore[attr-defined]


def run_cmd(cmd: list[str], name: str) -> int:
    """Run ``cmd`` while logging start/end markers and outputs."""
    LOG.info("[INFO] START %s cmd=%s", name, " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    LOG.info("[INFO] END %s (rc=%s, %.1fs)", name, result.returncode, duration)
    if result.stdout:
        LOG.info("[INFO] %s stdout:\n%s", name, result.stdout.strip())
    if result.stderr:
        LOG.info("[INFO] %s stderr:\n%s", name, result.stderr.strip())
    return result.returncode


def _split_args(raw: str, label: str) -> list[str]:
    if not raw:
        return []
    try:
        return shlex.split(raw)
    except ValueError as exc:
        LOG.error("Failed to parse %s: %s", label, exc)
        return []


def _record_health(stage: str) -> dict[str, Any]:
    report = run_health_check(write=True)
    trading = report.get("trading", {}) if isinstance(report, dict) else {}
    data = report.get("data", {}) if isinstance(report, dict) else {}
    LOG.info(
        "[INFO] HEALTH trading_ok=%s data_ok=%s stage=%s trading_status=%s data_status=%s",
        trading.get("ok"),
        data.get("ok"),
        stage,
        trading.get("status"),
        data.get("status"),
    )
    return report


def emit(event: str, **payload: Any) -> None:
    """Append a structured event to ``logs/execute_events.jsonl``."""

    record: dict[str, Any] = {"event": str(event), **payload}
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())

    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def emit_metric(name: str, value: Any) -> None:
    """Persist a lightweight pipeline metric and log it."""

    LOG.info("[INFO] METRIC %s value=%s", name, value)
    emit("METRIC", name=name, value=value)
    try:
        existing: dict[str, Any] = {}
        if PIPELINE_METRICS_PATH.exists():
            existing_payload = json.loads(PIPELINE_METRICS_PATH.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict):
                existing.update(existing_payload)
        existing[str(name)] = value
        existing["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        PIPELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PIPELINE_METRICS_PATH.write_text(json.dumps(existing, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort metric cache
        LOG.debug("[INFO] emit_metric failed to persist %s: %s", name, exc)


def _merge_metrics_defaults(base: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    defaults: Mapping[str, Any] = {
        "last_run_utc": "",
        "rows": 0,
        "symbols_in": 0,
        "symbols_with_bars": 0,
        "bars_rows_total": 0,
        "http": {"429": 0, "404": 0, "empty_pages": 0},
        "cache": {"batches_hit": 0, "batches_miss": 0},
        "universe_prefix_counts": {},
        "timings": {},
        "status": "ok",
        "auth_reason": "",
        "auth_missing": [],
        "auth_hint": "",
    }

    for key, value in defaults.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if isinstance(node, MutableMapping):
                for nested_key, nested_value in value.items():
                    node.setdefault(nested_key, nested_value)
            else:  # pragma: no cover - defensive fallback
                base[key] = dict(value)
        else:
            base.setdefault(key, value)

    if base.get("status") != "auth_error":
        base["status"] = "ok"
    base.setdefault("auth_reason", "")
    if not isinstance(base.get("auth_missing"), list):
        base["auth_missing"] = []
    base.setdefault("auth_hint", "")
    return base


def _count_candidate_rows(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for row in reader if any(field.strip() for field in row))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.warning("[INFO] failed to count rows in %s: %s", path, exc)
        return 0


def _read_screener_rows(metrics_path: pathlib.Path) -> int:
    if not metrics_path.exists():
        return 0
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse guard
        LOG.warning("[INFO] could not parse screener_metrics.json: %s", exc)
        return 0
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    try:
        return int(rows or 0)
    except Exception:
        return 0


def _maybe_fallback(project_root: Path) -> int:
    LOG.info("[INFO] FALLBACK_CHECK start")
    cmd = [sys.executable, "-m", "scripts.fallback_candidates"]
    inline_rows: Optional[int] = None
    try:
        subprocess.check_call(cmd, cwd=str(project_root))
    except subprocess.CalledProcessError as exc:
        LOG.warning("[INFO] fallback_candidates exited rc=%s during fallback", exc.returncode)
        inline_rows = _fallback_inline(project_root)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.warning("[INFO] fallback_candidates launch failed: %s", exc)
        inline_rows = _fallback_inline(project_root)

    data_dir = project_root / "data"
    top = data_dir / "top_candidates.csv"
    latest = data_dir / "latest_candidates.csv"
    if inline_rows is not None:
        rows = inline_rows
    else:
        if top.exists():
            try:
                copyfile(top, latest)
            except Exception as exc:  # pragma: no cover - defensive copy
                LOG.warning("[INFO] FALLBACK_CHECK copy_failed: %s", exc)
        rows = _count_candidate_rows(latest)
        if rows == 0 and inline_rows is None:
            inline_rows = _fallback_inline(project_root)
            rows = inline_rows

    LOG.info("[INFO] FALLBACK_CHECK rows_out=%s source=fallback", rows or 0)
    return rows or 0


def _fallback_inline(project_root: Path) -> int:
    try:
        prepared, _source = generate_candidates(project_root, max_rows=3)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.warning("[INFO] FALLBACK_CHECK inline generation failed: %s", exc)
        prepared = pd.DataFrame(columns=LATEST_COLUMNS)
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    top = data_dir / "top_candidates.csv"
    latest = data_dir / "latest_candidates.csv"
    prepared = prepared.reindex(columns=LATEST_COLUMNS, fill_value=pd.NA)
    prepared.to_csv(top, index=False)
    prepared.to_csv(latest, index=False)
    return len(prepared)


def refresh_latest_candidates() -> dict[str, Any]:
    """Refresh ``latest_candidates.csv`` and ensure metrics scaffolding exists."""

    src = os.path.join("data", "top_candidates.csv")
    dst = os.path.join("data", "latest_candidates.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    row_count = 0
    raw_row_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    if os.path.exists(src):
        try:
            copyfile(src, dst)
        except Exception as exc:  # pragma: no cover - defensive copy fallback
            LOG.warning("[INFO] failed to copy %s to %s: %s", src, dst, exc)
        try:
            df = pd.read_csv(src)
            raw_row_count = len(df.index)
        except Exception as exc:
            LOG.error("[INFO] failed to read %s: %s", src, exc)
            df = pd.DataFrame(columns=LATEST_COLUMNS)
        if raw_row_count > 0:
            prepared = normalize_candidate_df(df, now_ts=now_iso)
            prepared["source"] = "screener"
            prepared.to_csv(dst, index=False)
            row_count = len(prepared)
        else:
            pd.DataFrame(columns=LATEST_COLUMNS).to_csv(dst, index=False)
        try:
            size = os.path.getsize(dst)
        except OSError:  # pragma: no cover - defensive guard
            size = 0
        LOG.info(
            "[INFO] refreshed latest_candidates.csv rows=%s size=%s",
            row_count,
            size,
        )
    else:
        pd.DataFrame(columns=LATEST_COLUMNS).to_csv(dst, index=False)
        LOG.info("[INFO] top_candidates.csv missing; wrote header-only latest_candidates.csv")

    metrics_path = pathlib.Path("data/screener_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing: MutableMapping[str, Any] = {}
    if metrics_path.exists():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8")) or {}
            if isinstance(parsed, dict):
                existing.update(parsed)
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOG.warning("[INFO] could not parse screener_metrics.json: %s", exc)
    merged = _merge_metrics_defaults(existing)
    if row_count == 0:
        row_count = _count_candidate_rows(pathlib.Path(dst))
    merged["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    merged["rows"] = row_count
    status_value = str(merged.get("status") or "").upper()
    if row_count == 0:
        if raw_row_count > 0:
            merged["status"] = "ok"
            merged["candidate_reason"] = "HAVE_CANDIDATES"
        else:
            if status_value in ("", "OK", "ZERO_CANDIDATES"):
                merged["status"] = "ZERO_CANDIDATES"
            merged["candidate_reason"] = "ZERO_CANDIDATES"
    else:
        if status_value == "ZERO_CANDIDATES":
            merged["status"] = "ok"
        merged["candidate_reason"] = "HAVE_CANDIDATES"
    metrics_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    write_metrics_summary(
        overrides={"rows": row_count, "status": merged.get("status", "ok")}
    )

    return dict(merged)


def copy_latest_candidates() -> None:  # pragma: no cover - backward compatibility alias
    refresh_latest_candidates()


def write_metrics_summary(*, overrides: Mapping[str, object] | None = None) -> None:
    """Write a minimal metrics_summary.csv for the dashboard overview."""
    meta: dict[str, object] = {
        "last_run_utc": "",
        "symbols_in": 0,
        "symbols_with_bars": 0,
        "bars_rows_total": 0,
        "rows": 0,
        "status": "ok",
        "auth_reason": "",
        "auth_missing": "",
        "auth_hint": "",
    }

    metrics_path = pathlib.Path("data/screener_metrics.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if isinstance(metrics, dict):
                for key in meta:
                    meta[key] = metrics.get(key, meta[key])
                status_value = metrics.get("status")
                if status_value:
                    meta["status"] = status_value
                reason_value = metrics.get("auth_reason")
                if reason_value:
                    meta["auth_reason"] = reason_value
                missing_value = metrics.get("auth_missing")
                if isinstance(missing_value, (list, tuple)):
                    meta["auth_missing"] = ",".join(
                        str(item) for item in missing_value if str(item).strip()
                    )
                elif isinstance(missing_value, str):
                    meta["auth_missing"] = missing_value
                hint_value = metrics.get("auth_hint")
                if hint_value:
                    meta["auth_hint"] = hint_value
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOG.warning("[INFO] could not parse screener_metrics.json: %s", exc)

    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            meta[key] = value

    write_metrics_summary_row(
        {
            "last_run_utc": meta.get("last_run_utc", ""),
            "symbols_in": meta.get("symbols_in", 0),
            "with_bars": meta.get("symbols_with_bars", 0),
            "bars_rows": meta.get("bars_rows_total", 0),
            "candidates": meta.get("rows", 0),
            "status": meta.get("status", "ok"),
            "auth_reason": meta.get("auth_reason", ""),
            "auth_missing": meta.get("auth_missing", ""),
            "auth_hint": meta.get("auth_hint", ""),
        }
    )
    LOG.info("[INFO] wrote metrics_summary.csv")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JBRAVO pipeline")
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated list of steps to run (default env PIPE_STEPS or screener,backtest,metrics)",
    )
    parser.add_argument(
        "--reload-web",
        choices=("true", "false"),
        default="true",
        help="Reload the web application when the pipeline finishes (default: true)",
    )
    parser.add_argument(
        "--screener-args",
        default=os.getenv("JBR_SCREENER_ARGS", ""),
        help="Extra arguments passed to scripts.screener (default env JBR_SCREENER_ARGS)",
    )
    parser.add_argument(
        "--backtest-args",
        default=os.getenv("JBR_BACKTEST_ARGS", ""),
        help="Extra arguments passed to scripts.backtest (default env JBR_BACKTEST_ARGS)",
    )
    parser.add_argument(
        "--metrics-args",
        default=os.getenv("JBR_METRICS_ARGS", ""),
        help="Extra arguments passed to scripts.metrics (default env JBR_METRICS_ARGS)",
    )
    parser.add_argument(
        "--execute-args",
        default=os.getenv("JBR_EXEC_ARGS", ""),
        help="Extra arguments passed to scripts.execute_trades (default env JBR_EXEC_ARGS)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def determine_steps(requested: Optional[str]) -> list[str]:
    raw = requested or os.environ.get("PIPE_STEPS", "screener,execute,backtest,metrics")
    steps = [step.strip().lower() for step in raw.split(",") if step.strip()]
    return steps or ["screener", "backtest", "metrics"]


def latest_candidates_has_rows(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if any(field.strip() for field in row):
                    return True
    except Exception as exc:  # pragma: no cover - defensive parsing
        LOG.warning("[INFO] failed to inspect %s: %s", path, exc)
    return False


def run_execute_step(cmd: list[str], *, candidate_rows: int | None = None) -> int:
    latest_path = pathlib.Path("data") / "latest_candidates.csv"
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    start = time.time()
    LOG.info("[INFO] START EXECUTE %s", cmd_str)
    if candidate_rows is None:
        has_rows = latest_candidates_has_rows(latest_path)
    else:
        has_rows = candidate_rows > 0
    if not has_rows:
        duration = time.time() - start
        LOG.info("[INFO] EXECUTE_SKIP_NO_CANDIDATES rows=0")
        LOG.info("[INFO] END EXECUTE rc=0 duration=%.1fs", duration)
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    LOG.info("[INFO] END EXECUTE rc=%s duration=%.1fs", result.returncode, duration)
    if result.stdout:
        LOG.info("[INFO] EXECUTE stdout:\n%s", result.stdout.strip())
    if result.stderr:
        LOG.info("[INFO] EXECUTE stderr:\n%s", result.stderr.strip())
    return result.returncode


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_env()
    configure_logging()
    try:
        creds_snapshot = assert_alpaca_creds()
    except AlpacaCredentialsError as exc:
        missing = list(dict.fromkeys(list(exc.missing) + list(exc.whitespace)))
        LOG.error(
            "[ERROR] ALPACA_CREDENTIALS_INVALID reason=%s missing=%s whitespace=%s sanitized=%s",
            exc.reason,
            ",".join(exc.missing) or "",
            ",".join(exc.whitespace) or "",
            json.dumps(exc.sanitized, sort_keys=True),
        )
        write_auth_error_artifacts(
            reason=exc.reason,
            sanitized=exc.sanitized,
            missing=missing,
            metrics_path=pathlib.Path("data") / "screener_metrics.json",
            summary_path=pathlib.Path("data") / "metrics_summary.csv",
        )
        return 2

    LOG.info(
        "[INFO] ALPACA_CREDENTIALS_OK sanitized=%s",
        json.dumps(creds_snapshot, sort_keys=True),
    )

    args = parse_args(argv)
    extras = {
        "screener": _split_args(args.screener_args, "screener_args"),
        "backtest": _split_args(args.backtest_args, "backtest_args"),
        "metrics": _split_args(args.metrics_args, "metrics_args"),
        "execute": _split_args(args.execute_args, "execute_args"),
    }
    LOG.info(
        "[INFO] PIPELINE_ARGS screener_raw=%s backtest_raw=%s metrics_raw=%s execute_raw=%s",
        args.screener_args,
        args.backtest_args,
        args.metrics_args,
        args.execute_args,
    )
    LOG.info(
        "[INFO] PIPELINE_ARGS_PARSED screener=%s backtest=%s metrics=%s execute=%s",
        extras["screener"],
        extras["backtest"],
        extras["metrics"],
        extras["execute"],
    )

    steps = determine_steps(args.steps)
    health_report = _record_health("start")
    trading_status = {}
    if isinstance(health_report, dict):
        trading_status = health_report.get("trading", {}) or {}
    if trading_status.get("status") == 401:
        LOG.error(
            "[ERROR] Alpaca auth failed (401). Check: (1) paper vs live base URL, "
            "(2) fresh key/secret, (3) whitespace/CRLF in .env."
        )
        raise SystemExit(2)
    LOG.info("[INFO] PIPELINE_START steps=%s", steps)

    rc = 0
    start = time.time()
    artifacts_written = False
    latest_refreshed = False
    pipeline_metrics: dict[str, Any] = {}
    latest_path = pathlib.Path("data") / "latest_candidates.csv"
    candidate_rows = _count_candidate_rows(latest_path)

    def ensure_candidates(force: bool = False) -> int:
        nonlocal candidate_rows
        current = _count_candidate_rows(latest_path)
        if force or current <= 0:
            current = _maybe_fallback(BASE_DIR)
        candidate_rows = current
        return candidate_rows

    if "screener" in steps:
        step_start = time.time()
        cmd = [
            sys.executable,
            "-m",
            "scripts.screener",
            "--mode",
            "screener",
            "--feed",
            "iex",
        ]
        if extras["screener"]:
            cmd.extend(extras["screener"])
        rc_scr = run_cmd(cmd, "SCREENER")
        if rc_scr != 0 and rc == 0:
            rc = rc_scr
            LOG.error(
                "[INFO] SCREENER failed rc=%s; continuing to write minimal artifacts",
                rc_scr,
            )
        try:
            pipeline_metrics = refresh_latest_candidates()
            candidate_rows = _count_candidate_rows(latest_path)
            emit_metric("CANDIDATE_ROWS", candidate_rows)
            artifacts_written = True
            latest_refreshed = True
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOG.error("[INFO] failed to refresh screener artifacts after SCREENER step: %s", exc)

        screener_rows = _read_screener_rows(pathlib.Path("data") / "screener_metrics.json")
        LOG.info("[INFO] SCREENER rows=%s", screener_rows)
        if screener_rows == 0:
            ensure_candidates(force=True)
        else:
            ensure_candidates()

    if "execute" in steps:
        if not latest_refreshed:
            try:
                pipeline_metrics = refresh_latest_candidates()
                candidate_rows = _count_candidate_rows(latest_path)
                emit_metric("CANDIDATE_ROWS", candidate_rows)
                latest_refreshed = True
            except Exception as exc:  # pragma: no cover - defensive safeguard
                LOG.error("[INFO] failed to refresh artifacts before EXECUTE step: %s", exc)
        rows_out = ensure_candidates()
        if rows_out == 0:
            LOG.info("[INFO] EXECUTE_SKIP_NO_CANDIDATES rows=0")
        else:
            cmd = [sys.executable, "-m", "scripts.execute_trades"]
            if extras["execute"]:
                cmd.extend(extras["execute"])
            rc_exec = run_execute_step(cmd, candidate_rows=candidate_rows)
            if rc_exec != 0 and rc == 0:
                rc = rc_exec
            if EXECUTE_METRICS_PATH.exists():
                try:
                    execute_metrics = json.loads(
                        EXECUTE_METRICS_PATH.read_text(encoding="utf-8")
                    )
                except Exception as exc:  # pragma: no cover - defensive metrics parsing
                    LOG.warning("[INFO] failed to read execute metrics: %s", exc)
                else:
                    LOG.info("EXECUTE SUMMARY %s", json.dumps(execute_metrics, sort_keys=True))

    if "backtest" in steps:
        ensure_candidates()
        cmd = [sys.executable, "-m", "scripts.backtest"]
        if extras["backtest"]:
            cmd.extend(extras["backtest"])
        rc_bt = run_cmd(cmd, "BACKTEST")
        if rc_bt != 0 and rc == 0:
            rc = rc_bt

    if "metrics" in steps:
        ensure_candidates()
        cmd = [sys.executable, "-m", "scripts.metrics"]
        if extras["metrics"]:
            cmd.extend(extras["metrics"])
        rc_mx = run_cmd(cmd, "METRICS")
        if rc_mx != 0 and rc == 0:
            rc = rc_mx

    if not artifacts_written:
        try:
            pipeline_metrics = refresh_latest_candidates()
            candidate_rows = _count_candidate_rows(latest_path)
            emit_metric("CANDIDATE_ROWS", candidate_rows)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOG.error("[INFO] final artifact refresh failed: %s", exc)
        ensure_candidates()

    ensure_candidates()
    duration = time.time() - start
    LOG.info("[INFO] PIPELINE_END rc=%s duration=%.1fs", rc, duration)
    sm_path = BASE_DIR / "data" / "screener_metrics.json"
    metrics_source: dict[str, Any] = {}
    if isinstance(pipeline_metrics, Mapping) and pipeline_metrics:
        metrics_source = dict(pipeline_metrics)
    if not metrics_source and sm_path.exists():
        try:
            loaded = json.loads(sm_path.read_text())
        except Exception:
            loaded = None
        if isinstance(loaded, Mapping):
            metrics_source = dict(loaded)
    timings = metrics_source.get("timings", {}) if isinstance(metrics_source, Mapping) else {}
    if not isinstance(timings, Mapping):
        timings = {}
    m = SimpleNamespace(
        symbols_in=metrics_source.get("symbols_in", "na"),
        symbols_with_bars=metrics_source.get("symbols_with_bars", "na"),
        rows=metrics_source.get("rows", "na"),
        t_fetch=timings.get("fetch_secs", "na"),
        t_features=timings.get("feature_secs", "na"),
        t_rank=timings.get("rank_secs", "na"),
        t_gates=timings.get("gates_secs", "na"),
    )
    LOG.info(
        "[INFO] PIPELINE_SUMMARY symbols_in=%s with_bars=%s rows=%s fetch_secs=%s feature_secs=%s rank_secs=%s gate_secs=%s",
        m.symbols_in,
        m.symbols_with_bars,
        m.rows,
        m.t_fetch,
        m.t_features,
        m.t_rank,
        m.t_gates,
    )
    _record_health("end")

    if args.reload_web.lower() == "true":
        domain = os.environ.get("PYTHONANYWHERE_DOMAIN", "")
        cmd = ["pa_reload_webapp"]
        if domain:
            cmd.append(domain)
        try:
            subprocess.check_call(cmd)
            LOG.info("[INFO] AUTO-RELOAD ok domain=%s", domain or "(default)")
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOG.info("[INFO] AUTO-RELOAD failed: %s", exc)
            wsgi_path = Path("/var/www/raspatrick_pythonanywhere_com_wsgi.py")
            try:
                wsgi_path.touch()
                LOG.info(
                    "[INFO] AUTO-RELOAD fallback touch ok: %s",
                    wsgi_path,
                )
            except Exception as exc2:  # pragma: no cover - defensive fallback
                LOG.info(
                    "[INFO] AUTO-RELOAD fallback touch failed: %s",
                    exc2,
                )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
