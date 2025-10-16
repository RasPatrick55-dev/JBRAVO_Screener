import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from scripts.fallback_candidates import CANONICAL_COLUMNS, build_latest_candidates, normalize_candidate_df
from scripts.utils.env import load_env
from utils import write_csv_atomic
from utils.telemetry import emit_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG = logging.getLogger("pipeline")
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


def _resolve_base_dir(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    cwd = Path.cwd()
    if cwd != PROJECT_ROOT and (cwd / "data").exists():
        return cwd
    return BASE_DIR


def refresh_latest_candidates(base_dir: Path | None = None) -> pd.DataFrame:
    base = _resolve_base_dir(base_dir)
    data_dir = base / "data"
    top_path = data_dir / "top_candidates.csv"
    latest_path = data_dir / "latest_candidates.csv"
    metrics_path = data_dir / "screener_metrics.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    if top_path.exists() and top_path.stat().st_size > 0:
        copyfile(str(top_path), str(latest_path))
        try:
            frame = pd.read_csv(top_path)
        except Exception:
            frame = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
        normalized = normalize_candidate_df(frame)
        normalized = normalized[list(CANONICAL_COLUMNS)]
        write_csv_atomic(str(latest_path), normalized)
        _write_refresh_metrics(metrics_path)
        return normalized
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
        "--exec-args",
        default=os.getenv("JBR_EXEC_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.execute_trades",
    )
    parser.add_argument(
        "--backtest-args",
        default=os.getenv("JBR_BACKTEST_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.backtest",
    )
    parser.add_argument(
        "--metrics-args",
        default=os.getenv("JBR_METRICS_ARGS", ""),
        help="Extra CLI arguments forwarded to scripts.metrics",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def determine_steps(raw: Optional[str]) -> list[str]:
    default = "screener,backtest,metrics"
    target = raw or os.environ.get("PIPE_STEPS", default)
    steps = [part.strip().lower() for part in target.split(",") if part.strip()]
    return steps or default.split(",")


def run_step(name: str, cmd: Sequence[str], *, timeout: Optional[float] = None) -> tuple[int, float]:
    started = time.time()
    LOG.info("START %s cmd=%s", name, shlex.join(cmd))
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
    domain = os.environ.get("PYTHONANYWHERE_DOMAIN", "").strip()
    cmd = ["pa_reload_webapp"]
    if domain:
        cmd.append(domain)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        LOG.info("[INFO] DASH RELOAD method=pa rc=0 domain=%s", domain or "(default)")
        return
    except FileNotFoundError:
        LOG.info(
            "[INFO] DASH RELOAD method=pa rc=missing_tool domain=%s",
            domain or "(default)",
        )
    except subprocess.CalledProcessError as exc:
        LOG.info("[INFO] DASH RELOAD method=pa rc=%s domain=%s", exc.returncode, domain or "(default)")
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.info(
            "[INFO] DASH RELOAD method=pa rc=ERR domain=%s detail=%s",
            domain or "(default)",
            exc,
        )

    target = domain.replace(".", "_") if domain else ""
    path = Path(f"/var/www/{target}_wsgi.py") if target else DEFAULT_WSGI_PATH
    try:
        path.touch()
        LOG.info("[INFO] DASH RELOAD method=touch local rc=0 path=%s", path)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.warning(
            "[WARN] DASH RELOAD failed method=touch path=%s detail=%s",
            path,
            exc,
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    loaded_files, missing_keys = load_env(REQUIRED_ENV_KEYS)
    configure_logging()
    files_repr = f"[{', '.join(loaded_files)}]" if loaded_files else "[]"
    LOG.info("[INFO] ENV_LOADED files=%s", files_repr)
    if missing_keys:
        LOG.error("[ERROR] ENV_MISSING_KEYS=%s", f"[{', '.join(missing_keys)}]")
        raise SystemExit(2)
    args = parse_args(argv)
    steps = determine_steps(args.steps)
    LOG.info("PIPELINE_START steps=%s", ",".join(steps))

    _ensure_latest_headers()

    extras = {
        "screener": _split_args(args.screener_args),
        "exec": _split_args(args.exec_args),
        "backtest": _split_args(args.backtest_args),
        "metrics": _split_args(args.metrics_args),
    }

    started = time.time()
    metrics: dict[str, Any] = {}
    symbols_in = 0
    symbols_with_bars = 0
    rows = 0
    stage_times: dict[str, float] = {}
    rc = 0

    metrics_rows: int | None = None
    try:
        if "screener" in steps:
            cmd = [sys.executable, "-m", "scripts.screener", "--mode", "screener"]
            if extras["screener"]:
                cmd.extend(extras["screener"])
            rc_scr, secs = run_step("screener", cmd, timeout=60 * 20)
            stage_times["screener"] = secs
            if rc_scr:
                rc = rc_scr
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
                LOG.info("[INFO] FALLBACK_CHECK start reason=no_candidates")
                frame, _ = build_latest_candidates(PROJECT_ROOT, max_rows=1)
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
            LOG.info("[INFO] FALLBACK_CHECK rows_out=%d source=fallback", rows)

        if "backtest" in steps:
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.backtest"]
            if extras["backtest"]:
                cmd.extend(extras["backtest"])
            rc_bt, secs = run_step("backtest", cmd, timeout=60 * 3)
            stage_times["backtest"] = secs
            if rc_bt and not rc:
                rc = rc_bt

        if "metrics" in steps:
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            trades_path = DATA_DIR / "trades_log.csv"
            if not trades_path.exists():
                LOG.warning(
                    "[WARN] METRICS_TRADES_LOG_MISSING path=%s",
                    trades_path,
                )
            else:
                cmd = [sys.executable, "-m", "scripts.metrics"]
                if extras["metrics"]:
                    cmd.extend(extras["metrics"])
                rc_mt, secs = run_step("metrics", cmd, timeout=60 * 3)
                stage_times["metrics"] = secs
                if rc_mt and not rc:
                    rc = rc_mt
        if "exec" in steps:
            min_rows = rows or (metrics_rows if metrics_rows else 0) or 1
            rows = ensure_candidates(min_rows)
            cmd = [sys.executable, "-m", "scripts.execute_trades"]
            exec_args = extras["exec"]
            if not any(arg.startswith("--time-window") for arg in exec_args):
                cmd.extend(["--time-window", "auto"])
            if exec_args:
                cmd.extend(exec_args)
            rc_exec, secs = run_step("exec", cmd, timeout=60 * 10)
            stage_times["exec"] = secs
            if rc_exec and not rc:
                rc = rc_exec
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("PIPELINE_FATAL: %s", exc)
        rc = 1
    finally:
        fetch_secs = _extract_timing(metrics, "fetch_secs")
        feature_secs = _extract_timing(metrics, "feature_secs")
        rank_secs = _extract_timing(metrics, "rank_secs")
        gate_secs = _extract_timing(metrics, "gate_secs")
        summary_rows = metrics_rows if metrics_rows is not None else rows
        LOG.info(
            "PIPELINE_SUMMARY symbols_in=%s with_bars=%s rows=%s fetch_secs=%.3f feature_secs=%.3f rank_secs=%.3f gate_secs=%.3f",
            symbols_in,
            symbols_with_bars,
            summary_rows,
            fetch_secs,
            feature_secs,
            rank_secs,
            gate_secs,
        )
        duration = time.time() - started
        LOG.info("PIPELINE_END rc=%s duration=%.1f", rc, duration)
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
