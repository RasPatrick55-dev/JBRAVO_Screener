import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from scripts.fallback_candidates import CANONICAL_COLUMNS, build_latest_candidates, normalize_candidate_df
from utils.env import load_env
from utils import write_csv_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG = logging.getLogger("pipeline")
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
SCREENER_METRICS_PATH = DATA_DIR / "screener_metrics.json"
LATEST_CANDIDATES = DATA_DIR / "latest_candidates.csv"
TOP_CANDIDATES = DATA_DIR / "top_candidates.csv"
DEFAULT_WSGI_PATH = Path("/var/www/raspatrick_pythonanywhere_com_wsgi.py")


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


def _ensure_trades_log() -> None:
    trades_path = DATA_DIR / "trades_log.csv"
    if trades_path.exists() and trades_path.stat().st_size > 0:
        return
    header = pd.DataFrame(columns=[
        "timestamp",
        "symbol",
        "action",
        "qty",
        "price",
        "order_id",
        "status",
        "net_pnl",
        "entry_time",
        "exit_time",
    ])
    write_csv_atomic(str(trades_path), header)


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
        LOG.info("[INFO] DASH_RELOAD method=pa rc=0 domain=%s", domain or "(default)")
        return
    except FileNotFoundError:
        LOG.info("[INFO] DASH_RELOAD method=pa rc=ERR detail=missing_tool")
    except subprocess.CalledProcessError as exc:
        LOG.info(
            "[INFO] DASH_RELOAD method=pa rc=ERR detail=rc%s", exc.returncode
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.info("[INFO] DASH_RELOAD method=pa rc=ERR detail=%s", exc)

    target = domain.replace(".", "_") if domain else ""
    if target:
        path = Path(f"/var/www/{target}_wsgi.py")
    else:
        path = DEFAULT_WSGI_PATH
    try:
        path.touch()
        LOG.info("[INFO] DASH_RELOAD method=touch rc=0 path=%s", path)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.info("[INFO] DASH_RELOAD method=touch rc=ERR path=%s detail=%s", path, exc)


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_env()
    configure_logging()
    args = parse_args(argv)
    steps = determine_steps(args.steps)
    LOG.info("PIPELINE_START steps=%s", ",".join(steps))

    _ensure_latest_headers()

    extras = {
        "screener": _split_args(args.screener_args),
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

    try:
        if "screener" in steps:
            cmd = [sys.executable, "-m", "scripts.screener", "--mode", "screener"]
            if extras["screener"]:
                cmd.extend(extras["screener"])
            rc_scr, secs = run_step("screener", cmd, timeout=60 * 20)
            stage_times["screener"] = secs
            metrics = _read_json(SCREENER_METRICS_PATH)
            symbols_in = int(metrics.get("symbols_in", 0) or 0)
            symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
            top_frame = _load_top_candidates()
            if top_frame.empty:
                frame, source = build_latest_candidates(PROJECT_ROOT)
                write_csv_atomic(str(TOP_CANDIDATES), frame)
                rows = int(len(frame.index))
                LOG.info(
                    "[INFO] FALLBACK_CHECK reason=no_candidates rows_out=%d source=%s",
                    rows,
                    source,
                )
            else:
                rows = _write_latest_from_frame(top_frame, source="screener")
        else:
            metrics = _read_json(SCREENER_METRICS_PATH)
            symbols_in = int(metrics.get("symbols_in", 0) or 0)
            symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
            rows = ensure_candidates(0)

        if "backtest" in steps:
            rows = ensure_candidates(rows or 1)
            cmd = [sys.executable, "-m", "scripts.backtest"]
            if extras["backtest"]:
                cmd.extend(extras["backtest"])
            rc_bt, secs = run_step("backtest", cmd, timeout=60 * 3)
            stage_times["backtest"] = secs

        if "metrics" in steps:
            rows = ensure_candidates(rows or 1)
            _ensure_trades_log()
            cmd = [sys.executable, "-m", "scripts.metrics"]
            if extras["metrics"]:
                cmd.extend(extras["metrics"])
            rc_mt, secs = run_step("metrics", cmd, timeout=60 * 3)
            stage_times["metrics"] = secs

        rc = 0
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("PIPELINE_FATAL: %s", exc)
        rc = 1
    finally:
        fetch_secs = _extract_timing(metrics, "fetch_secs")
        feature_secs = _extract_timing(metrics, "feature_secs")
        rank_secs = _extract_timing(metrics, "rank_secs")
        gate_secs = _extract_timing(metrics, "gate_secs")
        LOG.info(
            "PIPELINE_SUMMARY symbols_in=%s with_bars=%s rows=%s fetch_secs=%.3f feature_secs=%.3f rank_secs=%.3f gate_secs=%.3f",
            symbols_in,
            symbols_with_bars,
            rows,
            fetch_secs,
            feature_secs,
            rank_secs,
            gate_secs,
        )
        duration = time.time() - started
        LOG.info("PIPELINE_END rc=%s duration=%.1fs", rc, duration)
        _reload_dashboard(args.reload_web.lower() == "true")
        sys.exit(rc)


if __name__ == "__main__":
    raise SystemExit(main())
