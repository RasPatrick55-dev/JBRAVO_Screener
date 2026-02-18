from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from scripts.fallback_candidates import CANON, CANONICAL_COLUMNS
from scripts import docs_consistency_check as docs_checker
from utils.alerts import send_alert

BASE_DIR = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("dashboard_consistency")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

PIPELINE_MARKERS = (
    "PIPELINE_START",
    "PIPELINE_SUMMARY",
    "FALLBACK_CHECK",
    "PIPELINE_END",
    "DASH RELOAD",
)

EXECUTION_TOKENS = (
    "BUY_SUBMIT",
    "BUY_FILL",
    "BUY_CANCELLED",
    "TRAIL_SUBMIT",
    "TRAIL_CONFIRMED",
    "TRAIL_FAILED",
    "TIME_WINDOW",
    "CASH",
    "ZERO_QTY",
    "PRICE_BOUNDS",
    "MAX_POSITIONS",
)

EXECUTION_SKIP_TOKENS = (
    "TIME_WINDOW",
    "CASH",
    "ZERO_QTY",
    "PRICE_BOUNDS",
    "MAX_POSITIONS",
)

CANDIDATE_CANONICAL_LOWER = {"symbol", "score"}


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN check
            return None
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


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    return None
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _path_info(path: Path) -> dict[str, Any]:
    info = {"path": str(path), "present": path.exists()}
    if info["present"]:
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["mtime_utc"] = _isoformat(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
    return info


def _safe_read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    info = _path_info(path)
    if not info["present"]:
        return {}, info
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            info["present"] = True
            return data if isinstance(data, dict) else {}, info
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read JSON %s: %s", path, exc)
        info["error"] = str(exc)
        return {}, info


def _safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info = _path_info(path)
    if not info["present"]:
        return None, info
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read CSV %s: %s", path, exc)
        info["error"] = str(exc)
        return None, info
    info["present"] = True
    info["rows"] = int(len(df.index))
    info["columns"] = list(df.columns)
    return df, info


def _read_tail(path: Path, max_bytes: int = 256_000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= max_bytes:
                handle.seek(0)
                data = handle.read()
            else:
                handle.seek(-max_bytes, os.SEEK_END)
                data = handle.read()
        return data.decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to tail log %s: %s", path, exc)
        return ""


def _coerce_token_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.endswith("s"):
        try:
            numeric = value[:-1]
            if numeric:
                return float(numeric)
        except ValueError:
            pass
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith("[") and value.endswith("]")
    ):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _extract_timestamp(line: str) -> str | None:
    if not isinstance(line, str):
        return None
    match = re.match(r"(?P<prefix>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?P<frac>[\.,]\d+)?", line)
    if not match:
        return None
    prefix = match.group("prefix").replace(" ", "T")
    frac = match.group("frac") or ""
    if frac.startswith(","):
        frac = "." + frac[1:]
    candidate = prefix + frac
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            dt = datetime.strptime(candidate, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _isoformat(dt)


def _parse_key_values(tail: str) -> dict[str, Any]:
    kv: dict[str, Any] = {}
    for part in tail.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key] = _coerce_token_value(value)
    return kv


def _parse_pipeline_tokens(text: str) -> dict[str, Any]:
    tokens: dict[str, list[dict[str, Any]]] = {marker: [] for marker in PIPELINE_MARKERS}
    if not text:
        return {marker: {"present": False, "count": 0} for marker in PIPELINE_MARKERS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        for marker in PIPELINE_MARKERS:
            if marker not in line:
                continue
            entry: dict[str, Any] = {
                "line": line,
                "timestamp_utc": _extract_timestamp(raw_line),
                "data": {},
            }
            if marker == "PIPELINE_START":
                tail = line.split("PIPELINE_START", 1)[1].strip()
                kv = _parse_key_values(tail)
                steps = kv.get("steps")
                if isinstance(steps, str):
                    entry["data"] = {"steps": [step for step in steps.split(",") if step]}
                else:
                    entry["data"] = kv
            elif marker in {"PIPELINE_SUMMARY", "FALLBACK_CHECK", "PIPELINE_END", "DASH RELOAD"}:
                tail = line.split(marker, 1)[1].strip()
                entry["data"] = _parse_key_values(tail)
            tokens[marker].append(entry)

    result: dict[str, Any] = {}
    for marker in PIPELINE_MARKERS:
        entries = tokens.get(marker) or []
        last_entry = entries[-1] if entries else {}
        result[marker] = {
            "present": bool(entries),
            "count": len(entries),
            "last_line": last_entry.get("line"),
            "timestamp_utc": last_entry.get("timestamp_utc"),
            "data": last_entry.get("data", {}),
        }
    return result


def _parse_executor_tokens(text: str) -> dict[str, Any]:
    tokens: dict[str, list[dict[str, Any]]] = {token: [] for token in EXECUTION_TOKENS}
    if not text:
        return {
            token: {"present": False, "count": 0, "latest": {}, "entries": []}
            for token in EXECUTION_TOKENS
        }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        for token in EXECUTION_TOKENS:
            if token not in line:
                continue
            entry: dict[str, Any] = {
                "line": line,
                "timestamp_utc": _extract_timestamp(raw_line),
                "data": {},
                "tail": "",
            }
            tail = line.split(token, 1)[1].strip() if token in line else ""
            entry["tail"] = tail
            if tail:
                entry["data"] = _parse_key_values(tail)
            tokens[token].append(entry)

    parsed: dict[str, Any] = {}
    for token, entries in tokens.items():
        parsed[token] = {
            "present": bool(entries),
            "count": len(entries),
            "latest": entries[-1] if entries else {},
            "entries": entries,
        }
    return parsed


def _numeric(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return 0.0


def _gather_universe(metrics: Mapping[str, Any], summary_data: Mapping[str, Any]) -> dict[str, Any]:
    def _get(key: str) -> int:
        for source in (metrics, summary_data):
            if not isinstance(source, Mapping):
                continue
            value = source.get(key)
            try:
                if value is None:
                    continue
                return int(float(value))
            except Exception:
                continue
        return 0

    bars_total = 0
    for candidate_key in ("bars_rows_total", "bars_total", "bars", "bars_count"):
        if isinstance(metrics, Mapping) and candidate_key in metrics:
            try:
                bars_total = int(float(metrics[candidate_key]))
                break
            except Exception:
                continue

    source = None
    if isinstance(metrics, Mapping):
        source = metrics.get("source") or metrics.get("latest_source")
    if not source and isinstance(summary_data, Mapping):
        source = summary_data.get("source")

    required = _get("symbols_with_required_bars")
    if not required:
        required = _get("symbols_with_bars")
    any_bars = _get("symbols_with_any_bars")
    if not any_bars:
        any_bars = _get("symbols_with_bars_any") or required
    required_bars_value = _get("required_bars") or _get("bars_required")
    universe = {
        "symbols_in": _get("symbols_in"),
        "symbols_with_required_bars": required,
        "symbols_with_any_bars": any_bars,
        "symbols_with_bars": required,
        "rows": _get("rows"),
        "bars_rows_total": bars_total,
        "source": source,
        "required_bars": required_bars_value,
    }
    return universe


def _analyze_candidates(df: pd.DataFrame | None, info: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "present": bool(info.get("present")),
        "path": info.get("path"),
        "row_count": 0,
        "columns": info.get("columns", []),
        "missing_canonical": sorted(
            c for c in CANDIDATE_CANONICAL_LOWER if c not in {str(col).lower() for col in info.get("columns", [])}
        ),
        "first_rows": [],
        "canonical": None,
        "missing_score_breakdown": True,
        "canonical_sequence": [],
    }
    raw_columns = list(info.get("columns", []) or [])
    canonical_sequence: list[str] = []
    missing_score_breakdown = True
    for column in raw_columns:
        key = str(column).strip()
        canonical = CANON.get(key, CANON.get(key.lower(), key.lower()))
        canonical_sequence.append(canonical)
        if canonical == "score_breakdown":
            missing_score_breakdown = False
    analysis["canonical_sequence"] = canonical_sequence
    analysis["missing_score_breakdown"] = missing_score_breakdown
    if raw_columns:
        canonical_list = list(CANONICAL_COLUMNS)
        sequence_slice = canonical_sequence[: len(canonical_list)]
        matches_order = sequence_slice == canonical_list
        has_all = all(name in canonical_sequence for name in canonical_list)
        analysis["canonical"] = matches_order and has_all and not missing_score_breakdown
    if df is None or df.empty:
        return analysis
    analysis["row_count"] = int(len(df.index))
    preview = df.head(3).to_dict(orient="records")
    analysis["first_rows"] = json.loads(json.dumps(preview, default=str))
    return analysis


def _gate_pressure(metrics: Mapping[str, Any]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    total = 0
    if not isinstance(metrics, Mapping):
        return {"breakdown": breakdown, "total": total}
    gate_breakdown = metrics.get("gate_breakdown")
    if isinstance(gate_breakdown, Mapping):
        for key, value in gate_breakdown.items():
            if isinstance(value, (int, float)):
                breakdown[str(key)] = int(value)
                total += int(value)
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if "gate" not in str(key).lower():
            continue
        if str(key) in breakdown:
            continue
        breakdown[str(key)] = int(value)
        total += int(value)
    return {"breakdown": breakdown, "total": total}


def _timings(metrics: Mapping[str, Any], summary_data: Mapping[str, Any]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for key in ("fetch_secs", "feature_secs", "rank_secs", "gate_secs"):
        value = None
        for source in (metrics, summary_data):
            if isinstance(source, Mapping) and key in source:
                value = source.get(key)
                break
        timings[key] = _numeric(value)
    return timings


def _parse_executor_log(text: str) -> dict[str, Any]:
    tokens = _parse_executor_tokens(text)
    skip_reasons: dict[str, int] = {}
    for reason in EXECUTION_SKIP_TOKENS:
        count = tokens.get(reason, {}).get("count", 0)
        if count:
            skip_reasons[reason] = int(count)
    summary = {
        "orders_submitted": tokens.get("BUY_SUBMIT", {}).get("count", 0),
        "orders_filled": tokens.get("BUY_FILL", {}).get("count", 0),
        "orders_canceled": tokens.get("BUY_CANCELLED", {}).get("count", 0),
        "trailing_attached": tokens.get("TRAIL_CONFIRMED", {}).get("count", 0),
        "skip_reasons": skip_reasons,
        "tokens": tokens,
    }
    return summary


def _executor_summary(metrics: Mapping[str, Any], log_text: str) -> dict[str, Any]:
    summary_from_log = _parse_executor_log(log_text)
    data = {
        "orders_submitted": summary_from_log["orders_submitted"],
        "orders_filled": summary_from_log["orders_filled"],
        "orders_canceled": summary_from_log["orders_canceled"],
        "trailing_attached": summary_from_log["trailing_attached"],
        "skip_reasons": dict(summary_from_log["skip_reasons"]),
        "metrics_present": bool(metrics),
        "tokens": summary_from_log.get("tokens", {}),
    }
    if isinstance(metrics, Mapping) and metrics:
        for key in ("orders_submitted", "orders_filled", "orders_canceled", "trailing_attached"):
            if key in metrics:
                data[key] = int(_numeric(metrics.get(key)))
        skip = metrics.get("skip_reasons")
        if isinstance(skip, Mapping):
            merged: dict[str, int] = dict(data.get("skip_reasons", {}))
            for reason, count in skip.items():
                try:
                    merged[str(reason)] = int(_numeric(count))
                except Exception:
                    continue
            data["skip_reasons"] = merged
    return data


def _predictions_summary(pred_df: pd.DataFrame | None, pred_info: dict[str, Any], ranker_json: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "predictions_present": bool(pred_info.get("present")),
        "ranker_eval_present": bool(ranker_json),
        "prediction_rows": 0,
        "prediction_columns": pred_info.get("columns", []),
        "decile_lifts": {},
    }
    if pred_df is not None and not pred_df.empty:
        summary["prediction_rows"] = int(len(pred_df.index))
        if "symbol" in pred_df.columns:
            summary["unique_symbols"] = int(pred_df["symbol"].nunique())
        summary["prediction_columns"] = list(pred_df.columns)
    if isinstance(ranker_json, Mapping):
        deciles = ranker_json.get("deciles")
        if isinstance(deciles, Iterable):
            for bucket in deciles:
                if not isinstance(bucket, Mapping):
                    continue
                name = bucket.get("name") or bucket.get("decile")
                lift = bucket.get("lift") or bucket.get("return")
                if name is None or lift is None:
                    continue
                try:
                    summary["decile_lifts"][str(name)] = float(lift)
                except Exception:
                    summary["decile_lifts"][str(name)] = lift
    return summary


def _prefix_sanity(frames: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    for label in ("latest", "scored", "top"):
        frame = frames.get(label)
        if frame is not None and not frame.empty and "symbol" in frame.columns:
            symbols = frame["symbol"].astype(str)
            prefixes = {symbol[:1].upper() for symbol in symbols if symbol}
            prefixes.discard("")
            return {
                "source": label,
                "unique_prefixes": sorted(prefixes),
                "ok": len(prefixes) > 1,
                "row_count": int(len(frame.index)),
            }
    return {"source": None, "unique_prefixes": [], "ok": False, "row_count": 0}


def _build_kpis(universe: Mapping[str, Any], candidates: Mapping[str, Any], gate: Mapping[str, Any], timings: Mapping[str, Any], executor: Mapping[str, Any], predictions: Mapping[str, Any], trades_present: bool) -> dict[str, Any]:
    kpis = {
        "symbols_in": universe.get("symbols_in", 0),
        "symbols_with_bars": universe.get("symbols_with_bars", 0),
        "symbols_with_any_bars": universe.get("symbols_with_any_bars", 0),
        "symbols_with_required_bars": universe.get("symbols_with_required_bars", universe.get("symbols_with_bars", 0)),
        "candidate_rows": candidates.get("latest", {}).get("row_count")
        or candidates.get("top", {}).get("row_count")
        or candidates.get("scored", {}).get("row_count")
        or 0,
        "candidate_source": universe.get("source") or "unknown",
        "gate_fail_total": gate.get("total", 0),
        "fetch_secs": timings.get("fetch_secs", 0.0),
        "feature_secs": timings.get("feature_secs", 0.0),
        "rank_secs": timings.get("rank_secs", 0.0),
        "gate_secs": timings.get("gate_secs", 0.0),
        "orders_submitted": executor.get("orders_submitted", 0),
        "orders_filled": executor.get("orders_filled", 0),
        "orders_canceled": executor.get("orders_canceled", 0),
        "trailing_attached": executor.get("trailing_attached", 0),
        "trades_log_present": trades_present,
        "predictions_rows": predictions.get("prediction_rows", 0),
    }
    return kpis


def _write_kpis_csv(path: Path, kpis: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(kpis):
            writer.writerow([key, kpis[key]])


def _write_findings(path: Path, report: Mapping[str, Any]) -> None:
    checks = report.get("checks", {})
    timestamps = checks.get("timestamps", {})
    candidates = checks.get("candidates", {})
    executor = checks.get("executor", {})
    trades = checks.get("trades_log", {})
    prefix = checks.get("prefix_sanity", {})
    universe = checks.get("universe", {})

    lines = [
        f"Generated at: {report.get('generated_at_utc', 'n/a')}",
        f"Screener last run: {timestamps.get('screener_last_run_utc', 'n/a')} (pipeline end {timestamps.get('pipeline_end_utc', 'n/a')})",
    ]
    if universe:
        lines.append(
            "Any bars fetched: {any_bars} | Required bars (>= {required_bars}): {required_bars_count} | Scored symbols: {symbols_in}".format(
                any_bars=universe.get("symbols_with_any_bars", 0),
                required_bars=universe.get("required_bars", 0) or 0,
                required_bars_count=universe.get("symbols_with_required_bars", 0),
                symbols_in=universe.get("symbols_in", 0),
            )
        )
    latest = candidates.get("latest", {})
    source = checks.get("universe", {}).get("source") or "unknown"
    lines.append(
        f"Candidates: {latest.get('row_count', 0)} rows from {source}; fallback used: {source.startswith('fallback') if isinstance(source, str) else False}"
    )
    canonical_flag = latest.get("canonical")
    missing_breakdown = bool(latest.get("missing_score_breakdown"))
    if canonical_flag is None:
        lines.append("candidates_header=canonical:unknown")
    else:
        line = f"candidates_header=canonical:{str(bool(canonical_flag)).lower()}"
        if not canonical_flag:
            details: list[str] = []
            if missing_breakdown:
                details.append("missing_score_breakdown")
            columns_snapshot = latest.get("columns") or latest.get("canonical_sequence")
            if columns_snapshot:
                details.append(f"columns={columns_snapshot}")
            if details:
                line = f"{line} (" + ", ".join(str(part) for part in details) + ")"
        lines.append(line)
    lines.append(
        f"Trades log present: {trades.get('present', False)}; Orders submitted (today): {executor.get('orders_submitted', 0)}"
    )
    if prefix.get("ok") is False and prefix.get("row_count", 0) > 0:
        lines.append(
            f"Prefix guard triggered: symbols use prefixes {prefix.get('unique_prefixes', [])} from source {prefix.get('source')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_evidence(
    report: Mapping[str, Any], *, base_dir: Path | str = BASE_DIR, evidence_dir: Path | str | None = None
) -> Path:
    base = Path(base_dir)
    destination = Path(evidence_dir) if evidence_dir else base / "reports" / "evidence"
    destination.mkdir(parents=True, exist_ok=True)

    pipeline_tail = _read_tail(base / "logs" / "pipeline.log")
    (destination / "pipeline_tail.log").write_text(pipeline_tail, encoding="utf-8")

    execute_tail = _read_tail(base / "logs" / "execute_trades.log")
    (destination / "execute_tail.log").write_text(execute_tail, encoding="utf-8")

    checks = report.get("checks", {}) if isinstance(report, Mapping) else {}
    candidates = checks.get("candidates", {}) if isinstance(checks, Mapping) else {}

    pipeline_tokens = checks.get("pipeline_tokens", {}) if isinstance(checks, Mapping) else {}
    (destination / "pipeline_tokens.json").write_text(
        json.dumps(pipeline_tokens, indent=2, default=str), encoding="utf-8"
    )

    executor_tokens = checks.get("executor", {}).get("tokens", {}) if isinstance(checks, Mapping) else {}
    (destination / "executor_tokens.json").write_text(
        json.dumps(executor_tokens, indent=2, default=str), encoding="utf-8"
    )

    header_snapshot: dict[str, Any] = {}
    for label in ("latest", "top", "scored"):
        info = candidates.get(label, {}) if isinstance(candidates, Mapping) else {}
        header_snapshot[label] = {
            "columns": info.get("columns"),
            "canonical_sequence": info.get("canonical_sequence"),
            "missing_canonical": info.get("missing_canonical"),
            "first_rows": info.get("first_rows"),
        }
    (destination / "candidate_headers.json").write_text(
        json.dumps(header_snapshot, indent=2, default=str), encoding="utf-8"
    )

    screener_metrics, _ = _safe_read_json(base / "data" / "screener_metrics.json")
    execute_metrics, _ = _safe_read_json(base / "data" / "execute_metrics.json")
    metrics_summary_df, _ = _safe_read_csv(base / "data" / "metrics_summary.csv")
    metrics_snapshot: dict[str, Any] = {
        "screener_metrics": screener_metrics,
        "execute_metrics": execute_metrics,
        "metrics_summary_latest": [],
    }
    if metrics_summary_df is not None and not metrics_summary_df.empty:
        latest_row = metrics_summary_df.tail(1).to_dict(orient="records")
        metrics_snapshot["metrics_summary_latest"] = json.loads(
            json.dumps(latest_row, default=str)
        )
    (destination / "metrics_snapshot.json").write_text(
        json.dumps(metrics_snapshot, indent=2, default=str), encoding="utf-8"
    )

    reports_root = base / "reports"
    evidence_copies = {
        "dashboard_consistency.json": "dashboard_consistency.json",
        "dashboard_kpis.csv": "dashboard_kpis.csv",
        "dashboard_findings.txt": "dashboard_findings.txt",
        "docs_findings.txt": "docs_findings.txt",
    }
    for src_name, dest_name in evidence_copies.items():
        src_path = reports_root / src_name
        if not src_path.exists():
            continue
        try:
            shutil.copy2(src_path, destination / dest_name)
        except Exception:
            LOGGER.warning("Failed to copy evidence artifact %s", src_path)

    return destination


def _assert_api_payload(base: Path) -> str | None:
    try:
        os.environ["JBRAVO_HOME"] = str(base)
        data_io = importlib.import_module("dashboards.data_io")
        data_io = importlib.reload(data_io)
    except Exception as exc:  # pragma: no cover - defensive import guard
        return f"[API] Unable to load dashboard modules: {exc}"
    try:
        loader_snapshot = data_io.screener_health()
        api_snapshot = data_io.health_payload_for_api()
    except Exception as exc:  # pragma: no cover - defensive loader guard
        return f"[API] Failed to evaluate loader payloads: {exc}"
    if loader_snapshot != api_snapshot:
        return "[API] /api/health payload differs from screener_health() output"
    return None


def run_assertions(base_dir: Path) -> list[str]:
    errors: list[str] = []
    data_dir = base_dir / "data"

    metrics, _ = _safe_read_json(data_dir / "screener_metrics.json")
    if not isinstance(metrics, Mapping):
        metrics = {}

    latest_df, latest_info = _safe_read_csv(data_dir / "latest_candidates.csv")
    _, top_info = _safe_read_csv(data_dir / "top_candidates.csv")
    top_rows = _coerce_int(top_info.get("rows"))
    rows_metric = _coerce_int(metrics.get("rows"))
    if top_rows is None or rows_metric is None or top_rows != rows_metric:
        errors.append(
            f"[PARITY] top_candidates.csv rows={top_rows} does not match screener_metrics.json rows={rows_metric}"
        )
    latest_rows = _coerce_int(latest_info.get("rows"))
    if latest_rows in (None, 0):
        errors.append("[FALLBACK] latest_candidates.csv empty (header-only)")

    for key in ("bars_rows_total_fetch", "symbols_with_any_bars", "symbols_with_required_bars"):
        if _coerce_int(metrics.get(key)) is None:
            errors.append(f"[FIELDS] screener_metrics.json missing numeric {key}")

    bars_rows_total_fetch = _coerce_int(metrics.get("bars_rows_total_fetch"))
    bars_rows_total = _coerce_int(metrics.get("bars_rows_total"))
    if bars_rows_total_fetch is not None and bars_rows_total is not None:
        if bars_rows_total_fetch != bars_rows_total:
            diff = bars_rows_total_fetch - bars_rows_total
            pct = (diff / bars_rows_total * 100) if bars_rows_total else None
            pct_str = f"{pct:.2f}%" if pct is not None else "n/a"
            LOGGER.warning(
                "[WARN] bars_rows_total_fetch=%s differs from bars_rows_total=%s (diff=%s pct=%s)",
                bars_rows_total_fetch,
                bars_rows_total,
                diff,
                pct_str,
            )

    conn_payload, _ = _safe_read_json(data_dir / "connection_health.json")
    trading_ok = _coerce_bool(conn_payload.get("trading_ok")) if isinstance(conn_payload, Mapping) else None
    data_ok = _coerce_bool(conn_payload.get("data_ok")) if isinstance(conn_payload, Mapping) else None
    if trading_ok is not True or data_ok is not True:
        errors.append(
            f"[CONN] connection_health.json requires trading_ok && data_ok for green badge (found trading_ok={trading_ok} data_ok={data_ok})"
        )

    api_error = _assert_api_payload(base_dir)
    if api_error:
        errors.append(api_error)

    return errors


def generate_report(base_dir: Path | str = BASE_DIR, reports_dir: Path | str | None = None) -> dict[str, Any]:
    base = Path(base_dir)
    reports_path = Path(reports_dir) if reports_dir else base / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)

    screener_metrics, screener_info = _safe_read_json(base / "data" / "screener_metrics.json")
    metrics_summary_df, metrics_summary_info = _safe_read_csv(base / "data" / "metrics_summary.csv")
    latest_df, latest_info = _safe_read_csv(base / "data" / "latest_candidates.csv")
    top_df, top_info = _safe_read_csv(base / "data" / "top_candidates.csv")
    scored_df, scored_info = _safe_read_csv(base / "data" / "scored_candidates.csv")
    execute_metrics, execute_info = _safe_read_json(base / "data" / "execute_metrics.json")
    predictions_df, predictions_info = _safe_read_csv(base / "data" / "predictions" / "latest.csv")
    ranker_eval, ranker_info = _safe_read_json(base / "data" / "ranker_eval" / "latest.json")
    stage_fetch, stage_fetch_info = _safe_read_json(base / "data" / "screener_stage_fetch.json")
    stage_post, stage_post_info = _safe_read_json(base / "data" / "screener_stage_post.json")
    conn_health, conn_info = _safe_read_json(base / "data" / "connection_health.json")

    pipeline_log_text = _read_tail(base / "logs" / "pipeline.log")
    execute_log_text = _read_tail(base / "logs" / "execute_trades.log")

    pipeline_tokens = _parse_pipeline_tokens(pipeline_log_text)
    summary_data = pipeline_tokens.get("PIPELINE_SUMMARY", {}).get("data", {})

    universe = _gather_universe(screener_metrics, summary_data)
    candidates = {
        "latest": _analyze_candidates(latest_df, latest_info),
        "top": _analyze_candidates(top_df, top_info),
        "scored": _analyze_candidates(scored_df, scored_info),
    }
    gate = _gate_pressure(screener_metrics)
    timings = _timings(screener_metrics, summary_data)
    executor = _executor_summary(execute_metrics, execute_log_text)
    predictions = _predictions_summary(predictions_df, predictions_info, ranker_eval)
    prefix_guard = _prefix_sanity({"latest": latest_df, "scored": scored_df, "top": top_df})

    universe_rows = int(universe.get("rows", 0) or 0)
    top_rows = int(candidates["top"].get("row_count") or 0)
    if universe_rows == 0 and top_rows == 0:
        parity = "empty"
    elif universe_rows == top_rows:
        parity = "match"
    else:
        parity = "mismatch"
    LOGGER.info(
        "[CHECK] rows_final=%s top_candidates=%s parity=%s",
        universe_rows,
        top_rows,
        parity,
    )
    LOGGER.info(
        "[CHECK] connection_health present=%s trading_ok=%s data_ok=%s",
        conn_info.get("present", False),
        conn_health.get("trading_ok") if isinstance(conn_health, Mapping) else None,
        conn_health.get("data_ok") if isinstance(conn_health, Mapping) else None,
    )
    fetch_present = stage_fetch_info.get("present", False)
    post_present = stage_post_info.get("present", False)
    if fetch_present and post_present:
        split_state = "fetch+post"
    elif fetch_present:
        split_state = "fetch-only"
    else:
        split_state = "missing"
    LOGGER.info(
        "[CHECK] stage_metrics=%s fetch_symbols=%s post_candidates=%s",
        split_state,
        stage_fetch.get("symbols_with_bars_fetch") if isinstance(stage_fetch, Mapping) else None,
        stage_post.get("candidates_final") if isinstance(stage_post, Mapping) else None,
    )

    trades_log_path = base / "data" / "trades_log.csv"
    trades_info = _path_info(trades_log_path)

    timestamps = {
        "screener_last_run_utc": screener_metrics.get("last_run_utc") if isinstance(screener_metrics, Mapping) else None,
        "metrics_summary_mtime_utc": metrics_summary_info.get("mtime_utc"),
        "pipeline_start_utc": pipeline_tokens.get("PIPELINE_START", {}).get("timestamp_utc"),
        "pipeline_summary_utc": pipeline_tokens.get("PIPELINE_SUMMARY", {}).get("timestamp_utc"),
        "pipeline_end_utc": pipeline_tokens.get("PIPELINE_END", {}).get("timestamp_utc"),
    }

    candidates_ok = bool(
        (candidates["latest"].get("row_count") or 0) > 0
        or (candidates["top"].get("row_count") or 0) > 0
        or (isinstance(universe.get("source"), str) and universe["source"].startswith("fallback"))
    )

    metrics_summary = {
        "present": bool(metrics_summary_info.get("present")),
        "rows": metrics_summary_info.get("rows", 0),
        "columns": metrics_summary_info.get("columns", []),
        "latest_row": [],
    }
    if metrics_summary_df is not None and not metrics_summary_df.empty:
        latest_row = metrics_summary_df.tail(1).to_dict(orient="records")
        metrics_summary["latest_row"] = json.loads(json.dumps(latest_row, default=str))

    report = {
        "generated_at_utc": _isoformat(_utc_now()),
        "inputs": {
            "screener_metrics": screener_info,
            "metrics_summary": metrics_summary_info,
            "latest_candidates": latest_info,
            "top_candidates": top_info,
            "scored_candidates": scored_info,
            "execute_metrics": execute_info,
            "predictions_latest": predictions_info,
            "ranker_eval": ranker_info,
            "pipeline_log": _path_info(base / "logs" / "pipeline.log"),
            "execute_log": _path_info(base / "logs" / "execute_trades.log"),
            "trades_log": trades_info,
            "stage_fetch": stage_fetch_info,
            "stage_post": stage_post_info,
            "connection_health": conn_info,
        },
        "checks": {
            "timestamps": timestamps,
            "universe": universe,
            "candidates": candidates,
            "candidates_ok": candidates_ok,
            "gate_pressure": gate,
            "timings": timings,
            "pipeline_tokens": pipeline_tokens,
            "executor": executor,
            "trades_log": {"present": trades_info.get("present", False)},
            "predictions": predictions,
            "prefix_sanity": prefix_guard,
            "metrics_summary": metrics_summary,
        },
    }

    kpis = _build_kpis(universe, candidates, gate, timings, executor, predictions, trades_info.get("present", False))

    dashboard_json = reports_path / "dashboard_consistency.json"
    dashboard_csv = reports_path / "dashboard_kpis.csv"
    findings_txt = reports_path / "dashboard_findings.txt"

    with dashboard_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    _write_kpis_csv(dashboard_csv, kpis)
    _write_findings(findings_txt, report)

    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate dashboard consistency against pipeline artifacts")
    parser.add_argument("--base", default=str(BASE_DIR), help="Base directory for the repository")
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Optional directory for report outputs (defaults to <base>/reports)",
    )
    parser.add_argument(
        "--write-evidence",
        action="store_true",
        help="Collect an evidence bundle with tokens, log snippets, and metrics",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Optional directory for evidence outputs (defaults to <reports>/evidence)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.base).resolve()
    target_reports_dir = (
        Path(args.reports_dir).resolve() if args.reports_dir else base / "reports"
    )
    report = generate_report(base_dir=base, reports_dir=target_reports_dir)
    LOGGER.info("Dashboard consistency report written to %s", target_reports_dir)
    LOGGER.debug("Report summary: %s", json.dumps(report, indent=2, default=str))
    if args.write_evidence:
        evidence_target = (
            Path(args.evidence_dir).resolve()
            if args.evidence_dir
            else target_reports_dir / "evidence"
        )
        evidence_path = collect_evidence(report, base_dir=base, evidence_dir=evidence_target)
        LOGGER.info("Dashboard evidence bundle written to %s", evidence_path)
    failures = run_assertions(base)
    docs_failures: list[str] = []
    docs_root = base / "docs"
    if not docs_root.exists():
        fallback_docs_root = BASE_DIR / "docs"
        if fallback_docs_root.exists():
            docs_root = fallback_docs_root
    try:
        docs_failures, docs_findings_path = docs_checker.run_docs_consistency(
            base_dir=base,
            docs_dir=docs_root,
            reports_dir=target_reports_dir,
            cli_reference_path=docs_root / "reference" / "cli_reference.md",
            fail_on_missing_docs=False,
        )
        LOGGER.info("Docs consistency findings written to %s", docs_findings_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        docs_failures = [f"[DOCS] docs consistency check failed: {exc}"]
        docs_findings_path = target_reports_dir / "docs_findings.txt"
        docs_findings_path.parent.mkdir(parents=True, exist_ok=True)
        docs_findings_path.write_text(
            "DOCS_CONSISTENCY status=FAIL\n"
            f"Findings:\n- [DOCS] docs consistency check failed: {exc}\n",
            encoding="utf-8",
        )
        LOGGER.warning("Docs consistency check failed: %s", exc)
    failures.extend(docs_failures)
    critical_issues = list(failures)

    tokens = (report.get("checks", {}) or {}).get("pipeline_tokens", {})
    pipeline_end = tokens.get("PIPELINE_END", {})
    end_data = pipeline_end.get("data", {}) if isinstance(pipeline_end, Mapping) else {}
    if not pipeline_end.get("present"):
        critical_issues.append("PIPELINE_END missing")
    else:
        rc_value = end_data.get("rc")
        try:
            rc_int = int(rc_value)
        except Exception:
            rc_int = None
        if rc_int not in (0, None):
            critical_issues.append(f"PIPELINE_END rc={rc_value}")

    executor_checks = (report.get("checks", {}) or {}).get("executor", {})
    if not executor_checks.get("metrics_present"):
        critical_issues.append("execute_metrics.json missing or unreadable")

    conn_info = (report.get("inputs", {}) or {}).get("connection_health", {})
    if not conn_info.get("present"):
        critical_issues.append("connection_health.json missing")

    if critical_issues:
        try:
            send_alert(
                "JBRAVO dashboard consistency FAIL",
                {
                    "issues": critical_issues,
                    "generated_at": report.get("generated_at_utc"),
                },
            )
        except Exception:
            LOGGER.debug("DASHBOARD_ALERT_FAILED", exc_info=True)
    if failures:
        for failure in failures:
            LOGGER.error(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
