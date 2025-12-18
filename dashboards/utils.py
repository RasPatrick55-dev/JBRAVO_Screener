from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict


def safe_tail_text(path: Path, max_lines: int = 200) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        logger.warning("TAIL_FAIL path=%s err=missing", path)
        return ""
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("TAIL_FAIL path=%s err=%s", path, exc)
        return ""
    trimmed = [line.rstrip("\n") for line in lines[-max_lines:]]
    return "\n".join(trimmed)


def file_stat(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        mtime_iso = None
        try:
            from datetime import datetime

            mtime_iso = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            mtime_iso = None
        return {"exists": True, "mtime_iso": mtime_iso, "size_bytes": stat.st_size}
    except FileNotFoundError:
        return {"exists": False, "mtime_iso": None, "size_bytes": None}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("FILE_STAT_FAIL path=%s err=%s", path, exc)
        return {"exists": False, "mtime_iso": None, "size_bytes": None}


def list_recent_files(dir_path: Path, pattern: str, limit: int = 30) -> list[Path]:
    if not dir_path.exists():
        return []
    try:
        files = sorted(
            dir_path.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:  # pragma: no cover - defensive logging
        return []
    return files[:limit]


def parse_pipeline_tokens(path: Path, limit_runs: int = 10) -> list[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        logger.warning("PIPELINE_TOKEN_PARSE_FAIL path=%s err=%s", path, "missing")
        return []
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("PIPELINE_TOKEN_PARSE_FAIL path=%s err=%s", path, exc)
        return []

    runs: list[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if "PIPELINE START" in line:
            if current:
                runs.append(current)
            current = {"start": line}
        elif "PIPELINE END" in line and current is not None:
            current["end"] = line
            runs.append(current)
            current = None
        elif "PIPELINE SUMMARY" in line and current is not None:
            current["summary"] = line
        elif "FALLBACK_CHECK" in line and current is not None:
            current.setdefault("tokens", []).append(line)

    if current:
        runs.append(current)

    parsed: list[Dict[str, Any]] = []
    for run in runs[::-1][:limit_runs]:
        start_match = re.search(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", run.get("start", ""))
        end_match = re.search(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", run.get("end", ""))
        summary = run.get("summary", "")
        summary_match = re.search(
            r"symbols_in=(?P<symbols_in>\d+)\s+with_bars=(?P<with_bars>\d+)\s+rows=(?P<rows>\d+)(?:[^\n\r]*?(?:\sbar_rows|\sbars_rows_total)=(?P<bars_rows_total>\d+))?",
            summary,
        )
        rc_match = re.search(r"rc=(?P<rc>-?\d+)", run.get("end", ""))
        start_ts = start_match.group("ts") if start_match else None
        end_ts = end_match.group("ts") if end_match else None
        duration = None
        if start_ts and end_ts:
            try:
                from datetime import datetime

                dt_start = datetime.fromisoformat(start_ts)
                dt_end = datetime.fromisoformat(end_ts)
                duration = (dt_end - dt_start).total_seconds()
            except Exception:
                duration = None

        parsed.append(
            {
                "start_time": start_ts,
                "end_time": end_ts,
                "rc": int(rc_match.group("rc")) if rc_match else None,
                "duration": duration,
                "symbols_in": int(summary_match.group("symbols_in")) if summary_match else None,
                "with_bars": int(summary_match.group("with_bars")) if summary_match else None,
                "rows": int(summary_match.group("rows")) if summary_match else None,
                "bars_rows_total": int(summary_match.group("bars_rows_total")) if summary_match and summary_match.group("bars_rows_total") else None,
                "source": "fallback" if any("FALLBACK_CHECK" in t for t in run.get("tokens", [])) else "screener",
            }
        )

    return parsed

logger = logging.getLogger(__name__)


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, "non-object payload")
    except FileNotFoundError:
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, "missing")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("METRICS_READ_FAIL json path=%s err=%s", path, exc)
    return {}


def safe_read_metrics_csv(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if isinstance(row, dict):
            return row
    except FileNotFoundError:
        logger.warning("METRICS_READ_FAIL csv path=%s err=%s", path, "missing")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("METRICS_READ_FAIL csv path=%s err=%s", path, exc)
    return {}


def parse_pipeline_summary(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        logger.warning("PIPELINE_SUMMARY_PARSE_FAIL path=%s err=%s", path, "missing")
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("PIPELINE_SUMMARY_PARSE_FAIL path=%s err=%s", path, exc)
        return {}

    for line in reversed(text.splitlines()):
        if "PIPELINE_SUMMARY" not in line:
            continue
        match = re.search(
            r"symbols_in=(\d+)\s+with_bars=(\d+)\s+rows=(\d+)(?:[^\n\r]*?(?:\sbar_rows|\sbars_rows_total)=(\d+))?",
            line,
        )
        if match:
            payload = {
                "last_run_utc": None,
                "symbols_in": int(match.group(1)),
                "symbols_with_bars": int(match.group(2)),
                "rows": int(match.group(3)),
            }
            if match.group(4):
                payload["bars_rows_total"] = int(match.group(4))
            return payload
        break
    return {}


def coerce_kpi_types(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def _raw_value(*keys: str) -> Any:
        for key in keys:
            if key in metrics:
                return metrics.get(key)
        return None

    result: Dict[str, Any] = {}
    int_fields = {
        "symbols_in": ("symbols_in",),
        "symbols_with_bars": ("symbols_with_bars", "with_bars"),
        "bars_rows_total": ("bars_rows_total",),
        "rows": ("rows", "rows_out"),
    }

    for canonical, aliases in int_fields.items():
        value = _raw_value(*aliases)
        if value in ("", None):
            result[canonical] = None
            continue
        try:
            result[canonical] = int(value)
        except (TypeError, ValueError):
            result[canonical] = None

    result["last_run_utc"] = (
        _raw_value("last_run_utc", "timestamp", "last_run") or None
    )
    return result
