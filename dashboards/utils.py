from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo


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


def tail_lines(path: Path, max_lines: int) -> list[str]:
    """Return up to ``max_lines`` from the end of the file efficiently.

    The implementation avoids reading the entire file into memory by
    iteratively seeking from the end in fixed-size blocks until enough
    lines are collected or the start of the file is reached.
    """

    if max_lines <= 0:
        return []

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = bytearray()
            lines_found = 0
            block_size = 8192

            while position > 0 and lines_found <= max_lines:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                buffer[:0] = handle.read(read_size)
                lines_found = buffer.count(b"\n")

            text = buffer.decode("utf-8", errors="ignore")
    except FileNotFoundError:
        logger.warning("TAIL_FAIL path=%s err=missing", path)
        return []
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("TAIL_FAIL path=%s err=%s", path, exc)
        return []

    return text.splitlines()[-max_lines:]


def safe_tail_lines(path: Path, max_lines: int = 2000) -> list[str]:
    """Return up to ``max_lines`` lines from the end of ``path``.

    Falls back to an empty list on any exception while logging the
    failure for observability.
    """

    try:
        return tail_lines(path, max_lines)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("TAIL_FAIL path=%s err=%s", path, exc)
        return []


_LOG_TS_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:,(?P<ms>\d{3}))?"
)


def _parse_timestamp_to_tz(line: str, tz: timezone) -> datetime | None:
    match = _LOG_TS_RE.search(line)
    if not match:
        return None
    ts_text = match.group("ts").replace(" ", "T")
    ms = match.group("ms")
    if ms:
        ts_text = f"{ts_text}.{ms}"
    try:
        dt = datetime.fromisoformat(ts_text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(tz)
    except Exception:
        return None


def parse_timed_events_from_logs(
    pipeline_path: Path, execute_path: Path, ny_tz: str = "America/New_York"
) -> list[dict]:
    """Parse key timed events from pipeline and execute logs for the current day.

    The function reads bounded tails of both logs to avoid loading large files into
    memory and returns a list of dictionaries sorted by timestamp (descending)
    containing normalized timeline entries.
    """

    try:
        tz = ZoneInfo(ny_tz)
        tz_label = ny_tz
    except Exception:
        tz = timezone.utc
        tz_label = "UTC (tzdata unavailable)"

    today = datetime.now(tz).date()

    severity_overrides = {
        "AUTH_FAIL": "error",
        "API_FAIL": "error",
        "FALLBACK_CHECK": "warn",
        "BUY_CANCELLED": "warn",
        "SKIP": "warn",
        "EXECUTE_SKIP": "warn",
    }

    def _severity_for(event: str) -> str:
        return severity_overrides.get(event, "info")

    def _search_first(text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            found = re.search(pattern, text)
            if found:
                return found.group(1)
        return None

    def _extract_fields(text: str) -> list[tuple[str, str]]:
        """Return ordered key/value pairs found in ``text``.

        This helper prioritizes the commonly requested keys and falls back to
        generic ``key=value`` pairs, preserving the order of appearance.
        """

        fields: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(key: str, value: str | None) -> None:
            if value is None:
                return
            norm_key = key.strip()
            if not norm_key or norm_key in seen:
                return
            seen.add(norm_key)
            fields.append((norm_key, value.strip()))

        _add("symbol", _search_first(text, [r"\bsymbol[:=]\s*([A-Z0-9\.-]+)", r"\bticker[:=]\s*([A-Z0-9\.-]+)"]))
        _add(
            "symbol",
            _search_first(
                text,
                [
                    r"\bfor\s+([A-Z]{1,6})(?:\b|,)",
                    r"\b([A-Z]{1,6}) order",
                ],
            ),
        )
        _add("qty", _search_first(text, [r"\bqty[:=]\s*([\d\.]+)", r"\bquantity[:=]\s*([\d\.]+)"]))
        _add("order_id", _search_first(text, [r"\border[_ ]?id[:=]\s*([\w-]+)"]))
        _add("rc", _search_first(text, [r"\brc[:=]\s*(-?\d+)"]))
        _add("reason", _search_first(text, [r"\breason[:=]\s*([^;|]+)", r"\breason\s+(.+)"]))

        for match in re.finditer(r"([A-Za-z_][\w]*)[:=]\s*([^\s,;]+)", text):
            key = match.group(1)
            value = match.group(2)
            _add(key, value)

        return fields

    def _details_text(text: str, fields: list[tuple[str, str]]) -> str:
        if fields:
            return " ".join(f"{k}={v}" for k, v in fields)
        cleaned = text.strip()
        return cleaned or "-"

    def _classify_event(line: str, source: str) -> tuple[str | None, str]:
        upper_line = line.upper()
        if source == "pipeline":
            mapping = {
                "PIPELINE START": "PIPELINE_START",
                "PIPELINE SUMMARY": "PIPELINE_SUMMARY",
                "PIPELINE END": "PIPELINE_END",
                "FALLBACK_CHECK": "FALLBACK_CHECK",
                "DASH RELOAD": "DASH_RELOAD",
            }
        else:
            mapping = {
                "EXEC_START": "EXEC_START",
                "EXECUTE SUMMARY": "EXECUTE_SUMMARY",
                "EXECUTE_SKIP": "EXECUTE_SKIP",
                "SKIP REASON": "SKIP",
                "AUTH FAIL": "AUTH_FAIL",
                "API_FAIL": "API_FAIL",
                "BUY_SUBMIT": "BUY_SUBMIT",
                "BUY_FILL": "BUY_FILL",
                "BUY_CANCELLED": "BUY_CANCELLED",
                "TRAIL_SUBMIT": "TRAIL_SUBMIT",
                "TRAIL_CONFIRMED": "TRAIL_CONFIRMED",
            }
        for token, label in mapping.items():
            if token in upper_line:
                return label, token

        if source == "execute":
            heuristics = [
                ("EXEC_START", [r"STARTING\s+PRE-MARKET\s+TRADE\s+EXECUTION"]),
                (
                    "EXECUTE_SUMMARY",
                    [r"SCRIPT\s+COMPLETE", r"EXECUTE\s+SUMMARY"],
                ),
                ("AUTH_FAIL", [r"AUTH(ENTICATION)?\s+FAILED", r"AUTH\s+ERROR"]),
                ("API_FAIL", [r"API\s+ERROR", r"HTTPERROR", r"CONNECTION\s+ERROR"]),
                (
                    "BUY_SUBMIT",
                    [r"SUBMITTING\s+LIMIT\s+BUY\s+ORDER", r"PLAC(ING|ED)\s+BUY\s+ORDER"],
                ),
                ("BUY_FILL", [r"ORDER\s+FILLED", r"FILL\s+CONFIRMED"]),
                ("BUY_CANCELLED", [r"CANCELLED\s+ORDER", r"ORDER\s+CANCELLED", r"ORDER\s+REJECTED"]),
                ("TRAIL_SUBMIT", [r"TRAILING\s+STOP", r"TRAIL\s+SUBMIT"]),
                ("TRAIL_CONFIRMED", [r"TRAIL\s+CONFIRMED", r"TRAILING\s+STOP\s+CREATED"]),
                ("SKIP", [r"SKIP\s+REASON", r"SKIPPING\s"]),
            ]
            for label, patterns in heuristics:
                for pattern in patterns:
                    if re.search(pattern, upper_line):
                        return label, pattern

        return None, ""

    def _normalize_event(line: str, source: str) -> dict | None:
        dt_local = _parse_timestamp_to_tz(line, tz)
        if not dt_local:
            return None
        event, marker = _classify_event(line, source)
        if not event:
            return None
        marker_upper = marker.upper() if marker else event
        try:
            idx = line.upper().index(marker_upper)
            tail = line[idx + len(marker_upper) :]
        except ValueError:
            tail = line
        fields = _extract_fields(tail)
        symbol = None
        for key, value in fields:
            if key.lower() in ("symbol", "ticker"):
                symbol = value
                break
        return {
            "dt": dt_local,
            "time_str": dt_local.strftime("%H:%M:%S"),
            "source": source,
            "event": event,
            "symbol": symbol,
            "severity": _severity_for(event),
            "details": _details_text(tail, fields),
            "tz_label": tz_label,
        }

    events: list[dict] = []

    for line in safe_tail_lines(pipeline_path, 2000):
        event = _normalize_event(line.strip(), "pipeline")
        if event:
            events.append(event)

    for line in safe_tail_lines(execute_path, 8000):
        event = _normalize_event(line.strip(), "execute")
        if event:
            events.append(event)

    today_events = [entry for entry in events if entry["dt"].date() == today]
    today_events.sort(key=lambda e: e["dt"], reverse=True)
    return today_events


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
