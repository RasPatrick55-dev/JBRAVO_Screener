"""Read-only log evidence. Log text is never broker transaction evidence."""

from datetime import datetime, timezone
import json
import re

STALE_AFTER_SECONDS = 48 * 3600


_TIMESTAMP = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:[.,][0-9]+)?(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?"
)
_LOG_TIMESTAMP = re.compile(r"^\s*\[?(" + _TIMESTAMP + r")(?=$|[\s\]])")


def _parse_timestamp(value, *, allow_naive=False):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(_TIMESTAMP, value)
    if not match:
        return None
    offset = match["offset"]
    if not offset and not allow_naive:
        return None
    # fromisoformat normalizes offsets such as +00:60 instead of rejecting them.
    if offset and offset != "Z" and (int(offset[1:3]) > 23 or int(offset[4:6]) > 59):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace(",", ".").replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _unique_json_fields(pairs):
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate JSON field")
        fields[key] = value
    return fields


def event_time(text):
    # Only a leading log timestamp uses the existing producer UTC convention.
    # Never borrow arbitrary dates from a message, path, or nested JSON data.
    match = _LOG_TIMESTAMP.match(text)
    if match:
        return _parse_timestamp(match[1], allow_naive=True)
    # PythonAnywhere monitor output has a yearless display prefix, then a JSON
    # result. The explicit completion time supplies the year and offset; the
    # display prefix and fetch time must never be used to guess either one.
    json_start = text.find("{")
    if json_start < 0:
        return None
    try:
        payload = json.loads(text[json_start:], object_pairs_hook=_unique_json_fields)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_timestamp(payload.get("finished_at"))


def log_provenance(text, source):
    # Exact emitter signatures, restricted to known backtest/pipeline sources.
    # This is format attribution, not proof of a particular runtime or a fill.
    known_source = bool(
        re.search(
            r"(?:^|[ /])(?:pipeline|backtest)(?:\.log)?$|(?:^|[/ ])(?:run_primary_pipeline\.sh|run_pipeline\.py|step\.backtest\.out)(?:[ \x27\x22]|$)",
            source.lower(),
        )
    )
    simulator = re.search(
        r"(?:Opened \S+ @ \d+\.\d{2} \(\d+ shares\)|Closed \S+ @ \d+\.\d{2} reason=\S+|Scaled out \S+: sold \d+ @ \d+\.\d{2} reason=\S+)\s*$",
        text,
    )
    if known_source and simulator:
        return {
            "provenance": "simulation_format",
            "provenance_label": "Simulation-format match · runtime unverified",
            "provenance_evidence": "scripts/backtest_core.py in-memory simulator emitter signature; runtime attribution not independently verified",
        }
    if re.search(
        r" -- (?:Started task|Completed task, took .+ seconds, return code was -?\d+\.)\s*$", text
    ):
        return {
            "provenance": "scheduler",
            "provenance_label": "Scheduler event",
            "provenance_evidence": "PythonAnywhere task wrapper format",
        }
    return {
        "provenance": "unclassified",
        "provenance_label": "Unclassified · not broker evidence",
        "provenance_evidence": None,
    }


def log_receipt(rows, observed_at=None):
    now = observed_at or datetime.now(timezone.utc)
    times = [event_time(str(row.get("ts_utc") or "")) for row in rows]
    valid = [value for value in times if value is not None]
    latest = max(valid) if valid else None
    age = (now - latest).total_seconds() if latest else None
    status = (
        "unknown" if age is None or age < 0 else "stale" if age > STALE_AFTER_SECONDS else "recent"
    )
    return {
        "observed_at": now.isoformat(),
        "latest_event_at": latest.isoformat() if latest else None,
        "event_age_seconds": age,
        "freshness": status,
        "stale_after_seconds": STALE_AFTER_SECONDS,
        "unknown_timestamp_count": sum(value is None for value in times),
        "timestamp_basis": "Explicit offsets preserved; offset-free leading task/log timestamps interpreted as UTC (producer convention); JSON finished_at requires an explicit offset",
    }


def annotate_log_text(text, source):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        timestamp = event_time(line)
        rows.append(
            {
                "raw": line,
                "ts_utc": timestamp.isoformat() if timestamp else None,
                **log_provenance(line, source),
            }
        )
    return {"entries": rows, **log_receipt(rows)}
