"""Read-only log evidence. Log text is never broker transaction evidence."""

from datetime import datetime, timezone
import re

STALE_AFTER_SECONDS = 48 * 3600


def event_time(text):
    # Do not accept a valid prefix of a malformed timestamp/offset.
    match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})"
        r"([.,]\d+)?(Z|[+-]\d{2}:\d{2})?(?=$|[\s\]])",
        text,
    )
    if not match:
        return None
    try:
        fraction = (match[3] or "").replace(",", ".")
        value = f"{match[1]}T{match[2]}{fraction}{match[4] or '+00:00'}"
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


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
        "timestamp_basis": "Explicit offsets preserved; offset-free task/log timestamps interpreted as UTC (producer convention)",
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
