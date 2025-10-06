"""Generate the weekly execution retrospective report."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATE_DIR = REPORTS_DIR / "templates"
TEMPLATE_NAME = "weekly_retrospective.md.j2"
LOOKBACK_DAYS = 7
API_FAILURE_THRESHOLD = 5
API_RETRY_THRESHOLD = 10
LATENCY_P95_THRESHOLD_MS = 60_000


@dataclass
class MetricsDocument:
    """Container for execute metrics JSON payloads."""

    path: Path
    timestamp: datetime
    data: dict[str, Any]


def parse_timestamp(raw: Any) -> datetime | None:
    """Parse ``raw`` into an aware UTC ``datetime`` if possible."""

    if not raw:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if not isinstance(raw, str):
        return None

    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.endswith("Z") or candidate.endswith("z"):
        candidate = candidate[:-1] + "+00:00"

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(candidate, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def extract_event_timestamp(event: dict[str, Any]) -> datetime | None:
    """Return the timestamp embedded in an event payload."""

    for key in ("timestamp", "ts", "time", "created_at"):
        ts = parse_timestamp(event.get(key))
        if ts:
            return ts
    details = event.get("details")
    if isinstance(details, dict):
        for key in ("timestamp", "ts"):
            ts = parse_timestamp(details.get(key))
            if ts:
                return ts
    return None


def iter_jsonl_files() -> Iterable[Path]:
    """Yield candidate JSONL files from data and logs directories."""

    for root in {DATA_DIR, LOG_DIR}:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if path.is_file():
                yield path


def load_recent_events(since: datetime) -> list[dict[str, Any]]:
    """Load JSONL events newer than ``since``."""

    events: list[dict[str, Any]] = []
    for path in iter_jsonl_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = extract_event_timestamp(payload)
            if ts is None or ts < since:
                continue
            events.append(payload)
    return events


def load_recent_metrics(since: datetime) -> list[MetricsDocument]:
    """Load execute metrics JSON documents newer than ``since``."""

    metrics_docs: list[MetricsDocument] = []
    for root in {DATA_DIR, LOG_DIR}:
        if not root.exists():
            continue
        for path in root.rglob("execute_metrics*.json"):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ts = parse_timestamp(data.get("generated_at")) or parse_timestamp(data.get("timestamp"))
            if ts is None:
                ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if ts < since:
                continue
            metrics_docs.append(MetricsDocument(path=path, timestamp=ts, data=data))
    return metrics_docs


def percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile from ``values`` using linear interpolation."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (rank - lower_index)


def build_context(events: list[dict[str, Any]], metrics_docs: list[MetricsDocument]) -> dict[str, Any]:
    """Aggregate metrics and prepare the template context."""

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=LOOKBACK_DAYS)

    event_counts = Counter()
    latencies_ms: list[float] = []
    exit_events = Counter()
    trailing_attaches = Counter()

    for event in events:
        name = str(event.get("event", "")).strip() or "unknown"
        event_counts[name] += 1

        latency_value = event.get("latency_ms")
        if latency_value is None and isinstance(event.get("details"), dict):
            latency_value = event["details"].get("latency_ms")
        if isinstance(latency_value, (int, float)):
            latencies_ms.append(float(latency_value))

        symbol = event.get("symbol")
        if not symbol and isinstance(event.get("details"), dict):
            symbol = event["details"].get("symbol")
        if not symbol:
            meta = event.get("meta")
            if isinstance(meta, dict):
                symbol = meta.get("symbol")

        if symbol:
            if name in {"EXIT_SUBMIT", "EXIT_FINAL", "EXIT_ORDER", "POSITION_EXIT"}:
                exit_events[symbol] += 1
            if name == "TRAILING_STOP_ATTACH":
                trailing_attaches[symbol] += 1

    orders_submitted = sum(doc.data.get("orders_submitted", 0) for doc in metrics_docs)
    if not orders_submitted:
        orders_submitted = event_counts.get("ORDER_SUBMIT", 0)

    api_failures = event_counts.get("API_ERROR", 0)
    if not api_failures:
        api_failures = sum(doc.data.get("api_failures", 0) for doc in metrics_docs)

    api_retries = event_counts.get("RETRY", 0)

    totals = {
        "orders_submitted": int(orders_submitted),
        "api_retries": int(api_retries),
        "api_failures": int(api_failures),
    }

    if latencies_ms:
        latency_p50 = int(round(median(latencies_ms)))
        latency_p95 = int(round(percentile(latencies_ms, 95)))
    else:
        # Fallback to most recent metrics document if available
        latency_p50 = 0
        latency_p95 = 0
        if metrics_docs:
            latest = max(metrics_docs, key=lambda doc: doc.timestamp)
            latency_p50 = int(latest.data.get("order_latency_ms_p50", 0) or 0)
            latency_p95 = int(latest.data.get("order_latency_ms_p95", 0) or 0)

    latency = {
        "p50": latency_p50,
        "p95": latency_p95,
    }

    combined_symbols = set(exit_events) | set(trailing_attaches)
    top_symbols = []
    for symbol in combined_symbols:
        exits = exit_events.get(symbol, 0)
        attaches = trailing_attaches.get(symbol, 0)
        top_symbols.append(
            {
                "symbol": symbol,
                "exits": int(exits),
                "trailing_stops": int(attaches),
                "total": int(exits + attaches),
            }
        )
    top_symbols.sort(key=lambda item: (-item["total"], -item["exits"], item["symbol"]))
    top_symbols = top_symbols[:5]

    anomalies: list[str] = []
    if totals["api_failures"] > API_FAILURE_THRESHOLD:
        anomalies.append(
            f"API failures exceeded threshold ({totals['api_failures']} > {API_FAILURE_THRESHOLD})."
        )
    if totals["api_retries"] > API_RETRY_THRESHOLD:
        anomalies.append(
            f"Elevated API retries observed ({totals['api_retries']} > {API_RETRY_THRESHOLD})."
        )
    if latency["p95"] > LATENCY_P95_THRESHOLD_MS:
        anomalies.append(
            "Order latency p95 is above 60 seconds; investigate broker/API responsiveness."
        )

    context = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "week_start": since.date().isoformat(),
        "week_end": now.date().isoformat(),
        "totals": totals,
        "latency": latency,
        "top_symbols": top_symbols,
        "anomalies": anomalies,
    }
    return context


def render_report(context: dict[str, Any]) -> Path:
    """Render the Jinja template to the reports directory."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template(TEMPLATE_NAME)

    report_date = context["week_end"]
    output_path = REPORTS_DIR / f"weekly_{report_date}.md"
    output_path.write_text(template.render(**context), encoding="utf-8")
    return output_path


def main() -> None:
    """Load inputs, aggregate metrics, and write the report."""

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=LOOKBACK_DAYS)
    events = load_recent_events(since)
    metrics_docs = load_recent_metrics(since)
    context = build_context(events, metrics_docs)
    output_path = render_report(context)
    print(f"Weekly retrospective written to {output_path}")


if __name__ == "__main__":
    main()
