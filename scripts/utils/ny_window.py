"""Utility to display the U.S. equities pre-market window in multiple zones."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


PREMARKET_START = time(4, 0)
PREMARKET_END = time(9, 30)
NY_TZ = ZoneInfo("America/New_York")
CHICAGO_TZ = ZoneInfo("America/Chicago")


def _resolve_reference_date(raw: str) -> datetime:
    raw = (raw or "now").strip().lower()
    if raw in {"", "now", "today"}:
        return datetime.now(tz=NY_TZ)
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=NY_TZ)
        return parsed.astimezone(NY_TZ)
    except ValueError as exc:
        raise SystemExit(f"Invalid --when value '{raw}': {exc}")


def compute_window(reference: datetime) -> dict[str, tuple[datetime, datetime]]:
    ref_date = reference.astimezone(NY_TZ).date()
    start_ny = datetime.combine(ref_date, PREMARKET_START, tzinfo=NY_TZ)
    end_ny = datetime.combine(ref_date, PREMARKET_END, tzinfo=NY_TZ)
    start_utc = start_ny.astimezone(timezone.utc)
    end_utc = end_ny.astimezone(timezone.utc)
    start_chi = start_ny.astimezone(CHICAGO_TZ)
    end_chi = end_ny.astimezone(CHICAGO_TZ)
    return {
        "New York": (start_ny, end_ny),
        "Chicago": (start_chi, end_chi),
        "UTC": (start_utc, end_utc),
    }


def format_window(name: str, window: tuple[datetime, datetime]) -> str:
    start, end = window
    return f"{name}: {start.isoformat()} -> {end.isoformat()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show pre-market window timing")
    parser.add_argument(
        "--when",
        default="now",
        help="Reference point (ISO date or 'now', default: now)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = _resolve_reference_date(args.when)
    windows = compute_window(reference)
    print(f"[INFO] PREMARKET_REFERENCE {reference.date().isoformat()}")
    for label, window in windows.items():
        print(f"[INFO] {format_window(label, window)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
