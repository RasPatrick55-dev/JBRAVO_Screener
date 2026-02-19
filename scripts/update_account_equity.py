"""Refresh ``data/account_equity.csv`` from Alpaca portfolio history."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from scripts.utils.env import load_env, trading_base_url

FIELDNAMES = ["timestamp", "equity", "cash", "buying_power", "source"]
DEFAULT_OUTPUT = Path("data/account_equity.csv")


def _bool_arg(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _alpaca_headers() -> dict[str, str]:
    try:
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
    except KeyError as exc:  # pragma: no cover - validated by load_env
        raise RuntimeError(f"Missing Alpaca credential: {exc.args[0]}") from exc
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _fetch_portfolio_history(
    base_url: str, period: str, timeframe: str, extended_hours: bool
) -> dict:
    params = {
        "period": period,
        "timeframe": timeframe,
        "intraday_reporting": "extended_hours" if extended_hours else "market_hours",
    }
    resp = requests.get(
        f"{base_url}/v2/account/portfolio/history",
        headers=_alpaca_headers(),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_account_snapshot(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/v2/account", headers=_alpaca_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def _write_rows(rows: Iterable[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_history_rows(history: dict) -> list[dict[str, object]]:
    timestamps = history.get("timestamp") or []
    equities = history.get("equity") or []
    rows: list[dict[str, object]] = []
    for raw_ts, raw_equity in zip(timestamps, equities):
        try:
            ts = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        except (TypeError, ValueError):
            continue
        equity_value = ""
        try:
            equity_value = float(raw_equity) if raw_equity is not None else ""
        except (TypeError, ValueError):
            equity_value = ""
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "equity": equity_value,
                "cash": "",
                "buying_power": "",
                "source": "alpaca:portfolio_history",
            }
        )
    return rows


def _snapshot_row(snapshot: dict) -> dict[str, object]:
    def _coerce(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    now = datetime.now(timezone.utc).isoformat()
    return {
        "timestamp": now,
        "equity": _coerce(snapshot.get("equity")),
        "cash": _coerce(snapshot.get("cash")),
        "buying_power": _coerce(snapshot.get("buying_power")),
        "source": "alpaca:account_snapshot",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update account equity history")
    parser.add_argument("--period", default="1M", help="Lookback window (e.g. 1M, 3M, 1A)")
    parser.add_argument("--timeframe", default="1D", help="Aggregation period (1D, 1H, etc.)")
    parser.add_argument(
        "--extended-hours",
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        metavar="BOOL",
        help="Include extended hours (true/false). Defaults to false.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Destination CSV (default: data/account_equity.csv)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _, missing = load_env()
    if missing:
        raise SystemExit(f"Missing required environment keys: {', '.join(missing)}")

    base_url = trading_base_url().rstrip("/")
    history = _fetch_portfolio_history(
        base_url, args.period, args.timeframe, bool(args.extended_hours)
    )
    snapshot = _fetch_account_snapshot(base_url)

    rows = _build_history_rows(history)
    rows.append(_snapshot_row(snapshot))
    _write_rows(rows, Path(args.out))
    print(f"Wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
