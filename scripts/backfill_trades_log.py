from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import requests
from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_COLUMNS = [
    "symbol",
    "qty",
    "entry_price",
    "exit_price",
    "entry_time",
    "exit_time",
    "net_pnl",
    "side",
    "order_type",
    "exit_reason",
    "mfe_pct",
    "exit_efficiency_pct",
]


class AccountActivityUnavailable(RuntimeError):
    """Raised when account activity endpoints are not available."""


def _bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
    }


def _alpaca_base_url() -> str:
    return (os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if ts is pd.NaT:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    return None


def _isoformat(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _request_with_backoff(fn, *, max_attempts: int = 5, logger: logging.Logger) -> Any:
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive fallback
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 and attempt < max_attempts:
                logger.warning("Rate limited (429). Backing off for %.2fs (attempt %s/%s)", delay, attempt, max_attempts)
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            raise


def _http_get_with_backoff(
    session: requests.Session,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    logger: logging.Logger,
    max_attempts: int = 5,
) -> requests.Response:
    delay = 1.0
    last_response: Optional[requests.Response] = None

    for attempt in range(1, max_attempts + 1):
        resp = session.get(url, headers=headers, params=params, timeout=30)
        last_response = resp
        if resp.status_code == 429 and attempt < max_attempts:
            logger.warning("Hit 429 from Alpaca. Sleeping %.2fs before retry %s/%s", delay, attempt, max_attempts)
            time.sleep(delay)
            delay = min(delay * 2, 16)
            continue
        if 500 <= resp.status_code < 600 and attempt < max_attempts:
            logger.warning(
                "Server error %s from Alpaca. Sleeping %.2fs before retry %s/%s", resp.status_code, delay, attempt, max_attempts
            )
            time.sleep(delay)
            delay = min(delay * 2, 16)
            continue
        return resp

    return last_response  # pragma: no cover - safety return


def fetch_account_fill_events(
    start: datetime,
    end: datetime,
    *,
    session: Optional[requests.Session] = None,
    logger: logging.Logger,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    """Pull fill activities from Alpaca's account activities endpoint."""

    session = session or requests.Session()
    base = _alpaca_base_url()
    url = f"{base}/v2/account/activities"
    headers = _alpaca_headers()

    params: Dict[str, Any] = {
        "activity_types": "FILL",
        "after": start.isoformat(),
        "until": end.isoformat(),
        "page_size": 100,
    }
    events: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        if page_token:
            params["page_token"] = page_token

        resp = _http_get_with_backoff(session, url, headers=headers, params=params, logger=logger, max_attempts=max_attempts)
        if resp is None:
            raise AccountActivityUnavailable("No response from account activities endpoint")
        if resp.status_code in (403, 404):
            raise AccountActivityUnavailable(f"Account activities unavailable (status={resp.status_code})")
        resp.raise_for_status()

        try:
            payload = resp.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise AccountActivityUnavailable(f"Unable to parse account activities payload: {exc}") from exc

        records: Iterable[Mapping[str, Any]]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("activities"), list):
            records = payload.get("activities", [])
        else:
            raise AccountActivityUnavailable("Unexpected account activities payload shape")

        for item in records:
            if not isinstance(item, Mapping):
                continue
            ts = _coerce_datetime(item.get("transaction_time") or item.get("date"))
            if ts is None:
                continue
            events.append(
                {
                    "symbol": _safe_str(item.get("symbol")).upper(),
                    "side": _safe_str(item.get("side")).lower(),
                    "qty": _safe_float(item.get("qty") or item.get("quantity")),
                    "price": _safe_float(item.get("price")),
                    "timestamp": ts,
                    "order_type": _safe_str(item.get("order_type") or item.get("type")),
                }
            )

        page_token = resp.headers.get("Next-Page-Token") or resp.headers.get("next-page-token")
        if not page_token and isinstance(payload, dict):
            page_token = _safe_str(payload.get("next_page_token"))
        if not page_token:
            break

    logger.info("Fetched %s fill activities from account endpoint", len(events))
    return events


def _clean_order_side(side: Any) -> str:
    if isinstance(side, OrderSide):
        return side.value
    return _safe_str(getattr(side, "value", side)).lower()


def fetch_order_fill_events(
    client: TradingClient,
    start: datetime,
    end: datetime,
    *,
    logger: logging.Logger,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    """Fallback: reconstruct fills from closed orders."""

    events: List[Dict[str, Any]] = []
    cursor_until = end

    while True:
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            after=start,
            until=cursor_until,
            direction=Sort.DESC,
            nested=False,
        )

        def _run_query():
            return client.get_orders(filter=request)

        orders = _request_with_backoff(_run_query, max_attempts=max_attempts, logger=logger)
        if not orders:
            break

        for order in orders:
            filled_at = getattr(order, "filled_at", None)
            if filled_at and filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
            ts = _coerce_datetime(filled_at)
            if ts is None or ts < start:
                continue
            events.append(
                {
                    "symbol": _safe_str(getattr(order, "symbol", "")).upper(),
                    "side": _clean_order_side(getattr(order, "side", "")),
                    "qty": _safe_float(getattr(order, "filled_qty", 0.0)),
                    "price": _safe_float(getattr(order, "filled_avg_price", 0.0)),
                    "timestamp": ts,
                    "order_type": _safe_str(getattr(order, "order_type", "")),
                }
            )

        last_order = orders[-1]
        cursor_until = getattr(last_order, "submitted_at", None) or getattr(last_order, "created_at", None) or getattr(last_order, "filled_at", None)
        if cursor_until is None or cursor_until <= start:
            break
        if cursor_until.tzinfo is None:
            cursor_until = cursor_until.replace(tzinfo=timezone.utc)

    logger.info("Fetched %s fill events from orders fallback", len(events))
    return events


def _net_pnl(entry_side: str, entry_price: float, exit_price: float, qty: float) -> float:
    if entry_side == "sell":
        return (entry_price - exit_price) * qty
    return (exit_price - entry_price) * qty


def build_trades_from_events(events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    positions: Dict[str, List[Dict[str, Any]]] = {}
    trades: List[Dict[str, Any]] = []

    sorted_events = sorted(
        (
            e
            for e in events
            if _safe_str(e.get("symbol"))
            and _safe_float(e.get("qty")) > 0
            and _coerce_datetime(e.get("timestamp")) is not None
        ),
        key=lambda e: _coerce_datetime(e.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
    )

    for event in sorted_events:
        symbol = _safe_str(event.get("symbol")).upper()
        side = _safe_str(event.get("side")).lower()
        qty = _safe_float(event.get("qty"))
        price = _safe_float(event.get("price"))
        ts = _coerce_datetime(event.get("timestamp"))
        order_type = _safe_str(event.get("order_type"))

        if not symbol or not side or qty <= 0 or price == 0 or ts is None:
            continue

        position = positions.setdefault(symbol, [])
        remaining = qty

        if position and position[0]["side"] != side:
            while remaining > 0 and position:
                lot = position[0]
                close_qty = min(remaining, lot["qty"])
                trades.append(
                    {
                        "symbol": symbol,
                        "qty": close_qty,
                        "entry_price": lot["price"],
                        "exit_price": price,
                        "entry_time": _isoformat(lot["timestamp"]),
                        "exit_time": _isoformat(ts),
                        "net_pnl": _net_pnl(lot["side"], lot["price"], price, close_qty),
                        "side": lot["side"],
                        "order_type": order_type or lot.get("order_type", ""),
                        "exit_reason": "",
                    }
                )
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] <= 0:
                    position.pop(0)

        if remaining > 0:
            position.append(
                {
                    "side": side,
                    "qty": remaining,
                    "price": price,
                    "timestamp": ts,
                    "order_type": order_type,
                }
            )

    return trades


def _trade_key(record: Mapping[str, Any]) -> str:
    return "|".join(
        [
            _safe_str(record.get("symbol")).upper(),
            _safe_str(record.get("entry_time")),
            _safe_str(record.get("exit_time")),
            _safe_str(record.get("qty")),
        ]
    )


def merge_trades(existing: pd.DataFrame, new_trades: List[Dict[str, Any]], *, merge_existing: bool = True) -> pd.DataFrame:
    new_df = pd.DataFrame(new_trades)
    if new_df.empty:
        if merge_existing:
            return existing.copy()
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    existing = existing.copy()
    if merge_existing and not existing.empty:
        existing_keys = {_trade_key(row) for row in existing.to_dict(orient="records")}
        filtered = [row for row in new_trades if _trade_key(row) not in existing_keys]
        combined = pd.concat([existing, pd.DataFrame(filtered)], ignore_index=True, sort=False)
    elif merge_existing:
        combined = new_df
    else:
        combined = new_df

    all_columns = list(dict.fromkeys(list(existing.columns) + DEFAULT_COLUMNS + list(new_df.columns)))
    for col in all_columns:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined[all_columns]
    return combined


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def gather_fill_events(
    start: datetime,
    end: datetime,
    *,
    logger: logging.Logger,
    session: Optional[requests.Session] = None,
    client: Optional[TradingClient] = None,
) -> List[Dict[str, Any]]:
    try:
        return fetch_account_fill_events(start, end, session=session, logger=logger)
    except AccountActivityUnavailable:
        logger.info("Account activities unavailable; falling back to closed orders")

    client = client or TradingClient(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        paper=True,
        url_override=_alpaca_base_url(),
    )
    return fetch_order_fill_events(client, start, end, logger=logger)


def backfill(days: int, out_path: Path, merge: bool) -> None:
    logger = logging.getLogger("backfill_trades_log")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    logger.info("Fetching trade fills between %s and %s", start.isoformat(), end.isoformat())

    events = gather_fill_events(start, end, logger=logger)
    trades = build_trades_from_events(events)
    logger.info("Built %s trades from %s fill events", len(trades), len(events))

    if merge and out_path.exists():
        existing_df = pd.read_csv(out_path)
    else:
        existing_df = pd.DataFrame(columns=DEFAULT_COLUMNS)

    merged_df = merge_trades(existing_df, trades, merge_existing=merge)
    _atomic_write(merged_df, out_path)
    logger.info("Wrote %s rows to %s", len(merged_df), out_path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill trades_log.csv from Alpaca paper history")
    parser.add_argument("--days", type=int, default=365, help="Lookback window in days")
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "data" / "trades_log.csv"), help="Output CSV path")
    parser.add_argument("--merge", type=_bool_arg, default=True, help="Merge with existing trades_log.csv")
    args = parser.parse_args(argv)

    destination = Path(args.out)
    if not destination.is_absolute():
        destination = BASE_DIR / destination

    backfill(args.days, destination, args.merge)


if __name__ == "__main__":
    main()
