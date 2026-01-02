"""Robust trade execution orchestrator for JBRAVO."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, time as dtime
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, unquote

from zoneinfo import ZoneInfo
from dateutil.parser import isoparse

NY = ZoneInfo("America/New_York")

import pandas as pd
import psycopg2
import requests

from scripts import db
from scripts.fallback_candidates import CANONICAL_COLUMNS, normalize_candidate_df
from scripts.utils.env import load_env
from utils import atomic_write_bytes, write_csv_atomic
from utils.alerts import send_alert
from utils.env import (
    AlpacaCredentialsError,
    AlpacaUnauthorizedError,
    assert_alpaca_creds,
    get_alpaca_creds,
    write_auth_error_artifacts,
)
import utils.telemetry as telemetry

try:  # pragma: no cover - import guard for optional dependency
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderDirection, OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        TrailingStopOrderRequest,
    )
except Exception:  # pragma: no cover - lightweight fallback for tests
    TradingClient = None  # type: ignore

    class APIError(Exception):
        def __init__(self, message: str = "", code: int | None = None) -> None:
            super().__init__(message)
            self.code = code or 0

    class _Enum(str):
        def __new__(cls, value: str):
            return str.__new__(cls, value)

    class OrderSide:  # type: ignore
        BUY = _Enum("buy")
        SELL = _Enum("sell")

    class OrderDirection:  # type: ignore
        ASC = _Enum("asc")
        DESC = _Enum("desc")

    class TimeInForce:  # type: ignore
        DAY = _Enum("day")
        GTC = _Enum("gtc")

    class QueryOrderStatus:  # type: ignore
        OPEN = _Enum("open")
        CLOSED = _Enum("closed")
        ALL = _Enum("all")

    class LimitOrderRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class TrailingStopOrderRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class GetOrdersRequest:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


logger = logging.getLogger("execute_trades")
LOGGER = logger

ALLOCATION_MODE_TAG = "[INFO] ALLOCATION_MODE"
ALLOC_SPLIT_TAG = "[INFO] ALLOC_SPLIT"
ALLOC_SPLIT_SKIPPED_TAG = "[WARN] ALLOC_SPLIT_SKIPPED"

REQUIRED = list(CANONICAL_COLUMNS)


def _canonicalize_candidate_header(
    df: Optional[pd.DataFrame], base_dir: Path | str | None
) -> pd.DataFrame:
    """Best-effort canonicalization for executor input."""

    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED)

    frame = df.copy()

    def _normalize_name(name: Any) -> str:
        key = str(name).strip().lower()
        key = re.sub(r"[^a-z0-9]+", "_", key)
        return key.strip("_")

    frame.columns = [_normalize_name(col) for col in frame.columns]

    rename_map = {
        "price": "close",
        "last": "close",
        "entry": "entry_price",
        "entryprice": "entry_price",
        "avg_daily_dollar_volume_20d": "adv20",
        "avg_dollar_vol_20d": "adv20",
        "avgdailydollarvolume20d": "adv20",
        "avgdollarvol20d": "adv20",
        "atrp_14": "atrp",
        "scorebreakdown": "score_breakdown",
    }
    for key, value in rename_map.items():
        if key in frame.columns and value not in frame.columns:
            frame.rename(columns={key: value}, inplace=True)

    if "symbol" not in frame.columns:
        return pd.DataFrame(columns=REQUIRED)

    frame["symbol"] = (
        frame["symbol"].astype("string").fillna("").str.strip().str.upper()
    )
    frame = frame.loc[frame["symbol"].str.len() > 0]

    if frame.empty:
        return pd.DataFrame(columns=REQUIRED)

    now_value = datetime.now(timezone.utc).isoformat()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = now_value
    else:
        ts_series = frame["timestamp"].astype("string").fillna("")
        frame["timestamp"] = ts_series.replace({"": now_value})

    numeric_columns = ("score", "close", "volume", "universe_count", "entry_price", "adv20", "atrp")
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    base_path: Optional[Path] = None
    if base_dir is not None:
        try:
            base_path = Path(base_dir)
        except TypeError:
            base_path = Path(str(base_dir))
    if base_path and base_path.is_file():
        base_path = base_path.parent.parent
    scored = pd.DataFrame()
    if base_path:
        scored_path = base_path / "data" / "scored_candidates.csv"
        if scored_path.exists():
            try:
                scored = pd.read_csv(scored_path)
            except Exception:
                scored = pd.DataFrame()

    if ("close" not in frame.columns or frame["close"].isna().all()) and "entry_price" in frame.columns:
        frame.loc[:, "close"] = frame["entry_price"]

    if not scored.empty and "symbol" in scored.columns:
        scored = scored.copy()
        scored["symbol"] = scored["symbol"].astype("string").str.upper().str.strip()
        scored = scored.loc[scored["symbol"].str.len() > 0]
        scored = scored.drop_duplicates(subset=["symbol"], keep="last")
        scored.set_index("symbol", inplace=True)
        if "score_breakdown" in scored.columns and "score_breakdown" not in frame.columns:
            frame = frame.join(scored["score_breakdown"], on="symbol")

    if "close" not in frame.columns:
        frame["close"] = pd.NA
    if "entry_price" not in frame.columns:
        frame["entry_price"] = pd.NA

    close_series = pd.to_numeric(frame["close"], errors="coerce")
    entry_series = pd.to_numeric(frame["entry_price"], errors="coerce")
    entry_missing = entry_series.isna()
    entry_series.loc[entry_missing] = close_series.loc[entry_missing]
    close_missing = close_series.isna()
    close_series.loc[close_missing] = entry_series.loc[close_missing]
    frame["close"] = close_series
    frame["entry_price"] = entry_series

    if "score_breakdown" in frame.columns:
        frame["score_breakdown"] = (
            frame["score_breakdown"].astype("string").fillna("").replace({"": "{}", "fallback": "{}"})
        )
    else:
        frame["score_breakdown"] = "{}"

    if "source" in frame.columns:
        frame["source"] = frame["source"].astype("string").fillna("").replace({"": "screener"})
    else:
        frame["source"] = "screener"

    volume_series = frame["volume"] if "volume" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["volume"] = pd.to_numeric(volume_series, errors="coerce").fillna(0)

    adv_series = frame["adv20"] if "adv20" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["adv20"] = pd.to_numeric(adv_series, errors="coerce").fillna(0.0)

    atrp_series = frame["atrp"] if "atrp" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["atrp"] = pd.to_numeric(atrp_series, errors="coerce").fillna(0.0)

    universe_series = (
        frame["universe_count"] if "universe_count" in frame.columns else pd.Series(pd.NA, index=frame.index)
    )
    frame["universe_count"] = pd.to_numeric(universe_series, errors="coerce").fillna(0).astype(int)

    for column in REQUIRED:
        if column not in frame.columns:
            frame[column] = pd.NA

    ordered = REQUIRED + [column for column in frame.columns if column not in REQUIRED]
    return frame[ordered]


# --- MPV helpers (Alpaca equities) ---
#  >= $1.00  →  two decimals
#  <  $1.00  →  four decimals    (per Alpaca docs)
#  Buys round DOWN, sells round UP to maintain trader intent and avoid 42210000.
def _tick_for(price: float) -> Decimal:
    try:
        p = Decimal(str(price))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0.01")
    if p >= Decimal("1.00"):
        return Decimal("0.01")
    return Decimal("0.0001")


def round_to_tick(x: float) -> float:
    try:
        tick = _tick_for(x)
        # Alpaca enforces SEC Rule 612 (no sub-penny increments for US equities).
        # Quantize down so we do not trip 422 "sub-penny increment" errors when submitting.
        return float(Decimal(str(x)).quantize(tick, rounding=ROUND_DOWN))
    except (InvalidOperation, ValueError, TypeError):
        return float(x)


def _mpv_step(price: float) -> Decimal:
    return _tick_for(price)


def _enforce_order_price_ticks(request: Any) -> None:
    """Round known price fields to valid ticks prior to API submission."""

    price_fields = (
        "limit_price",
        "stop_price",
        "stop_limit_price",
        "stop_loss_price",
        "take_profit_price",
        "trail_price",
    )
    for field in price_fields:
        if not hasattr(request, field):
            continue
        value = getattr(request, field)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        rounded = round_to_tick(numeric)
        setattr(request, field, rounded)


def normalize_price_for_alpaca(price: float, side: str) -> float:
    side = (side or "").lower()
    d = Decimal(str(price))
    step = _mpv_step(price)
    rounding = ROUND_DOWN if side == "buy" else ROUND_UP
    normalized = d.quantize(step, rounding=rounding)
    return float(normalized)


# --- Canonicalize incoming candidates to the executor's schema ---


def _round_limit_price(price: float) -> float:
    """Round ``price`` to Alpaca-friendly precision (two decimals >= $1)."""

    try:
        value = float(price)
    except (TypeError, ValueError):
        return 0.0
    try:
        return round_to_tick(value)
    except (InvalidOperation, ValueError, TypeError):
        decimals = 4 if value < 1 else 2
        return round(value + 1e-9, decimals)


def _coerce_trade_timestamp(timestamp: Any | None) -> str:
    """Return an ISO-8601 UTC timestamp string for ``timestamp``."""

    if timestamp is None or timestamp == "":
        return datetime.now(timezone.utc).isoformat()
    if isinstance(timestamp, datetime):
        dt = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    try:
        parsed = pd.to_datetime(timestamp, utc=True)
    except Exception:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(parsed, pd.Series):
        if parsed.empty:
            return datetime.now(timezone.utc).isoformat()
        parsed = parsed.iloc[0]
    if pd.isna(parsed):
        return datetime.now(timezone.utc).isoformat()
    if isinstance(parsed, pd.Timestamp):
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(timezone.utc)
        return parsed.tz_convert(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def record_executed_trade(
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    status: str,
    order_id: str,
    order_type: str,
    timestamp: Any | None = None,
    event_type: str | None = None,
    raw: Mapping[str, Any] | None = None,
) -> None:
    """Append a trade execution event to ``executed_trades.csv``."""

    side_value = (side or "").lower()
    event_label = (event_type or "").upper() or None
    try:
        qty_value = float(qty)
    except (TypeError, ValueError):
        qty_value = 0.0
    try:
        price_value = float(price)
    except (TypeError, ValueError):
        price_value = 0.0
    event_time_value = _coerce_trade_timestamp(timestamp)
    exit_price = price_value if side_value == "sell" else 0.0
    entry_price = price_value if side_value == "buy" else 0.0
    current_price = price_value if side_value == "buy" else exit_price
    raw_payload = dict(raw or {})
    if event_label and "event_type" not in raw_payload:
        raw_payload["event_type"] = event_label
    exit_time_value = event_time_value if side_value == "sell" else ""

    row = {
        "order_id": str(order_id or ""),
        "symbol": str(symbol or "").upper(),
        "qty": qty_value,
        "avg_entry_price": entry_price,
        "current_price": current_price,
        "unrealized_pl": 0.0,
        "entry_price": entry_price,
        "entry_time": event_time_value,
        "exit_price": exit_price,
        "exit_time": exit_time_value,
        "net_pnl": 0.0,
        "pnl": 0.0,
        "order_status": str(status or ""),
        "order_type": str(order_type or ""),
        "side": side_value,
        "event_type": event_label,
        "raw": raw_payload or None,
    }

    existing: pd.DataFrame
    if EXECUTED_TRADES_PATH.exists():
        try:
            existing = pd.read_csv(EXECUTED_TRADES_PATH)
        except Exception:
            LOGGER.exception(
                "Failed to read existing executed trades log at %s",
                EXECUTED_TRADES_PATH,
            )
            existing = pd.DataFrame(columns=EXECUTED_TRADES_COLUMNS)
    else:
        existing = pd.DataFrame(columns=EXECUTED_TRADES_COLUMNS)

    for column in EXECUTED_TRADES_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    updated = pd.concat(
        [existing[EXECUTED_TRADES_COLUMNS], pd.DataFrame([row])],
        ignore_index=True,
    )

    try:
        write_csv_atomic(str(EXECUTED_TRADES_PATH), updated[EXECUTED_TRADES_COLUMNS])
    except Exception:
        LOGGER.exception(
            "Failed to append executed trade for %s to %s",
            symbol,
            EXECUTED_TRADES_PATH,
        )
    event_for_log = event_label or row["order_status"]
    try:
        db_ok = db.insert_executed_trade(row)
        if db_ok:
            LOGGER.info(
                "[INFO] DB_WRITE_OK table=executed_trades event=%s order_id=%s",
                event_for_log or "",
                row["order_id"],
            )
        else:
            LOGGER.warning(
                "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
                event_for_log or "",
                row["order_id"],
                "db_disabled",
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
            event_for_log or "",
            row["order_id"],
            exc,
        )
    order_event_payload = {
        "symbol": row["symbol"],
        "qty": row["qty"],
        "order_id": row["order_id"],
        "status": row["order_status"],
        "event_type": event_for_log or "UNKNOWN",
        "event_time": event_time_value,
        "raw": raw_payload or None,
    }
    try:
        db.insert_order_event(order_event_payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            "[WARN] DB_WRITE_FAILED table=order_events event=%s order_id=%s err=%s",
            event_for_log or "",
            row["order_id"],
            exc,
        )
    if (event_label or "").upper() == "BUY_FILL":
        try:
            trade_ok = db.upsert_trade_on_buy_fill(
                row["symbol"],
                row["qty"],
                row["order_id"],
                event_time_value,
                price_value,
            )
            if not trade_ok:
                LOGGER.warning(
                    "[WARN] DB_TRADE_UPSERT_FAILED symbol=%s order_id=%s",
                    row["symbol"],
                    row["order_id"],
                )
            else:
                open_trades = db.count_open_trades()
                LOGGER.info("OPEN_TRADES count=%s", open_trades)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "[WARN] DB_TRADE_UPSERT_FAILED symbol=%s order_id=%s err=%s",
                row["symbol"],
                row["order_id"],
                exc,
            )
    if side_value == "sell" and (event_label or "").upper().endswith("FILL"):
        try:
            close_ok = db.close_trade_on_sell_fill(
                row["symbol"],
                row["order_id"],
                event_time_value,
                price_value,
                (event_label or "").upper(),
            )
            if not close_ok:
                LOGGER.warning(
                    "[WARN] DB_TRADE_CLOSE_FAILED symbol=%s order_id=%s",
                    row["symbol"],
                    row["order_id"],
                )
            else:
                open_trades = db.count_open_trades()
                LOGGER.info("OPEN_TRADES count=%s", open_trades)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "[WARN] DB_TRADE_CLOSE_FAILED symbol=%s order_id=%s err=%s",
                row["symbol"],
                row["order_id"],
                exc,
            )


def _isoformat_or_none(timestamp: Any) -> str | None:
    if timestamp in (None, ""):
        return None
    if isinstance(timestamp, datetime):
        dt = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    try:
        parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
    except Exception:
        return None
    if isinstance(parsed, pd.Series):
        if parsed.empty:
            return None
        parsed = parsed.iloc[0]
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(timezone.utc)
        return parsed.tz_convert(timezone.utc).isoformat()
    return None


def _order_snapshot(order: Any) -> Dict[str, Any]:
    if order is None:
        return {}
    fields = (
        "id",
        "symbol",
        "qty",
        "side",
        "status",
        "type",
        "limit_price",
        "filled_qty",
        "filled_avg_price",
        "submitted_at",
        "created_at",
        "client_order_id",
        "trail_percent",
    )
    snapshot: Dict[str, Any] = {}
    for field in fields:
        if isinstance(order, Mapping):
            value = order.get(field)
        else:
            value = getattr(order, field, None)
        iso_value = _isoformat_or_none(value)
        snapshot[field] = iso_value if iso_value is not None else value
    return {key: value for key, value in snapshot.items() if value not in (None, "", [], {}, ())}


def alpaca_list_orders_http(
    lookback_days: int | None = None,
    limit: int = 500,
    *,
    after_iso: str | None = None,
) -> list[dict]:
    """
    Return orders from Alpaca REST /v2/orders using APCA_API_BASE_URL.
    Must work in paper mode. No alpaca-py dependency.
    """
    load_env()
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError("missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")
    url = f"{(base_url or '').rstrip('/')}/v2/orders"
    computed_after = after_iso
    if computed_after is None:
        computed_after = (datetime.now(timezone.utc) - timedelta(days=max(0, lookback_days or 0))).isoformat()
    params = {
        "status": "all",
        "after": computed_after,
        "direction": "desc",
        "limit": limit,
        "nested": "true",
    }
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    response = requests.get(url, headers=headers, params=params, timeout=15)
    if response.status_code != 200:
        response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    return []


def log_trade_event_db(
    *,
    event_type: str,
    symbol: str,
    qty: float,
    order_id: str,
    status: str,
    entry_price: float | None = None,
    entry_time: Any | None = None,
    raw: Mapping[str, Any] | None = None,
) -> None:
    event_label = (event_type or "").upper() or "UNKNOWN"
    try:
        qty_value = float(qty)
    except (TypeError, ValueError):
        qty_value = 0.0
    try:
        entry_price_value = float(entry_price) if entry_price is not None else 0.0
    except (TypeError, ValueError):
        entry_price_value = 0.0
    entry_time_value = entry_time or datetime.now(timezone.utc)
    raw_payload = dict(raw or {})
    raw_payload.setdefault("event_type", event_label)
    row = {
        "symbol": symbol,
        "qty": qty_value,
        "entry_time": entry_time_value,
        "entry_price": entry_price_value,
        "order_id": order_id,
        "status": status,
        "event_type": event_label,
        "raw": raw_payload,
    }
    try:
        db_ok = db.insert_executed_trade(row)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
            event_label,
            order_id or "",
            exc,
        )
        return
    if db_ok:
        LOGGER.info(
            "[INFO] DB_WRITE_OK table=executed_trades event=%s order_id=%s",
            event_label,
            order_id or "",
        )
    else:
        LOGGER.warning(
            "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
            event_label,
            order_id or "",
            "db_disabled",
        )
    order_event_payload = {
        "symbol": symbol,
        "qty": qty_value,
        "order_id": order_id,
        "status": status,
        "event_type": event_label,
        "event_time": entry_time_value,
        "raw": raw_payload or None,
    }
    try:
        db.insert_order_event(order_event_payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            "[WARN] DB_WRITE_FAILED table=order_events event=%s order_id=%s err=%s",
            event_label,
            order_id or "",
            exc,
        )


def log_info(tag: str, **kv: Any) -> None:
    payload = " ".join(f"{key}={kv[key]}" for key in sorted(kv)) if kv else ""
    message = f"{tag} {payload}".strip()
    LOGGER.info(message)
 
 
def _log_call(label, fn, *args, **kwargs):
    t0 = _time.perf_counter()
    try:
        res = fn(*args, **kwargs)
        logger.debug("[CALL] %s ok=1 dt=%.3fs", label, _time.perf_counter() - t0)
        return res, None
    except Exception as e:
        logger.warning("[CALL] %s ok=0 dt=%.3fs err=%r", label, _time.perf_counter() - t0, e)
        return None, e
LOG_PATH = Path("logs") / "execute_trades.log"
METRICS_PATH = Path("data") / "execute_metrics.json"
EXECUTED_TRADES_PATH = Path("data") / "executed_trades.csv"
_EXECUTE_START_UTC: datetime | None = None
_EXECUTE_FINISH_UTC: datetime | None = None


def write_execute_metrics(
    payload: Mapping[str, Any] | None,
    *,
    status: str | None = None,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> Dict[str, Any]:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload or {})
    if status:
        payload["status"] = status
    enriched = _canonicalize_execute_metrics(
        payload,
        start_dt=start_dt,
        end_dt=end_dt or datetime.now(timezone.utc),
    )
    atomic_write_bytes(
        METRICS_PATH,
        json.dumps(enriched, indent=2, sort_keys=True).encode("utf-8"),
    )
    log_info(
        "METRICS_UPDATED",
        path=str(METRICS_PATH),
        run_started_utc=enriched.get("run_started_utc"),
        run_finished_utc=enriched.get("run_finished_utc"),
        status=enriched.get("status"),
    )
    return enriched


def _write_execute_metrics_error(
    message: str,
    *,
    status: str = "error",
    rc: int | None = None,
    **extra: Any,
) -> None:
    """Persist an error snapshot so dashboards surface a clear banner."""

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    except Exception:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    payload: Dict[str, Any] = dict(existing)
    payload["status"] = status
    error_block: Dict[str, Any] = {
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rc is not None:
        try:
            error_block["rc"] = int(rc)
        except Exception:
            error_block["rc"] = rc
    for key, value in extra.items():
        if value is None:
            continue
        try:
            json.dumps({key: value})
            error_block[key] = value
        except TypeError:
            error_block[key] = str(value)
    payload["error"] = error_block
    payload.setdefault("skips", {})
    payload.setdefault("orders_submitted", 0)
    payload.setdefault("orders_filled", 0)
    payload.setdefault("trailing_attached", 0)
    payload["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("auth_ok", False)
    return write_execute_metrics(
        payload,
        status=status,
        start_dt=_EXECUTE_START_UTC,
        end_dt=datetime.now(timezone.utc),
    )
EXECUTED_TRADES_COLUMNS: list[str] = [
    "order_id",
    "symbol",
    "qty",
    "avg_entry_price",
    "current_price",
    "unrealized_pl",
    "entry_price",
    "entry_time",
    "exit_price",
    "exit_time",
    "net_pnl",
    "pnl",
    "order_status",
    "order_type",
    "side",
]
DEFAULT_BAR_DIRECTORIES: Sequence[Path] = (
    Path("data") / "daily",
    Path("data") / "bars" / "daily",
    Path("data") / "bars",
    Path("data") / "cache" / "bars",
)

REQUIRED_COLUMNS = [
    "symbol",
    "close",
    "score",
    "universe_count",
    "score_breakdown",
]

OPTIONAL_COLUMNS = {"atrp", "exchange", "adv20", "entry_price"}
SKIP_REASON_ORDER: tuple[str, ...] = (
    "TIME_WINDOW",
    "CASH",
    "ZERO_QTY",
    "PRICE_BOUNDS",
    "MAX_POSITIONS",
    "API_FAIL",
    "DATA_MISSING",
    "EXISTING_POSITION",
    "OPEN_ORDER",
    "NO_CANDIDATES",
)
SKIP_REASON_KEYS = set(SKIP_REASON_ORDER)
IMPORT_SENTINEL_ENV = "JBRAVO_IMPORT_SENTINEL"
DATA_URL_ENV_VARS = ("APCA_API_DATA_URL", "ALPACA_API_DATA_URL")
DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
REQUIRED_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)

_ENV_FILES_LOADED: list[str] = []


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _mask_key_hint(raw: Optional[str]) -> str:
    if not raw:
        return "missing"
    try:
        value = str(raw).strip()
    except Exception:
        return "invalid"
    if not value:
        return "missing"
    if len(value) <= 4:
        return value[0] + "***"
    prefix = value[:4]
    suffix = value[-2:]
    return f"{prefix}…{suffix}"


def _log_account_probe(
    creds_snapshot: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    env_files = [
        Path(path).name
        for path in _ENV_FILES_LOADED
        if isinstance(path, str) and path.strip()
    ]
    has_env_file = any(name.lower().endswith(".env") for name in env_files)

    key, secret, base_url, _ = get_alpaca_creds()
    key_hint = _mask_key_hint(key)
    if (not key or key_hint == "missing") and isinstance(creds_snapshot, Mapping):
        prefix = str(creds_snapshot.get("key_prefix") or "").strip()
        if prefix:
            key_hint = prefix

    auth_ok = False
    bp_display = "n/a"
    trading_base = base_url or ""
    if not trading_base and isinstance(creds_snapshot, Mapping):
        base_urls = creds_snapshot.get("base_urls")
        if isinstance(base_urls, Mapping):
            trading_base = str(base_urls.get("trading") or "")
    trading_base = (trading_base or "https://paper-api.alpaca.markets").rstrip("/")

    if key and secret:
        url = f"{trading_base}/v2/account"
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if isinstance(payload, Mapping):
                    buying_power = payload.get("buying_power")
                    if buying_power not in (None, ""):
                        bp_display = str(buying_power)
                auth_ok = True
            else:
                bp_display = f"status={response.status_code}"
        except Exception as exc:  # pragma: no cover - best-effort logging
            bp_display = f"error={exc}"
    else:
        bp_display = "missing-creds"

    LOGGER.info(
        "[INFO] AUTH_CHECK auth_ok=%s buying_power=%s base_url=%s env_loaded=%s key_hint=%s",
        bool(auth_ok),
        bp_display,
        base_url,
        "yes" if has_env_file else "no",
        key_hint,
    )
    return bool(auth_ok), str(bp_display)


def _paper_only_guard(trading_client: Any | None, base_url: str | None = "") -> None:
    if trading_client is None:
        return

    base = (os.getenv("APCA_API_BASE_URL") or base_url or "").lower()
    if "paper-api.alpaca.markets" not in base:
        logger.error(
            "PAPER_ONLY guard: APCA_API_BASE_URL is not a paper endpoint: %s",
            base,
        )
        sys.exit(2)


def _load_execute_metrics() -> Optional[Dict[str, Any]]:
    if not METRICS_PATH.exists():
        return None
    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive metrics parsing
        LOGGER.warning("Failed to load execute metrics: %s", exc)
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    LOGGER.warning(
        "Execute metrics file contained unexpected payload type: %s",
        type(payload).__name__,
    )
    return None


def _normalize_space_keys(payload: Mapping[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    for key in list(enriched.keys()):
        if isinstance(key, str) and " " in key:
            normalized = key.replace(" ", "_")
            enriched.setdefault(normalized, enriched[key])
    return enriched


def _has_auth_missing(value: Any) -> bool:
    if value in (None, "", False):
        return False
    if isinstance(value, (list, tuple, set, Mapping)):
        return len(value) > 0
    return bool(value)


def _canonicalize_execute_metrics(
    payload: Mapping[str, Any], *, start_dt: datetime | None = None, end_dt: datetime | None = None
) -> Dict[str, Any]:
    enriched = _normalize_space_keys(payload)

    timestamp = None
    for key in ("last_run_utc", "last_run", "run_utc", "timestamp"):
        value = enriched.get(key)
        if isinstance(value, str) and value.strip():
            timestamp = value
            break
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    if start_dt is None:
        start_dt = _EXECUTE_START_UTC or end_dt

    start_iso = None
    for key in ("run_started_utc", "run_started", "start_utc", "start"):
        value = enriched.get(key)
        if isinstance(value, str) and value.strip():
            start_iso = value
            break
    end_iso = None
    for key in ("run_finished_utc", "run_finished", "finished_utc", "finish_utc"):
        value = enriched.get(key)
        if isinstance(value, str) and value.strip():
            end_iso = value
            break

    start_iso = start_dt.isoformat() if start_dt is not None else start_iso or end_dt.isoformat()
    end_iso = end_dt.isoformat() if end_dt is not None else end_iso or start_iso
    enriched["timestamp"] = timestamp or end_iso
    enriched["last_run_utc"] = enriched.get("last_run_utc") or enriched["timestamp"]
    enriched["run_started_utc"] = start_iso
    enriched["run_finished_utc"] = end_iso

    auth_missing = enriched.get("auth_missing")
    auth_ok = enriched.get("auth_ok")
    if auth_ok is None:
        auth_ok = not (
            _has_auth_missing(auth_missing)
            or str(enriched.get("status", "")).lower() == "auth_error"
        )
    auth_reason = enriched.get("auth_reason")
    if not auth_ok:
        if not auth_reason:
            error_block = enriched.get("error")
            if isinstance(error_block, Mapping):
                auth_reason = error_block.get("reason") or error_block.get("message")
        auth_reason = auth_reason or "auth_failed"
        status_value = str(enriched.get("status", "")).strip().lower()
        if status_value in ("", "ok"):
            enriched["status"] = "error"
    else:
        auth_reason = None
    enriched["auth_ok"] = bool(auth_ok)
    enriched["auth_reason"] = auth_reason

    fills_value = enriched.get("orders_filled", enriched.get("fills", 0))
    trails_value = enriched.get("trailing_attached", enriched.get("trails", 0))
    try:
        fills = int(fills_value)
    except Exception:
        fills = 0
    try:
        trails = int(trails_value)
    except Exception:
        trails = 0

    enriched["fills"] = fills
    enriched["trails"] = trails
    enriched["orders_filled"] = enriched.get("orders_filled", fills)
    enriched["trailing_attached"] = enriched.get("trailing_attached", trails)

    if start_dt is not None:
        try:
            enriched["duration_sec"] = round((end_dt - start_dt).total_seconds(), 3)
        except Exception:
            enriched["duration_sec"] = 0.0
    else:
        enriched.setdefault("duration_sec", 0.0)

    step_signals = [
        enriched.get("orders_submitted", 0),
        enriched.get("orders_filled", 0),
        enriched.get("trailing_attached", 0),
        enriched.get("orders_canceled", 0),
    ]
    skips_block = enriched.get("skips") if isinstance(enriched.get("skips"), Mapping) else {}
    try:
        step_signals.append(sum(int(v) for v in skips_block.values()))
    except Exception:
        step_signals.append(0)
    if any(value not in (None, "", 0, 0.0) for value in step_signals):
        if enriched.get("duration_sec", 0.0) < 1.0:
            enriched["duration_sec"] = 1.0

    enriched.setdefault("status", "ok")
    metrics_defaults: Dict[str, Any] = {
        "open_positions": 0,
        "open_orders": 0,
        "configured_max_positions": 0,
        "max_total_positions": 0,
        "risk_limited_max_positions": 0,
        "slots_total": 0,
        "allowed_new_positions": 0,
        "in_window": False,
    }
    for key, default in metrics_defaults.items():
        value = enriched.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            enriched[key] = default
    for key in (
        "open_positions",
        "open_orders",
        "configured_max_positions",
        "max_total_positions",
        "risk_limited_max_positions",
        "slots_total",
        "allowed_new_positions",
    ):
        try:
            enriched[key] = int(enriched.get(key, 0))
        except Exception:
            enriched[key] = metrics_defaults[key]
    exit_reason_val = enriched.get("exit_reason")
    if exit_reason_val is None:
        enriched["exit_reason"] = None
    else:
        enriched["exit_reason"] = str(exit_reason_val or "UNKNOWN")
    enriched["in_window"] = bool(enriched.get("in_window"))

    return enriched


def _count_csv_rows(path: Path) -> int:
    """Return the number of data rows in a CSV (excluding the header)."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle)
    except FileNotFoundError:
        return 0
    except Exception:
        return 0
    return max(total - 1, 0)


_HHMM_RE = re.compile(r"^\d{4}$")


def _ny_now() -> datetime:
    return datetime.now(NY)


def _parse_hhmm(raw: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(raw, str):
        candidate = raw.strip()
        if _HHMM_RE.fullmatch(candidate):
            hour = int(candidate[:2])
            minute = int(candidate[2:])
            if 0 <= hour < 24 and 0 <= minute < 60:
                return hour, minute
    return default


def _premarket_bounds_components() -> tuple[tuple[int, int], tuple[int, int]]:
    default_start = (7, 0)
    default_end = (9, 30)
    start = _parse_hhmm(os.getenv("JBRAVO_PREMARKET_START"), default_start)
    end = _parse_hhmm(os.getenv("JBRAVO_PREMARKET_END"), default_end)
    if start >= end:
        return default_start, default_end
    return start, end


def _premarket_bounds_naive() -> tuple[dtime, dtime]:
    (sh, sm), (eh, em) = _premarket_bounds_components()
    return dtime(sh, sm), dtime(eh, em)


def _premarket_bounds_tz() -> tuple[dtime, dtime]:
    start, end = _premarket_bounds_naive()
    return start.replace(tzinfo=NY), end.replace(tzinfo=NY)


def _premarket_bounds_strings() -> tuple[str, str]:
    start, end = _premarket_bounds_naive()
    return start.strftime("%H:%M"), end.strftime("%H:%M")


def write_premarket_snapshot(
    base_dir: Path | str = Path("."),
    *,
    probe_payload: Mapping[str, Any] | None = None,
    started_utc: str | None = None,
    finished_utc: str | None = None,
) -> Path:
    """Persist a lightweight snapshot of the latest wrapper run."""

    base_path = Path(base_dir)
    data_dir = base_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_execute_metrics() or {}

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    skip_block = metrics.get("skips") if isinstance(metrics.get("skips"), Mapping) else {}
    skip_counts = {str(k): _as_int(v) for k, v in (skip_block or {}).items()}
    in_window = skip_counts.get("TIME_WINDOW", 0) == 0

    auth_ok = metrics.get("status") != "auth_error"
    buying_power = metrics.get("buying_power")
    if isinstance(probe_payload, Mapping):
        auth_ok = bool(probe_payload.get("auth_ok", auth_ok))
        buying_power = probe_payload.get("buying_power", buying_power)

    try:
        buying_power = float(str(buying_power).replace(",", "")) if buying_power not in (None, "") else None
    except Exception:
        buying_power = None

    candidates_in = _count_csv_rows(data_dir / "latest_candidates.csv")
    start_str, end_str = _premarket_bounds_strings()
    timestamp_started = (
        started_utc
        or metrics.get("start_utc")
        or metrics.get("started_utc")
        or metrics.get("last_run_utc")
    )
    timestamp_finished = finished_utc or datetime.now(timezone.utc).isoformat()
    snapshot = {
        "in_window": bool(in_window),
        "auth_ok": bool(auth_ok),
        "candidates_in": int(candidates_in),
        "skip_counts": skip_counts,
        "window_label": f"{start_str}-{end_str} ET",
        "started_utc": timestamp_started,
        "finished_utc": timestamp_finished,
    }
    if buying_power is not None:
        snapshot["buying_power"] = buying_power
    for key in ("orders_submitted", "orders_filled", "orders_canceled", "trailing_attached"):
        if key in metrics:
            snapshot[key] = _as_int(metrics.get(key))

    target = data_dir / "last_premarket_run.json"
    try:
        target.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        LOGGER.warning("[WARN] PREMARKET_SNAPSHOT_WRITE_FAILED path=%s", target, exc_info=True)
    return target


def _clock_is_in_window(client: Any, window: str) -> tuple[bool, str]:
    if client is None or not hasattr(client, "get_clock"):
        return False, "00:00"
    try:
        clk = client.get_clock()
        ny = ZoneInfo("America/New_York")
        now_ny = (
            clk.timestamp.astimezone(ny)
            if getattr(clk, "timestamp", None) is not None
            else None
        )
        if now_ny is None:
            now_ny = datetime.now(ny)
        hhmm = now_ny.strftime("%H:%M")
        target = (window or "auto").lower()
        start_str, end_str = _premarket_bounds_strings()
        if target == "premarket":
            in_window = start_str <= hhmm < end_str
        elif target == "regular":
            in_window = "09:30" <= hhmm < "16:00"
        elif target == "any":
            in_window = True
        else:  # auto and unknown default to premarket guard
            in_window = start_str <= hhmm < end_str
        log_info(
            "MARKET_TIME",
            ny_now=str(now_ny),
            window=target,
            in_window=in_window,
            premarket_bounds=f"{start_str}-{end_str}",
        )
        return in_window, hhmm
    except Exception as exc:  # pragma: no cover - defensive logging
        log_info("CLOCK_FETCH_FAILED", error=str(exc)[:200])
        return False, "00:00"


def _resolve_time_window(requested: str):
    req = (requested or "auto").strip().lower()
    now = _ny_now()
    tnow = now.timetz()
    prem_open, prem_close = _premarket_bounds_tz()
    reg_open = dtime(9, 30, tzinfo=NY)
    reg_close = dtime(16, 0, tzinfo=NY)

    in_pre = prem_open <= tnow < prem_close
    in_reg = reg_open <= tnow < reg_close

    if req == "premarket":
        return "premarket", in_pre, now
    if req == "regular":
        return "regular", in_reg, now
    if req == "any":
        return "any", True, now

    if in_pre:
        return "premarket", True, now
    if in_reg:
        return "regular", True, now
    return "closed", False, now


def _wait_until_submit_at(target: Optional[str]) -> None:
    value = (target or "").strip()
    if not value:
        return
    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError:
        LOGGER.warning("[WARN] submit_at_ny invalid value=%s -> skipping wait", value)
        return
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        LOGGER.warning("[WARN] submit_at_ny out of range value=%s -> skipping wait", value)
        return
    ny = ZoneInfo("America/New_York")
    target_dt = datetime.now(ny).replace(hour=hour, minute=minute, second=0, microsecond=0)
    now_ny = datetime.now(ny)
    if now_ny >= target_dt:
        LOGGER.info(
            "[INFO] SUBMIT_AT passed target -> proceeding ny_now=%s target=%s",
            now_ny.isoformat(),
            target_dt.isoformat(),
        )
        return
    while now_ny < target_dt:
        remaining = (target_dt - now_ny).total_seconds()
        LOGGER.info(
            "[INFO] WAIT_UNTIL submit_at_ny=%s remaining=%ds",
            value,
            int(max(1, math.ceil(remaining))),
        )
        sleep_for = min(60, max(5, remaining / 4))
        time.sleep(sleep_for)
        now_ny = datetime.now(ny)


def _record_auth_error(
    reason: str,
    sanitized: Mapping[str, object],
    missing: Iterable[str] | None = None,
) -> None:
    write_auth_error_artifacts(
        reason=reason,
        sanitized=sanitized,
        missing=missing or [],
        metrics_path=METRICS_PATH,
        summary_path=Path("data") / "metrics_summary.csv",
    )


def _fetch_latest_close_from_alpaca(symbol: str) -> Optional[float]:
    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return None
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/bars/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.latest_bar", symbol=symbol)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.latest_bar", f"{symbol}: {exc}")
        return None
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.latest_bar",
            f"{symbol} unauthorized status={response.status_code}",
        )
        return None
    if response.status_code != 200:
        _warn_context(
            "alpaca.latest_bar",
            f"{symbol} status={response.status_code}",
        )
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.latest_bar", f"{symbol} decode_failed: {exc}")
        return None
    if not isinstance(payload, Mapping):
        return None
    bar = payload.get("bar")
    if isinstance(bar, Mapping):
        close_value = bar.get("c") if "c" in bar else bar.get("close")
        if close_value is None:
            return None
        try:
            return float(close_value)
        except (TypeError, ValueError):
            return None
    return None


def _fetch_prevclose_snapshot(symbol: str) -> Optional[float]:
    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return None
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/snapshot"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.snapshot_prevclose", symbol=symbol, feed=params.get("feed"))
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.snapshot_prevclose", f"{symbol}: {exc}")
        return None
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.snapshot_prevclose", f"{symbol} unauthorized status={response.status_code}"
        )
        return None
    if response.status_code != 200:
        _warn_context(
            "alpaca.snapshot_prevclose", f"{symbol} status={response.status_code}"
        )
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.snapshot_prevclose", f"{symbol} decode_failed: {exc}")
        return None
    snapshot = None
    if isinstance(payload, Mapping):
        snapshot = payload.get("snapshot", payload)
    if not isinstance(snapshot, Mapping):
        return None
    previous = snapshot.get("previousDailyBar") or snapshot.get("prevDailyBar")
    if not isinstance(previous, Mapping):
        return None
    close_value = previous.get("c") if "c" in previous else previous.get("close")
    if close_value is None:
        return None
    try:
        return float(close_value)
    except (TypeError, ValueError):
        return None


def _fetch_prev_close_from_alpaca(symbol: str) -> Optional[float]:
    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return None
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {"timeframe": "1Day", "limit": "2", "feed": "iex"}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.prevclose", symbol=symbol)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.prevclose", f"{symbol}: {exc}")
        return None
    if response.status_code in (401, 403):
        _warn_context("alpaca.prevclose", f"{symbol} unauthorized status={response.status_code}")
        return None
    if response.status_code != 200:
        _warn_context("alpaca.prevclose", f"{symbol} status={response.status_code}")
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.prevclose", f"{symbol} decode_failed: {exc}")
        return None
    bars = payload.get("bars") if isinstance(payload, Mapping) else None
    if not isinstance(bars, list) or len(bars) < 2:
        return None
    prior = bars[-2]
    if not isinstance(prior, Mapping):
        return None
    close_value = prior.get("c") if "c" in prior else prior.get("close")
    if close_value is None:
        return None
    try:
        return float(close_value)
    except (TypeError, ValueError):
        return None


def _fetch_latest_quote_from_alpaca(
    symbol: str, *, feed: str | None = None
) -> Dict[str, Optional[float | str]]:
    api_key, api_secret, _, default_feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return {}
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/quotes/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    resolved_feed = feed or default_feed
    params: Dict[str, str] = {}
    if resolved_feed:
        params["feed"] = str(resolved_feed)
    log_info("alpaca.latest_quote", symbol=symbol, feed=params.get("feed"))
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
    except Exception as exc:
        _warn_context("alpaca.latest_quote", f"{symbol}: {exc}")
        return {}
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.latest_quote",
            f"{symbol} unauthorized status={response.status_code}",
        )
        return {}
    if response.status_code != 200:
        _warn_context(
            "alpaca.latest_quote",
            f"{symbol} status={response.status_code}",
        )
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.latest_quote", f"{symbol} decode_failed: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        return {}
    quote = payload.get("quote")
    if not isinstance(quote, Mapping):
        return {}

    def _as_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(str(value))
            except Exception:
                return None

    ask = _as_float(quote.get("ap") if "ap" in quote else quote.get("ask_price"))
    bid = _as_float(quote.get("bp") if "bp" in quote else quote.get("bid_price"))
    ts = quote.get("t") or quote.get("timestamp")
    resolved = params.get("feed") or payload.get("feed")
    snapshot: Dict[str, Optional[float | str]] = {
        "ask": ask,
        "bid": bid,
        "timestamp": ts,
        "feed": resolved,
    }
    return snapshot


def _fetch_latest_trade_from_alpaca(
    symbol: str, *, feed: str | None = None
) -> Dict[str, Optional[float | str]]:
    api_key, api_secret, _, default_feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return {}
    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/trades/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    resolved_feed = feed or default_feed
    params: Dict[str, str] = {}
    if resolved_feed:
        params["feed"] = str(resolved_feed)
    log_info("alpaca.latest_trade", symbol=symbol, feed=params.get("feed"))
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
    except Exception as exc:
        _warn_context("alpaca.latest_trade", f"{symbol}: {exc}")
        return {}
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.latest_trade",
            f"{symbol} unauthorized status={response.status_code}",
        )
        return {}
    if response.status_code != 200:
        _warn_context(
            "alpaca.latest_trade",
            f"{symbol} status={response.status_code}",
        )
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.latest_trade", f"{symbol} decode_failed: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        return {}
    trade = payload.get("trade")
    if not isinstance(trade, Mapping):
        return {}

    def _as_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(str(value))
            except Exception:
                return None

    price = _as_float(trade.get("p") if "p" in trade else trade.get("price"))
    ts = trade.get("t") or trade.get("timestamp")
    resolved = params.get("feed") or payload.get("feed")
    return {"price": price, "timestamp": ts, "feed": resolved}


def _fetch_latest_daily_bars(symbols: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    unique = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    if not unique:
        return {}

    api_key, api_secret, _, feed = get_alpaca_creds()
    if not api_key or not api_secret:
        return {}

    base_url = None
    for env_name in DATA_URL_ENV_VARS:
        candidate = os.getenv(env_name)
        if candidate:
            base_url = candidate
            break
    base_url = base_url or DEFAULT_DATA_BASE_URL
    url = f"{base_url.rstrip('/')}/v2/stocks/bars/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: Dict[str, str] = {"symbols": ",".join(unique), "timeframe": "1Day"}
    if feed:
        params["feed"] = str(feed)
    log_info("alpaca.bars_latest", symbols=len(unique))
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:
        _warn_context("alpaca.bars_latest", f"request_failed: {exc}")
        return {}
    if response.status_code in (401, 403):
        _warn_context(
            "alpaca.bars_latest",
            f"unauthorized status={response.status_code}",
        )
        return {}
    if response.status_code != 200:
        _warn_context(
            "alpaca.bars_latest",
            f"status={response.status_code}",
        )
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        _warn_context("alpaca.bars_latest", f"decode_failed: {exc}")
        return {}
    bars = payload.get("bars") if isinstance(payload, Mapping) else None
    results: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(bars, Mapping):
        for symbol in unique:
            entry = bars.get(symbol) or bars.get(symbol.lower()) or bars.get(symbol.upper())
            if not isinstance(entry, Mapping):
                continue
            close_value = entry.get("c") if "c" in entry else entry.get("close")
            volume_value = entry.get("v") if "v" in entry else entry.get("volume")
            close: Optional[float]
            volume: Optional[float]
            try:
                close = float(close_value) if close_value is not None else None
            except (TypeError, ValueError):
                close = None
            try:
                volume = float(volume_value) if volume_value is not None else None
            except (TypeError, ValueError):
                volume = None
            results[symbol] = {"close": close, "volume": volume, "source": "alpaca"}
    return results


class CandidateLoadError(RuntimeError):
    """Raised when candidate data cannot be loaded or validated."""


def load_candidates_from_db(
    *,
    metrics: ExecutionMetrics | None = None,
    record_skip: Optional[Callable[..., Any]] = None,
    diagnostic: bool = False,
) -> list[Dict[str, Any]]:
    """Load the latest screener candidates from PostgreSQL."""

    LOGGER.info("Loading candidates from DB")

    def _record_missing_config() -> list[Dict[str, Any]]:
        LOGGER.error("[ERROR] Missing DB config: set JBRAVO_DB_* or DATABASE_URL")
        if metrics is not None:
            metrics.auth_ok = False
            metrics.auth_reason = "candidate_load_error"
            metrics.record_skip("DATA_MISSING", count=1)
        if record_skip is not None:
            try:
                record_skip("DATA_MISSING", count=1, detail="missing_db_config")
            except TypeError:
                record_skip("DATA_MISSING", count=1)
        return []

    host = os.getenv("JBRAVO_DB_HOST")
    port = os.getenv("JBRAVO_DB_PORT")
    name = os.getenv("JBRAVO_DB_NAME")
    user = os.getenv("JBRAVO_DB_USER")
    password = os.getenv("JBRAVO_DB_PASS")
    database_url = os.getenv("DATABASE_URL")

    config_source = None
    if host and name and user and password:
        config_source = "JBRAVO_DB_*"
        port = port or "5432"
    elif database_url:
        parsed = urlparse(database_url)
        if parsed.scheme.startswith("postgresql"):
            host = parsed.hostname
            port = str(parsed.port or "5432")
            name = (parsed.path or "").lstrip("/")
            user = parsed.username
            password = unquote(parsed.password or "")
            if host and name and user and password:
                config_source = "DATABASE_URL"
                LOGGER.info("[INFO] DB_PASSWORD decoded_from_url=true")

    if not config_source:
        return _record_missing_config()

    connection = None
    masked_user = (user or "")[:1] + "***" if user else "***"
    masked_pass = "***" if not password else "***"
    LOGGER.info("[INFO] DB_CONFIG source=%s", config_source)
    LOGGER.info(
        "[INFO] DB_CONNECT target=host=%s port=%s db=%s user=%s pass=%s",
        host,
        port or "5432",
        name,
        masked_user,
        masked_pass,
    )

    def _record_no_db_candidates(detail: str) -> None:
        LOGGER.info("[INFO] NO_DB_CANDIDATES %s", detail)
        if record_skip is not None:
            try:
                record_skip("NO_CANDIDATES", count=1, detail=detail)
            except TypeError:
                record_skip("NO_CANDIDATES", count=1)
        if metrics is not None:
            metrics.record_skip("NO_CANDIDATES", count=1)
            if metrics.exit_reason is None:
                metrics.exit_reason = "NO_CANDIDATES"

    max_timestamp_value: Any = None
    try:
        connection = psycopg2.connect(
            host=host,
            port=port or "5432",
            dbname=name,
            user=user,
            password=password,
        )
        if diagnostic:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("select 1")
                    cursor.fetchone()
                LOGGER.info("DB_PING ok=true")
            except Exception as exc:  # pragma: no cover - diagnostic logging
                LOGGER.info("DB_PING ok=false err=%s", exc)
        LOGGER.info("[INFO] DB_QUERY max_timestamp")
        max_df = pd.read_sql_query(
            "SELECT MAX(timestamp) AS max_ts FROM screener_candidates;",
            connection,
        )
        if not max_df.empty and "max_ts" in max_df.columns:
            max_timestamp_value = max_df["max_ts"].iloc[0]
        max_ts_display = (
            max_timestamp_value.isoformat()
            if hasattr(max_timestamp_value, "isoformat")
            else (str(max_timestamp_value) if max_timestamp_value is not None else "NULL")
        )
        LOGGER.info("[INFO] DB_MAX_TIMESTAMP %s", max_ts_display)
        if max_timestamp_value is None or pd.isna(max_timestamp_value):
            LOGGER.info("[INFO] DB_CANDIDATES_LOADED count=0 max_timestamp=%s", max_ts_display)
            _record_no_db_candidates("max_timestamp_null")
            return []
        LOGGER.info("[INFO] DB_QUERY latest_batch order=score_desc")
        query = """
            SELECT
                run_date, symbol, score, exchange, close, volume,
                universe_count, score_breakdown, entry_price, adv20, atrp, source, timestamp
            FROM screener_candidates
            WHERE timestamp = %s
            ORDER BY score DESC;
        """
        df = pd.read_sql_query(query, connection, params=(max_timestamp_value,))
    except Exception as exc:
        raise CandidateLoadError(f"Failed to load candidates from database: {exc}") from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                LOGGER.debug("Failed to close DB connection", exc_info=True)
    records = df.to_dict(orient="records")
    max_ts_display = (
        max_timestamp_value.isoformat()
        if hasattr(max_timestamp_value, "isoformat")
        else (str(max_timestamp_value) if max_timestamp_value is not None else "NULL")
    )
    LOGGER.info(
        "[INFO] DB_CANDIDATES_LOADED count=%s max_timestamp=%s",
        len(records),
        max_ts_display,
    )
    if not records:
        _record_no_db_candidates("latest_batch_empty")
        return []
    if diagnostic:
        sample = [{"symbol": rec.get("symbol"), "score": rec.get("score")} for rec in records[:5]]
        LOGGER.info("[DIAGNOSTIC] DB_SAMPLE symbols_scores=%s", sample)
    return records


class AlpacaAuthFailure(RuntimeError):
    """Raised when Alpaca auth probes fail."""


def emit_import_sentinel() -> None:
    """Emit an import sentinel event when requested via environment flag."""

    if os.environ.get(IMPORT_SENTINEL_ENV) != "1":
        return
    version = telemetry.get_version()
    cmd = [
        sys.executable,
        "-m",
        "bin.emit_event",
        "IMPORT_SENTINEL",
        "component=execute_trades",
        f"version={version}",
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:  # pragma: no cover - telemetry best effort
        LOGGER.debug("Import sentinel emission failed: %s", exc)


emit_import_sentinel()


def _format_breakdown(value: Any, score: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:  # pragma: no cover - fallback for unserialisable payload
            pass
    if score is not None and not pd.isna(score):
        try:
            return json.dumps({"score": float(score)})
        except (TypeError, ValueError):
            return json.dumps({"score": score})
    return json.dumps({})


def _apply_candidate_defaults(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    frame = df.copy()
    messages: List[str] = []

    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.upper()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    for column in missing_columns:
        frame[column] = pd.NA
        messages.append(f"[WARN] MISSING_{column.upper()} column (defaulted)")

    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    if "entry_price" in frame.columns:
        entry_series = pd.to_numeric(frame["entry_price"], errors="coerce")
        close_series = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame.columns else pd.Series(dtype="float64")
        missing_entry = entry_series.isna()
        if "close" in frame.columns and not close_series.empty:
            filled = close_series.reindex_like(entry_series)
            entry_series = entry_series.fillna(filled)
        frame["entry_price"] = entry_series
        if missing_entry.any():
            messages.append("[WARN] DEFAULTED_ENTRY_PRICE_FROM_CLOSE")

    if "score" not in frame.columns or frame["score"].isna().all():
        if "Score" in frame.columns:
            frame["score"] = frame.get("Score")
            messages.append("[WARN] derived score column from Score header (defaulted)")
        else:
            frame["score"] = pd.NA
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")

    if "close" not in frame.columns:
        frame["close"] = pd.NA
    close_series = pd.to_numeric(frame.get("close"), errors="coerce")

    if "entry_price" not in frame.columns:
        frame["entry_price"] = pd.NA
    entry_series = pd.to_numeric(frame.get("entry_price"), errors="coerce")

    close_fallback_mask = close_series.isna() & entry_series.notna()
    if close_fallback_mask.any():
        close_series.loc[close_fallback_mask] = entry_series.loc[close_fallback_mask]
        messages.append(
            f"[WARN] close fallback from entry_price applied to {int(close_fallback_mask.sum())} rows"
        )
    frame["close"] = close_series
    frame["entry_price"] = entry_series

    if "universe_count" not in frame.columns:
        frame["universe_count"] = pd.NA
    universe_series = pd.to_numeric(frame.get("universe_count"), errors="coerce")
    default_universe = int(frame.shape[0])
    universe_missing = universe_series.isna()
    if universe_missing.any():
        universe_series.loc[universe_missing] = default_universe
        messages.append("[WARN] universe_count defaulted to candidate row count")
    frame["universe_count"] = universe_series.fillna(default_universe).astype(int)

    if "score_breakdown" not in frame.columns:
        frame["score_breakdown"] = [None] * frame.shape[0]
        messages.append("[WARN] score_breakdown populated from score (defaulted)")
    frame["score_breakdown"] = [
        _format_breakdown(value, score)
        for value, score in zip(frame.get("score_breakdown"), frame["score"])
    ]

    frame["exchange"] = (
        frame.get("exchange", pd.Series(dtype="string")).astype("string").fillna("").str.upper()
    )

    return frame, messages


@dataclass
class ExecutorConfig:
    source_path: Path = Path("data/latest_candidates.csv")
    source_type: str = "csv"
    allocation_pct: float = 0.05
    max_positions: int = 7
    max_new_positions: int = 7
    entry_buffer_bps: int = 75
    limit_buffer_pct: float = 0.5
    max_gap_pct: float = _env_float("MAX_GAP_PCT", 3.0)
    ref_buffer_pct: float = _env_float("REF_BUFFER_PCT", 0.5)
    trailing_percent: float = 3.0
    cancel_after_min: int = 35
    max_poll_secs: int = 60
    extended_hours: bool = True
    time_window: str = "auto"
    market_timezone: str = "America/New_York"
    dry_run: bool = False
    min_adv20: int = 2_000_000
    min_price: float = 1.0
    max_price: float = 1_000.0
    log_json: bool = False
    bar_directories: Sequence[Path] = DEFAULT_BAR_DIRECTORIES
    min_order_usd: float = 200.0
    allow_bump_to_one: bool = True
    allow_fractional: bool = False
    position_sizer: str = "notional"
    atr_target_pct: float = 0.02
    submit_at_ny: str = "07:00"
    price_source: str = "prevclose"
    price_band_pct: float = 10.0
    price_band_action: str = "clamp"
    chase_interval_minutes: int = 5
    max_chase_count: int = 6
    max_chase_gap_pct: float = 5.0
    chase_enabled: bool = False
    diagnostic: bool = False
    reconcile_only: bool = False
    reconcile_auto: bool = True
    reconcile_lookback_days: int = 14
    reconcile_limit: int = 500
    reconcile_use_watermark: bool = True
    reconcile_overlap_secs: int = 300


@dataclass
class ExecutionMetrics:
    symbols_in: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_canceled: int = 0
    trailing_attached: int = 0
    open_positions: int = 0
    open_orders: int = 0
    allowed_new_positions: int = 0
    open_buy_orders: int = 0
    max_total_positions: int = 0
    configured_max_positions: int = 0
    risk_limited_max_positions: int = 0
    slots_total: int = 0
    api_retries: int = 0
    api_failures: int = 0
    auth_ok: bool = False
    auth_reason: str | None = None
    exit_reason: str | None = None
    in_window: bool = False
    latency_samples: List[float] = field(default_factory=list)
    skipped_reasons: Counter = field(default_factory=Counter)
    status: str = "ok"
    _error_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Maintain backward compatibility with legacy attribute name
        self.skipped_by_reason = self.skipped_reasons

    def record_skip(self, reason: str, count: int = 1) -> None:
        count = int(count)
        if count <= 0:
            return
        key = reason.upper()
        self.skipped_reasons[key] = self.skipped_reasons.get(key, 0) + count

    def record_latency(self, seconds: float) -> None:
        if seconds > 0:
            self.latency_samples.append(seconds)

    def record_error(self, message: str, **details: Any) -> None:
        payload = {"message": message}
        for key, value in details.items():
            if value is None:
                continue
            try:
                json.dumps({key: value})  # validate serialisable
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._error_info = payload
        if self.status != "auth_error":
            self.status = "error"

    def percentile(self, p: float) -> float:
        if not self.latency_samples:
            return 0.0
        ordered = sorted(self.latency_samples)
        if len(ordered) == 1:
            return round(ordered[0], 3)
        rank = (len(ordered) - 1) * p
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return round(ordered[int(rank)], 3)
        fraction = rank - lower
        value = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        return round(value, 3)

    def as_dict(self) -> Dict[str, Any]:
        skip_payload = {
            key: int(self.skipped_reasons.get(key, 0)) for key in SKIP_REASON_ORDER
        }
        for key, value in sorted(self.skipped_reasons.items()):
            if key not in skip_payload:
                skip_payload[key] = int(value)
        max_total_positions = self.max_total_positions or self.configured_max_positions
        payload: Dict[str, Any] = {
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
            "symbols_in": self.symbols_in,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_canceled": self.orders_canceled,
            "trailing_attached": self.trailing_attached,
            "open_positions": self.open_positions,
            "open_orders": self.open_orders,
            "allowed_new_positions": self.allowed_new_positions,
            "max_total_positions": max_total_positions,
            "configured_max_positions": self.configured_max_positions,
            "risk_limited_max_positions": self.risk_limited_max_positions,
            "slots_total": self.slots_total,
            "open_buy_orders": self.open_buy_orders,
            "api_retries": self.api_retries,
            "api_failures": self.api_failures,
            "auth_ok": bool(self.auth_ok),
            "auth_reason": self.auth_reason,
            "exit_reason": None if self.exit_reason is None else str(self.exit_reason or "UNKNOWN"),
            "in_window": bool(self.in_window),
            "latency_secs": {
                "p50": self.percentile(0.5),
                "p95": self.percentile(0.95),
            },
            "skips": skip_payload,
            "status": self.status,
        }
        if self._error_info:
            payload["error"] = dict(self._error_info)
        return payload

    def flush(self) -> None:
        return None

    @property
    def skips(self) -> Counter:
        return self.skipped_reasons


def compute_limit_price(row: Dict[str, Any], buffer_bps: int = 75) -> float:
    base_price = row.get("entry_price") if row.get("entry_price") not in (None, "") else row.get("close")
    if base_price is None or pd.isna(base_price):
        raise ValueError("Row must contain either entry_price or close")
    base_price_f = float(base_price)
    limit = base_price_f * (1 + buffer_bps / 10_000)
    return float(round_to_tick(limit))


def _sanitize_atr_pct(value: Any) -> float:
    try:
        atr = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(atr) or atr <= 0:
        return 0.0
    if atr > 1.0:
        atr /= 100.0
    return max(0.0, atr)


def _compute_qty(
    buying_power: float,
    limit_price: float,
    allocation_pct: float,
    min_order_usd: float,
    *,
    target_notional: float | None = None,
) -> int:
    limit_floor = max(0.01, float(limit_price or 0))
    notional_budget = (
        max(0.0, buying_power) * max(0.0, allocation_pct)
        if target_notional is None
        else max(0.0, float(target_notional))
    )
    alloc_qty = math.floor(notional_budget / limit_floor)
    min_qty = (
        math.floor(max(0.0, min_order_usd) / limit_floor)
        if min_order_usd
        else 0
    )
    qty = max(alloc_qty, min_qty)
    return max(0, qty)


def compute_quantity(buying_power: float, allocation_pct: float, limit_price: float) -> int:
    if buying_power <= 0 or allocation_pct <= 0:
        return 0
    notional_cap = buying_power * allocation_pct
    if notional_cap <= 0:
        return 0
    qty = int(notional_cap // limit_price)
    if qty <= 0 and notional_cap >= limit_price:
        qty = 1
    if qty <= 0:
        return 0
    max_affordable = int(buying_power // limit_price)
    if max_affordable <= 0:
        return 0
    qty = min(qty, max_affordable)
    if qty < 1:
        return 0
    return qty


class DailyBarCache:
    def __init__(self, search_paths: Sequence[Path]) -> None:
        self.search_paths = search_paths
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _candidate_paths(self, symbol: str) -> Iterator[Path]:
        for directory in self.search_paths:
            candidate = directory / f"{symbol.upper()}.csv"
            yield candidate

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        key = symbol.upper()
        if key in self._cache:
            return self._cache[key]
        for path in self._candidate_paths(key):
            if path.exists():
                try:
                    frame = pd.read_csv(path)
                    self._cache[key] = frame
                    return frame
                except Exception:  # pragma: no cover - defensive parsing
                    break
        self._cache[key] = None
        return None

    def atr_percent(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        required = {"high", "low", "close"}
        if not required.issubset(lower_map.keys()):
            return None
        try:
            highs = pd.to_numeric(frame[lower_map["high"]], errors="coerce")
            lows = pd.to_numeric(frame[lower_map["low"]], errors="coerce")
            closes = pd.to_numeric(frame[lower_map["close"]], errors="coerce")
        except KeyError:
            return None
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}).dropna()
        if df.empty or len(df) < 15:
            return None
        df.sort_index(inplace=True)
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=14).mean()
        latest_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else None
        latest_close = df["close"].iloc[-1]
        if latest_atr is None or latest_close <= 0:
            return None
        return round(float(latest_atr) / float(latest_close) * 100, 3)

    def adv20(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        volume_column = lower_map.get("volume") or lower_map.get("vol")
        if not close_column or not volume_column:
            return None
        try:
            closes = pd.to_numeric(frame[close_column], errors="coerce")
            volumes = pd.to_numeric(frame[volume_column], errors="coerce")
        except Exception:
            return None
        df = pd.DataFrame({"close": closes, "volume": volumes}).dropna()
        if df.empty:
            return None
        window = df.tail(20)
        if window.empty:
            return None
        adv = (window["close"] * window["volume"]).mean()
        if pd.isna(adv):
            return None
        return float(adv)

    def latest_close(self, symbol: str) -> Optional[float]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        if not close_column or close_column not in frame.columns:
            return None
        try:
            closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
        except Exception:  # pragma: no cover - defensive conversion
            return None
        if closes.empty:
            return None
        return float(closes.iloc[-1])

    def latest_bar(self, symbol: str) -> Optional[Dict[str, Optional[float]]]:
        frame = self.load(symbol)
        if frame is None or frame.empty:
            return None
        lower_map = {col.lower(): col for col in frame.columns}
        close_column = lower_map.get("close")
        volume_column = lower_map.get("volume") or lower_map.get("vol")
        close_value: Optional[float] = None
        volume_value: Optional[float] = None
        if close_column and close_column in frame.columns:
            try:
                closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
                if not closes.empty:
                    close_value = float(closes.iloc[-1])
            except Exception:  # pragma: no cover - defensive conversion
                close_value = None
        if volume_column and volume_column in frame.columns:
            try:
                volumes = pd.to_numeric(frame[volume_column], errors="coerce").dropna()
                if not volumes.empty:
                    volume_value = float(volumes.iloc[-1])
            except Exception:  # pragma: no cover - defensive conversion
                volume_value = None
        if close_value is None and volume_value is None:
            return None
        return {"close": close_value, "volume": volume_value, "source": "cache"}


class OptionalFieldHydrator:
    def __init__(self, client: Any, cache: DailyBarCache) -> None:
        self.client = client
        self.cache = cache
        self.exchange_cache: Dict[str, Optional[str]] = {}
        self.latest_bar_cache: Dict[str, Dict[str, Optional[float]]] = {}
        self._missing_logged: set[Tuple[str, str]] = set()
        self.missing_counts: Counter[str] = Counter()
        self._summary_logged = False
        self._entry_summary_logged = False
        self._entry_defaults = 0

    def _log_missing(
        self, field: str, symbol: str, detail: str = "", *, qualifier: str = "defaulted"
    ) -> None:
        key = (field.lower(), symbol.upper())
        if key in self._missing_logged:
            return
        qualifier_text = qualifier.strip()
        if qualifier_text:
            parts = [f"[WARN] MISSING_{field.upper()} ({qualifier_text})"]
        else:
            parts = [f"[WARN] MISSING_{field.upper()}"]
        if symbol:
            parts.append(f"symbol={symbol.upper()}")
        if detail:
            parts.append(detail)
        LOGGER.warning(" ".join(parts))
        self._missing_logged.add(key)
        self.missing_counts[field.upper()] += 1
        if field.lower() == "entry_price" and "defaulted from close" in qualifier_text.lower():
            self._entry_defaults += 1

    def get_exchange(self, symbol: str) -> Optional[str]:
        key = symbol.upper()
        if key in self.exchange_cache:
            return self.exchange_cache[key]
        exchange: Optional[str] = None
        if self.client is not None:
            try:
                asset = self.client.get_asset(key)
                exchange = getattr(asset, "exchange", None)
            except Exception:  # pragma: no cover - API errors surfaced elsewhere
                exchange = None
        self.exchange_cache[key] = exchange
        return exchange

    @staticmethod
    def _value_missing(value: Any) -> bool:
        if value in (None, ""):
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def prime_latest_bars(self, records: Iterable[Mapping[str, Any]]) -> None:
        symbols_to_fetch: List[str] = []
        for row in records:
            raw_symbol = row.get("symbol")
            symbol = str(raw_symbol).upper() if raw_symbol not in (None, "") else ""
            if not symbol:
                continue
            needs_close = self._value_missing(row.get("close"))
            needs_volume = self._value_missing(row.get("volume"))
            cache_entry = self.latest_bar_cache.get(symbol)
            if cache_entry:
                if not needs_close or cache_entry.get("close") is not None:
                    needs_close = False
                if not needs_volume or cache_entry.get("volume") is not None:
                    needs_volume = False
            if not needs_close and not needs_volume:
                continue
            local = self.cache.latest_bar(symbol)
            if local:
                cached = self.latest_bar_cache.get(symbol, {})
                if local.get("close") is not None:
                    cached["close"] = local.get("close")
                if local.get("volume") is not None:
                    cached["volume"] = local.get("volume")
                cached["source"] = "cache"
                self.latest_bar_cache[symbol] = cached
                if (
                    (not needs_close or cached.get("close") is not None)
                    and (not needs_volume or cached.get("volume") is not None)
                ):
                    continue
            symbols_to_fetch.append(symbol)

        unique_symbols = sorted(set(symbols_to_fetch))
        remote_payload = _fetch_latest_daily_bars(unique_symbols) if unique_symbols else {}
        for symbol, payload in remote_payload.items():
            cached = self.latest_bar_cache.get(symbol, {})
            if "close" in payload and payload.get("close") is not None:
                cached["close"] = payload.get("close")
            if "volume" in payload and payload.get("volume") is not None:
                cached["volume"] = payload.get("volume")
            if payload:
                cached["source"] = payload.get("source") or "alpaca"
            self.latest_bar_cache[symbol] = cached

    def latest_bar(self, symbol: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
        key = symbol.upper()
        if key in self.latest_bar_cache:
            cached = self.latest_bar_cache[key]
            return (
                cached.get("close"),
                cached.get("volume"),
                cached.get("source"),
            )
        local = self.cache.latest_bar(symbol)
        if local:
            self.latest_bar_cache[key] = dict(local)
            cached = self.latest_bar_cache[key]
            return (
                cached.get("close"),
                cached.get("volume"),
                cached.get("source"),
            )
        price = _fetch_latest_close_from_alpaca(symbol)
        if price is not None:
            cached = {"close": price, "volume": None, "source": "alpaca"}
            self.latest_bar_cache[key] = cached
            return price, None, "alpaca"
        return None, None, None

    def log_missing_summary(self) -> None:
        if not self.missing_counts or self._summary_logged:
            return
        summary = " ".join(
            f"{field.lower()}={count}" for field, count in sorted(self.missing_counts.items())
        )
        if summary:
            LOGGER.info("[INFO] DEFAULTED_OPTIONALS %s", summary)
        if self._entry_defaults and not self._entry_summary_logged:
            LOGGER.warning(
                "[WARN] entry_price defaulted from close on %d rows", self._entry_defaults
            )
            self._entry_summary_logged = True
        self._summary_logged = True

    def hydrate(self, row: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(row)
        symbol = enriched.get("symbol", "").upper()
        if symbol:
            exchange_value = enriched.get("exchange")
            if exchange_value in (None, "") or pd.isna(exchange_value):
                exchange = self.get_exchange(symbol)
                enriched["exchange"] = exchange
                self._log_missing(
                    "exchange",
                    symbol,
                    detail=f"source={'asset' if exchange else 'default'}",
                )
            close_value = enriched.get("close")
            if self._value_missing(close_value):
                fallback = enriched.get("entry_price")
                fallback_source: Optional[str] = None
                source_label = "entry_price"
                if self._value_missing(fallback):
                    latest_close, _, fallback_source = self.latest_bar(symbol)
                    fallback = latest_close
                    source_label = fallback_source or "default"
                if fallback is not None and not pd.isna(fallback):
                    enriched["close"] = fallback
                    self._log_missing(
                        "close",
                        symbol,
                        detail=f"source={source_label}",
                        qualifier="defaulted",
                    )
                else:
                    self._log_missing(
                        "close",
                        symbol,
                        detail="source=unavailable",
                        qualifier="missing",
                    )
            volume_value = enriched.get("volume")
            if self._value_missing(volume_value):
                _, latest_volume, source_label = self.latest_bar(symbol)
                if latest_volume is not None and not pd.isna(latest_volume):
                    enriched["volume"] = latest_volume
                    self._log_missing(
                        "volume",
                        symbol,
                        detail=f"source={source_label or 'default'}",
                        qualifier="defaulted",
                    )
                else:
                    self._log_missing(
                        "volume",
                        symbol,
                        detail="source=unavailable",
                        qualifier="missing",
                    )
            atr_value = enriched.get("atrp")
            if self._value_missing(atr_value):
                atr = self.cache.atr_percent(symbol)
                enriched["atrp"] = atr
                self._log_missing(
                    "atrp",
                    symbol,
                    detail=f"source={'bars' if atr is not None else 'default'}",
                )
            adv_value = enriched.get("adv20")
            if self._value_missing(adv_value):
                adv = self.cache.adv20(symbol)
                enriched["adv20"] = adv
                self._log_missing(
                    "adv20",
                    symbol,
                    detail=f"source={'bars' if adv is not None else 'default'}",
                )
        entry_value = enriched.get("entry_price")
        if self._value_missing(entry_value):
            close_value = enriched.get("close")
            if not self._value_missing(close_value):
                enriched["entry_price"] = close_value
                self._log_missing(
                    "entry_price",
                    symbol,
                    qualifier="defaulted from close",
                )
            else:
                self._log_missing(
                    "entry_price",
                    symbol,
                    detail="source=unavailable",
                    qualifier="missing",
                )
        if "adv20" not in enriched or pd.isna(enriched.get("adv20")):
            enriched["adv20"] = None
        return enriched


class TradeExecutor:
    def __init__(
        self,
        config: ExecutorConfig,
        client: Any,
        metrics: ExecutionMetrics,
        *,
        sleep_fn: Optional[Any] = None,
        base_url: str = "",
        account_snapshot: Optional[Mapping[str, Any]] = None,
        clock_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.metrics = metrics
        self.sleep = sleep_fn or time.sleep
        self.bar_cache = DailyBarCache(config.bar_directories)
        self.hydrator = OptionalFieldHydrator(client, self.bar_cache)
        self.log_json = config.log_json
        self._tz_fallback_logged = False
        self._clock_warning_logged = False
        self._base_url = base_url
        self.account_snapshot: Optional[Mapping[str, Any]] = account_snapshot
        self.clock_snapshot: Optional[Mapping[str, Any]] = clock_snapshot
        self._last_buying_power_raw: Any = None
        self._prev_close_cache: Dict[str, Optional[float]] = {}
        self._ranking_key: str = "score"

    def reconcile_closed_trades(self, *, lookback_days: int | None = None, limit: int | None = None) -> None:
        watermark_enabled = bool(getattr(self.config, "reconcile_use_watermark", True))
        if not db.db_enabled():
            if watermark_enabled:
                LOGGER.warning("[WARN] RECONCILE_WATERMARK_DISABLED reason=db_unavailable")
            return
        engine = db.get_engine()
        if engine is None:
            if watermark_enabled:
                LOGGER.warning("[WARN] RECONCILE_WATERMARK_DISABLED reason=db_unavailable")
            return

        try:
            lookback_days = int(
                lookback_days
                if lookback_days is not None
                else getattr(self.config, "reconcile_lookback_days", 7)
            )
        except Exception:
            lookback_days = 7
        lookback_days = max(0, lookback_days)
        try:
            limit = int(limit if limit is not None else getattr(self.config, "reconcile_limit", 500))
        except Exception:
            limit = 500
        limit = max(1, limit)
        try:
            overlap_secs = int(getattr(self.config, "reconcile_overlap_secs", 300))
        except Exception:
            overlap_secs = 300
        overlap_secs = max(0, overlap_secs)
        now_utc = datetime.now(timezone.utc)
        after_time = now_utc - timedelta(days=lookback_days)
        decorate_window_days = min(2, max(1, lookback_days or 0))
        decorate_after_time = now_utc - timedelta(days=decorate_window_days)
        orders_cache: Dict[str, Any] = {"orders": None, "error": False}
        fetch_after_dt = after_time
        fetch_after_iso = fetch_after_dt.isoformat()

        if watermark_enabled:
            try:
                reconcile_state = db.get_reconcile_state(engine)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("[WARN] RECONCILE_WATERMARK_DISABLED reason=db_error err=%s", exc)
                reconcile_state = {}
                watermark_enabled = False
            if watermark_enabled:
                last_after = (
                    db.normalize_ts(reconcile_state.get("last_after"), field="last_after")
                    if reconcile_state
                    else None
                )
                if last_after is not None:
                    fetch_after_dt = last_after
                else:
                    fetch_after_dt = after_time
                fetch_after_iso = fetch_after_dt.isoformat()
            else:
                fetch_after_dt = after_time
                fetch_after_iso = fetch_after_dt.isoformat()

        def _fetch_orders() -> list[dict]:
            cached = orders_cache.get("orders")
            if cached is not None:
                return cached
            if orders_cache.get("error"):
                return []
            try:
                orders = alpaca_list_orders_http(
                    lookback_days=lookback_days,
                    limit=limit,
                    after_iso=fetch_after_iso if watermark_enabled else None,
                )
                orders_cache["orders"] = orders or []
                if watermark_enabled:
                    LOGGER.info(
                        "[INFO] RECONCILE_WATERMARK after=%s fetched_orders=%s",
                        fetch_after_iso,
                        len(orders_cache["orders"]),
                    )
                else:
                    LOGGER.info(
                        "[INFO] RECONCILE_FETCH after=%s fetched_orders=%s",
                        fetch_after_iso,
                        len(orders_cache["orders"]),
                    )
                return orders_cache["orders"]
            except requests.HTTPError as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", "ERR")
                err_text = getattr(response, "text", None) or str(exc)
                LOGGER.warning(
                    "[WARN] RECONCILE_ALPACA_FAIL action=http_orders status=%s err=%s",
                    status,
                    err_text,
                )
            except Exception as exc:
                LOGGER.warning(
                    "[WARN] RECONCILE_ALPACA_FAIL action=http_orders status=ERR err=%s",
                    exc,
                )
            orders_cache["error"] = True
            orders_cache["orders"] = []
            return []

        def _best_order_timestamp(order: Mapping[str, Any]) -> datetime | None:
            for field in ("updated_at", "filled_at", "submitted_at", "created_at"):
                ts = db.normalize_ts(order.get(field), field=field)
                if ts is not None:
                    return ts
            return None

        def _latest_filled_sell_order(symbol: str) -> tuple[Any | None, datetime | None, Dict[str, Any]]:
            stats: Dict[str, Any] = {
                "after_iso": fetch_after_iso,
                "fetched_orders": 0,
                "symbol_sells": 0,
                "filled_sells": 0,
                "orders_error": False,
            }

            orders = _fetch_orders()
            stats["orders_error"] = bool(orders_cache.get("error"))
            stats["fetched_orders"] = len(orders or [])
            symbol_upper = symbol.upper()
            filled_sells: list[tuple[Any, datetime]] = []

            for order in orders or []:
                if str(order.get("symbol", "")).upper() != symbol_upper:
                    continue
                side_value = str(order.get("side", "")).lower()
                if side_value != "sell":
                    continue
                stats["symbol_sells"] += 1
                status_value = str(order.get("status", "")).lower()
                filled_at_value = order.get("filled_at")
                if status_value != "filled" or filled_at_value is None:
                    continue
                try:
                    filled_at = isoparse(str(filled_at_value))
                except Exception:
                    continue
                if filled_at.tzinfo is None:
                    filled_at = filled_at.replace(tzinfo=timezone.utc)
                else:
                    filled_at = filled_at.astimezone(timezone.utc)
                filled_sells.append((order, filled_at))

            stats["filled_sells"] = len(filled_sells)
            if not filled_sells:
                return None, None, stats

            best_order, best_filled_at = max(filled_sells, key=lambda item: item[1])
            return best_order, best_filled_at, stats

        try:
            open_trades = db.get_open_trades(engine)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("[WARN] RECONCILE_DB_FAIL stage=open_trades err=%s", exc)
            open_trades = []
        LOGGER.info("[INFO] RECONCILE_START open_trades=%s", len(open_trades))
        if self.client is None:
            tc = get_trading_client()
            if tc is None:
                LOGGER.warning("RECONCILE_ALPACA_FAIL action=get_trading_client err=unavailable")
                return
            self.client = tc

        closed_count = 0
        open_by_symbol: Dict[str, list[dict[str, Any]]] = {}

        if open_trades:
            try:
                log_info("alpaca.get_positions")
                positions = self.client.get_all_positions()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning(
                    "[WARN] RECONCILE_ALPACA_FAIL symbol=ALL err=%s err_type=%s",
                    exc,
                    type(exc).__name__,
                )
                positions = []
            open_symbols_on_alpaca = {
                str(getattr(pos, "symbol", "")).upper()
                for pos in positions
                if getattr(pos, "qty", None) not in (None, "", 0, "0", 0.0)
            }

            for trade in open_trades:
                symbol = str(trade.get("symbol", "")).upper()
                if not symbol:
                    continue
                open_by_symbol.setdefault(symbol, []).append(trade)
            for trades in open_by_symbol.values():
                trades.sort(
                    key=lambda t: db.normalize_ts(t.get("entry_time")) or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )

            for symbol, trades in list(open_by_symbol.items()):
                remaining_trades: list[dict[str, Any]] = []
                for trade in trades:
                    trade_id = trade.get("trade_id")
                    if symbol not in open_symbols_on_alpaca:
                        try:
                            closed = db.close_trade(
                                engine=engine,
                                trade_id=trade_id,
                                exit_order_id=None,
                                exit_time=datetime.now(timezone.utc),
                                exit_price=None,
                                exit_reason="POSITION_CLOSED",
                            )
                        except Exception as exc:  # pragma: no cover - defensive guard
                            LOGGER.warning(
                                "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=close_trade_position err=%s",
                                trade_id,
                                symbol,
                                exc,
                            )
                            closed = False
                        if closed:
                            closed_count += 1
                            LOGGER.info(
                                "[INFO] RECONCILE_CLOSE_BY_POSITION trade_id=%s symbol=%s",
                                trade_id,
                                symbol,
                            )
                    else:
                        remaining_trades.append(trade)
                open_by_symbol[symbol] = remaining_trades

            for symbol, trades in open_by_symbol.items():
                if not trades:
                    continue
                orders = _fetch_orders()
                symbol_orders: list[tuple[Mapping[str, Any], datetime]] = []
                for order in orders:
                    if str(order.get("symbol", "")).upper() != symbol:
                        continue
                    if str(order.get("side", "")).lower() != "sell":
                        continue
                    if str(order.get("status", "")).lower() != "filled":
                        continue
                    filled_at_raw = order.get("filled_at")
                    if filled_at_raw in (None, ""):
                        continue
                    try:
                        filled_at_dt = isoparse(str(filled_at_raw))
                    except Exception:
                        continue
                    if filled_at_dt.tzinfo is None:
                        filled_at_dt = filled_at_dt.replace(tzinfo=timezone.utc)
                    else:
                        filled_at_dt = filled_at_dt.astimezone(timezone.utc)
                    symbol_orders.append((order, filled_at_dt))
                symbol_orders.sort(key=lambda item: item[1], reverse=True)

                for order, filled_at_dt in symbol_orders:
                    if not trades:
                        break
                    trade = trades.pop(0)
                    trade_id = trade.get("trade_id")
                    exit_time_raw = filled_at_dt or datetime.now(timezone.utc)
                    exit_price_raw = order.get("filled_avg_price")
                    try:
                        exit_price = float(exit_price_raw) if exit_price_raw is not None else None
                    except (TypeError, ValueError):
                        exit_price = None
                    exit_reason = "TRAIL_STOP" if order.get("type") == "trailing_stop" else "SELL_FILL"
                    order_id = str(order.get("id") or order.get("order_id") or "")
                    try:
                        db.insert_order_event(
                            engine=engine,
                            event_type="SELL_FILL",
                            symbol=symbol,
                            qty=order.get("filled_qty", order.get("qty")),
                            order_id=order_id,
                            status=str(order.get("status", "")),
                            event_time=exit_time_raw,
                            raw=_order_snapshot(order),
                        )
                    except Exception as exc:  # pragma: no cover - defensive guard
                        LOGGER.warning(
                            "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=order_event err=%s",
                            trade_id,
                            symbol,
                            exc,
                        )
                    try:
                        closed = db.close_trade(engine, trade_id, order_id, exit_time_raw, exit_price, exit_reason)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        LOGGER.warning(
                            "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=close_trade err=%s",
                            trade_id,
                            symbol,
                            exc,
                        )
                        closed = False
                    if closed:
                        closed_count += 1
                        LOGGER.info(
                            "[INFO] RECONCILE_CLOSE trade_id=%s symbol=%s exit_price=%s reason=%s",
                            trade_id,
                            symbol,
                            "" if exit_price is None else exit_price,
                            exit_reason,
                        )
                    else:
                        LOGGER.warning(
                            "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=close_trade err=%s",
                            trade_id,
                            symbol,
                            "close_failed",
                        )

        decorated_count = 0
        try:
            trades_to_decorate = db.get_closed_trades_missing_exit(
                engine=engine, updated_after=decorate_after_time
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("[WARN] RECONCILE_DB_FAIL stage=get_closed_trades err=%s", exc)
            trades_to_decorate = []

        latest_sell_order_by_symbol: Dict[str, tuple[Any | None, datetime | None, Dict[str, Any]]] = {}
        for trade in trades_to_decorate:
            trade_id = trade.get("trade_id")
            symbol = str(trade.get("symbol", "")).upper()
            if trade_id is None or not symbol:
                continue
            if symbol not in latest_sell_order_by_symbol:
                latest_sell_order_by_symbol[symbol] = _latest_filled_sell_order(symbol)
            order, filled_at, stats = latest_sell_order_by_symbol[symbol]
            if order is None:
                LOGGER.warning(
                    "[WARN] RECONCILE_DECORATE_MISS trade_id=%s symbol=%s after_iso=%s lookback_days=%s fetched_orders=%s tsla_sell_orders=%s filled_sells=%s orders_error=%s",
                    trade_id,
                    symbol,
                    stats.get("after_iso"),
                    lookback_days,
                    stats.get("fetched_orders", 0),
                    stats.get("symbol_sells", 0),
                    stats.get("filled_sells", 0),
                    stats.get("orders_error", False),
                )
                continue

            exit_time_raw = filled_at
            exit_price_raw = order.get("filled_avg_price")
            try:
                exit_price = float(exit_price_raw) if exit_price_raw is not None else None
            except (TypeError, ValueError):
                exit_price = None
            order_type = str(order.get("type", "")).lower()
            exit_reason = "TRAIL_STOP" if order_type == "trailing_stop" else "SELL_FILL"
            order_id = str(order.get("id") or order.get("order_id") or "")

            try:
                db.insert_order_event(
                    engine=engine,
                    event_type="SELL_FILL",
                    symbol=symbol,
                    qty=order.get("filled_qty", order.get("qty")),
                    order_id=order_id,
                    status=str(order.get("status", "")),
                    event_time=exit_time_raw,
                    raw=_order_snapshot(order),
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning(
                    "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=order_event err=%s",
                    trade_id,
                    symbol,
                    exc,
                )

            realized_pnl = None
            try:
                entry_price_value = trade.get("entry_price")
                qty_value = trade.get("qty")
                if exit_price is not None and entry_price_value is not None and qty_value is not None:
                    realized_pnl = (float(exit_price) - float(entry_price_value)) * float(qty_value)
            except Exception:
                realized_pnl = None

            try:
                decorated = db.decorate_trade_exit(
                    engine=engine,
                    trade_id=trade_id,
                    exit_order_id=order_id,
                    exit_time=exit_time_raw,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    realized_pnl=realized_pnl,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning(
                    "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=decorate_exit err=%s",
                    trade_id,
                    symbol,
                    exc,
                )
                decorated = False

            if decorated:
                decorated_count += 1
                LOGGER.info(
                    "[INFO] RECONCILE_DECORATE_SELL trade_id=%s symbol=%s exit_order_id=%s exit_price=%s reason=%s",
                    trade_id,
                    symbol,
                    order_id,
                    "" if exit_price is None else exit_price,
                    exit_reason,
                )
            else:
                LOGGER.warning(
                    "[WARN] RECONCILE_DB_FAIL trade_id=%s symbol=%s stage=decorate_exit err=%s",
                    trade_id,
                    symbol,
                    "decorate_failed",
                )
        if watermark_enabled:
            if orders_cache.get("orders") is None and not orders_cache.get("error"):
                _fetch_orders()
            orders_for_watermark = orders_cache.get("orders") or []
            max_ts: datetime | None = None
            for order in orders_for_watermark:
                ts = _best_order_timestamp(order)
                if ts is None:
                    continue
                if max_ts is None or ts > max_ts:
                    max_ts = ts
            new_last_after = fetch_after_dt
            if max_ts is not None:
                new_last_after = max_ts - timedelta(seconds=overlap_secs)
            try:
                updated = db.set_reconcile_state(engine, new_last_after, now_utc)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("[WARN] RECONCILE_WATERMARK_DISABLED reason=write_failed err=%s", exc)
                updated = False
            if updated:
                LOGGER.info("[INFO] RECONCILE_WATERMARK_UPDATE last_after=%s", new_last_after.isoformat())
        LOGGER.info("[INFO] RECONCILE_END closed=%s decorated=%s", closed_count, decorated_count)

    def log_info(self, event: str, **payload: Any) -> None:
        log_info(event, **payload)
        if self.log_json:
            try:
                LOGGER.info(json.dumps({"event": event, **payload}))
            except TypeError:  # pragma: no cover - serialization guard
                LOGGER.info(json.dumps({"event": event, "message": str(payload)}))

    def _log_diagnostic_snapshot(self) -> None:
        if not getattr(self.config, "diagnostic", False):
            return
        LOGGER.info(
            "[DIAGNOSTIC_EXECUTE] exit_reason=%s allowed_new_positions=%s open_positions=%s open_orders=%s slots_total=%s in_window=%s",
            str(self.metrics.exit_reason or "UNKNOWN"),
            int(self.metrics.allowed_new_positions),
            int(self.metrics.open_positions),
            int(self.metrics.open_orders),
            int(self.metrics.slots_total),
            bool(self.metrics.in_window),
        )

    def _market_label(self) -> str:
        tz_name = str(self.config.market_timezone or "America/New_York")
        mapping = {
            "America/New_York": "NY",
            "America/Chicago": "Chicago",
        }
        if tz_name in mapping:
            return mapping[tz_name]
        suffix = tz_name.split("/")[-1].strip()
        return suffix or tz_name

    def record_skip_reason(
        self,
        reason: str,
        *,
        symbol: str = "",
        detail: str = "",
        count: int = 1,
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        reason_key = reason.upper()
        parts = [f"[INFO] {reason_key}"]
        if symbol:
            parts.append(f"symbol={symbol}")
        if detail:
            parts.append(str(detail))
        if count > 1:
            parts.append(f"count={count}")
        LOGGER.info(" ".join(parts))

        self.metrics.record_skip(reason_key, count=count)
        if aliases:
            for alias in aliases:
                if alias:
                    self.metrics.record_skip(alias.upper(), count=count)
        payload: Dict[str, Any] = {"reason": reason_key}
        if symbol:
            payload["symbol"] = symbol
        if detail:
            payload["detail"] = detail
        if count > 1:
            payload["count"] = count
        self.log_info("SKIP", **payload)

    def resolve_limit_price(self, symbol: str, row: Mapping[str, Any]) -> Optional[float]:
        key = symbol.upper()
        if not key:
            return None
        if key in self._prev_close_cache:
            return self._prev_close_cache[key]

        snapshot_price = _fetch_prevclose_snapshot(key)
        if snapshot_price is not None and snapshot_price > 0:
            LOGGER.info(
                "[INFO] LIMIT_SRC snapshot symbol=%s prevclose=%.4f",
                key,
                snapshot_price,
            )
            self._prev_close_cache[key] = snapshot_price
            return snapshot_price

        bar_price = _fetch_prev_close_from_alpaca(key)
        if bar_price is not None and bar_price > 0:
            LOGGER.info(
                "[INFO] LIMIT_SRC bars symbol=%s prevclose=%.4f",
                key,
                bar_price,
            )
            self._prev_close_cache[key] = bar_price
            return bar_price

        try:
            entry_candidate = float(row.get("entry_price"))
            if math.isnan(entry_candidate):
                entry_candidate = None
        except Exception:
            entry_candidate = None

        if entry_candidate is not None and entry_candidate > 0:
            LOGGER.info(
                "[INFO] LIMIT_SRC entry symbol=%s prevclose=%.4f",
                key,
                entry_candidate,
            )
            self._prev_close_cache[key] = entry_candidate
            return entry_candidate

        self._prev_close_cache[key] = None
        return None

    def _apply_price_band(
        self,
        symbol: str,
        anchor_price: float,
        price_ref: float,
        quote_snapshot: Mapping[str, Any] | None,
        trade_snapshot: Mapping[str, Any] | None,
    ) -> tuple[Optional[float], bool]:
        if anchor_price is None or anchor_price <= 0:
            return price_ref, False
        band_pct = max(0.0, float(self.config.price_band_pct)) / 100.0
        if band_pct <= 0:
            return price_ref, False
        action = (self.config.price_band_action or "clamp").lower()
        floor = anchor_price * (1 - band_pct)
        ceiling = anchor_price * (1 + band_pct)
        live_points: list[tuple[str, float]] = []
        if isinstance(quote_snapshot, Mapping):
            bid = quote_snapshot.get("bid")
            ask = quote_snapshot.get("ask")
            try:
                bid_f = float(bid) if bid not in (None, "") else 0.0
                ask_f = float(ask) if ask not in (None, "") else 0.0
            except (TypeError, ValueError):
                bid_f = 0.0
                ask_f = 0.0
            if bid_f > 0 and ask_f > 0 and not math.isnan(bid_f) and not math.isnan(ask_f):
                mid = (bid_f + ask_f) / 2
                if mid > 0 and not math.isnan(mid):
                    live_points.append(("mid", mid))
        if isinstance(trade_snapshot, Mapping):
            trade_price = trade_snapshot.get("price")
            try:
                trade_f = float(trade_price) if trade_price not in (None, "") else 0.0
            except (TypeError, ValueError):
                trade_f = 0.0
            if trade_f > 0 and not math.isnan(trade_f):
                live_points.append(("trade", trade_f))
        if not live_points:
            return price_ref, False
        for label, live in live_points:
            if live > ceiling:
                if action == "skip":
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=f"{label}_gt_band",
                    )
                    return None, True
                new_ref = min(max(price_ref, floor), ceiling)
                LOGGER.info(
                    "[INFO] PRICE_BAND_CLAMP symbol=%s anchor=%.4f live_src=%s live=%.4f limit=%.4f pct=%.2f",
                    symbol,
                    anchor_price,
                    label,
                    live,
                    new_ref,
                    band_pct * 100,
                )
                return new_ref, False
            if live < floor:
                if action == "skip":
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=f"{label}_lt_band",
                    )
                    return None, True
                new_ref = min(max(price_ref, floor), ceiling)
                LOGGER.info(
                    "[INFO] PRICE_BAND_CLAMP symbol=%s anchor=%.4f live_src=%s live=%.4f limit=%.4f pct=%.2f",
                    symbol,
                    anchor_price,
                    label,
                    live,
                    new_ref,
                    band_pct * 100,
                )
                return new_ref, False
        return price_ref, False

    def _resolve_market_timezone(self) -> ZoneInfo:
        tz_name = self.config.market_timezone or "America/New_York"
        try:
            return ZoneInfo(tz_name)
        except Exception:
            if not self._tz_fallback_logged:
                LOGGER.warning(
                    "[WARN] invalid market timezone '%s'; falling back to America/New_York",
                    tz_name,
                )
                self._tz_fallback_logged = True
            try:
                return ZoneInfo("America/New_York")
            except Exception:  # pragma: no cover - ZoneInfo fallback safety
                return ZoneInfo("UTC")

    def _log_clock_warning(self, status: str) -> None:
        if self._clock_warning_logged:
            return
        LOGGER.warning(
            "[WARN] clock_fetch_failed status=%s -> using tz_fallback=America/New_York",
            status,
        )
        self._clock_warning_logged = True

    def _probe_trading_clock(self) -> Optional[Any]:
        api_key, api_secret, base_url, _ = get_alpaca_creds()
        base = (base_url or "").strip()
        if not api_key or not api_secret or not base:
            return None
        url = f"{base.rstrip('/')}/v2/clock"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        log_info("alpaca.clock_probe")
        try:
            response = requests.get(url, headers=headers, timeout=5)
        except Exception as exc:
            self._log_clock_warning("ERR")
            _warn_context("alpaca.clock_probe", str(exc))
            LOGGER.debug("Trading clock probe failed: %s", exc, exc_info=False)
            return None
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                return None
            if isinstance(payload, Mapping):
                return SimpleNamespace(**payload)
            return None
        self._log_clock_warning(str(response.status_code))
        _warn_context(
            "alpaca.clock_probe",
            f"status={response.status_code}",
        )
        return None

    def _get_trading_clock(self) -> Optional[Any]:
        if self.clock_snapshot is not None:
            if isinstance(self.clock_snapshot, Mapping):
                return SimpleNamespace(**self.clock_snapshot)
            return None
        if self.client is None or not hasattr(self.client, "get_clock"):
            return self._probe_trading_clock()
        try:
            log_info("alpaca.get_clock")
            return self.client.get_clock()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            status_label = str(status) if status else "ERR"
            self._log_clock_warning(status_label)
            if status not in (401, 403):
                _warn_context("alpaca.get_clock", str(exc))
            return self._probe_trading_clock()

    def evaluate_time_window(self, *, log: bool = True) -> Tuple[bool, str, str]:
        tz = self._resolve_market_timezone()
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz)
        ny_now_dt = now_utc.astimezone(ZoneInfo("America/New_York"))
        ny_now = ny_now_dt.isoformat()
        current_time = dtime(
            now_local.hour, now_local.minute, now_local.second, now_local.microsecond
        )
        ny_time = dtime(
            ny_now_dt.hour,
            ny_now_dt.minute,
            ny_now_dt.second,
            ny_now_dt.microsecond,
        )

        premarket_start, premarket_end = _premarket_bounds_naive()
        regular_start = dtime(9, 30)
        regular_end = dtime(16, 0)
        postmarket_end = dtime(20, 0)

        within_premarket = premarket_start <= current_time < premarket_end
        within_regular = regular_start <= current_time < regular_end
        within_post = regular_end <= current_time < postmarket_end
        ny_within_premarket = premarket_start <= ny_time < premarket_end
        ny_within_regular = regular_start <= ny_time < regular_end

        mode = (self.config.time_window or "auto").lower()
        resolved_window = mode
        auto_branch = None
        if mode == "auto":
            if premarket_start <= ny_time < premarket_end:
                resolved_window = "premarket"
                auto_branch = "premarket"
            elif ny_within_regular:
                resolved_window = "regular"
                auto_branch = "regular"
            else:
                resolved_window = "regular"
                auto_branch = "regular"
            log_info("TIME_WINDOW_AUTO", branch=auto_branch, ny_time=ny_now)
        elif mode not in {"premarket", "regular", "any"}:
            resolved_window = "any"

        clock = self._get_trading_clock()
        clock_is_open = bool(getattr(clock, "is_open", False))

        message: str
        allowed = False
        market_label = self._market_label()

        if resolved_window == "premarket":
            if not self.config.extended_hours:
                allowed = False
                message = f"outside premarket ({market_label})"
            else:
                allowed = ny_within_premarket if mode == "auto" else within_premarket
                message = (
                    f"premarket window open ({market_label})"
                    if allowed
                    else f"outside premarket ({market_label})"
                )
        elif resolved_window == "regular":
            allowed = ny_within_regular if mode == "auto" else within_regular
            message = (
                f"regular session ({market_label})"
                if allowed
                else f"outside regular session ({market_label})"
            )
        else:  # any
            allowed = True
            message = f"any window ({market_label})"

        if not allowed and clock is not None and resolved_window in {"premarket", "regular"}:
            # When Alpaca reports the venue open, treat it as authoritative for overrides.
            if (
                resolved_window == "premarket"
                and self.config.extended_hours
                and clock_is_open
                and not within_regular
            ):
                allowed = True
                message = f"premarket window open ({market_label})"
            elif resolved_window == "regular" and clock_is_open and within_regular:
                allowed = True
                message = f"regular session ({market_label})"

        if log:
            log_info(
                "MARKET_TIME",
                ny_now=ny_now,
                mode=mode,
                resolved=resolved_window,
                in_window=bool(allowed),
            )
            log_info("TIME_WINDOW", message=message)
        return allowed, message, resolved_window

    def _load_latest_candidates_from_db(self) -> tuple[pd.DataFrame, Optional[str]]:
        records = load_candidates_from_db(
            metrics=self.metrics,
            record_skip=self.record_skip_reason,
            diagnostic=bool(getattr(self.config, "diagnostic", False)),
        )
        df = pd.DataFrame(records)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype("string").str.upper()
        run_label: Optional[str] = None
        if "run_date" in df.columns and not df["run_date"].isna().all():
            try:
                run_label = str(df["run_date"].dropna().iloc[0])
            except Exception:
                run_label = None
        LOGGER.info("[INFO] DB_CANDIDATES rows=%s", len(df))
        return df, run_label

    def load_candidates(self, *, rank: bool = True) -> pd.DataFrame:
        source_type = (self.config.source_type or "csv").strip().lower()
        if source_type == "db":
            LOGGER.info("[INFO] CANDIDATE_SOURCE db")
        df: pd.DataFrame
        base_dir: Path | None = None
        if source_type == "db":
            df, _ = self._load_latest_candidates_from_db()
            base_dir = Path.cwd()
        elif source_type == "csv":
            path = self.config.source_path
            if not path.exists():
                raise CandidateLoadError(f"Candidate file not found: {path}")
            df = pd.read_csv(path, dtype={"symbol": "string"})
            try:
                base_dir = path.resolve().parent.parent
            except Exception:
                base_dir = Path.cwd()
        else:
            raise CandidateLoadError(f"Unknown candidate source: {source_type}")
        raw_columns = list(df.columns)
        LOGGER.info("[INFO] CANDIDATE_COLUMNS raw=%s", raw_columns)
        normalized_columns = [str(column).strip() for column in raw_columns]
        LOGGER.info("[INFO] CANDIDATE_COLUMNS normalized=%s", normalized_columns)
        df.columns = normalized_columns
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype("string").str.upper()
        if df.empty:
            LOGGER.info("[INFO] NO_CANDIDATES_IN_SOURCE")
            return df
        if base_dir is None:
            base_dir = Path.cwd()
        df = _canonicalize_candidate_header(df, base_dir)
        preserved_score_cols = [
            column
            for column in df.columns
            if re.match(r"model_score", str(column), flags=re.IGNORECASE)
        ]
        preserved_scores = df[preserved_score_cols].copy() if preserved_score_cols else None
        if df.empty:
            LOGGER.info("[INFO] NO_CANONICAL_CANDIDATES")
            return df
        missing = [
            column
            for column in ("symbol", "score", "close")
            if column not in df.columns or df[column].isna().all()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise CandidateLoadError(f"Missing required columns: {joined}")
        normalized = normalize_candidate_df(df, now_ts=None)
        if preserved_scores is not None:
            preserved_scores = preserved_scores.reset_index(drop=True)
            for column in preserved_score_cols:
                if column in normalized.columns:
                    continue
                try:
                    normalized[column] = preserved_scores[column]
                except Exception:
                    normalized[column] = pd.NA
        missing_required = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
        if missing_required:
            joined = ", ".join(sorted(missing_required))
            raise CandidateLoadError(f"Missing required columns: {joined}")
        df, warnings = _apply_candidate_defaults(normalized)
        for message in warnings:
            LOGGER.warning(message)
        if rank:
            return self._rank_candidates(df)
        return df

    def _rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("RANK_CANDIDATES_ENTER rows=%d cols=%d", len(df), len(df.columns))

        def _as_numeric(series: pd.Series | None) -> pd.Series:
            return (
                pd.to_numeric(series, errors="coerce")
                if series is not None
                else pd.Series(dtype="float")
            )

        df = df.copy()
        normalized_columns = [str(column).strip() for column in df.columns]
        df.columns = normalized_columns

        score_series = _as_numeric(df.get("score"))
        df["score"] = score_series
        score_non_null = int(score_series.notna().sum())

        key_label = "score"
        ranking_series = score_series
        non_null = score_non_null
        reason = None

        ms_col = "model_score_5d"
        ms_col_actual = next(
            (column for column in df.columns if str(column).strip() == ms_col), None
        )
        if ms_col_actual is not None:
            series = df[ms_col_actual]
            series_numeric = pd.to_numeric(series, errors="coerce")
            df[ms_col_actual] = series_numeric
            non_null_model = int(series_numeric.notna().sum())
            total = len(df)
            logger.info(
                "MODEL_SCORE_DIAG col=%s dtype=%s non_null=%d total=%d sample=%s",
                ms_col_actual,
                str(getattr(series, "dtype", "unknown")),
                non_null_model,
                total,
                series.head(3).tolist(),
            )
            if non_null_model > 0:
                key_label = ms_col_actual
                ranking_series = series_numeric
                non_null = non_null_model
                reason = None
            else:
                reason = "all_nan_model_score"
        else:
            reason = "missing_model_score_column"

        self._ranking_key = key_label
        if reason:
            LOGGER.info(
                "[INFO] CANDIDATE_RANKING key=%s rows=%d non_null=%d reason=%s",
                key_label,
                len(df),
                non_null,
                reason,
            )
        else:
            LOGGER.info(
                "[INFO] CANDIDATE_RANKING key=%s rows=%d non_null=%d",
                key_label,
                len(df),
                non_null,
            )

        ranked = df.assign(_rank_val=ranking_series, _score_sort=score_series)
        ranked = ranked.sort_values(
            by=["_rank_val", "_score_sort"],
            ascending=[False, False],
            na_position="last",
        )
        ranked = ranked.drop(columns=["_rank_val", "_score_sort"]).reset_index(drop=True)
        return ranked

    def hydrate_candidates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records = df.to_dict(orient="records")
        self.hydrator.prime_latest_bars(records)
        enriched = [self.hydrator.hydrate(record) for record in records]
        self.hydrator.log_missing_summary()
        return enriched

    def guard_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        buffer_pct = 0.0
        if self.config.extended_hours:
            try:
                buffer_pct = max(0.0, float(self.config.limit_buffer_pct)) / 100.0
            except (TypeError, ValueError):
                buffer_pct = 0.0
        min_threshold = max(0.0, float(self.config.min_price) * (1 - buffer_pct))
        max_threshold = float(self.config.max_price) * (1 + buffer_pct)
        for record in records:
            raw_symbol = record.get("symbol")
            symbol = ""
            if raw_symbol not in (None, "") and not pd.isna(raw_symbol):
                symbol = str(raw_symbol).upper()
            price = record.get("entry_price")
            if price in (None, "") or pd.isna(price):
                price = record.get("close")
            if price in (None, "") or pd.isna(price):
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail="missing_price",
                    aliases=("MISSING_PRICE", "DATA_MISSING"),
                )
                continue
            price_f = float(price)
            if price_f < min_threshold:
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail=f"lt_min({price_f:.2f}) thr={min_threshold:.2f}",
                    aliases=("PRICE_LT_MIN",),
                )
                continue
            if price_f > max_threshold:
                self.record_skip_reason(
                    "PRICE_BOUNDS",
                    symbol=symbol,
                    detail=f"gt_max({price_f:.2f}) thr={max_threshold:.2f}",
                    aliases=("PRICE_GT_MAX",),
                )
                continue
            adv = record.get("adv20")
            if adv not in (None, "") and not pd.isna(adv):
                adv_value = pd.to_numeric(pd.Series([adv]), errors="coerce").iloc[0]
                if (
                    not pd.isna(adv_value)
                    and float(adv_value) > 0
                    and float(adv_value) < self.config.min_adv20
                ):
                    detail = f"adv20_lt_min({float(adv_value):.0f})"
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=detail,
                        aliases=("ADV20_LT_MIN",),
                    )
                    continue
            filtered.append(record)
        return filtered

    def _log_top_candidates(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_positions: int,
        base_alloc_pct: float,
        source_df: pd.DataFrame | None = None,
    ) -> None:
        if not records:
            return
        rank_key = self._ranking_key or "score"
        frame = pd.DataFrame.from_records(records)
        if "_rank_val" in frame.columns:
            rank_series = pd.to_numeric(frame["_rank_val"], errors="coerce")
        else:
            rank_series = pd.to_numeric(frame.get(rank_key, pd.Series(dtype="float")), errors="coerce")
        symbols = frame.get("symbol", pd.Series(dtype="string"))
        picks: list[tuple[str, Any]] = []
        for idx in range(len(frame)):
            symbol_raw = symbols.iloc[idx] if idx < len(symbols) else ""
            symbol = str(symbol_raw).upper()
            raw_value = rank_series.iloc[idx] if idx < len(rank_series) else None
            if pd.isna(raw_value):
                rank_value: Any = None
            else:
                rank_value = round(float(raw_value), 6)
            picks.append((symbol, rank_value))
        LOGGER.info("[INFO] CANDIDATE_PICK key=%s top=%s", rank_key, picks)

        top_n = min(max_positions, len(picks))
        picked_df = frame.head(top_n) if top_n > 0 else pd.DataFrame()
        weight_frame = source_df if source_df is not None else frame
        weight_frame = weight_frame.head(top_n) if top_n > 0 else weight_frame.iloc[0:0]
        weight_symbols = weight_frame.get("symbol", pd.Series(dtype="string"))
        weight_series = pd.to_numeric(
            weight_frame.get("alloc_weight", pd.Series(dtype="float")), errors="coerce"
        )
        weights: dict[str, float] = {}
        for sym_raw, weight_val in zip(weight_symbols, weight_series):
            sym = str(sym_raw).upper()
            if not sym or pd.isna(weight_val):
                continue
            try:
                weight_float = float(weight_val)
            except Exception:
                continue
            if not math.isfinite(weight_float):
                continue
            weights[sym] = weight_float
        picked_symbols: list[str] = [str(sym).upper() for sym in picked_df.get("symbol", [])]
        picked_weights_found = sum(1 for sym in picked_symbols if sym in weights)

        weight_columns = list(weight_frame.columns)[:15]
        LOGGER.info(
            "[INFO] ALLOC_WEIGHT_COLUMNS present=%s columns=%s",
            "alloc_weight" in weight_frame.columns,
            weight_columns,
        )

        LOGGER.info(
            "ALLOC_WEIGHT_DIAG total_rows=%d weights_non_null=%d picked=%d picked_weights_found=%d sample=%s",
            len(weight_frame),
            int(pd.notna(weight_series).sum()),
            top_n,
            picked_weights_found,
            list(weights.items())[:3],
        )

        LOGGER.info(
            "ALLOC_BLOCK_ENTER rows=%d top_n=%d base_alloc_pct=%s",
            len(frame),
            top_n,
            base_alloc_pct,
        )
        if picked_weights_found > 0:
            split: list[tuple[str, float, float]] = []
            for symbol in picked_symbols:
                if symbol not in weights:
                    continue
                weight_float = float(weights[symbol])
                split.append(
                    (symbol, round(weight_float, 6), round(base_alloc_pct * weight_float, 6))
                )
            LOGGER.info(
                "ALLOCATION_MODE mode=weighted base_alloc_pct=%s weights_col=alloc_weight",
                base_alloc_pct,
            )
            LOGGER.info("ALLOC_SPLIT top=%s", split)
        else:
            LOGGER.info("ALLOCATION_MODE mode=flat base_alloc_pct=%s", base_alloc_pct)
            LOGGER.warning("ALLOC_SPLIT_SKIPPED reason=missing_or_invalid_alloc_weight")

    def execute(
        self,
        df: pd.DataFrame,
        *,
        prefiltered: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        diagnostic_mode = bool(getattr(self.config, "diagnostic", False))
        self.metrics.symbols_in = len(df)
        if prefiltered is not None:
            candidates = prefiltered
        else:
            records = self.hydrate_candidates(df)
            candidates = self.guard_candidates(records)
        allowed, status, resolved_window = self.evaluate_time_window()
        self.metrics.in_window = bool(allowed)
        if self.config.dry_run:
            LOGGER.info(
                "[INFO] DRY_RUN=True — no orders will be submitted, but still perform sizing and emit skip reasons"
            )

        existing_positions = self.fetch_existing_positions()
        open_order_symbols, open_buy_order_count, open_order_total = self.fetch_open_order_symbols()
        self.metrics.open_buy_orders = open_buy_order_count
        self.metrics.open_orders = open_order_total
        account_buying_power = self.fetch_buying_power()
        try:
            max_positions = int(self.config.max_positions)
        except (TypeError, ValueError):
            max_positions = 0
        max_positions = max(1, max_positions)
        self.metrics.max_total_positions = max_positions
        self.metrics.configured_max_positions = max_positions
        self.metrics.risk_limited_max_positions = max_positions
        try:
            max_new_positions = int(self.config.max_new_positions)
        except (TypeError, ValueError):
            max_new_positions = max_positions
        max_new_positions = max(1, max_new_positions)
        open_count = len(existing_positions)
        slots_total = max(0, max_positions - open_count)
        allowed_new = min(slots_total, max_new_positions)
        self.metrics.open_positions = open_count
        self.metrics.allowed_new_positions = allowed_new
        self.metrics.slots_total = slots_total
        LOGGER.info(
            "POSITION_LIMIT max_total=%d open=%d slots_total=%d max_new=%d allowed_new=%d",
            max_positions,
            open_count,
            slots_total,
            max_new_positions,
            allowed_new,
        )
        if not candidates:
            self.metrics.exit_reason = "NO_CANDIDATES"
            self.metrics.record_skip("NO_CANDIDATES", count=max(1, len(df)))
            LOGGER.info("No candidates passed guardrails; nothing to do.")
            self.persist_metrics()
            self._log_diagnostic_snapshot()
            self.log_summary()
            return 0
        if slots_total <= 0:
            self.record_skip_reason("MAX_POSITIONS", count=max(1, len(candidates)))
            self.metrics.exit_reason = "MAX_POSITIONS"
            self.persist_metrics()
            self._log_diagnostic_snapshot()
            self.log_summary()
            return 0
        queue: list[dict[str, Any]] = []
        for record in candidates:
            symbol = str(record.get("symbol", "")).upper()
            if not symbol:
                continue
            if symbol in existing_positions:
                self.record_skip_reason("EXISTING_POSITION", symbol=symbol)
                continue
            queue.append(record)
        if not queue:
            LOGGER.info(
                "No available slots after filtering existing positions (holdings=%d max=%d)",
                open_count,
                max_positions,
            )
            if not self.metrics.exit_reason or self.metrics.exit_reason == "UNKNOWN":
                self.metrics.exit_reason = "NO_SLOTS"
            self.persist_metrics()
            self._log_diagnostic_snapshot()
            self.log_summary()
            return 0
        if len(queue) > allowed_new:
            queue = queue[:allowed_new]
        slot_hint = max(1, min(allowed_new, len(queue)))
        candidates = queue
        try:
            allocation_pct = float(self.config.allocation_pct)
        except (TypeError, ValueError):
            allocation_pct = 0.0
        allocation_pct = max(0.0, allocation_pct)

        self._log_top_candidates(
            candidates,
            max_positions=max_positions,
            base_alloc_pct=allocation_pct,
            source_df=df,
        )

        weight_map: dict[str, float] = {}
        weight_mode = False
        if candidates and all("alloc_weight" in record for record in candidates):
            raw_weights: dict[str, float] = {}
            for record in candidates:
                symbol = str(record.get("symbol", "")).upper()
                try:
                    weight_val = float(record.get("alloc_weight"))
                except (TypeError, ValueError):
                    weight_val = math.nan
                if not symbol or math.isnan(weight_val) or weight_val < 0:
                    raw_weights = {}
                    break
                raw_weights[symbol] = weight_val
            total_weight = sum(raw_weights.values()) if raw_weights else 0.0
            if total_weight > 0:
                weight_mode = True
                weight_map = {k: v / total_weight for k, v in raw_weights.items()}

        if not allowed:
            self.record_skip_reason("TIME_WINDOW", detail=status, count=len(candidates))
            self.metrics.exit_reason = "TIME_WINDOW"
            self.persist_metrics()
            self._log_diagnostic_snapshot()
            self.log_summary()
            return 0
        bp_raw = self._last_buying_power_raw
        if bp_raw is None:
            bp_raw = account_buying_power
        bp_display: str
        try:
            bp_display = f"{float(bp_raw):.2f}"
        except Exception:
            bp_display = str(bp_raw)
        if account_buying_power <= 0 or bp_raw in (None, "", 0, "0"):
            LOGGER.info("EXECUTE_SKIP reason=CASH buying_power=%s", bp_display)
            self.record_skip_reason(
                "CASH",
                detail=f"buying_power={bp_display}",
                count=max(1, len(candidates)),
            )
            self.metrics.exit_reason = "CASH"
            self.persist_metrics()
            self._log_diagnostic_snapshot()
            self.log_summary()
            return 0
        limit_buffer_ratio = max(0.0, float(self.config.limit_buffer_pct)) / 100.0
        max_gap_ratio = max(0.0, float(self.config.max_gap_pct)) / 100.0
        ref_buffer_ratio = max(0.0, float(self.config.ref_buffer_pct)) / 100.0
        premarket_active = resolved_window == "premarket" and self.config.extended_hours
        preferred_feed = (
            os.getenv("LIMIT_PRICE_FEED") or os.getenv("ALPACA_DATA_FEED") or None
        )

        price_mode = (self.config.price_source or "entry").lower()
        min_prevclose = max(1.0, float(self.config.min_price or 0.0))

        submitted_new = 0
        for record in candidates:
            symbol = record.get("symbol", "").upper()
            if not symbol:
                continue
            if symbol in open_order_symbols:
                self.record_skip_reason("OPEN_ORDER", symbol=symbol)
                continue
            if submitted_new >= allowed_new:
                break
            if len(existing_positions) + submitted_new >= max_positions:
                self.record_skip_reason("MAX_POSITIONS", symbol=symbol)
                break

            anchor_label = "entry"
            anchor_price: Optional[float] = None
            price_f = 0.0
            price_ref = 0.0
            price_src = "entry"
            price_ref_src = "entry"
            reference_price: Optional[float] = None
            limit_source = "prev_close_only"
            mode = (
                price_mode
                if price_mode in {"prevclose", "entry", "close", "blended"}
                else "entry"
            )
            if mode in {"prevclose", "blended"}:
                anchor_price = self.resolve_limit_price(symbol, record)
                if anchor_price is None or anchor_price <= 0:
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail="prevclose_unavailable",
                    )
                    continue
                if anchor_price < min_prevclose:
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=f"prevclose_lt_min({anchor_price:.2f})",
                    )
                    continue
                price_f = anchor_price
                price_ref = anchor_price
                reference_price = anchor_price
                anchor_label = "prevclose"
                price_src = anchor_label
                price_ref_src = anchor_label
            else:
                if mode == "close":
                    price_value = record.get("close")
                    fallback_value = record.get("entry_price")
                    anchor_label = "close"
                else:
                    price_value = record.get("entry_price")
                    fallback_value = record.get("close")
                    anchor_label = "entry"
                if price_value in (None, "") or (isinstance(price_value, float) and math.isnan(price_value)):
                    price_value = fallback_value
                    anchor_label = "entry" if mode == "close" else "close"
                price_series = pd.to_numeric(pd.Series([price_value]), errors="coerce").fillna(0.0)
                price_f = float(price_series.iloc[0])
                if price_f <= 0:
                    self.record_skip_reason(
                        "ZERO_QTY",
                        symbol=symbol,
                        detail="invalid_price",
                        aliases=("DATA_MISSING",),
                    )
                    continue
                price_ref = price_f
                anchor_price = price_f
            price_src = anchor_label
            price_ref_src = anchor_label

            atr_pct = _sanitize_atr_pct(record.get("atrp"))
            atr_scale = 1.0
            min_order = max(0.0, float(self.config.min_order_usd))

            quote_snapshot: Dict[str, Optional[float | str]] = {}
            trade_snapshot: Dict[str, Optional[float | str]] = {}
            live_reference_needed = premarket_active and mode == "prevclose"
            if mode == "blended" or live_reference_needed:
                trade_snapshot = _fetch_latest_trade_from_alpaca(symbol, feed=preferred_feed)
                trade_price: Optional[float] = None
                trade_feed = None
                if isinstance(trade_snapshot, Mapping):
                    raw_trade = trade_snapshot.get("price")
                    if raw_trade is not None:
                        try:
                            trade_price = float(raw_trade)
                        except (TypeError, ValueError):
                            trade_price = None
                    trade_feed = trade_snapshot.get("feed")
                if trade_price is not None and not math.isnan(trade_price) and trade_price > 0:
                    reference_price = trade_price
                    price_ref_src = f"trade[{str(trade_feed or 'default').strip()}]"
                    limit_source = "ref_price"
                else:
                    quote_snapshot = _fetch_latest_quote_from_alpaca(symbol, feed=preferred_feed)
                    quote_feed = None
                    ask_price: Optional[float] = None
                    if isinstance(quote_snapshot, Mapping):
                        quote_feed = quote_snapshot.get("feed")
                        raw_ask = quote_snapshot.get("ask")
                        try:
                            ask_price = float(raw_ask) if raw_ask is not None else None
                        except (TypeError, ValueError):
                            ask_price = None
                    if ask_price is not None and not math.isnan(ask_price) and ask_price > 0:
                        reference_price = ask_price
                        price_ref_src = f"ask[{str(quote_feed or 'default').strip()}]"
                        limit_source = "ref_price"
                if reference_price is None or reference_price <= 0 or math.isnan(reference_price):
                    reference_price = anchor_price
                    price_ref_src = anchor_label
                    limit_source = "prev_close_only"
                else:
                    limit_source = "blended" if mode == "blended" else "ref_price"
                price_ref = reference_price or 0.0
                quote_feed = (
                    quote_snapshot.get("feed") if isinstance(quote_snapshot, Mapping) else None
                )
                trade_feed = (
                    trade_snapshot.get("feed") if isinstance(trade_snapshot, Mapping) else None
                )
                quote_ts = (
                    quote_snapshot.get("timestamp")
                    if isinstance(quote_snapshot, Mapping)
                    else None
                )
                trade_ts = (
                    trade_snapshot.get("timestamp")
                    if isinstance(trade_snapshot, Mapping)
                    else None
                )
                LOGGER.info(
                    "PRICE_REF symbol=%s anchor=%.4f ref=%.4f src=%s feed=%s q_ts=%s t_ts=%s",
                    symbol,
                    price_f,
                    price_ref,
                    price_ref_src,
                    str(quote_feed or trade_feed or preferred_feed or ""),
                    str(quote_ts or ""),
                    str(trade_ts or ""),
                )

            if mode in {"prevclose", "blended"}:
                adjusted_ref, skip_due_band = self._apply_price_band(
                    symbol,
                    float(anchor_price or price_ref),
                    price_ref,
                    quote_snapshot,
                    trade_snapshot,
                )
                if skip_due_band:
                    continue
                if adjusted_ref is not None:
                    price_ref = adjusted_ref

            if mode in {"prevclose", "blended"}:
                anchor_val = max(0.0, float(anchor_price or 0.0))
                reference_val = price_ref if price_ref > 0 else anchor_val
                limit_cap = anchor_val * (1 + max_gap_ratio)
                ref_buffered = reference_val * (1 + ref_buffer_ratio)
                limit_px = min(limit_cap, ref_buffered)
                if limit_px > anchor_val * 1.5:
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=f"limit_gt_cap({limit_px:.2f}>{anchor_val * 1.5:.2f})",
                    )
                    continue
                if limit_px < 1.0:
                    self.record_skip_reason(
                        "PRICE_BOUNDS",
                        symbol=symbol,
                        detail=f"limit_lt_floor({limit_px:.2f})",
                    )
                    continue
                LOGGER.info(
                    "[INFO] LIMIT_SRC %s symbol=%s anchor=%.4f ref=%.4f limit=%.4f gap_pct=%.2f ref_pct=%.2f src=%s",
                    limit_source,
                    symbol,
                    anchor_val,
                    reference_val,
                    limit_px,
                    max_gap_ratio * 100,
                    ref_buffer_ratio * 100,
                    price_ref_src,
                )
            elif premarket_active:
                limit_px = price_ref * (1 + limit_buffer_ratio)
            else:
                limit_px = price_f * (1 + limit_buffer_ratio)
            # Guard against Alpaca 422 errors (Rule 612 minimum increments).
            limit_px = round_to_tick(limit_px)
            limit_px = max(limit_px, 0.01)
            effective_alloc_pct = allocation_pct
            if weight_mode:
                effective_alloc_pct = allocation_pct * weight_map.get(symbol, 0.0)
            base_notional = max(0.0, effective_alloc_pct * max(account_buying_power, 0.0))
            target_notional = base_notional
            if (self.config.position_sizer or "").lower() == "atr":
                target_pct = max(0.0, float(self.config.atr_target_pct))
                if target_pct > 0 and atr_pct > 0:
                    atr_scale = min(1.0, target_pct / atr_pct)
                    target_notional = base_notional * atr_scale
            target_notional = max(target_notional, min_order)
            qty = _compute_qty(
                account_buying_power,
                limit_px,
                effective_alloc_pct,
                min_order,
                target_notional=target_notional,
            )
            if qty < 1 and self.config.allow_bump_to_one and not self.config.allow_fractional:
                qty = 1 if limit_px > 0 and target_notional >= limit_px else qty
            if qty < 1:
                if not self.config.allow_fractional and target_notional >= min_order > 0:
                    _warn_context(
                        "sizing",
                        f"ZERO_QTY post min-notional symbol={symbol} limit={limit_px:.2f}",
                    )
                    if limit_px > 0 and min_order < limit_px:
                        LOGGER.info(
                            "[INFO] SIZING_HINT symbol=%s required_min_order_usd=%.2f current_min=%.2f",
                            symbol,
                            limit_px,
                            min_order,
                        )
                LOGGER.info(
                    "[INFO] ZERO_QTY_AFTER_BUMP symbol=%s limit=%.2f min_usd=%.2f",
                    symbol,
                    limit_px,
                    min_order,
                )
                self.record_skip_reason("ZERO_QTY", symbol=symbol)
                continue
            LOGGER.debug(
                "CALC symbol=%s price=%.2f bp=%.2f alloc_pct=%.4f slots=%d notional=%.2f limit_px=%.2f qty=%d sizer=%s atr=%.4f scale=%.3f",
                symbol,
                price_f,
                account_buying_power,
                effective_alloc_pct,
                slot_hint,
                target_notional,
                limit_px,
                qty,
                (self.config.position_sizer or "notional"),
                atr_pct,
                atr_scale,
            )

            if mode in {"prevclose", "blended"}:
                limit_price_raw = limit_px
            else:
                limit_price_raw = compute_limit_price(record, self.config.entry_buffer_bps)
            limit_price = max(limit_price_raw, limit_px)
            normalized_limit = normalize_price_for_alpaca(limit_price, "buy")
            rounded_limit = _round_limit_price(normalized_limit)
            notional = qty * rounded_limit
            if notional > account_buying_power:
                detail = f"required={notional:.2f} available={account_buying_power:.2f}"
                self.record_skip_reason("CASH", symbol=symbol, detail=detail)
                continue

            if diagnostic_mode:
                self.log_info(
                    "DIAGNOSTIC_ORDER",
                    symbol=symbol,
                    qty=qty,
                    limit_price=f"{rounded_limit:.4f}",
                    limit_raw=f"{limit_price:.8f}",
                )
                if not self.metrics.exit_reason or self.metrics.exit_reason == "UNKNOWN":
                    self.metrics.exit_reason = "DIAGNOSTIC"
                continue

            if self.config.dry_run:
                self.log_info(
                    "DRY_RUN_ORDER",
                    symbol=symbol,
                    qty=qty,
                    limit_price=f"{rounded_limit:.4f}",
                    limit_raw=f"{limit_price:.8f}",
                )
                if not self.metrics.exit_reason or self.metrics.exit_reason == "UNKNOWN":
                    self.metrics.exit_reason = "DRY_RUN"
                continue

            outcome = self.execute_order(
                symbol,
                qty,
                rounded_limit,
                raw_limit=limit_price,
                price_src=price_src,
                anchor_price=anchor_price,
                preferred_feed=preferred_feed,
            )
            if outcome.get("filled_qty", 0) > 0:
                existing_positions.add(symbol)
                account_buying_power = max(0.0, account_buying_power - notional)
            if outcome.get("submitted"):
                open_order_symbols.add(symbol)
                submitted_new += 1

        if not self.metrics.exit_reason or self.metrics.exit_reason == "UNKNOWN":
            if self.metrics.orders_submitted > 0:
                self.metrics.exit_reason = "EXECUTED"
            elif diagnostic_mode:
                self.metrics.exit_reason = "DIAGNOSTIC"
            else:
                self.metrics.exit_reason = "NO_ACTION"

        self.persist_metrics()
        self._log_diagnostic_snapshot()
        self.log_summary()
        return 0

    def fetch_existing_positions(self) -> set[str]:
        symbols: set[str] = set()
        if self.client is None:
            return symbols
        try:
            log_info("alpaca.get_positions")
            positions = self.client.get_all_positions()
            for pos in positions:
                symbol = getattr(pos, "symbol", "")
                qty = getattr(pos, "qty", None)
                try:
                    qty_val = float(qty)
                except Exception:
                    qty_val = None
                if qty_val is not None and qty_val == 0:
                    continue
                if symbol:
                    symbols.add(symbol.upper())
        except Exception as exc:
            _warn_context("alpaca.get_positions", str(exc))
        return symbols

    @staticmethod
    def _coerce_buying_power(value: Any) -> Optional[float]:
        if value in (None, "", False):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            try:
                numeric = float(str(value))
            except Exception:
                return None
        if math.isnan(numeric):
            return None
        return numeric

    def _get_orders(self, request: Any) -> Any:
        try:
            if request is None:
                return self.client.get_orders()
            return self.client.get_orders(filter=request)
        except TypeError:
            if request is None:
                return self.client.get_orders()
            return self.client.get_orders(filter=request)

    def fetch_open_order_symbols(self) -> tuple[set[str], int, int]:
        symbols: set[str] = set()
        open_buy_orders = 0
        total_orders = 0
        if self.client is None:
            return symbols, open_buy_orders, total_orders
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            log_info("alpaca.get_orders", status=getattr(request, "status", ""))
            orders = self._get_orders(request)
            for order in orders or []:
                total_orders += 1
                symbol = getattr(order, "symbol", "")
                if symbol:
                    symbols.add(symbol.upper())
                side = getattr(order, "side", "")
                if str(side).lower() == "buy":
                    open_buy_orders += 1
        except Exception as exc:
            _warn_context("alpaca.get_orders", str(exc))
        return symbols, open_buy_orders, total_orders

    def fetch_buying_power(self) -> float:
        def _extract(snapshot: Mapping[str, Any], source: str) -> Optional[float]:
            snapshot_map = dict(snapshot)
            for field in ("cash", "buying_power", "non_marginable_buying_power"):
                raw_value = snapshot_map.get(field)
                parsed = self._coerce_buying_power(raw_value)
                if parsed is None:
                    continue
                self._last_buying_power_raw = raw_value
                if field != "buying_power" and "buying_power" not in snapshot_map:
                    snapshot_map["buying_power"] = raw_value
                if field != "cash":
                    LOGGER.info(
                        "[INFO] BUYING_POWER_FALLBACK source=%s field=%s", source, field
                    )
                self.account_snapshot = snapshot_map
                return parsed
            return None

        if isinstance(self.account_snapshot, Mapping):
            parsed_snapshot = _extract(self.account_snapshot, "snapshot")
            if parsed_snapshot is not None:
                return parsed_snapshot
        if self.client is None:
            _warn_context("alpaca.buying_power", "client_unavailable")
            return 0.0
        try:
            log_info("alpaca.get_account")
            account = self.client.get_account()
        except Exception as exc:
            _warn_context("alpaca.get_account", str(exc))
            return 0.0
        snapshot: Dict[str, Any] = {}
        for field in ("cash", "buying_power", "non_marginable_buying_power"):
            buying_power = getattr(account, field, None)
            snapshot[field] = buying_power
        parsed = _extract(snapshot, "account")
        if parsed is not None:
            return parsed
        _warn_context("alpaca.buying_power", "Unable to fetch buying power")
        self.account_snapshot = snapshot
        return 0.0

    def _record_buy_execution(
        self,
        symbol: str,
        qty: float,
        avg_price: Any,
        order_id: str,
        status: str,
        filled_at: Any | None,
    ) -> None:
        try:
            price_value = float(avg_price)
        except (TypeError, ValueError):
            price_value = 0.0
        try:
            record_executed_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                price=price_value,
                status=status,
                order_id=order_id,
                order_type="limit",
                timestamp=filled_at or datetime.now(timezone.utc),
                event_type="BUY_FILL",
                raw={
                    "event_type": "BUY_FILL",
                    "filled_at": _isoformat_or_none(filled_at),
                    "status": status,
                },
            )
            LOGGER.info(
                "DB_WRITE_OK table=executed_trades event=BUY_FILL order_id=%s symbol=%s",
                order_id,
                symbol,
            )
        except Exception as exc:
            LOGGER.warning(
                "DB_WRITE_FAILED table=executed_trades event=BUY_FILL order_id=%s symbol=%s err=%s",
                order_id,
                symbol,
                exc,
            )
            LOGGER.exception("Failed to record executed buy trade for %s", symbol)

    def _record_trailing_submit(
        self,
        symbol: str,
        qty: float,
        avg_price: Any,
        order_id: str,
        status: str,
        submitted_at: Any | None,
        parent_order_id: str | None,
    ) -> None:
        try:
            price_value = float(avg_price)
        except (TypeError, ValueError):
            price_value = 0.0
        try:
            record_executed_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                price=price_value,
                status="submitted",
                order_id=order_id,
                order_type="trailing_stop",
                timestamp=submitted_at or datetime.now(timezone.utc),
                event_type="TRAIL_SUBMIT",
                raw={
                    "event_type": "TRAIL_SUBMIT",
                    "parent_order_id": parent_order_id,
                    "trail_percent": self.config.trailing_percent,
                    "status": status,
                    "submitted_at": _isoformat_or_none(submitted_at),
                },
            )
        except Exception:
            LOGGER.exception("Failed to record trailing stop submit for %s", symbol)

    def _parse_snapshot_time(self, timestamp: Any) -> Optional[datetime]:
        if timestamp in (None, ""):
            return None
        try:
            parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()

    def _fetch_reference_price(
        self, symbol: str, preferred_feed: str | None
    ) -> tuple[Optional[float], Optional[datetime], str]:
        trade_snapshot = _fetch_latest_trade_from_alpaca(symbol, feed=preferred_feed)
        trade_price: Optional[float] = None
        trade_feed = None
        if isinstance(trade_snapshot, Mapping):
            try:
                raw_trade = trade_snapshot.get("price")
                trade_price = float(raw_trade) if raw_trade is not None else None
            except (TypeError, ValueError):
                trade_price = None
            trade_feed = trade_snapshot.get("feed")

        quote_snapshot: Dict[str, Optional[float | str]] = {}
        if trade_price is None or math.isnan(trade_price) or trade_price <= 0:
            quote_snapshot = _fetch_latest_quote_from_alpaca(symbol, feed=preferred_feed)
            quote_feed = None
            ask_price: Optional[float] = None
            if isinstance(quote_snapshot, Mapping):
                quote_feed = quote_snapshot.get("feed")
                raw_ask = quote_snapshot.get("ask")
                try:
                    ask_price = float(raw_ask) if raw_ask is not None else None
                except (TypeError, ValueError):
                    ask_price = None
            else:
                quote_feed = None
            if ask_price is not None and not math.isnan(ask_price) and ask_price > 0:
                ts = quote_snapshot.get("timestamp") if isinstance(quote_snapshot, Mapping) else None
                return ask_price, self._parse_snapshot_time(ts), f"ask[{str(quote_feed or 'default').strip()}]"

        if trade_price is None or math.isnan(trade_price) or trade_price <= 0:
            return None, None, ""
        trade_ts = trade_snapshot.get("timestamp") if isinstance(trade_snapshot, Mapping) else None
        return trade_price, self._parse_snapshot_time(trade_ts), f"trade[{str(trade_feed or 'default').strip()}]"

    def _has_capacity_for_symbol(self, symbol: str) -> bool:
        try:
            max_positions = int(self.config.max_positions)
        except (TypeError, ValueError):
            return True
        if max_positions <= 0:
            return True
        positions = self.fetch_existing_positions()
        if symbol.upper() in positions:
            return True
        return len(positions) < max_positions

    def execute_order(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        *,
        raw_limit: Optional[float] = None,
        price_src: str = "",
        anchor_price: Optional[float] = None,
        preferred_feed: str | None = None,
    ) -> Dict[str, Any]:
        outcome: Dict[str, Any] = {"submitted": False, "filled_qty": 0.0}
        raw_limit_price = float(raw_limit if raw_limit is not None else limit_price)
        normalized_limit = normalize_price_for_alpaca(raw_limit_price, "buy")
        limit_price = _round_limit_price(normalized_limit)
        if getattr(self.config, "diagnostic", False):
            self.log_info(
                "DIAGNOSTIC_SKIP_SUBMIT",
                symbol=symbol,
                qty=str(qty),
                limit=f"{limit_price:.4f}",
                limit_raw=f"{raw_limit_price:.8f}",
                price_src=str(price_src or ""),
            )
            return outcome
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type="limit",
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=self.config.extended_hours,
        )
        # Alpaca rejects sub-penny increments (Rule 612); keep limit/stop/take-profit fields compliant.
        _enforce_order_price_ticks(order_request)
        setattr(order_request, "_normalized_limit", True)
        self.log_info(
            "BUY_INIT",
            symbol=symbol,
            qty=str(qty),
            limit=f"{limit_price:.4f}",
            limit_raw=f"{raw_limit_price:.8f}",
            price_src=str(price_src or ""),
        )
        submitted_order = self.submit_with_retries(order_request)
        if submitted_order is None:
            self.record_skip_reason("API_FAIL", symbol=symbol, detail="submit_failed")
            return outcome
        outcome["submitted"] = True
        order_id = str(getattr(submitted_order, "id", ""))
        self.metrics.orders_submitted += 1
        extended_hours_flag = "true" if self.config.extended_hours else "false"
        self.log_info(
            "BUY_SUBMIT",
            symbol=symbol,
            qty=str(qty),
            limit=f"{limit_price:.4f}",
            limit_raw=f"{raw_limit_price:.8f}",
            ext=extended_hours_flag,
            order_id=order_id or "",
            price_src=str(price_src or ""),
        )
        submitted_at = getattr(submitted_order, "submitted_at", None)
        if submitted_at is None:
            submitted_at = getattr(submitted_order, "created_at", None)
        request_payload = {
            "limit_price": limit_price,
            "extended_hours": bool(self.config.extended_hours),
            "time_in_force": str(getattr(order_request, "time_in_force", "day")),
            "qty": qty,
        }
        log_trade_event_db(
            event_type="BUY_SUBMIT",
            symbol=symbol,
            qty=qty,
            order_id=order_id,
            status="submitted",
            entry_price=limit_price,
            entry_time=submitted_at or datetime.now(timezone.utc),
            raw={
                "event_type": "BUY_SUBMIT",
                "submitted_at": _isoformat_or_none(submitted_at),
                "request": request_payload,
                "response": _order_snapshot(submitted_order),
            },
        )

        fill_deadline = datetime.now(timezone.utc) + timedelta(minutes=self.config.cancel_after_min)
        submit_ts = time.time()
        max_poll_seconds = max(0, int(getattr(self.config, "max_poll_secs", 0) or 0))
        poll_timeout_ts = submit_ts + max_poll_seconds if max_poll_seconds > 0 else None
        current_limit = limit_price

        filled_qty = 0.0
        filled_avg_price = None
        status = "open"
        last_order_snapshot: Any | None = None

        chase_enabled = bool(self.config.chase_enabled)
        chase_interval = max(1, int(self.config.chase_interval_minutes)) * 60
        max_chase_count = max(0, int(self.config.max_chase_count))
        chase_gap_ratio = max(0.0, float(self.config.max_chase_gap_pct)) / 100.0
        chase_buffer_ratio = max(0.0, float(self.config.ref_buffer_pct)) / 100.0
        chase_attempts = 0
        chase_halted = False
        next_chase_ts: Optional[float] = (
            submit_ts + chase_interval if chase_enabled else None
        )

        def _chase_order(current_id: str) -> tuple[str, float, float] | None:
            nonlocal chase_attempts, chase_halted
            if anchor_price is None or anchor_price <= 0:
                self.log_info("SKIP_CHASE", symbol=symbol, reason="no_anchor")
                chase_halted = True
                return None
            if not self._has_capacity_for_symbol(symbol):
                self.log_info("SKIP_CHASE", symbol=symbol, reason="max_positions")
                chase_halted = True
                return None
            ref_price, ref_ts, ref_src = self._fetch_reference_price(symbol, preferred_feed)
            if ref_price is None or ref_price <= 0:
                self.log_info("SKIP_CHASE", symbol=symbol, reason="no_reference")
                chase_halted = True
                return None
            if ref_ts is None:
                self.log_info("SKIP_CHASE", symbol=symbol, reason="stale_reference")
                chase_halted = True
                return None
            if datetime.now(timezone.utc) - ref_ts > timedelta(
                minutes=max(5, self.config.chase_interval_minutes * 2)
            ):
                self.log_info("SKIP_CHASE", symbol=symbol, reason="stale_reference")
                chase_halted = True
                return None
            candidate_limit = min(
                ref_price * (1 + chase_buffer_ratio),
                anchor_price * (1 + chase_gap_ratio),
            )
            normalized_candidate = normalize_price_for_alpaca(candidate_limit, "buy")
            candidate_limit = _round_limit_price(normalized_candidate)
            if candidate_limit <= current_limit:
                self.log_info("SKIP_CHASE", symbol=symbol, reason="no_improvement")
                return None

            self.cancel_order(current_id, symbol)
            self.metrics.orders_canceled += 1
            chase_attempts += 1
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                type="limit",
                limit_price=candidate_limit,
                time_in_force=TimeInForce.DAY,
                extended_hours=self.config.extended_hours,
            )
            _enforce_order_price_ticks(request)
            setattr(request, "_normalized_limit", True)
            chased_order = self.submit_with_retries(request)
            if chased_order is None:
                self.log_info("SKIP_CHASE", symbol=symbol, reason="submit_failed")
                return None
            chase_order_id = str(getattr(chased_order, "id", current_id))
            self.metrics.orders_submitted += 1
            self.log_info(
                "CHASE_SUBMIT",
                symbol=symbol,
                limit=f"{candidate_limit:.4f}",
                attempt=str(chase_attempts),
                source=ref_src or "",
                order_id=chase_order_id,
            )
            chased_submitted_at = getattr(chased_order, "submitted_at", None) or getattr(
                chased_order, "created_at", None
            )
            log_trade_event_db(
                event_type="BUY_SUBMIT",
                symbol=symbol,
                qty=qty,
                order_id=chase_order_id,
                status="submitted",
                entry_price=candidate_limit,
                entry_time=chased_submitted_at or datetime.now(timezone.utc),
                raw={
                    "event_type": "BUY_SUBMIT",
                    "submitted_at": _isoformat_or_none(chased_submitted_at),
                    "request": {
                        "limit_price": candidate_limit,
                        "extended_hours": bool(self.config.extended_hours),
                        "time_in_force": str(getattr(request, "time_in_force", "day")),
                        "qty": qty,
                        "chase_attempt": chase_attempts,
                    },
                    "response": _order_snapshot(chased_order),
                },
            )
            return chase_order_id, time.time(), candidate_limit

        while datetime.now(timezone.utc) < fill_deadline:
            try:
                log_info("alpaca.get_order", order_id=order_id)
                order = self.client.get_order_by_id(order_id)
            except Exception as exc:
                _warn_context("alpaca.get_order", f"{order_id}: {exc}")
                break
            now_ts = time.time()
            if poll_timeout_ts is not None and now_ts >= poll_timeout_ts:
                LOGGER.warning(
                    "[WARN] POLL_TIMEOUT order_id=%s waited_secs=%.1f",
                    order_id,
                    now_ts - submit_ts,
                )
                break
            last_order_snapshot = order
            status = str(getattr(order, "status", "")).lower()
            filled_qty = float(getattr(order, "filled_qty", filled_qty) or 0)
            filled_avg_price = getattr(order, "filled_avg_price", filled_avg_price)
            if status == "filled":
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_info(
                    "BUY_FILL",
                    symbol=symbol,
                    qty=f"{filled_qty:.0f}",
                    filled_qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                filled_at = getattr(order, "filled_at", None)
                self._record_buy_execution(
                    symbol,
                    filled_qty,
                    filled_avg_price,
                    order_id,
                    status,
                    filled_at,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price, order_id)
                outcome["filled_qty"] = filled_qty
                return outcome
            if status in {"canceled", "expired", "rejected"}:
                break
            if filled_qty >= qty:
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_info(
                    "BUY_FILL",
                    symbol=symbol,
                    qty=f"{filled_qty:.0f}",
                    filled_qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                filled_at = getattr(order, "filled_at", None)
                status_label = status or "filled"
                self._record_buy_execution(
                    symbol,
                    filled_qty,
                    filled_avg_price,
                    order_id,
                    status_label,
                    filled_at,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price, order_id)
                outcome["filled_qty"] = filled_qty
                return outcome
            now_ts = time.time()
            if (
                chase_enabled
                and next_chase_ts is not None
                and now_ts >= next_chase_ts
                and chase_attempts < max_chase_count
            ):
                chased = _chase_order(order_id)
                if chased is not None:
                    order_id, submit_ts, current_limit = chased
                    next_chase_ts = submit_ts + chase_interval
                    continue
                next_chase_ts = None if chase_halted else now_ts + chase_interval
            self.sleep(5)

        # Deadline reached or not fully filled
        remaining = max(qty - filled_qty, 0)
        if remaining > 0:
            self.cancel_order(order_id, symbol)
            self.metrics.orders_canceled += 1
            self.log_info(
                "BUY_CANCELLED",
                symbol=symbol,
                remaining_qty=f"{remaining:.0f}",
                status=status,
                order_id=order_id,
            )
        if filled_qty > 0:
            latency = time.time() - submit_ts
            self.metrics.orders_filled += 1
            self.metrics.record_latency(latency)
            self.log_info(
                "BUY_FILL",
                symbol=symbol,
                qty=f"{filled_qty:.0f}",
                filled_qty=f"{filled_qty:.0f}",
                avg_price=f"{float(filled_avg_price or 0):.2f}",
                partial="true",
                order_id=order_id,
            )
            status_label = status or "partially_filled"
            filled_at = getattr(last_order_snapshot, "filled_at", None)
            self._record_buy_execution(
                symbol,
                filled_qty,
                filled_avg_price,
                order_id,
                status_label,
                filled_at,
            )
            self.attach_trailing_stop(symbol, filled_qty, filled_avg_price, order_id)
            outcome["filled_qty"] = filled_qty
        return outcome

    def cancel_order(self, order_id: str, symbol: str) -> None:
        if getattr(self.config, "diagnostic", False):
            self.log_info("DIAGNOSTIC_CANCEL_SKIP", order_id=order_id or "", symbol=symbol)
            return
        if self.client is None or not order_id:
            return
        try:
            if hasattr(self.client, "cancel_order_by_id"):
                log_info("alpaca.cancel_order", order_id=order_id)
                self.client.cancel_order_by_id(order_id)
            elif hasattr(self.client, "cancel_order"):
                log_info("alpaca.cancel_order", order_id=order_id)
                self.client.cancel_order(order_id)
            else:  # pragma: no cover - defensive fallback
                LOGGER.warning("Client has no cancel method; unable to cancel %s", order_id)
        except Exception as exc:
            _warn_context("alpaca.cancel_order", f"{order_id}: {exc}")

    def attach_trailing_stop(
        self, symbol: str, qty: float, avg_price: Optional[Any], parent_order_id: str | None = None
    ) -> None:
        if getattr(self.config, "diagnostic", False):
            self.log_info("DIAGNOSTIC_TRAIL_SKIP", symbol=symbol, qty=str(qty))
            return
        if qty <= 0 or self.client is None:
            return
        qty_int = int(qty) if int(qty) == qty else math.floor(qty)
        if qty_int <= 0:
            return
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty_int,
            side=OrderSide.SELL,
            trail_percent=self.config.trailing_percent,
            time_in_force=TimeInForce.GTC,
        )
        trail_value = self.config.trailing_percent
        if float(trail_value).is_integer():
            trail_display = str(int(trail_value))
        else:
            trail_display = f"{trail_value:g}"
        LOGGER.info(
            "TRAIL_SUBMIT symbol=%s trail_pct=%s route=trailing_stop",
            symbol,
            trail_display,
        )
        self.log_info(
            "TRAIL_SUBMIT",
            symbol=symbol,
            qty=str(qty_int),
            trail_pct=trail_display,
            route="trailing_stop",
        )
        trailing_order = self.submit_with_retries(request)
        if trailing_order is None:
            LOGGER.error("[ERROR] TRAIL_FAILED symbol=%s reason=submit_failed", symbol)
            self.log_info("TRAIL_FAILED", symbol=symbol, reason="submit_failed")
            self.record_skip_reason("API_FAIL", symbol=symbol, detail="trail_submit_failed")
            return
        self.metrics.trailing_attached += 1
        order_id = str(getattr(trailing_order, "id", ""))
        status = str(getattr(trailing_order, "status", ""))
        submitted_at = getattr(trailing_order, "submitted_at", None)
        if submitted_at is None:
            submitted_at = getattr(trailing_order, "created_at", None)
        LOGGER.info(
            "TRAIL_CONFIRMED symbol=%s qty=%s order_id=%s",
            symbol,
            qty_int,
            order_id,
        )
        self.log_info(
            "TRAIL_CONFIRMED",
            symbol=symbol,
            qty=str(qty_int),
            order_id=order_id,
        )
        self._record_trailing_submit(
            symbol,
            float(qty_int),
            avg_price,
            order_id,
            status,
            submitted_at,
            parent_order_id,
        )

    def _submit_order(self, request: Any) -> Any:
        try:
            return self.client.submit_order(order_data=request)
        except TypeError:
            return self.client.submit_order(order_data=request)

    def submit_with_retries(self, request: Any) -> Optional[Any]:
        if getattr(self.config, "diagnostic", False):
            request_type = getattr(request, "__class__", type("")).__name__
            self.log_info("DIAGNOSTIC_SUBMIT_SKIP", request_type=request_type)
            return None
        if self.client is None:
            return None
        attempts = 3
        backoff = 1.5
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                log_info("alpaca.submit_order", attempt=attempt)
                _enforce_order_price_ticks(request)
                result = self._submit_order(request)
                if attempt > 1:
                    self.log_info("API_RETRY_SUCCESS", attempt=attempt)
                return result
            except APIError as exc:
                last_error = exc
                error_message = str(exc)
                code = getattr(exc, "code", None)
                side_value = str(getattr(request, "side", "")).lower()
                if (
                    attempt < attempts
                    and code == 42210000
                    and "sub-penny increment" in error_message.lower()
                    and getattr(request, "_normalized_limit", False)
                    and not getattr(request, "_retry_tick_applied", False)
                    and hasattr(request, "limit_price")
                ):
                    current_limit = float(getattr(request, "limit_price") or 0.0)
                    step = float(_mpv_step(current_limit or 0.0))
                    step = step if step > 0 else 0.01
                    canonical_side = "buy" if side_value == "buy" else "sell"
                    if canonical_side == "buy":
                        adjusted_seed = max(current_limit - step, step)
                    else:
                        adjusted_seed = current_limit + step
                    new_limit = normalize_price_for_alpaca(adjusted_seed, canonical_side)
                    new_limit = _round_limit_price(new_limit)
                    request.limit_price = new_limit
                    setattr(request, "_retry_tick_applied", True)
                    LOGGER.warning("[RETRY_TICK] side=%s new_limit=%.4f", canonical_side, new_limit)
                    self.log_info(
                        "RETRY_TICK",
                        side=canonical_side,
                        new_limit=f"{new_limit:.4f}",
                    )
                    self.metrics.api_retries += 1
                    continue
                if attempt < attempts:
                    self.metrics.api_retries += 1
                    self.log_info("API_RETRY", attempt=attempt, error=error_message)
                    self.sleep(delay)
                    delay *= backoff
                    continue
                self.metrics.api_failures += 1
                _warn_context("alpaca.submit_order failed", str(exc))
            except Exception as exc:
                last_error = exc
                if isinstance(exc, AttributeError):
                    self.metrics.api_failures += 1
                    _warn_context("alpaca.submit_order failed", str(exc))
                    break
                if attempt < attempts:
                    self.metrics.api_retries += 1
                    self.log_info("API_RETRY", attempt=attempt, error=str(exc))
                    self.sleep(delay)
                    delay *= backoff
                    continue
                self.metrics.api_failures += 1
                _warn_context("alpaca.submit_order failed", str(exc))
        if last_error is not None:
            self.log_info("API_FAILURE", error=str(last_error))
        return None

    def log_summary(self) -> None:
        skips = {key.upper(): int(value) for key, value in self.metrics.skipped_reasons.items()}
        payload: Dict[str, Any] = {
            "orders_submitted": self.metrics.orders_submitted,
            "orders_filled": self.metrics.orders_filled,
            "trailing_attached": self.metrics.trailing_attached,
        }
        for key in SKIP_REASON_ORDER:
            payload[f"skips.{key}"] = int(skips.get(key, 0))
        extra_keys = sorted(key for key in skips.keys() if key not in SKIP_REASON_KEYS)
        for key in extra_keys:
            payload[f"skips.{key}"] = int(skips.get(key, 0))
        log_info("EXECUTE_SUMMARY", **payload)
        summary_skips = {key: int(value) for key, value in sorted(skips.items())}
        LOGGER.info(
            "EXECUTE_SUMMARY orders_submitted=%s orders_filled=%s skips=%s",
            self.metrics.orders_submitted,
            self.metrics.orders_filled,
            summary_skips,
        )

    def persist_metrics(self) -> None:
        payload = self.metrics.as_dict()
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if METRICS_PATH.exists():
            try:
                existing_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive merge
                _warn_context("metrics.persist", f"decode_failed: {exc}")
                existing_payload = {}
            if isinstance(existing_payload, Mapping):
                existing = dict(existing_payload)
        merged = dict(existing)
        merged.update(payload)
        existing_skips = existing.get("skips") if isinstance(existing.get("skips"), Mapping) else {}
        payload_skips = payload.get("skips") if isinstance(payload.get("skips"), Mapping) else {}
        skips_merged = dict(existing_skips)
        skips_merged.update(payload_skips)
        merged["skips"] = skips_merged
        merged["status"] = payload.get("status", merged.get("status", "ok"))
        if payload.get("error"):
            merged["error"] = payload["error"]
        else:
            merged.pop("error", None)
        merged["last_run_utc"] = payload.get("last_run_utc", datetime.now(timezone.utc).isoformat())
        try:
            write_execute_metrics(
                merged,
                start_dt=_EXECUTE_START_UTC,
                end_dt=datetime.now(timezone.utc),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            _warn_context("metrics.persist", f"write_failed: {exc}")
            return


def configure_logging(log_json: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stream_handler)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handlers.append(file_handler)

    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
    for handler in handlers:
        if log_json:
            handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def get_trading_client():
    """
    Return an Alpaca TradingClient in paper mode.
    Never hard-fail at import time; callers must handle None.
    """
    try:
        from alpaca.trading.client import TradingClient as AlpacaTradingClient
    except Exception as e:
        LOGGER.warning("TradingClient import failed: %s", e)
        return None

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "")

    if not key or not secret:
        LOGGER.warning("TradingClient unavailable: missing APCA_API_KEY_ID/APCA_API_SECRET_KEY")
        return None

    # We remain paper-mode only. If base_url contains paper, use paper=True.
    paper = "paper" in (base_url or "").lower()

    try:
        return AlpacaTradingClient(key, secret, paper=paper)
    except Exception as e:
        LOGGER.warning("TradingClient init failed: %s", e)
        return None


def _create_trading_client() -> tuple[Any | None, str, bool, bool | None]:
    _, _, base_url, _ = get_alpaca_creds()
    forced_raw = os.getenv("JBR_EXEC_PAPER")
    forced_mode: bool | None = None
    paper_mode: bool
    if forced_raw is not None:
        forced_mode = forced_raw.strip().lower() in {"1", "true", "yes", "on"}
        paper_mode = forced_mode
    else:
        env = (base_url or "paper").lower()
        paper_mode = "live" not in env
    resolved_base = (base_url or "https://paper-api.alpaca.markets").rstrip("/")
    if not resolved_base:
        resolved_base = "https://paper-api.alpaca.markets"
    client = get_trading_client()
    return client, resolved_base, paper_mode, forced_mode


def _ensure_trading_auth(
    base_url: str, creds_snapshot: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    api_key, api_secret, _, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        body = "missing credentials"
        LOGGER.error("[ERROR] ALPACA_AUTH_FAILED status=MISSING body=%s", body)
        _record_auth_error("unauthorized", creds_snapshot)
        raise AlpacaAuthFailure("missing credentials")

    resolved_base = (base_url or "https://paper-api.alpaca.markets").rstrip("/")
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    def fetch(path: str) -> Mapping[str, Any]:
        url = f"{resolved_base}{path}"
        context = f"alpaca.auth{path}"
        log_info(context)
        try:
            response = requests.get(url, headers=headers, timeout=10)
        except Exception as exc:
            body_text = str(exc)
            _warn_context(context, body_text)
            LOGGER.error(
                "[ERROR] ALPACA_AUTH_FAILED status=ERR body=%s endpoint=%s",
                body_text,
                path,
            )
            _record_auth_error("unauthorized", creds_snapshot)
            raise AlpacaAuthFailure(body_text) from exc
        if response.status_code != 200:
            raw_text = getattr(response, "text", "")
            body_text = (raw_text or "")[:500].replace("\n", " ")
            _warn_context(context, f"status={response.status_code}")
            LOGGER.error(
                "[ERROR] ALPACA_AUTH_FAILED status=%s body=%s endpoint=%s",
                response.status_code,
                body_text,
                path,
            )
            _record_auth_error("unauthorized", creds_snapshot)
            raise AlpacaAuthFailure(f"status={response.status_code} body={body_text}")
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return payload if isinstance(payload, Mapping) else {}

    account_payload = fetch("/v2/account")
    clock_payload = fetch("/v2/clock")
    return account_payload, clock_payload


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades for pipeline candidates")
    parser.add_argument(
        "--source",
        choices=("db", "csv"),
        default=ExecutorConfig.source_type,
        help="Candidate source: db or csv (default csv)",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=ExecutorConfig.source_path,
        help="Path to the candidate CSV file (used when --source=csv)",
    )
    parser.add_argument(
        "--allocation-pct",
        type=float,
        default=ExecutorConfig.allocation_pct,
        help="Fraction of buying power allocated per position (0-1)",
    )
    parser.add_argument(
        "--min-order-usd",
        type=float,
        default=ExecutorConfig.min_order_usd,
        help="Minimum USD notional target per slot before rounding",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=ExecutorConfig.max_positions,
        help="Maximum concurrent positions the executor will open",
    )
    parser.add_argument(
        "--max-new-positions",
        type=int,
        default=ExecutorConfig.max_new_positions,
        help="Maximum new positions the executor will attempt to open per run",
    )
    parser.add_argument(
        "--entry-buffer-bps",
        type=int,
        default=ExecutorConfig.entry_buffer_bps,
        help="Entry buffer in basis points added to the reference price",
    )
    parser.add_argument(
        "--limit-buffer-pct",
        type=float,
        default=ExecutorConfig.limit_buffer_pct,
        help="Percent tolerance added to entry/anchor prices (default 0.5)",
    )
    parser.add_argument(
        "--max-gap-pct",
        type=float,
        default=ExecutorConfig.max_gap_pct,
        help=(
            "Max percent above previous close allowed for prevclose anchoring (env: MAX_GAP_PCT)"
        ),
    )
    parser.add_argument(
        "--ref-buffer-pct",
        type=float,
        default=ExecutorConfig.ref_buffer_pct,
        help=(
            "Percent buffer added to live reference prices (env: REF_BUFFER_PCT, default 0.5)"
        ),
    )
    parser.add_argument(
        "--trailing-percent",
        type=float,
        default=ExecutorConfig.trailing_percent,
        help="Percent trail for the protective stop order",
    )
    parser.add_argument(
        "--cancel-after-min",
        type=int,
        default=ExecutorConfig.cancel_after_min,
        help="Minutes after regular market open to cancel unfilled orders",
    )
    parser.add_argument(
        "--max-poll-secs",
        type=int,
        default=ExecutorConfig.max_poll_secs,
        help="Maximum seconds to poll an order before emitting a timeout warning and moving on",
    )
    parser.add_argument(
        "--extended-hours",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.extended_hours,
        help="Whether to submit orders eligible for extended hours",
    )
    parser.add_argument(
        "--submit-at-ny",
        type=str,
        default=ExecutorConfig.submit_at_ny,
        help="Target HH:MM in America/New_York to start submitting (premarket)",
    )
    parser.add_argument(
        "--price-source",
        choices=("prevclose", "entry", "close", "blended"),
        default=ExecutorConfig.price_source,
        help=(
            "Anchor price source for limit orders (prevclose falls back to snapshot/bars/entry, blended uses prevclose and live reference prices)"
        ),
    )
    parser.add_argument(
        "--price-band-pct",
        type=float,
        default=ExecutorConfig.price_band_pct,
        help="Safety band percent for prevclose anchoring (default 10.0)",
    )
    parser.add_argument(
        "--price-band-action",
        choices=("clamp", "skip"),
        default=ExecutorConfig.price_band_action,
        help="Action when live prices deviate outside the band",
    )
    parser.add_argument(
        "--chase-interval-minutes",
        type=int,
        default=ExecutorConfig.chase_interval_minutes,
        help="Minutes to wait between chase attempts",
    )
    parser.add_argument(
        "--max-chase-count",
        type=int,
        default=ExecutorConfig.max_chase_count,
        help="Maximum number of chase attempts to reprice open orders",
    )
    parser.add_argument(
        "--max-chase-gap-pct",
        type=float,
        default=ExecutorConfig.max_chase_gap_pct,
        help="Max percent above anchor/prevclose allowed during chase repricing",
    )
    parser.add_argument(
        "--chase-enabled",
        action="store_true",
        default=ExecutorConfig.chase_enabled,
        help="Enable periodic chasing of unfilled limit orders",
    )
    parser.add_argument(
        "--dry-run",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.dry_run,
        help="If true, only log intended actions without submitting orders",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=ExecutorConfig.diagnostic,
        help="Diagnostic mode: run allocation logic and emit metrics without submitting or canceling orders",
    )
    parser.add_argument(
        "--allow-bump-to-one",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.allow_bump_to_one,
        help="Allow bumping position size to one share when sizing rounds to zero",
    )
    parser.add_argument(
        "--allow-fractional",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.allow_fractional,
        help="Allow fractional share orders when full shares are not affordable",
    )
    parser.add_argument(
        "--position-sizer",
        choices=("notional", "atr"),
        default=ExecutorConfig.position_sizer,
        help="Strategy used to determine order size (default: notional)",
    )
    parser.add_argument(
        "--atr-target-pct",
        type=float,
        default=ExecutorConfig.atr_target_pct,
        help="Target ATR%% used when position_sizer=atr (expressed as decimal, e.g. 0.02)",
    )
    parser.add_argument(
        "--min-adv20",
        type=int,
        default=ExecutorConfig.min_adv20,
        help="Minimum 20-day average dollar volume required",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=ExecutorConfig.min_price,
        help="Minimum allowed price for candidates",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=ExecutorConfig.max_price,
        help="Maximum allowed price for candidates",
    )
    parser.add_argument(
        "--log-json",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.log_json,
        help="Emit structured JSON logs in addition to human readable ones",
    )
    parser.add_argument(
        "--reconcile-only",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.reconcile_only,
        help="Run reconciliation of closed trades and exit without placing new orders",
    )
    parser.add_argument(
        "--reconcile-auto",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.reconcile_auto,
        help="Automatically reconcile closed trades at start and end of executor run (default: true)",
    )
    parser.add_argument(
        "--reconcile-lookback-days",
        type=int,
        default=ExecutorConfig.reconcile_lookback_days,
        help="Lookback window in days for reconciliation queries (default 14)",
    )
    parser.add_argument(
        "--reconcile-limit",
        type=int,
        default=ExecutorConfig.reconcile_limit,
        help="Maximum number of orders fetched during reconciliation (default 500)",
    )
    parser.add_argument(
        "--reconcile-use-watermark",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.reconcile_use_watermark,
        help="Use reconcile watermark stored in DB to fetch incremental orders (default: true)",
    )
    parser.add_argument(
        "--reconcile-overlap-secs",
        type=int,
        default=ExecutorConfig.reconcile_overlap_secs,
        help="Seconds of overlap when advancing reconcile watermark (default 300)",
    )
    parser.add_argument(
        "--time-window",
        choices=("premarket", "regular", "any", "auto"),
        default=ExecutorConfig.time_window,
        help="Trading time window gate controlling when orders may be submitted",
    )
    parser.add_argument(
        "--market-tz",
        "--market-timezone",
        dest="market_timezone",
        default=ExecutorConfig.market_timezone,
        help="IANA timezone name used for market window evaluation",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    market_timezone = args.market_timezone or ExecutorConfig.market_timezone
    return ExecutorConfig(
        source_path=args.source_path,
        source_type=(args.source or "csv").lower(),
        allocation_pct=args.allocation_pct,
        max_positions=args.max_positions,
        max_new_positions=args.max_new_positions,
        entry_buffer_bps=args.entry_buffer_bps,
        trailing_percent=args.trailing_percent,
        limit_buffer_pct=args.limit_buffer_pct,
        max_gap_pct=args.max_gap_pct,
        ref_buffer_pct=args.ref_buffer_pct,
        cancel_after_min=args.cancel_after_min,
        max_poll_secs=args.max_poll_secs,
        extended_hours=args.extended_hours,
        dry_run=args.dry_run,
        min_order_usd=args.min_order_usd,
        allow_bump_to_one=args.allow_bump_to_one,
        allow_fractional=args.allow_fractional,
        position_sizer=(args.position_sizer or "notional").lower(),
        atr_target_pct=args.atr_target_pct,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        log_json=args.log_json,
        time_window=args.time_window,
        market_timezone=market_timezone,
        submit_at_ny=args.submit_at_ny,
        price_source=args.price_source,
        price_band_pct=args.price_band_pct,
        price_band_action=args.price_band_action,
        chase_interval_minutes=args.chase_interval_minutes,
        max_chase_count=args.max_chase_count,
        max_chase_gap_pct=args.max_chase_gap_pct,
        chase_enabled=args.chase_enabled,
        diagnostic=bool(getattr(args, "diagnostic", False)),
        reconcile_auto=bool(getattr(args, "reconcile_auto", True)),
        reconcile_only=bool(getattr(args, "reconcile_only", False)),
        reconcile_lookback_days=int(
            getattr(args, "reconcile_lookback_days", ExecutorConfig.reconcile_lookback_days)
        ),
        reconcile_limit=int(getattr(args, "reconcile_limit", ExecutorConfig.reconcile_limit)),
        reconcile_use_watermark=bool(getattr(args, "reconcile_use_watermark", ExecutorConfig.reconcile_use_watermark)),
        reconcile_overlap_secs=int(
            getattr(args, "reconcile_overlap_secs", ExecutorConfig.reconcile_overlap_secs)
        ),
    )


def _seed_initial_metrics(metrics: ExecutionMetrics, config: ExecutorConfig) -> None:
    try:
        max_positions = int(config.max_positions)
    except (TypeError, ValueError):
        max_positions = 0
    max_positions = max(1, max_positions)
    metrics.max_total_positions = max_positions
    metrics.configured_max_positions = max_positions
    metrics.risk_limited_max_positions = max_positions
    metrics.open_positions = 0
    metrics.open_orders = 0
    metrics.allowed_new_positions = 0
    metrics.slots_total = 0
    metrics.exit_reason = None
    metrics.symbols_in = 0
    try:
        write_execute_metrics(
            metrics.as_dict(),
            start_dt=_EXECUTE_START_UTC,
            end_dt=_EXECUTE_START_UTC,
        )
    except Exception:
        LOGGER.debug("INITIAL_METRICS_WRITE_FAILED", exc_info=True)


_RECONCILE_DB_DISABLED_LOGGED = False


def _run_auto_reconcile(executor: TradeExecutor, config: ExecutorConfig, stage: str) -> None:
    global _RECONCILE_DB_DISABLED_LOGGED
    if not getattr(config, "reconcile_auto", True):
        return
    if not db.db_enabled():
        if not _RECONCILE_DB_DISABLED_LOGGED:
            LOGGER.info("[INFO] RECONCILE_AUTO_SKIP reason=db_disabled")
            _RECONCILE_DB_DISABLED_LOGGED = True
        return
    lookback_days = getattr(config, "reconcile_lookback_days", 7)
    limit = getattr(config, "reconcile_limit", 500)
    LOGGER.info(
        "[INFO] RECONCILE_AUTO_START stage=%s lookback_days=%s limit=%s",
        stage,
        lookback_days,
        limit,
    )
    try:
        executor.reconcile_closed_trades(lookback_days=lookback_days, limit=limit)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("[WARN] RECONCILE_AUTO_FAIL stage=%s err=%s", stage, exc)
    finally:
        LOGGER.info("[INFO] RECONCILE_AUTO_END stage=%s", stage)


def run_executor(
    config: ExecutorConfig,
    *,
    client: Optional[Any] = None,
    creds_snapshot: Mapping[str, Any] | None = None,
) -> int:
    global _EXECUTE_START_UTC
    if _EXECUTE_START_UTC is None:
        _EXECUTE_START_UTC = datetime.now(timezone.utc)
    configure_logging(config.log_json)
    metrics = ExecutionMetrics()
    _seed_initial_metrics(metrics, config)
    diagnostic_mode = bool(getattr(config, "diagnostic", False))
    loader = TradeExecutor(config, client, metrics)
    auto_reconcile_enabled = bool(getattr(config, "reconcile_auto", True)) and not getattr(config, "reconcile_only", False)
    if auto_reconcile_enabled:
        _run_auto_reconcile(loader, config, stage="start")
    try:
        if config.dry_run:
            banner = "=" * 72
            LOGGER.info(banner)
            LOGGER.info(
                "[INFO] DRY_RUN=True — no orders will be submitted, but still perform sizing and emit skip reasons"
            )
            LOGGER.info(banner)
        try:
            LOGGER.info("[INFO] CANDIDATE_SOURCE %s", str(config.source_type or "csv"))
            frame = loader.load_candidates(rank=False)
        except CandidateLoadError as exc:
            LOGGER.error("%s", exc)
            metrics.api_failures += 1
            metrics.record_skip("DATA_MISSING", count=1)
            metrics.record_error(
                "candidate_load_error",
                detail=str(exc),
                exception=exc.__class__.__name__,
            )
            loader.persist_metrics()
            return 1
        if str(config.source_type or "").lower() == "db":
            max_ts = ""
            if not frame.empty and "timestamp" in frame.columns:
                ts_series = frame["timestamp"].dropna().astype("string")
                if not ts_series.empty:
                    max_ts = ts_series.max()
            LOGGER.info("[INFO] DB_CANDIDATES_LOADED count=%s max_timestamp=%s", len(frame), max_ts)

        candidates_df = loader._rank_candidates(frame)

        try:
            base_alloc_pct = float(config.allocation_pct)
        except (TypeError, ValueError):
            base_alloc_pct = 0.0
        base_alloc_pct = max(0.0, base_alloc_pct)
        try:
            configured_max_positions = int(config.max_positions)
        except (TypeError, ValueError):
            configured_max_positions = 0
        configured_max_positions = max(1, configured_max_positions)
        metrics.max_total_positions = configured_max_positions
        metrics.configured_max_positions = configured_max_positions
        metrics.risk_limited_max_positions = configured_max_positions

        loader._log_top_candidates(
            candidates_df.to_dict(orient="records"),
            max_positions=configured_max_positions,
            base_alloc_pct=base_alloc_pct,
            source_df=candidates_df,
        )

        _wait_until_submit_at(config.submit_at_ny)
        configured_window = config.time_window or "auto"
        win, in_window, now_ny = _resolve_time_window(configured_window)
        trading_probe = client if client is not None else None
        clock_hhmm = now_ny.strftime("%H:%M")
        clock_window = win if win in {"premarket", "regular", "any"} else configured_window
        if trading_probe is not None and hasattr(trading_probe, "get_clock"):
            clock_allowed, hhmm = _clock_is_in_window(trading_probe, clock_window)
            if hhmm != "00:00":
                clock_hhmm = hhmm
            if clock_window == "any":
                in_window = clock_allowed
            elif clock_window in {"premarket", "regular"}:
                in_window = bool(in_window) and clock_allowed
            else:
                in_window = bool(in_window) and clock_allowed
            if hhmm == "00:00":
                log_info(
                    "MARKET_TIME",
                    ny_now=now_ny.isoformat(),
                    window=clock_window,
                    in_window=bool(in_window),
                )
        else:
            log_info(
                "MARKET_TIME",
                ny_now=now_ny.isoformat(),
                window=clock_window,
                in_window=bool(in_window),
            )

        log_info(
            "EXEC_START",
            dry_run=bool(config.dry_run),
            diagnostic=bool(config.diagnostic),
            time_window=configured_window,
            resolved=win,
            ny_now=now_ny.isoformat(),
            hhmm=clock_hhmm,
            in_window=bool(in_window),
            candidates=len(candidates_df),
            submit_at_ny=str(config.submit_at_ny or ""),
        )

        reconcile_exit_mode = bool(
            getattr(config, "reconcile_only", False)
            or (
                getattr(config, "reconcile_auto", True)
                and str(configured_window).lower() != "any"
            )
        )

        account_payload: Optional[Mapping[str, Any]] = None
        clock_payload: Optional[Mapping[str, Any]] = None
        skip_trading_client = False
        if client is not None:
            trading_client = client
            base_url = os.getenv("APCA_API_BASE_URL", "")
            paper_mode = None
        else:
            skip_trading_client = config.dry_run and not diagnostic_mode and not reconcile_exit_mode
            skip_trading_client = skip_trading_client or (
                not reconcile_exit_mode and not diagnostic_mode and candidates_df.empty
            )
            if skip_trading_client:
                trading_client = None
                base_url = ""
                paper_mode = None
            else:
                trading_client, base_url, paper_mode, forced_mode = _create_trading_client()
                forced_label = forced_mode if forced_mode is not None else "auto"
                LOGGER.info(
                    "[INFO] TRADING_MODE base=%s paper_mode=%s forced=%s",
                    base_url or "",
                    bool(paper_mode),
                    forced_label,
                )
        _paper_only_guard(trading_client, base_url)
        auth_ok = trading_client is None and skip_trading_client
        auth_reason = "ok"
        if trading_client is not None and client is None:
            try:
                account_payload, clock_payload = _ensure_trading_auth(base_url or "", creds_snapshot or {})
                auth_ok = True
            except AlpacaAuthFailure as exc:
                auth_ok = False
                auth_reason = str(exc) or "auth_failed"
        elif trading_client is not None:
            auth_ok = True
        metrics.auth_ok = bool(auth_ok)
        metrics.auth_reason = None if auth_ok else f"auth_failed:{auth_reason}"
        LOGGER.info("[INFO] AUTH_RESULT ok=%s reason=%s", bool(auth_ok), auth_reason)
        if not auth_ok:
            LOGGER.warning("[WARN] AUTH_FAIL detail=%s", auth_reason)
            metrics.exit_reason = "AUTH_FAIL"
            metrics.api_failures += 1
            metrics.record_skip("API_FAIL", count=max(1, len(candidates_df)))
            metrics.record_error(
                "auth_failure",
                stage="ensure_trading_auth",
                base_url=base_url or "",
                detail=auth_reason,
            )
            loader.persist_metrics()
            loader.log_summary()
            return 2

        executor = TradeExecutor(
            config,
            trading_client,
            metrics,
            base_url=base_url or "",
            account_snapshot=account_payload,
            clock_snapshot=clock_payload,
        )
        executor._ranking_key = loader._ranking_key

        if reconcile_exit_mode:
            metrics.exit_reason = "RECONCILE_ONLY"
            executor.reconcile_closed_trades()
            executor.persist_metrics()
            executor.log_summary()
            return 0

        metrics.in_window = bool(in_window)
        if not in_window:
            start_str, end_str = _premarket_bounds_strings()
            if not diagnostic_mode:
                loader.record_skip_reason(
                    "TIME_WINDOW",
                    count=len(candidates_df) if len(candidates_df) > 0 else 1,
                    detail=f"{start_str}-{end_str}",
                )
            metrics.exit_reason = "TIME_WINDOW"
            if not diagnostic_mode:
                metrics.flush()
                loader.persist_metrics()
                LOGGER.info(
                    "EXECUTE_SKIP reason=TIME_WINDOW count=%s",
                    len(candidates_df),
                )
                loader.log_summary()
                return 0

        if candidates_df.empty:
            if str(config.source_type).lower() == "db":
                LOGGER.info("SKIP NO_CANDIDATES source=db")
            if metrics.skipped_reasons.get("NO_CANDIDATES", 0) <= 0:
                loader.record_skip_reason("NO_CANDIDATES", count=1)
            metrics.exit_reason = "NO_CANDIDATES"
            loader.persist_metrics()
            LOGGER.info("EXECUTE_SKIP reason=NO_CANDIDATES")
            loader.log_summary()
            return 0

        records = executor.hydrate_candidates(candidates_df)
        filtered = executor.guard_candidates(records)
        return executor.execute(candidates_df, prefiltered=filtered)
    finally:
        if auto_reconcile_enabled:
            _run_auto_reconcile(loader, config, stage="end")


def load_candidates(path: Path) -> pd.DataFrame:
    config = ExecutorConfig(source_path=path, source_type="csv")
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    return executor.load_candidates()


def apply_guards(df: pd.DataFrame, config: ExecutorConfig, metrics: ExecutionMetrics) -> pd.DataFrame:
    executor = TradeExecutor(config, None, metrics)
    records = executor.hydrate_candidates(df)
    filtered = executor.guard_candidates(records)
    if not filtered:
        return df.iloc[0:0].copy()
    return pd.DataFrame(filtered)


def _bootstrap_env() -> list[str]:
    global _ENV_FILES_LOADED
    loaded_files, missing = load_env(REQUIRED_ENV_KEYS)
    _ENV_FILES_LOADED = list(loaded_files)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    missing_keys: list[str] = []
    try:
        files_repr = f"[{', '.join(loaded_files)}]" if loaded_files else "[]"
        LOGGER.info("[INFO] ENV_LOADED files=%s", files_repr)
        if missing:
            missing_keys = list(missing)
            LOGGER.error("[ERROR] ENV_MISSING_KEYS=%s", f"[{', '.join(missing_keys)}]")
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    if missing_keys:
        raise SystemExit(2)
    return loaded_files


def main(argv: Optional[Iterable[str]] = None) -> int:
    global _EXECUTE_START_UTC, _EXECUTE_FINISH_UTC
    if _EXECUTE_START_UTC is None:
        _EXECUTE_START_UTC = datetime.now(timezone.utc)
    _EXECUTE_FINISH_UTC = None
    _bootstrap_env()
    rc = 1
    metrics_payload: Dict[str, Any] | None = None
    status: str | None = None
    try:
        try:
            creds_snapshot = assert_alpaca_creds()
        except AlpacaCredentialsError as exc:
            missing = list(dict.fromkeys(list(exc.missing) + list(exc.whitespace)))
            LOGGER.error(
                "[ERROR] ALPACA_CREDENTIALS_INVALID reason=%s missing=%s whitespace=%s sanitized=%s",
                exc.reason,
                ",".join(exc.missing) or "",
                ",".join(exc.whitespace) or "",
                json.dumps(exc.sanitized, sort_keys=True),
            )
            log_info("SKIP", reason="AUTH", detail=exc.reason)
            _record_auth_error(exc.reason, exc.sanitized, missing)
            metrics_payload = _write_execute_metrics_error(
                "credentials_invalid",
                status="auth_error",
                rc=2,
                reason=exc.reason,
                missing=missing,
            )
            status = "auth_error"
            rc = 2
            return rc

        LOGGER.info(
            "[INFO] ALPACA_CREDENTIALS_OK sanitized=%s",
            json.dumps(creds_snapshot, sort_keys=True),
        )

        args = parse_args(argv)
        LOGGER.info(
            "[INFO] EXEC_CONFIG ext_hours=%s alloc=%.2f max_pos=%d trail_pct=%.1f",
            args.extended_hours,
            args.allocation_pct,
            args.max_positions,
            args.trailing_percent,
        )
        config = build_config(args)
        rc = run_executor(config, creds_snapshot=creds_snapshot)
        metrics_payload = _load_execute_metrics()
        if metrics_payload is None:
            fallback_metrics = ExecutionMetrics().as_dict()
            fallback_metrics["status"] = "ok" if rc == 0 else "error"
            metrics_payload = write_execute_metrics(
                fallback_metrics,
                start_dt=_EXECUTE_START_UTC,
                end_dt=datetime.now(timezone.utc),
            )
        status = metrics_payload.get("status") if isinstance(metrics_payload, Mapping) else None
        if metrics_payload is not None:
            def _as_int(value: Any) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            LOGGER.info(
                "EXECUTE END submitted=%d trailing=%d skips=%s",
                _as_int(metrics_payload.get("orders_submitted")),
                _as_int(metrics_payload.get("trailing_attached")),
                metrics_payload.get("skips", {}),
            )
            try:
                skip_block = metrics_payload.get("skips") if isinstance(metrics_payload.get("skips"), Mapping) else {}
                if not skip_block and isinstance(metrics_payload.get("skip_reasons"), Mapping):
                    skip_block = metrics_payload.get("skip_reasons")
                skip_counts = {str(k): _as_int(v) for k, v in (skip_block or {}).items()}
                candidates_in = _as_int(metrics_payload.get("symbols_in"))
                if candidates_in <= 0:
                    candidates_in = _count_csv_rows(Path("data") / "latest_candidates.csv")
                orders_submitted = _as_int(metrics_payload.get("orders_submitted"))
                non_time_skips = sum(count for reason, count in skip_counts.items() if reason != "TIME_WINDOW")
                time_only = non_time_skips == 0 and skip_counts.get("TIME_WINDOW", 0) > 0
                if orders_submitted == 0 and candidates_in > 0 and not time_only:
                    send_alert(
                        "JBRAVO pre-market: 0 orders submitted for non-TIME_WINDOW reasons",
                        {
                            "skips": skip_counts,
                            "buying_power": metrics_payload.get("buying_power"),
                            "status": metrics_payload.get("status"),
                            "candidates_in": candidates_in,
                            "run_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except Exception:
                LOGGER.debug("ALERT_EXECUTE_SKIPS_FAILED", exc_info=True)
        return rc
    except AlpacaUnauthorizedError as exc:
        LOGGER.error(
            '[ERROR] ALPACA_UNAUTHORIZED endpoint=%s feed=%s hint="check keys/base urls"',
            exc.endpoint or "",
            exc.feed or "",
        )
        log_info("SKIP", reason="AUTH", detail="unauthorized")
        _record_auth_error("unauthorized", creds_snapshot)
        metrics_payload = _write_execute_metrics_error(
            "unauthorized",
            status="auth_error",
            rc=2,
            endpoint=exc.endpoint or "",
            feed=exc.feed or "",
        )
        status = "auth_error"
        rc = 2
        return rc
    except Exception as exc:  # pragma: no cover - top-level guard
        LOGGER.exception("Executor failed: %s", exc)
        metrics_payload = _write_execute_metrics_error(
            "executor_exception",
            rc=1,
            exception=exc.__class__.__name__,
            detail=str(exc),
        )
        status = "error"
        try:
            send_alert(
                "JBRAVO execute_trades FAILED",
                {
                    "exception": repr(exc),
                    "run_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            LOGGER.debug("ALERT_EXECUTE_FATAL_FAILED", exc_info=True)
        rc = 1
        return rc
    finally:
        _EXECUTE_FINISH_UTC = datetime.now(timezone.utc)
        final_status = status or (metrics_payload.get("status") if isinstance(metrics_payload, Mapping) else None)
        if final_status is None:
            final_status = "ok" if rc == 0 else "error"
        try:
            metrics_payload = write_execute_metrics(
                metrics_payload,
                status=final_status,
                start_dt=_EXECUTE_START_UTC,
                end_dt=_EXECUTE_FINISH_UTC,
            )
        except Exception:
            LOGGER.debug("METRICS_WRITE_FAILED", exc_info=True)


def _warn_context(context: str, message: str) -> None:
    LOGGER.warning("[WARN] %s: %s", context, message)


# Keep the import-safe entrypoint at EOF so helpers above are always defined
if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
