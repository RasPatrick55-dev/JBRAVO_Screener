"""Robust trade execution orchestrator for JBRAVO."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from utils.env import get_alpaca_creds, load_env
import utils.telemetry as telemetry

try:  # pragma: no cover - import guard for optional dependency
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        TrailingStopOrderRequest,
    )
except Exception:  # pragma: no cover - lightweight fallback for tests
    TradingClient = None  # type: ignore

    class _Enum(str):
        def __new__(cls, value: str):
            return str.__new__(cls, value)

    class OrderSide:  # type: ignore
        BUY = _Enum("buy")
        SELL = _Enum("sell")

    class TimeInForce:  # type: ignore
        DAY = _Enum("day")
        GTC = _Enum("gtc")

    class QueryOrderStatus:  # type: ignore
        OPEN = _Enum("open")

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


LOGGER = logging.getLogger("execute_trades")
LOG_PATH = Path("logs") / "execute_trades.log"
METRICS_PATH = Path("data") / "execute_metrics.json"
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
IMPORT_SENTINEL_ENV = "JBRAVO_IMPORT_SENTINEL"


class CandidateLoadError(RuntimeError):
    """Raised when candidate data cannot be loaded or validated."""


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


@dataclass
class ExecutorConfig:
    source: Path = Path("data/latest_candidates.csv")
    allocation_pct: float = 0.03
    max_positions: int = 4
    entry_buffer_bps: int = 75
    trailing_percent: float = 3.0
    cancel_after_min: int = 35
    extended_hours: bool = True
    dry_run: bool = False
    min_adv20: int = 2_000_000
    min_price: float = 1.0
    max_price: float = 1_000.0
    log_json: bool = False
    bar_directories: Sequence[Path] = DEFAULT_BAR_DIRECTORIES


@dataclass
class ExecutionMetrics:
    symbols_in: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_canceled: int = 0
    trailing_attached: int = 0
    api_retries: int = 0
    api_failures: int = 0
    latency_samples: List[float] = field(default_factory=list)
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Maintain backward compatibility with legacy attribute name
        self.skipped_by_reason = self.skipped_reasons

    def record_skip(self, reason: str) -> None:
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def record_latency(self, seconds: float) -> None:
        if seconds > 0:
            self.latency_samples.append(seconds)

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
        return {
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
            "symbols_in": self.symbols_in,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_canceled": self.orders_canceled,
            "trailing_attached": self.trailing_attached,
            "api_retries": self.api_retries,
            "api_failures": self.api_failures,
            "latency_secs": {
                "p50": self.percentile(0.5),
                "p95": self.percentile(0.95),
            },
            "skipped": self.skipped_reasons,
        }


def round_to_tick(value: float, tick: float = 0.01) -> float:
    decimal_value = Decimal(str(value))
    quant = Decimal(str(tick))
    return float(decimal_value.quantize(quant, rounding=ROUND_HALF_UP))


def compute_limit_price(row: Dict[str, Any], buffer_bps: int = 75) -> float:
    base_price = row.get("entry_price") if row.get("entry_price") not in (None, "") else row.get("close")
    if base_price is None or pd.isna(base_price):
        raise ValueError("Row must contain either entry_price or close")
    base_price_f = float(base_price)
    limit = base_price_f * (1 + buffer_bps / 10_000)
    return round_to_tick(limit, 0.01)


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


class OptionalFieldHydrator:
    def __init__(self, client: Any, cache: DailyBarCache) -> None:
        self.client = client
        self.cache = cache
        self.exchange_cache: Dict[str, Optional[str]] = {}

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

    def hydrate(self, row: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(row)
        symbol = enriched.get("symbol", "").upper()
        if symbol:
            if "exchange" not in enriched or pd.isna(enriched.get("exchange")):
                enriched["exchange"] = self.get_exchange(symbol)
            if "atrp" not in enriched or pd.isna(enriched.get("atrp")):
                enriched["atrp"] = self.cache.atr_percent(symbol)
        if "entry_price" not in enriched or pd.isna(enriched.get("entry_price")):
            enriched["entry_price"] = enriched.get("close")
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
    ) -> None:
        self.config = config
        self.client = client
        self.metrics = metrics
        self.sleep = sleep_fn or time.sleep
        self.bar_cache = DailyBarCache(config.bar_directories)
        self.hydrator = OptionalFieldHydrator(client, self.bar_cache)
        self.log_json = config.log_json

    def log_event(self, event: str, **payload: Any) -> None:
        human = " ".join(f"{key}={value}" for key, value in payload.items())
        if human:
            LOGGER.info("%s %s", event, human)
        else:
            LOGGER.info("%s", event)
        if self.log_json:
            try:
                LOGGER.info(json.dumps({"event": event, **payload}))
            except TypeError:  # pragma: no cover - serialization guard
                LOGGER.info(json.dumps({"event": event, "message": str(payload)}))

    def load_candidates(self) -> pd.DataFrame:
        path = self.config.source
        if not path.exists():
            raise CandidateLoadError(f"Candidate file not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            LOGGER.info("[INFO] NO_CANDIDATES_IN_SOURCE")
            return df
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise CandidateLoadError(
                f"Candidate file missing required columns: {', '.join(missing)}"
            )
        for column in OPTIONAL_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df

    def hydrate_candidates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records = df.to_dict(orient="records")
        return [self.hydrator.hydrate(record) for record in records]

    def guard_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for record in records:
            price = record.get("entry_price")
            if price in (None, "") or pd.isna(price):
                price = record.get("close")
            if price in (None, "") or pd.isna(price):
                self.metrics.record_skip("MISSING_PRICE")
                continue
            price_f = float(price)
            if price_f < self.config.min_price:
                self.metrics.record_skip("PRICE_LT_MIN")
                continue
            if price_f > self.config.max_price:
                self.metrics.record_skip("PRICE_GT_MAX")
                continue
            adv = record.get("adv20")
            if adv not in (None, "") and not pd.isna(adv):
                adv_value = pd.to_numeric(pd.Series([adv]), errors="coerce").iloc[0]
                if not pd.isna(adv_value) and float(adv_value) < self.config.min_adv20:
                    self.metrics.record_skip("ADV20_LT_MIN")
                    continue
            filtered.append(record)
        return filtered

    def execute(
        self,
        df: pd.DataFrame,
        *,
        prefiltered: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        self.metrics.symbols_in = len(df)
        if prefiltered is not None:
            candidates = prefiltered
        else:
            records = self.hydrate_candidates(df)
            candidates = self.guard_candidates(records)
        if not candidates:
            LOGGER.info("No candidates passed guardrails; nothing to do.")
            self.persist_metrics()
            return 0

        if self.config.dry_run:
            LOGGER.info("Dry-run mode active; no orders will be submitted.")

        existing_positions = self.fetch_existing_positions()
        open_order_symbols = self.fetch_open_order_symbols()
        account_buying_power = self.fetch_buying_power()

        for record in candidates:
            symbol = record.get("symbol", "").upper()
            if not symbol:
                continue
            if symbol in existing_positions:
                self.metrics.record_skip("EXISTING_POSITION")
                continue
            if symbol in open_order_symbols:
                self.metrics.record_skip("OPEN_ORDER")
                continue
            if len(existing_positions) >= self.config.max_positions:
                self.metrics.record_skip("MAX_POSITIONS")
                break

            limit_price = compute_limit_price(record, self.config.entry_buffer_bps)
            qty = compute_quantity(account_buying_power, self.config.allocation_pct, limit_price)
            if qty < 1:
                self.metrics.record_skip("BUYING_POWER")
                continue

            notional = qty * limit_price
            if notional > account_buying_power:
                self.metrics.record_skip("BUYING_POWER")
                continue

            if self.config.dry_run:
                self.log_event("DRY_RUN_ORDER", symbol=symbol, qty=qty, limit_price=f"{limit_price:.2f}")
                continue

            outcome = self.execute_order(symbol, qty, limit_price)
            if outcome.get("filled_qty", 0) > 0:
                existing_positions.add(symbol)
                account_buying_power = max(0.0, account_buying_power - notional)
            if outcome.get("submitted"):
                open_order_symbols.add(symbol)

        self.persist_metrics()
        return 0

    def fetch_existing_positions(self) -> set[str]:
        symbols: set[str] = set()
        if self.client is None:
            return symbols
        try:
            positions = self.client.get_all_positions()
            for pos in positions:
                symbol = getattr(pos, "symbol", "")
                if symbol:
                    symbols.add(symbol.upper())
        except Exception as exc:
            LOGGER.warning("Failed to fetch positions: %s", exc)
        return symbols

    def fetch_open_order_symbols(self) -> set[str]:
        symbols: set[str] = set()
        if self.client is None:
            return symbols
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(request)
            for order in orders or []:
                symbol = getattr(order, "symbol", "")
                if symbol:
                    symbols.add(symbol.upper())
        except Exception as exc:
            LOGGER.warning("Failed to fetch open orders: %s", exc)
        return symbols

    def fetch_buying_power(self) -> float:
        if self.client is None:
            return 0.0
        try:
            account = self.client.get_account()
            buying_power = getattr(account, "buying_power", 0.0)
            return float(buying_power)
        except Exception as exc:
            LOGGER.warning("Failed to fetch buying power: %s", exc)
            return 0.0

    def execute_order(self, symbol: str, qty: int, limit_price: float) -> Dict[str, Any]:
        outcome: Dict[str, Any] = {"submitted": False, "filled_qty": 0.0}
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type="limit",
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=self.config.extended_hours,
        )
        submitted_order = self.submit_with_retries(order_request)
        if submitted_order is None:
            return outcome
        outcome["submitted"] = True
        order_id = str(getattr(submitted_order, "id", ""))
        self.metrics.orders_submitted += 1
        self.log_event("BUY_SUBMIT", symbol=symbol, qty=qty, limit=f"{limit_price:.2f}", order_id=order_id or "")

        fill_deadline = datetime.now(timezone.utc) + timedelta(minutes=self.config.cancel_after_min)
        submit_ts = time.time()

        filled_qty = 0.0
        filled_avg_price = None
        status = "open"

        while datetime.now(timezone.utc) < fill_deadline:
            try:
                order = self.client.get_order_by_id(order_id)
            except Exception as exc:
                LOGGER.warning("Failed to refresh order %s: %s", order_id, exc)
                break
            status = str(getattr(order, "status", "")).lower()
            filled_qty = float(getattr(order, "filled_qty", filled_qty) or 0)
            filled_avg_price = getattr(order, "filled_avg_price", filled_avg_price)
            if status == "filled":
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_event(
                    "BUY_FILL",
                    symbol=symbol,
                    qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
                outcome["filled_qty"] = filled_qty
                return outcome
            if status in {"canceled", "expired", "rejected"}:
                break
            if filled_qty >= qty:
                latency = time.time() - submit_ts
                self.metrics.orders_filled += 1
                self.metrics.record_latency(latency)
                self.log_event(
                    "BUY_FILL",
                    symbol=symbol,
                    qty=f"{filled_qty:.0f}",
                    avg_price=f"{float(filled_avg_price or 0):.2f}",
                    order_id=order_id,
                )
                self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
                outcome["filled_qty"] = filled_qty
                return outcome
            self.sleep(5)

        # Deadline reached or not fully filled
        remaining = max(qty - filled_qty, 0)
        if remaining > 0:
            self.cancel_order(order_id, symbol)
            self.metrics.orders_canceled += 1
            self.log_event(
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
            self.log_event(
                "BUY_FILL",
                symbol=symbol,
                qty=f"{filled_qty:.0f}",
                avg_price=f"{float(filled_avg_price or 0):.2f}",
                partial="true",
                order_id=order_id,
            )
            self.attach_trailing_stop(symbol, filled_qty, filled_avg_price)
            outcome["filled_qty"] = filled_qty
        return outcome

    def cancel_order(self, order_id: str, symbol: str) -> None:
        if self.client is None or not order_id:
            return
        try:
            if hasattr(self.client, "cancel_order_by_id"):
                self.client.cancel_order_by_id(order_id)
            elif hasattr(self.client, "cancel_order"):
                self.client.cancel_order(order_id)
            else:  # pragma: no cover - defensive fallback
                LOGGER.warning("Client has no cancel method; unable to cancel %s", order_id)
        except Exception as exc:
            LOGGER.warning("Failed to cancel %s for %s: %s", order_id, symbol, exc)

    def attach_trailing_stop(self, symbol: str, qty: float, avg_price: Optional[Any]) -> None:
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
        self.log_event(
            "TRAIL_SUBMIT",
            symbol=symbol,
            qty=str(qty_int),
            trail_percent=self.config.trailing_percent,
        )
        trailing_order = self.submit_with_retries(request)
        if trailing_order is None:
            return
        self.metrics.trailing_attached += 1
        order_id = str(getattr(trailing_order, "id", ""))
        self.log_event(
            "TRAIL_CONFIRMED",
            symbol=symbol,
            qty=str(qty_int),
            order_id=order_id,
        )

    def submit_with_retries(self, request: Any) -> Optional[Any]:
        if self.client is None:
            return None
        attempts = 3
        backoff = 1.5
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                if isinstance(request, LimitOrderRequest):
                    result = self.client.submit_order(request)
                else:
                    result = self.client.submit_order(request)
                if attempt > 1:
                    self.log_event("API_RETRY_SUCCESS", attempt=attempt)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    self.metrics.api_retries += 1
                    self.log_event("API_RETRY", attempt=attempt, error=str(exc))
                    self.sleep(delay)
                    delay *= backoff
                    continue
                self.metrics.api_failures += 1
                LOGGER.error("Order submission failed after retries: %s", exc)
        if last_error is not None:
            self.log_event("API_FAILURE", error=str(last_error))
        return None

    def persist_metrics(self) -> None:
        payload = self.metrics.as_dict()
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Metrics updated: %s", json.dumps(payload))


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


def _create_trading_client() -> Any:
    if TradingClient is None:
        raise RuntimeError("alpaca-py TradingClient is unavailable")
    api_key, api_secret, base_url, _ = get_alpaca_creds()
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials for trading client")
    env = (base_url or "paper").lower()
    paper = "live" not in env
    return TradingClient(api_key, api_secret, paper=paper)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades for pipeline candidates")
    parser.add_argument("--source", type=Path, default=ExecutorConfig.source, help="Path to the candidate CSV file")
    parser.add_argument(
        "--allocation-pct",
        type=float,
        default=ExecutorConfig.allocation_pct,
        help="Fraction of buying power allocated per position (0-1)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=ExecutorConfig.max_positions,
        help="Maximum concurrent positions the executor will open",
    )
    parser.add_argument(
        "--entry-buffer-bps",
        type=int,
        default=ExecutorConfig.entry_buffer_bps,
        help="Entry buffer in basis points added to the reference price",
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
        "--extended-hours",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.extended_hours,
        help="Whether to submit orders eligible for extended hours",
    )
    parser.add_argument(
        "--dry-run",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=ExecutorConfig.dry_run,
        help="If true, only log intended actions without submitting orders",
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
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    return ExecutorConfig(
        source=args.source,
        allocation_pct=args.allocation_pct,
        max_positions=args.max_positions,
        entry_buffer_bps=args.entry_buffer_bps,
        trailing_percent=args.trailing_percent,
        cancel_after_min=args.cancel_after_min,
        extended_hours=args.extended_hours,
        dry_run=args.dry_run,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        log_json=args.log_json,
    )


def run_executor(config: ExecutorConfig, *, client: Optional[Any] = None) -> int:
    configure_logging(config.log_json)
    metrics = ExecutionMetrics()
    loader = TradeExecutor(config, client, metrics)
    try:
        frame = loader.load_candidates()
    except CandidateLoadError as exc:
        LOGGER.error("%s", exc)
        metrics.api_failures += 1
        loader.persist_metrics()
        return 1

    if frame.empty:
        loader.persist_metrics()
        return 0

    records = loader.hydrate_candidates(frame)
    filtered = loader.guard_candidates(records)

    if client is not None:
        trading_client = client
    elif config.dry_run or not filtered:
        trading_client = None
    else:
        trading_client = _create_trading_client()

    executor = TradeExecutor(config, trading_client, metrics)
    return executor.execute(frame, prefiltered=filtered)


def load_candidates(path: Path) -> pd.DataFrame:
    config = ExecutorConfig(source=path)
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_env()
    args = parse_args(argv)
    config = build_config(args)
    try:
        return run_executor(config)
    except Exception as exc:  # pragma: no cover - top-level guard
        LOGGER.exception("Executor failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
