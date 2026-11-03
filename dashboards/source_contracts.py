"""Read-only dashboard contracts. Unknown financial data is never an empty balance.

Broker calls have a caller deadline and a fixed, non-queued worker budget. A timed
out read may finish in its worker; it cannot enqueue more work or submit orders.
Physical ledger history deliberately does not qualify performance or recovery.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
import math
import threading


_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dashboard-read")
_slots = threading.BoundedSemaphore(4)


def bounded_read(function, seconds=3):
    if not _slots.acquire(blocking=False):
        raise RuntimeError("source_busy")
    try:
        future = _workers.submit(function)
    except Exception:
        _slots.release()
        raise
    future.add_done_callback(lambda _: _slots.release())
    try:
        return future.result(timeout=seconds)
    except TimeoutError:
        raise RuntimeError("source_timeout") from None


def number(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def timestamp(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def expected_paper_read_identity(disposition, pin, *, broker_context, mode_present, mode_value):
    """Read-only expected DB target; no executor override or broker UUID claim.

    The caller must obtain broker_context from the existing strict SDK/config
    context guard. The canonical policy reader still verifies the connected DB.
    """
    if broker_context != "configured_paper_broker":
        return None
    if mode_present and (type(mode_value) is not str or mode_value != "true"):
        return None
    if type(pin) is not tuple or len(pin) != 3 or any(type(item) is not str for item in pin):
        return None
    try:
        identity = disposition.identity(*pin[:2])
        disposition.text_field(pin[2])
        prior = vars(disposition).get("_RUNTIME_IDENTITY")
        if prior is not None and (type(prior) is not tuple or len(prior) != 4 or
                any(type(item) is not str for item in prior) or prior != (*pin, "paper")):
            return None
        return identity
    except Exception:
        return None


def policy_visibility(connection_factory, disposition, identity):
    """Canonical policy evidence only. Never financial qualification or metrics."""
    unavailable = dict(trade_policy=None, policy_validation_status="UNAVAILABLE",
                       policy_revision=None, excluded_count=None, unresolved_trade_ids=None,
                       accounting_input_status="NOT_INSPECTED", policy_visibility_scope="POLICY_ONLY")
    connection = None
    try:
        if identity is None:
            return unavailable
        disposition.identity(*identity)
        connection = connection_factory()
        if connection is None or connection.autocommit or connection.get_transaction_status() != 0:
            return unavailable
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SET LOCAL statement_timeout='5000ms'")
            policy = disposition.read_policy(cursor, identity, scope="VALIDATED_PERFORMANCE")
            metadata = policy.metadata()
            # Original accounting populations were not queried here. Do not
            # turn this allocation subset into a claim about every trade.
            return dict(trade_policy={"policy": metadata}, policy_validation_status="VALIDATED_POLICY",
                        policy_revision=policy.revision, excluded_count=metadata["excluded_count"],
                        unresolved_trade_ids=None, accounting_input_status="NOT_INSPECTED",
                        policy_visibility_scope="POLICY_ONLY")
    except Exception:
        return unavailable
    finally:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass


def _sum_known(rows, key):
    values = [number(row.get(key)) for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def position_snapshot(client, *, read=bounded_read, now=None):
    now = now or datetime.now(timezone.utc)
    base = dict(source="alpaca", account_scope="configured_broker", observed_at=now.isoformat(),
                calculationSource="alpaca", db_source_of_truth=False,
                enrichment_status="deferred", ledger_status="not_queried")
    unknown = dict(totalShares=None, totalOpenPL=None, avgDaysHeld=None, totalCapturedPL=None)
    if client is None:
        return dict(base, ok=False, status="unavailable", reason="broker_client_unavailable",
                    positions=None, summary=unknown)
    try:
        raw = read(client.get_all_positions)
        if raw is None:
            raise ValueError("missing_broker_result")
        rows = []
        for position in raw:
            symbol = str(getattr(position, "symbol", "") or "").strip().upper()
            if not symbol:
                raise ValueError("invalid_broker_position")
            pct = number(getattr(position, "unrealized_plpc", None))
            rows.append(dict(symbol=symbol, qty=number(getattr(position, "qty", None)),
                entryPrice=number(getattr(position, "avg_entry_price", None)),
                currentPrice=number(getattr(position, "current_price", None)),
                dollarPL=number(getattr(position, "unrealized_pl", None)),
                percentPL=None if pct is None else pct * 100,
                daysHeld=None, trailingStop=None, capturedPL=None, sparklineData=[]))
    except Exception as failure:
        reason = str(failure) if type(failure) is RuntimeError and str(failure) in {"source_busy", "source_timeout"} else "broker_read_failed"
        return dict(base, ok=False, status="unavailable", reason=reason, positions=None, summary=unknown)
    partial = any(any(row[key] is None for key in ("qty", "entryPrice", "currentPrice", "dollarPL")) for row in rows)
    summary = dict(totalShares=_sum_known(rows, "qty"), totalOpenPL=_sum_known(rows, "dollarPL"),
                   avgDaysHeld=None, totalCapturedPL=None)
    return dict(base, ok=True, status="partial" if partial else "ready" if rows else "empty",
                reason="broker_fields_missing" if partial else None, positions=rows, summary=summary,
                reconciliation_required=None)


def clock_snapshot(clock, *, now=None):
    now = now or datetime.now(timezone.utc)
    base = dict(source="alpaca_exchange_clock", observed_at=None, is_open=None,
                next_open=None, next_close=None)
    if clock is None:
        return dict(base, status="unavailable")
    stamp = timestamp(getattr(clock, "timestamp", None))
    opened = getattr(clock, "is_open", None)
    if stamp is None or type(opened) is not bool:
        return dict(base, status="unavailable")
    age = (now - datetime.fromisoformat(stamp)).total_seconds()
    if not -5 <= age <= 60:
        return dict(base, status="stale", observed_at=stamp)
    return dict(base, status="ready", observed_at=stamp, is_open=opened,
                next_open=timestamp(getattr(clock, "next_open", None)),
                next_close=timestamp(getattr(clock, "next_close", None)))


_clock_lock = threading.Lock()
_clock_cache = None


def broker_clock(client, *, now=None, read=bounded_read):
    global _clock_cache
    now = now or datetime.now(timezone.utc)
    with _clock_lock:
        cached = _clock_cache
    if cached and cached[0] is client and 0 <= (now - cached[1]).total_seconds() < 5:
        return clock_snapshot(cached[2], now=now)
    try:
        clock = None if client is None else read(client.get_clock, seconds=2)
    except Exception:
        clock = None
    with _clock_lock:
        _clock_cache = (client, now, clock)
    return clock_snapshot(clock, now=now)


def physical_history(connection_factory, *, limit=25, before=None, now=None, derived_identity=None):
    """Keyset pagination over physical rows; no filtering or P&L reconstruction.

    There is no broker account identifier in the ledger schema. Its association
    with a broker account is explicitly unverified; this endpoint never joins
    broker orders or CSVs. Disposition failure leaves rows visible as partial.
    """
    now = now or datetime.now(timezone.utc)
    limit = max(1, min(100, int(limit)))
    base = dict(source="postgres", trade_view="PHYSICAL_HISTORY_UNFILTERED",
                account_scope="configured_postgres_ledger", broker_account_match="unverified",
                observed_at=now.isoformat(), source_as_of=None, freshness="unknown",
                performance_status="NOT_QUALIFIED_PHYSICAL_HISTORY", accounting_complete=False,
                recovery_complete=False, fees_status="UNKNOWN", net_status="UNKNOWN",
                entry_ready=False, limit=limit)
    connection = None
    derived_context, derived_data = {"status": "NOT_CONFIGURED"}, None
    try:
        connection = connection_factory()
        if connection is None:
            raise RuntimeError("database_unavailable")
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute("SET LOCAL statement_timeout = '2000ms'")
                cursor.execute("""SELECT trade_id, symbol, status, qty, entry_time, entry_price,
                    exit_time, exit_price, realized_pnl, exit_reason, updated_at
                    FROM public.trades WHERE (%s::bigint IS NULL OR trade_id < %s::bigint)
                    ORDER BY trade_id DESC LIMIT %s""", (before, before, limit + 1))
                records = cursor.fetchall()
                names = [description[0] for description in cursor.description]
                rows = [dict(zip(names, record)) for record in records[:limit]]
                cursor.execute("SAVEPOINT history_disposition")
                dispositions = {}
                disposition_ready = True
                try:
                    cursor.execute("""SELECT trade_id, state FROM public.trade_disposition_cases
                        WHERE state = 'ACTIVE' AND trade_id = ANY(%s)""", ([row["trade_id"] for row in rows],))
                    dispositions = {record[0]: record[1] for record in cursor.fetchall()}
                except Exception:
                    cursor.execute("ROLLBACK TO SAVEPOINT history_disposition")
                    disposition_ready = False
                cursor.execute("RELEASE SAVEPOINT history_disposition")
                if derived_identity is not None and any(row.get("trade_id") == 110 for row in rows):
                    cursor.execute("SAVEPOINT history_derived_costs")
                    try:
                        from scripts import observed_fee_display
                        derived_context, derived_data = observed_fee_display.read_for_history(cursor, rows, derived_identity, now=now)
                    except Exception:
                        cursor.execute("ROLLBACK TO SAVEPOINT history_derived_costs")
                        derived_context, derived_data = {"status": "UNAVAILABLE_OR_INVALIDATED"}, None
                    cursor.execute("RELEASE SAVEPOINT history_derived_costs")
        for row in rows:
            for key in ("qty", "entry_price", "exit_price", "realized_pnl"):
                row[key] = number(row.get(key))
            for key in ("entry_time", "exit_time", "updated_at"):
                row[key] = timestamp(row.get(key))
            row["disposition"] = ("ACTIVE_CASE" if row["trade_id"] in dispositions else "NO_ACTIVE_CASE") if disposition_ready else "UNKNOWN"
            row["accounting_status"] = "UNQUALIFIED_RECORDED_VALUES"
            if row.get("trade_id") == 110 and derived_data is not None:
                row["row_fingerprint"] = derived_data["bound_row"]["row_fingerprint"]
                row["derived_accounting"] = derived_data
        stamps = [row["updated_at"] for row in rows if row.get("updated_at")]
        return dict(base, ok=True, status=("partial" if not disposition_ready else "ready" if rows else "empty"),
                    rows=rows, has_more=len(records) > limit,
                    next_before=rows[-1]["trade_id"] if len(records) > limit and rows else None,
                    disposition_status="ready" if disposition_ready else "unavailable",
                    derived_read_context=derived_context,
                    source_as_of=max(stamps) if stamps else None)
    except Exception:
        return dict(base, ok=False, status="unavailable", reason="physical_history_unavailable",
                    rows=[], has_more=None, next_before=None, disposition_status="unknown")
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
