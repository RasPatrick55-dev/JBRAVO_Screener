"""Bounded SELECT-only strategy history, separate from the physical trade ledger.

The native execution store is deliberately not imported or constructed. Stored
broker order snapshots are evidence, not live fills, fee accounting or a license
to repair state. Native state is returned unchanged, even when inconsistent.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re
import time
from uuid import UUID
from threading import Event

from dashboards.source_contracts import bounded_read


SCHEMA = "jbravo.strategy-history.v1"
BASIS = "STORED_BROKER_ORDER_OBSERVATIONS"
MAX_SIGNALS, MAX_ORDERS, MAX_BYTES, ROW_BYTES = 50, 1000, 4 * 1024 * 1024, 192 * 1024
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}
LIVE = {
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "accepted_for_bidding",
    "held",
    "pending_replace",
    "pending_cancel",
    "stopped",
    "suspended",
    "calculated",
    "done_for_day",
}
STATES = {
    "PREPARED",
    "ENTRY_PENDING",
    "OPEN",
    "EXIT_PENDING",
    "CLOSED",
    "EXPIRED",
    "UNKNOWN",
    "DISPUTED",
}


class EvidenceError(ValueError):
    """Only fixed, module-owned reason codes may reach the public response."""


def need(value, reason):
    if not value:
        raise EvidenceError(reason)


def exact(value, *, positive=False):
    need(type(value) in (str, int, float, Decimal) and len(str(value)) <= 64, "invalid_number")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise EvidenceError("invalid_number") from None
    need(
        result.is_finite()
        and -12 <= result.as_tuple().exponent <= 15
        and (not result or -12 <= result.adjusted() <= 15)
        and result >= 0
        and (not positive or result > 0),
        "invalid_number",
    )
    return result


def text(value, maximum=128):
    need(
        type(value) is str and 0 < len(value) <= maximum and value == value.strip(),
        "invalid_identity",
    )
    return value


def parsed_timestamp(value):
    """Return UTC text and exact integer nanoseconds since the Unix epoch.

    Native observations retain Alpaca's 1-9 fractional digits. Parse only whole
    seconds with datetime (including on Python 3.10), and keep the fraction as
    an integer. Never round/truncate it or infer a zone for a naive value.
    """
    try:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return None
            parsed = value.astimezone(timezone.utc)
            fraction = f"{parsed.microsecond:06d}" if parsed.microsecond else ""
            parsed = parsed.replace(microsecond=0)
        elif type(value) is str:
            match = re.fullmatch(
                r"([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2})"
                r"(?:\.([0-9]{1,9}))?(Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
                value,
            )
            if match is None:
                return None
            main, fraction, offset = match.groups()
            parsed = datetime.fromisoformat(main + ("+00:00" if offset == "Z" else offset))
            parsed = parsed.astimezone(timezone.utc)
            fraction = fraction or ""
        else:
            return None
        nanos = int(fraction.ljust(9, "0")) if fraction else 0
        seconds = (
            (parsed.toordinal() - datetime(1970, 1, 1).toordinal()) * 86400
            + parsed.hour * 3600
            + parsed.minute * 60
            + parsed.second
        )
        # Keep the existing microsecond wire shape and every higher-precision
        # source digit. An explicit offset is converted, not silently dropped.
        suffix = "." + fraction.ljust(6, "0") if nanos or len(fraction) > 6 else ""
        stamp = parsed.isoformat(timespec="seconds")[:-6] + suffix + "+00:00"
        return stamp, seconds * 1_000_000_000 + nanos
    except (ValueError, TypeError, OverflowError):
        return None


def timestamp(value):
    parsed = parsed_timestamp(value)
    return parsed[0] if parsed is not None else None


def instant(value):
    parsed = parsed_timestamp(value)
    need(parsed is not None, "invalid_timestamp")
    return parsed[1]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def client_key(account, signal, role, sequence):
    return (
        "jbr2-"
        + hashlib.sha256(
            json.dumps([account, signal, role, sequence], separators=(",", ":")).encode()
        ).hexdigest()[:40]
    )


def connect_current_db(db):
    """Use the existing resolver, with a local connection timeout, no logging."""
    try:
        config = db._resolve_db_config()
        if config is None:
            return None
        options = {key: config.get(key) for key in ("host", "port", "dbname", "user")}
        options["connect_timeout"] = 2
        for key in ("password", "sslmode"):
            if config.get(key):
                options[key] = config[key]
        return db.psycopg2.connect(**options)
    except Exception:
        return None


class Reader:
    """Four fixed data queries; row and total wire-byte sentinel in SQL."""

    def __init__(self, cursor):
        self.cursor, self.deadline, self.bytes, self.queries = cursor, time.monotonic() + 4, 0, 0

    def rows(self, sql, params, cap):
        remaining = self.deadline - time.monotonic()
        need(remaining > 0 and self.queries < 4, "read_budget_exceeded")
        millis = max(1, min(1500, int(remaining * 1000)))
        self.cursor.execute(f"SET LOCAL statement_timeout = '{millis}ms'")
        self.queries += 1
        # Only module-owned SQL enters this method. LIMIT +1 is an overflow
        # sentinel, never a silent truncation of orders or duplicate evidence.
        wire = (
            "WITH q AS MATERIALIZED (" + sql + "), "
            "encoded AS MATERIALIZED (SELECT row_to_json(q)::text AS body FROM q), "
            "checked AS MATERIALIZED (SELECT body, count(*) OVER () AS n, "
            "sum(octet_length(body)) OVER () AS bytes FROM encoded) "
            f"SELECT CASE WHEN octet_length(body)<={ROW_BYTES} AND n<={cap} "
            f"AND bytes<={MAX_BYTES - self.bytes} THEN body ELSE NULL END, n, bytes FROM checked"
        )
        self.cursor.execute(wire, params)
        result = self.cursor.fetchmany(cap + 1)
        need(len(result) <= cap, "row_budget_exceeded")
        if not result:
            return []
        count, size = result[0][1:]
        need(
            type(count) is int
            and count == len(result)
            and type(size) is int
            and 0 < size <= MAX_BYTES - self.bytes,
            "read_budget_exceeded",
        )
        self.bytes += size
        rows = []
        for body, n, byte_count in result:
            need(
                type(body) is str
                and n == count
                and byte_count == size
                and len(body.encode()) <= ROW_BYTES,
                "row_budget_exceeded",
            )
            rows.append(json.loads(body))
        need(time.monotonic() <= self.deadline, "read_budget_exceeded")
        return rows


def read_snapshot(connection_factory, account, expected_identity, limit):
    connection = None
    try:
        connection = connection_factory()
        need(
            connection is not None
            and connection.autocommit is False
            and connection.get_transaction_status() == 0,
            "database_unavailable",
        )
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            cursor.execute("SET LOCAL lock_timeout = '500ms'")
            cursor.execute("SET LOCAL idle_in_transaction_session_timeout = '5000ms'")
            cursor.execute("SET LOCAL TIME ZONE 'UTC'")
            reader = Reader(cursor)
            identity = reader.rows(
                "SELECT deployment_id, database_id, database_name, "
                "current_database() AS connected_database FROM public.trade_disposition_policy "
                "WHERE singleton=1 LIMIT 2",
                (),
                1,
            )
            need(
                len(identity) == 1
                and (identity[0].get("deployment_id"), identity[0].get("database_id"))
                == expected_identity
                and type(identity[0].get("database_name")) is str
                and bool(identity[0]["database_name"])
                and identity[0]["database_name"] == identity[0].get("connected_database"),
                "database_identity_mismatch",
            )
            signals = reader.rows(
                "SELECT account_id, signal_id, symbol, state, payload, created_at, updated_at "
                "FROM public.strategy_execution_signals_app WHERE account_id=%s "
                "ORDER BY updated_at DESC, signal_id DESC LIMIT %s",
                (account, limit + 1),
                limit + 1,
            )
            signals.sort(key=lambda r: (instant(r["updated_at"]), r["signal_id"]), reverse=True)
            has_more = len(signals) > limit
            signals = signals[:limit]
            ids = [row["signal_id"] for row in signals]
            orders = reader.rows(
                "SELECT account_id, signal_id, client_order_id, role, sequence, operation, "
                "request_sha256, request, lineage, state, observation, claimed_at, updated_at "
                "FROM public.strategy_execution_orders_app WHERE account_id=%s AND signal_id=ANY(%s) "
                "ORDER BY claimed_at, client_order_id LIMIT %s",
                (account, ids, MAX_ORDERS + 1),
                MAX_ORDERS,
            )
            order_ids = list(
                {
                    str(row["observation"]["id"])
                    for row in orders
                    if isinstance(row.get("observation"), dict) and row["observation"].get("id")
                }
            )
            overlap = reader.rows(
                "SELECT trade_id, entry_order_id::text AS entry_order_id, "
                "exit_order_id::text AS exit_order_id FROM public.trades "
                "WHERE entry_order_id::text=ANY(%s) OR exit_order_id::text=ANY(%s) "
                "ORDER BY trade_id LIMIT %s",
                (order_ids, order_ids, MAX_ORDERS + 1),
                MAX_ORDERS,
            )
        return signals, orders, overlap, has_more
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()


def blank_row(signal):
    return dict(
        signal_id=signal.get("signal_id"),
        symbol=signal.get("symbol"),
        strategy_id=None,
        state=signal.get("state"),
        status="incomplete",
        reason="incomplete_evidence",
        entry_qty=None,
        exited_qty=None,
        remaining_qty=None,
        entry_avg_price=None,
        exit_avg_price=None,
        entry_time=None,
        exit_time=None,
        gross_realized_pnl=None,
        fees=None,
        net_realized_pnl=None,
        fees_status="UNKNOWN_UNALLOCATED",
        completion_status="UNVERIFIED",
        updated_at=timestamp(signal.get("updated_at")),
        entry_order_ids=[],
        exit_order_ids=[],
        evidence_basis=BASIS,
        ownership_scope="UNKNOWN",
        ownership_basis="STORED_SIGNAL_PAYLOAD_NOT_REVALIDATED",
    )


def project_signal(signal, records, account, now, *, include_exact=False):
    row = blank_row(signal)
    try:
        sid, symbol, payload = signal["signal_id"], signal["symbol"], signal["payload"]
        need(
            re.fullmatch("[0-9a-f]{64}", sid) and re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}", symbol),
            "invalid_identity",
        )
        need(
            signal["state"] in STATES
            and signal["account_id"] == account
            and payload["account_id"] == account,
            "signal_account_mismatch",
        )
        proposal, handoff = payload["proposal"], payload["handoff"]
        need(
            proposal["signal_id"] == sid
            and proposal["symbol"] == symbol
            and handoff["paper_only"] is True
            and proposal["strategy_id"] == handoff["strategy_contract"]["strategy_id"]
            and type(handoff["proposals"]) is list
            and sum(item == proposal for item in handoff["proposals"]) == 1,
            "signal_binding_mismatch",
        )
        row["strategy_id"] = text(proposal["strategy_id"])
        owner = payload.get("execution_owner")
        if (
            isinstance(owner, dict)
            and owner.get("schema") == "jbravo.execution-entry-recognition.v1"
            and owner.get("scope") == "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS"
            and owner.get("signal_id") == sid
        ):
            row["ownership_scope"] = "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS"
        elif "execution_owner" in payload or "ownership_history" in payload:
            row["ownership_scope"] = "UNKNOWN"
        elif "manual_paper_test" in handoff:
            row["ownership_scope"] = "MANUAL_PAPER_TEST"
        else:
            row["ownership_scope"] = "STRATEGY_SIGNAL"
        need(records, "order_evidence_missing")
        by_id, cancels, replacements = {}, [], []
        for record in records:
            role, sequence, operation = record["role"], record["sequence"], record["operation"]
            need(
                role in {"entry", "stop", "partial", "exit", "cancel"}
                and type(sequence) is int
                and sequence >= 0
                and operation in {"POST", "PATCH", "DELETE"},
                "order_binding_mismatch",
            )
            key = client_key(account, sid, role, sequence)
            need(
                record["account_id"] == account
                and record["signal_id"] == sid
                and record["client_order_id"] == key,
                "order_binding_mismatch",
            )
            descriptor, lineage = record["request"], record["lineage"]
            need(
                record["request_sha256"]
                == digest(dict(operation=operation, request=descriptor, lineage=lineage))
                and lineage["signal_id"] == sid
                and lineage["handoff_id"] == handoff["handoff_id"]
                and lineage["strategy_contract"] == handoff["strategy_contract"],
                "order_binding_mismatch",
            )
            observation = record["observation"]
            if record["state"] == "REJECTED":
                need(observation is None, "order_binding_mismatch")
                continue
            need(
                record["state"] == "OBSERVED" and type(observation) is dict, "order_outcome_unknown"
            )
            oid = text(observation["id"])
            need(
                observation["symbol"] == symbol and observation["status"] in TERMINAL | LIVE,
                "broker_observation_mismatch",
            )
            filled, quantity = (
                exact(observation["filled_qty"]),
                exact(observation["qty"], positive=True),
            )
            need(
                filled <= quantity and (observation["status"] != "filled" or filled == quantity),
                "fill_quantity_invalid",
            )
            price = exact(observation["filled_avg_price"], positive=True) if filled else None
            updated = instant(observation["updated_at"])
            need(updated <= instant(now + timedelta(seconds=5)), "future_observation")
            if observation.get("filled_at") is not None:
                need(
                    filled > 0 and instant(observation["filled_at"]) <= updated,
                    "fill_timestamp_after_observation",
                )
            observed = dict(observation, _qty=filled, _price=price, _updated=updated)
            if operation == "DELETE":
                need(role == "cancel" and descriptor["order_id"] == oid, "cancel_binding_mismatch")
                cancels.append(observed)
                continue
            order = descriptor["order"]
            need(
                role != "cancel"
                and observation["client_order_id"] == key
                and order["client_order_id"] == key
                and order["symbol"] == symbol
                and order["side"] == ("buy" if role == "entry" else "sell")
                and observation["side"] == order["side"]
                and quantity == exact(order["qty"], positive=True)
                and observation["type"] == order["type"]
                and observation["time_in_force"] == order["time_in_force"]
                and type(order["extended_hours"]) is bool
                and observation["extended_hours"] is order["extended_hours"],
                "broker_observation_mismatch",
            )
            for field in ("limit_price", "stop_price"):
                if field in order:
                    need(
                        exact(observation[field], positive=True)
                        == exact(order[field], positive=True),
                        "broker_price_mismatch",
                    )
            if filled and order["type"] == "limit":
                boundary = exact(order["limit_price"], positive=True)
                need(
                    price <= boundary if order["side"] == "buy" else price >= boundary,
                    "broker_price_mismatch",
                )
            need(oid not in by_id, "duplicate_broker_order")
            by_id[oid] = observed
            if operation == "PATCH":
                need(
                    role == "stop" and observation.get("replaces") == descriptor["order_id"],
                    "replacement_binding_mismatch",
                )
                replacements.append((descriptor["order_id"], oid))
        for canceled in cancels:
            prior = by_id.get(canceled["id"])
            need(
                prior is not None
                and all(
                    canceled.get(k) == prior.get(k)
                    for k in (
                        "client_order_id",
                        "symbol",
                        "side",
                        "type",
                        "time_in_force",
                        "extended_hours",
                    )
                )
                and exact(canceled["qty"]) == exact(prior["qty"]),
                "cancel_binding_mismatch",
            )
            for field in ("limit_price", "stop_price"):
                if prior.get(field) is not None:
                    need(
                        exact(canceled.get(field), positive=True)
                        == exact(prior[field], positive=True),
                        "cancel_binding_mismatch",
                    )
            if canceled["_qty"] and prior["type"] == "limit":
                boundary = exact(prior["limit_price"], positive=True)
                need(
                    canceled["_price"] <= boundary
                    if prior["side"] == "buy"
                    else canceled["_price"] >= boundary,
                    "broker_price_mismatch",
                )
            older, newer = sorted((prior, canceled), key=lambda item: item["_updated"])
            need(
                newer["_qty"] >= older["_qty"]
                and (
                    newer["_updated"] != older["_updated"]
                    or (newer["_qty"], newer["_price"], newer["status"])
                    == (older["_qty"], older["_price"], older["status"])
                ),
                "cumulative_fill_conflict",
            )
            need(
                newer["_qty"] != older["_qty"] or newer["_price"] == older["_price"],
                "cumulative_fill_conflict",
            )
            if older["status"] in TERMINAL:
                need(
                    newer["status"] == older["status"] and newer["_qty"] == older["_qty"],
                    "cumulative_fill_conflict",
                )
            if (
                older.get("filled_at") is not None
                and newer.get("filled_at") is not None
                and newer["_qty"] == older["_qty"]
            ):
                need(
                    instant(older["filled_at"]) == instant(newer["filled_at"]),
                    "cumulative_fill_conflict",
                )
            by_id[canceled["id"]] = newer
        for old, new in replacements:
            need(
                old in by_id
                and by_id[old].get("replaced_by") == new
                and by_id[old]["status"] == "replaced",
                "replacement_binding_mismatch",
            )
            # Replacement snapshots can repeat cumulative fills. Do not guess
            # incremental allocation across replaced orders.
            need(not by_id[old]["_qty"], "replacement_fill_ambiguity")
        buys = [o for o in by_id.values() if o["side"] == "buy"]
        sells = [o for o in by_id.values() if o["side"] == "sell" and o["_qty"]]
        need(len(buys) == 1, "entry_fill_evidence_missing_or_ambiguous")
        entry, bought = buys[0], buys[0]["_qty"]
        sold = sum((o["_qty"] for o in sells), Decimal(0))
        need(sold <= bought, "exit_quantity_exceeds_entry")
        if not bought:
            row.update(
                entry_qty=0,
                exited_qty=0,
                remaining_qty=0,
                entry_order_ids=[entry["id"]],
                status="partial",
                reason="entry_not_filled",
                completion_status="NO_ENTRY_FILL_OBSERVED",
            )
            if signal["state"] == "CLOSED":
                row["reason"] = "native_state_reconciliation_required"
            return row
        # A cumulative order snapshot is not a fill-by-fill timestamp history.
        # Only a real filled_at is used, never an updated/submitted fallback.
        entry_time = timestamp(entry.get("filled_at"))
        exit_times = [timestamp(o.get("filled_at")) for o in sells]
        closed = sold == bought and all(o["status"] in TERMINAL for o in by_id.values())
        if closed:
            need(
                entry_time is not None
                and all(exit_times)
                and all(
                    instant(t) >= instant(entry_time)
                    and instant(t) <= instant(now + timedelta(seconds=5))
                    for t in exit_times
                ),
                "fill_timestamp_missing_or_invalid",
            )
        with localcontext() as arithmetic:
            arithmetic.prec = 80
            proceeds = sum((o["_qty"] * o["_price"] for o in sells), Decimal(0))
            exact_exit_average = proceeds / sold if sold else None
            exit_average = float(exact_exit_average) if sold else None
            exact_gross = proceeds - bought * entry["_price"]
            gross = float(exact_gross)
        row.update(
            entry_qty=float(bought),
            exited_qty=float(sold),
            remaining_qty=float(bought - sold),
            entry_avg_price=float(entry["_price"]),
            exit_avg_price=exit_average,
            entry_time=entry_time,
            exit_time=max(exit_times, key=instant) if closed else None,
            entry_order_ids=[entry["id"]],
            exit_order_ids=sorted(o["id"] for o in sells),
            status="partial",
            reason="position_not_fully_exited",
            completion_status="FULL_EXIT_OBSERVED"
            if closed
            else "PARTIAL_EXIT_OBSERVED"
            if sold
            else "NO_EXIT_OBSERVED",
        )
        if closed and signal["state"] == "CLOSED":
            row.update(status="complete", reason=None, gross_realized_pnl=gross)
            if include_exact:
                row["_exact"] = dict(
                    qty=bought,
                    entry_price=entry["_price"],
                    exit_price=exact_exit_average,
                    gross=exact_gross,
                    order_ids=sorted(by_id),
                    filled_orders=[
                        dict(
                            id=o["id"],
                            side=o["side"],
                            qty=o["_qty"],
                            price=o["_price"],
                            filled_at=timestamp(o["filled_at"]),
                        )
                        for o in by_id.values()
                        if o["_qty"]
                    ],
                )
        elif closed or signal["state"] == "CLOSED":
            row["reason"] = "native_state_reconciliation_required"
        return row
    except (
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        InvalidOperation,
        RecursionError,
    ) as error:
        # Sanitized code only, no payload, account, DSN or provider exception.
        failed = blank_row(signal)
        failed.update(strategy_id=row["strategy_id"], ownership_scope=row["ownership_scope"])
        failed["reason"] = str(error) if type(error) is EvidenceError else "incomplete_evidence"
        return failed


def unavailable(base, reason):
    return dict(
        base,
        ok=False,
        status="unavailable",
        reason=reason,
        rows=[],
        has_more=None,
        legacy_overlap_count=0,
    )


def snapshot(
    connection_factory, client, expected_identity, *, limit=25, read=bounded_read, now=None
):
    now = now or datetime.now(timezone.utc)
    limit = max(1, min(MAX_SIGNALS, int(limit)))
    base = dict(
        schema_version=SCHEMA,
        source="postgres_strategy_execution",
        trade_view="STRATEGY_EXECUTION_HISTORY",
        observed_at=now.isoformat(),
        source_as_of=None,
        freshness="stored_observations_not_live",
        account_scope="configured_paper_broker",
        broker_account_match="unavailable",
        evidence_basis=BASIS,
        performance_status="NOT_QUALIFIED_STRATEGY_HISTORY",
        accounting_complete=False,
        fees_status="UNKNOWN_UNALLOCATED",
        net_status="UNKNOWN",
        limit=limit,
    )
    # Canonical read identity contains deployment_id/database_id only. The
    # configured three-field disposition binding ends in an evidence reference,
    # not a database name; never reinterpret that reference as one.
    if client is None or type(expected_identity) is not tuple or len(expected_identity) != 2:
        return unavailable(base, "paper_read_context_unavailable")
    failure_stage = "broker_account_read"
    database_task_entered = Event()
    try:
        account = read(client.get_account, seconds=2)
        need(type(account.id) in (str, UUID), "broker_account_identity_invalid")
        account_id = text(str(account.id))
        need(str(UUID(account_id)) == account_id, "broker_account_identity_invalid")
        failure_stage = "database_history_read"

        def read_database():
            database_task_entered.set()
            return read_snapshot(connection_factory, account_id, expected_identity, limit)

        signals, records, overlap, has_more = read(read_database, seconds=7)
    except Exception as error:
        # Fixed phases/codes only. Never return SQL/provider exception text,
        # configuration, account IDs, database names or evidence references.
        known = {
            "database_identity_mismatch",
            "database_unavailable",
            "read_budget_exceeded",
            "row_budget_exceeded",
            "broker_account_identity_invalid",
        }
        reason = (
            str(error)
            if type(error) is EvidenceError and str(error) in known
            else "strategy_history_unavailable"
        )
        if type(error) is RuntimeError and str(error) in {"source_busy", "source_timeout"}:
            reason = str(error)
        # This is the observed state at response time, not a promise that a
        # timed-out worker cannot subsequently start or finish its bounded read.
        return dict(
            unavailable(base, reason),
            failure_stage=failure_stage,
            database_task_entered=database_task_entered.is_set(),
        )
    base["broker_account_match"] = "verified"
    legacy_ids = {
        r.get(k) for r in overlap for k in ("entry_order_id", "exit_order_id") if r.get(k)
    }
    rows, overlap_count = [], 0
    for signal in signals:
        orders = [record for record in records if record["signal_id"] == signal["signal_id"]]
        row = project_signal(signal, orders, account_id, now)
        # A malformed observation cannot hide a signal by merely naming an old
        # ledger order. Suppress only identities whose durable binding passed.
        if legacy_ids.intersection(row["entry_order_ids"] + row["exit_order_ids"]):
            overlap_count += 1
            continue
        rows.append(row)
    stamps = [timestamp(r.get("updated_at")) for r in signals + records]
    base["source_as_of"] = max((s for s in stamps if s), key=instant, default=None)
    partial = any(
        r["status"] == "incomplete" or r["reason"] == "native_state_reconciliation_required"
        for r in rows
    )
    return dict(
        base,
        ok=True,
        status="partial" if partial else "ready" if rows else "empty",
        reason="incomplete_strategy_evidence" if partial else None,
        rows=rows,
        has_more=has_more,
        legacy_overlap_count=overlap_count,
    )
