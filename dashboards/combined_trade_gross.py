"""One bounded, read-only cohort for Trades gross/list/rank presentations.

Legacy account association remains unverified. Strategy observations are bound
to the current paper account; neither source proves broker completeness or net.
Native policy/execution readers are reused, never replaced or initialized.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import re
import time
from uuid import UUID

from dashboards import execution_gross_projection as E
from dashboards import execution_read_transport as T
from dashboards import recorded_trade_analytics as R
from dashboards import strategy_trade_history as H
from scripts import gross_paper_accounting as G, trade_disposition as D

SCHEMA = "jbravo.trades-overview.v1"
SCOPE = "COMBINED_STORED_EXECUTION_GROSS"
MAX_SIGNALS, MAX_ORDERS, MAX_CLAIMS = 200, 1000, 1000

STRATEGY_VISIBILITY_SQL = (
    "SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,c.relispartition,"
    "EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),"
    "pg_catalog.has_table_privilege(current_user,c.oid,'SELECT') FROM pg_catalog.pg_class c "
    "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND "
    "c.relname IN ('strategy_execution_signals_app','strategy_execution_orders_app') ORDER BY c.relname LIMIT 3"
)

# Fixed full populations with overflow sentinels, including non-CLOSED owners.
# Versions are internal; no account keys, payloads or policy records are public.
SIGNALS_SQL = E.guarded(
    "SELECT pg_catalog.jsonb_build_object('row',pg_catalog.to_jsonb(s),'version',s.xmin::text)::text AS body "
    "FROM ONLY public.strategy_execution_signals_app s WHERE account_id=%s "
    "ORDER BY signal_id LIMIT 201"
)
ORDERS_SQL = E.guarded(
    "SELECT pg_catalog.jsonb_build_object('row',pg_catalog.to_jsonb(o),'version',o.xmin::text)::text AS body "
    "FROM ONLY public.strategy_execution_orders_app o ORDER BY client_order_id LIMIT 1001"
)
CLAIMS_SQL = E.guarded(
    "SELECT pg_catalog.jsonb_build_object('row',pg_catalog.jsonb_build_object('trade_id',trade_id,"
    "'status',status,'entry_order_id',entry_order_id::text,'exit_order_id',exit_order_id::text),"
    "'version',xmin::text)::text AS body FROM ONLY public.trades ORDER BY trade_id LIMIT 1001"
)
# Batch only the four candidate-owned reads. Native policy readers and their
# final canonical replay remain untouched; the two full cohort passes remain.
STRATEGY_COHORT_SQL = (
    "WITH visibility(relname,relkind,relrowsecurity,relforcerowsecurity,relispartition,inherits,selectable) AS MATERIALIZED ("
    + STRATEGY_VISIBILITY_SQL
    + "), "
    "signals(body) AS MATERIALIZED (" + SIGNALS_SQL + "), "
    "orders(body) AS MATERIALIZED (" + ORDERS_SQL + "), "
    "claims(body) AS MATERIALIZED (" + CLAIMS_SQL + ") "
    "SELECT 'visibility'::text AS section,pg_catalog.row_number() OVER (ORDER BY relname)::text AS ordinal,"
    "pg_catalog.json_build_array(relname,relkind,relrowsecurity,relforcerowsecurity,relispartition,"
    "inherits,selectable)::text AS body FROM visibility UNION ALL "
    "SELECT 'signals',pg_catalog.row_number() OVER (ORDER BY body::jsonb->'row'->>'signal_id')::text,body FROM signals UNION ALL "
    "SELECT 'orders',pg_catalog.row_number() OVER (ORDER BY body::jsonb->'row'->>'client_order_id')::text,body FROM orders UNION ALL "
    "SELECT 'claims',pg_catalog.row_number() OVER (ORDER BY (body::jsonb->'row'->>'trade_id')::bigint)::text,body FROM claims"
)


# Fixed pairs of existing projection queries, never native policy queries.
GUARD_PAIR_SQL = (
    "WITH visibility(c0,c1,c2,c3,c4,c5,c6) AS MATERIALIZED ("
    + E.VISIBILITY_SQL
    + "), schema_rows(c0,c1,c2) AS MATERIALIZED ("
    + E.SCHEMA_SQL
    + ") "
    "SELECT 'visibility'::text AS section,pg_catalog.row_number() OVER (ORDER BY c0)::text AS ordinal,"
    "pg_catalog.json_build_array(c0,c1,c2,c3,c4,c5,c6)::text AS body FROM visibility UNION ALL "
    "SELECT 'schema',pg_catalog.row_number() OVER (ORDER BY c0,c1)::text,"
    "pg_catalog.json_build_array(c0,c1,c2)::text FROM schema_rows"
)
EXECUTION_PAIR_SQL = (
    "WITH owners(body) AS MATERIALIZED ("
    + E.OWNERS_SQL
    + "), fills(body) AS MATERIALIZED ("
    + E.FILLS_SQL
    + ") "
    "SELECT 'owners'::text AS section,pg_catalog.row_number() OVER "
    "(ORDER BY (body::jsonb->'row'->>'trade_id')::bigint)::text AS ordinal,body FROM owners UNION ALL "
    "SELECT 'fills',pg_catalog.row_number() OVER (ORDER BY body::jsonb->'row'->>'activity_id')::text,body FROM fills"
)


PLANS = {
    GUARD_PAIR_SQL: dict(kind="canonical", arity=3, order="q.c0,q.c1::bigint"),
    EXECUTION_PAIR_SQL: dict(kind="canonical", arity=3, order="q.c0,q.c1::bigint"),
    STRATEGY_COHORT_SQL: dict(kind="canonical", arity=3, order="q.c0,q.c1::bigint"),
    STRATEGY_VISIBILITY_SQL: dict(kind="canonical", arity=7, order="q.c0"),
    SIGNALS_SQL: dict(kind="canonical", arity=1, order="q.c0::jsonb->'row'->>'signal_id'"),
    ORDERS_SQL: dict(kind="canonical", arity=1, order="q.c0::jsonb->'row'->>'client_order_id'"),
    CLAIMS_SQL: dict(kind="canonical", arity=1, order="(q.c0::jsonb->'row'->>'trade_id')::bigint"),
}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def strategy_documents(cursor, sql, params, cap):
    """Keep JSON numeric types exactly as the native order-hash reader does.

    Disposition document normalization is deliberately not used here: changing
    a request float to Decimal/string would invalidate the native request hash.
    """
    cursor.execute(sql, params)
    wire = cursor.fetchall()
    H.need(len(wire) <= cap and all(type(row[0]) is str for row in wire), "population_overflow")
    return [json.loads(row[0]) for row in wire]


def strategy_cohort(cursor, account):
    cursor.execute(STRATEGY_COHORT_SQL, (account,))
    groups = {name: [] for name in ("visibility", "signals", "orders", "claims")}
    caps = dict(visibility=3, signals=MAX_SIGNALS, orders=MAX_ORDERS, claims=MAX_CLAIMS)
    for row in cursor.fetchall():
        H.need(
            len(row) == 3
            and row[0] in groups
            and type(row[1]) is str
            and row[1].isascii()
            and row[1].isdigit()
            and len(row[1]) <= 4
            and type(row[2]) is str,
            "strategy_cohort_wire_invalid",
        )
        section, ordinal, body = row
        H.need(int(ordinal) == len(groups[section]) + 1, "strategy_cohort_order_invalid")
        H.need(len(groups[section]) < caps[section], "population_overflow")
        # Keep strategy request numeric types; legacy claim normalization stays
        # identical to E.documents. Null/oversize SQL sentinels fail above.
        groups[section].append(D.document(body) if section == "claims" else json.loads(body))
    H.need(
        groups["visibility"]
        == [
            ["strategy_execution_orders_app", "r", False, False, False, False, True],
            ["strategy_execution_signals_app", "r", False, False, False, False, True],
        ],
        "strategy_source_visibility_unavailable",
    )
    return groups["signals"], groups["orders"], groups["claims"]


class LegacyInputCapture:
    """One-pass pairing; unchanged native readers consume exact original rows.

    Nothing is reused across cohort passes. Native policy replay is untouched.
    Each batch is fully charged/drained by T before its original rows are exposed.
    """

    def __init__(self, cursor):
        self.cursor, self.sql, self.fills = cursor, None, None
        self.pending = None
        self.rows = None
        H.need(
            not (
                {E.VISIBILITY_SQL, E.SCHEMA_SQL, E.OWNERS_SQL, E.FILLS_SQL} & cursor.canonical_sql
            ),
            "projection_pair_native_overlap",
        )

    def pair(self, sql, params, caps):
        self.cursor.execute(sql, params)
        groups = {key: [] for key in caps}
        for row in self.cursor.fetchall():
            H.need(
                len(row) == 3
                and row[0] in groups
                and type(row[1]) is str
                and row[1].isascii()
                and row[1].isdigit()
                and len(row[1]) <= 4
                and type(row[2]) is str,
                "projection_pair_wire_invalid",
            )
            section, ordinal, body = row
            H.need(int(ordinal) == len(groups[section]) + 1, "projection_pair_order_invalid")
            H.need(len(groups[section]) < caps[section], "population_overflow")
            if section in ("visibility", "schema"):
                values = json.loads(body)
                H.need(
                    type(values) is list
                    and len(values) == (7 if section == "visibility" else 3)
                    and all(type(v) in (str, int, bool, type(None)) for v in values),
                    "projection_pair_guard_invalid",
                )
                groups[section].append(tuple(values))
            else:
                groups[section].append((body,))
        return groups

    def execute(self, sql, params=None):
        H.need(self.rows is None, "projection_pair_unconsumed")
        self.sql = sql
        if self.pending is not None:
            expected, expected_params, rows = self.pending
            H.need(
                sql == expected and params == expected_params, "projection_pair_sequence_changed"
            )
            self.rows, self.pending = rows, None
        elif sql == E.VISIBILITY_SQL:
            H.need(params is None, "projection_pair_params_invalid")
            groups = self.pair(GUARD_PAIR_SQL, None, dict(visibility=3, schema=65))
            self.rows = groups["visibility"]
            self.pending = (E.SCHEMA_SQL, None, groups["schema"])
        elif sql == E.OWNERS_SQL:
            H.need(
                type(params) is tuple
                and len(params) == 3
                and all(type(p) is list for p in params)
                and params[1] == params[2],
                "projection_pair_params_invalid",
            )
            # Native read_inputs derives both owner queries and fills from this
            # same sorted order population. Require the subsequent call exactly.
            groups = self.pair(
                EXECUTION_PAIR_SQL,
                (*params, params[1]),
                dict(owners=E.MAX_OWNERS, fills=E.MAX_FILLS),
            )
            self.rows = groups["owners"]
            self.pending = (E.FILLS_SQL, (list(params[1]),), groups["fills"])
        else:
            H.need(
                sql in (E.LOCK_SQL, E.TRADES_SQL) and params is None,
                "projection_pair_sequence_changed",
            )
            return self.cursor.execute(sql, params)

    def fetchall(self):
        if self.rows is None:
            rows = self.cursor.fetchall()
        else:
            rows, self.rows = self.rows, None
        if self.sql == E.FILLS_SQL:
            self.fills, _ = E.split_versions([D.document(row[0]) for row in rows], "activity_id")
        return rows

    def __getattr__(self, name):
        return getattr(self.cursor, name)


def read_cohort(connection_factory, identity, account):
    """Same transaction and finite transport budget as native execution gross.

    Ten fixed cohort SELECTs plus the existing finite canonical policy plan
    and final replay. Every population/version is compared before returning.
    """
    connection = None
    started = time.monotonic()
    try:
        connection = connection_factory()
        H.need(
            connection is not None
            and connection.autocommit is False
            and connection.get_transaction_status() == 0,
            "database_unavailable",
        )
        with connection.cursor() as raw:
            raw.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY")
            cursor = T.Cursor(raw, started + 8, extra_plans={**E.READ_PLANS, **PLANS})
            # Exact native optional optimization, never a policy/row shortcut.
            reuse = getattr(cursor, "enable_readonly_guard_reuse", None)
            if callable(reuse):
                reuse()
            cursor.execute("SET LOCAL lock_timeout = '500ms'")
            cursor.execute("SET LOCAL idle_in_transaction_session_timeout = '8000ms'")
            policy = D.read_policy(cursor, identity, scope="VALIDATED_PERFORMANCE")

            def inputs():
                capture = LegacyInputCapture(cursor)
                records, execution = E.read_inputs(capture, policy)
                H.need(capture.fills is not None, "legacy_fill_capture_unavailable")
                signals, orders, claims = strategy_cohort(cursor, account)
                # split_versions also validates the mandatory xmin envelope.
                signal_rows, _ = E.split_versions(signals, "signal_id")
                order_rows, _ = E.split_versions(orders, "client_order_id")
                claim_rows, _ = E.split_versions(claims, "trade_id")
                binding = (
                    execution.records_hash,
                    execution.source_hash,
                    execution.policy_hash,
                    fingerprint([signals, orders, claims]),
                )
                return (
                    records,
                    execution,
                    signal_rows,
                    order_rows,
                    claim_rows,
                    capture.fills,
                    binding,
                )

            first = inputs()
            second = inputs()
            H.need(first[-1] == second[-1], "cohort_changed")
            cursor.recheck_canonical()
            cursor.check()
            return policy, *first[:-1], fingerprint(first[-1])
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()


def decimal_text(value):
    return format(value, "f") if value is not None else None


def optional_number(value):
    try:
        result = Decimal(str(value))
        return (
            result
            if (
                result.is_finite()
                and len(str(value)) <= 96
                and -80 <= result.as_tuple().exponent <= 80
                and (not result or -80 <= result.adjusted() <= 80)
            )
            else None
        )
    except (ValueError, TypeError, ArithmeticError):
        return None


def holding_days(entered, exited):
    if entered is None or exited is None or H.instant(exited) < H.instant(entered):
        return None
    return (datetime.fromisoformat(exited[:10]) - datetime.fromisoformat(entered[:10])).days


def common_row(key, source, symbol, state, *, trade_id=None, signal_id=None):
    return dict(
        row_id=key,
        source_kind=source,
        trade_id=trade_id,
        signal_id=signal_id,
        symbol=symbol,
        state=state,
        inclusion="UNKNOWN",
        reason=None,
        account_binding="verified" if source == "STRATEGY_EXECUTION" else "unverified",
        entry_time=None,
        exit_time=None,
        recorded_entry_time=None,
        recorded_exit_time=None,
        holding_days=None,
        qty=None,
        entry_price=None,
        exit_price=None,
        gross_pnl=None,
        recorded_pnl=None,
        fees=None,
        net_pnl=None,
        price_basis="STORED_ORDER_FILL_AVERAGES"
        if source == "STRATEGY_EXECUTION"
        else "RECORDED_LEDGER",
        ownership_scope=None,
    )


def legacy_rows(policy, records, execution, identity, now):
    """Apply the same per-row eligibility gates as the native gross projection."""
    execution.verify(policy, records)
    # Preserve the complete native policy/allocation validation, not just totals.
    native = R.project(policy, records, identity=identity, now=now, execution=execution)
    allocations = {item.trade_id: item for item in policy.allocations}
    rows, economics, order_sets = [], {}, {}
    for record in records:
        identifier = D.positive_id(record.get("trade_id"))
        row = common_row(
            f"legacy:{identifier}",
            "LEGACY_EXECUTION",
            record.get("symbol"),
            record.get("status"),
            trade_id=identifier,
        )
        row.update(
            recorded_entry_time=H.timestamp(record.get("entry_time")),
            recorded_exit_time=H.timestamp(record.get("exit_time")),
            recorded_pnl=decimal_text(optional_number(record.get("realized_pnl"))),
        )
        if not isinstance(row["symbol"], str) or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9./^-]{0,31}", row["symbol"]
        ):
            row["symbol"] = None
        order_sets[identifier] = (
            set(allocations[identifier].order_ids)
            if identifier in allocations
            else {
                value
                for key in ("entry_order_id", "exit_order_id")
                if (value := record.get(key)) not in (None, "")
            }
        )
        if identifier in policy.excluded_ids:
            row.update(inclusion="POLICY_EXCLUDED", reason="LEGACY_POLICY_EXCLUDED")
            rows.append(row)
            continue
        if identifier in allocations:
            G.A.validate_bound_trade(allocations[identifier], record)
        reason = execution.unknown.get(identifier)
        H.need(identifier in execution.gross or reason is not None, "legacy_projection_incomplete")
        clock = execution.completion.get(identifier)
        entered, recorded_exit = R._time(record.get("entry_time")), R._time(record.get("exit_time"))
        exited = clock.completed_at if clock is not None else None
        if not isinstance(record.get("symbol"), str) or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9./^-]{0,31}", record["symbol"]
        ):
            reason = "SYMBOL_UNKNOWN"
        if (
            entered is None
            or recorded_exit is None
            or recorded_exit < entered
            or recorded_exit > now
            or exited is not None
            and (exited < entered or exited > now)
        ):
            reason = "TIME_UNKNOWN_OR_INVALID"
        try:
            G._number(record.get("qty"), positive=False, nonzero=True)
        except D.TradePolicyError:
            reason = "QTY_UNKNOWN_OR_INVALID"
        if reason:
            row["reason"] = reason
        else:
            economics[row["row_id"]] = execution.gross[identifier]
            row.update(
                inclusion="INCLUDED",
                gross_pnl=decimal_text(execution.gross[identifier]),
                entry_time=H.timestamp(clock.started_at),
                exit_time=H.timestamp(exited),
                qty=decimal_text(optional_number(record.get("qty"))),
                entry_price=decimal_text(optional_number(record.get("entry_price"))),
                exit_price=decimal_text(optional_number(record.get("exit_price"))),
            )
            row["holding_days"] = holding_days(row["entry_time"], row["exit_time"])
        rows.append(row)
    # Cross-check duplicated eligibility logic against the native full cohort.
    known = [row for row in rows if row["inclusion"] == "INCLUDED"]
    H.need(len(known) == native["coverage"]["included"], "legacy_eligibility_changed")
    total = G._gross_sum(list(economics.values()))
    expected = next(row for row in native["rows"] if row["key"] == "all")["totalPL"]
    H.need(
        expected is None and not known or expected is not None and Decimal(expected) == total,
        "legacy_eligibility_changed",
    )
    return rows, economics, order_sets


def ownership_claims(policy, claims, orders):
    physical, strategy = defaultdict(set), defaultdict(set)
    reserved = D.document(policy.allocation_metadata_json).get("reserved_order_owners")
    H.need(type(reserved) is dict, "policy_reservations_unavailable")
    for order, owner in reserved.items():
        physical[E.order_id(order)].add(D.positive_id(owner))
    seen = set()
    for claim in claims:
        identifier = D.positive_id(claim.get("trade_id"))
        H.need(identifier not in seen, "duplicate_legacy_claim")
        seen.add(identifier)
        for field in ("entry_order_id", "exit_order_id"):
            if claim.get(field) not in (None, ""):
                physical[E.order_id(claim[field])].add(identifier)
    seen = set()
    for order in orders:
        key = H.text(order.get("client_order_id"))
        H.need(key not in seen, "duplicate_strategy_claim")
        seen.add(key)
        owner = (H.text(order.get("account_id")), H.text(order.get("signal_id")))
        observation, request = order.get("observation"), order.get("request")
        if isinstance(observation, dict) and observation.get("id") not in (None, ""):
            strategy[E.order_id(observation["id"])].add(owner)
        if isinstance(request, dict) and request.get("order_id") not in (None, ""):
            strategy[E.order_id(request["order_id"])].add(owner)
    return physical, strategy


def identical_fill_economics(prior, strategy, exact, order_ids, fills):
    """Never deduplicate by an equal gross number alone.

    Native legacy clocks are the canonical range basis. Require exact instant
    equality, including nanoseconds; a merely truncation-compatible clock stays
    uncertain rather than changing range membership or inventing precision.
    """
    if fills is None or prior["symbol"] != strategy["symbol"]:
        return False
    if any(
        H.instant(prior[key]) != H.instant(strategy[key]) for key in ("entry_time", "exit_time")
    ):
        return False
    unique = {}
    for fill in fills:
        if fill.get("order_id") in order_ids:
            key = fill["activity_id"]
            if key in unique and unique[key] != fill:
                return False
            unique[key] = fill
    incoming = [fill for fill in unique.values() if fill.get("side") == "buy"]
    outgoing = [fill for fill in unique.values() if fill.get("side") == "sell"]
    if not incoming or not outgoing or len(incoming) + len(outgoing) != len(unique):
        return False
    if set(fill["order_id"] for fill in unique.values()) != order_ids:
        return False
    for fill in unique.values():
        if fill.get("symbol") != strategy["symbol"]:
            return False
        # A retained native raw clock must agree exactly, not merely share its
        # microsecond projection. Ambiguity never proves identical completion.
        if H.timestamp(fill.get("provider_transaction_time")) is None or H.instant(
            fill["provider_transaction_time"]
        ) != H.instant(fill["transaction_time"]):
            return False

    def quantity(group):
        return G._gross_sum([E.exact(fill["qty"], positive=True) for fill in group])

    def cash(group):
        return G._gross_sum(
            [
                G._gross_from_prices(
                    E.exact(fill["price"], positive=True),
                    Decimal(0),
                    E.exact(fill["qty"], positive=True),
                )
                for fill in group
            ]
        )

    per_order = exact["filled_orders"]
    if {order["id"] for order in per_order} != order_ids:
        return False
    for order in per_order:
        group = [fill for fill in unique.values() if fill["order_id"] == order["id"]]
        if (
            not group
            or any(fill["side"] != order["side"] for fill in group)
            or quantity(group) != order["qty"]
            or cash(group) != G._gross_from_prices(order["price"], Decimal(0), order["qty"])
            or max(H.instant(fill["transaction_time"]) for fill in group)
            != H.instant(order["filled_at"])
        ):
            return False
    cost = G._gross_from_prices(exact["entry_price"], Decimal(0), exact["qty"])
    proceeds = G._gross_sum([cost, exact["gross"]])
    return (
        quantity(incoming) == quantity(outgoing) == exact["qty"]
        and cash(incoming) == cost
        and cash(outgoing) == proceeds
        and min(H.instant(fill["transaction_time"]) for fill in incoming)
        == H.instant(strategy["entry_time"])
        and max(H.instant(fill["transaction_time"]) for fill in outgoing)
        == H.instant(strategy["exit_time"])
    )


def project(
    policy,
    records,
    execution,
    signals,
    orders,
    claims,
    identity,
    account,
    now,
    cohort_id,
    *,
    legacy_fills=None,
):
    H.need(
        len(signals) <= MAX_SIGNALS and len(orders) <= MAX_ORDERS and len(claims) <= MAX_CLAIMS,
        "population_overflow",
    )
    rows, economics, legacy_orders = legacy_rows(policy, records, execution, identity, now)
    physical, strategy = ownership_claims(policy, claims, orders)
    legacy_by_id = {row["trade_id"]: row for row in rows}
    H.need(set(legacy_by_id) <= {row["trade_id"] for row in claims}, "legacy_claims_incomplete")
    grouped = defaultdict(list)
    for order in orders:
        grouped[(order["account_id"], order["signal_id"])].append(order)
    source_rows, seen, deduplicated = [], set(), 0
    not_closed = 0
    for signal in signals:
        sid = signal["signal_id"]
        H.need(
            type(sid) is str
            and re.fullmatch("[0-9a-f]{64}", sid)
            and sid not in seen
            and signal["account_id"] == account
            and signal["state"] in H.STATES,
            "strategy_population_invalid",
        )
        seen.add(sid)
        value = H.project_signal(signal, grouped[(account, sid)], account, now, include_exact=True)
        exact = value.pop("_exact", None)
        value.update(combined_inclusion="NOT_CLOSED", combined_reason="NATIVE_POSITION_NOT_CLOSED")
        source_rows.append(value)
        if signal["state"] != "CLOSED":
            not_closed += 1
            continue
        row = common_row(
            "strategy:" + sid, "STRATEGY_EXECUTION", value["symbol"], value["state"], signal_id=sid
        )
        row["ownership_scope"] = value["ownership_scope"]
        reason = value["reason"] or "STRATEGY_EVIDENCE_INCOMPLETE"
        if (
            value["status"] == "complete"
            and value["completion_status"] == "FULL_EXIT_OBSERVED"
            and exact
        ):
            order_ids = set(exact["order_ids"])
            conflicting = any(strategy[oid] != {(account, sid)} for oid in order_ids)
            legacy_owners = set().union(*(physical[oid] for oid in order_ids))
            if conflicting:
                reason = "COMPETING_STRATEGY_ORDER_CLAIM"
            elif legacy_owners:
                economic_ids = set(value["entry_order_ids"] + value["exit_order_ids"])
                owner = next(iter(legacy_owners)) if len(legacy_owners) == 1 else None
                prior = legacy_by_id.get(owner)
                identical = (
                    prior is not None
                    and prior["inclusion"] == "INCLUDED"
                    and economic_ids == legacy_orders[owner]
                    and all(physical[oid] == {owner} for oid in economic_ids)
                    and economics[prior["row_id"]] == exact["gross"]
                    and optional_number(prior["qty"]) == exact["qty"]
                    and identical_fill_economics(prior, value, exact, economic_ids, legacy_fills)
                )
                if identical:
                    deduplicated += 1
                    value.update(
                        combined_inclusion="EXCLUDED", combined_reason="IDENTICAL_LEGACY_EXECUTION"
                    )
                    continue
                reason = "LEGACY_ORDER_CLAIM_RESERVED"
                if any(owner in policy.excluded_ids for owner in legacy_owners):
                    row["inclusion"] = "POLICY_EXCLUDED"
                    reason = "LEGACY_POLICY_ORDER_RESERVED"
            else:
                row.update(
                    inclusion="INCLUDED",
                    reason=None,
                    entry_time=value["entry_time"],
                    exit_time=value["exit_time"],
                    qty=decimal_text(exact["qty"]),
                    entry_price=decimal_text(exact["entry_price"]),
                    exit_price=decimal_text(exact["exit_price"]),
                    gross_pnl=decimal_text(exact["gross"]),
                )
                row["holding_days"] = holding_days(row["entry_time"], row["exit_time"])
                economics[row["row_id"]] = exact["gross"]
        if row["inclusion"] != "INCLUDED":
            row["reason"] = reason
        value.update(
            combined_inclusion="INCLUDED"
            if row["inclusion"] == "INCLUDED"
            else "EXCLUDED"
            if row["inclusion"] == "POLICY_EXCLUDED"
            else "UNKNOWN",
            combined_reason=row["reason"],
        )
        rows.append(row)
    # A strategy claim must not make an ambiguous physical owner appear unique.
    # Exact duplicate representations above are safe; other cross-source claims
    # remove the legacy contribution rather than pick a financial winner.
    for row in rows:
        if row["source_kind"] != "LEGACY_EXECUTION" or row["inclusion"] != "INCLUDED":
            continue
        related = set().union(*(strategy[oid] for oid in legacy_orders[row["trade_id"]]))
        if related and any(
            owner != account
            or not any(
                s["signal_id"] == sid and s["combined_reason"] == "IDENTICAL_LEGACY_EXECUTION"
                for s in source_rows
            )
            for owner, sid in related
        ):
            row.update(inclusion="UNKNOWN", reason="COMPETING_STRATEGY_ORDER_CLAIM", gross_pnl=None)
            economics.pop(row["row_id"], None)
    return summarize(rows, economics, source_rows, not_closed, deduplicated, now, cohort_id)


def summarize(rows, economics, source_rows, not_closed, deduplicated, now, cohort_id):
    def selected(row, key):
        if key == "all":
            return True
        stamp = row["exit_time"] if row["inclusion"] == "INCLUDED" else None
        return stamp is None or H.instant(now - timedelta(days=R.PERIODS[key])) <= H.instant(
            stamp
        ) <= H.instant(now)

    def coverage(group):
        def counts(items):
            return dict(
                included=sum(r["inclusion"] == "INCLUDED" for r in items),
                excluded=sum(r["inclusion"] == "POLICY_EXCLUDED" for r in items),
                unknown=sum(r["inclusion"] == "UNKNOWN" for r in items),
            )

        return dict(
            **counts(group),
            deduplicated=deduplicated,
            strategy_not_closed=not_closed,
            by_source={
                name: counts([r for r in group if r["source_kind"] == source])
                for name, source in [
                    ("legacy", "LEGACY_EXECUTION"),
                    ("strategy", "STRATEGY_EXECUTION"),
                ]
            },
        )

    stats, rankings = [], {}
    for key in R.PERIODS:
        group = [row for row in rows if selected(row, key)]
        known = [row for row in group if row["inclusion"] == "INCLUDED"]
        covered = coverage(group)
        total = G._gross_sum([economics[row["row_id"]] for row in known])
        symbol_values = defaultdict(list)
        for row in known:
            symbol_values[row["symbol"]].append(economics[row["row_id"]])
        symbols = {symbol: G._gross_sum(values) for symbol, values in symbol_values.items()}
        ordered = sorted(known, key=lambda row: (economics[row["row_id"]], row["row_id"]))

        def extreme(row):
            return dict(symbol=row["symbol"] if row else "--", pl=row["gross_pnl"] if row else None)

        stats.append(
            dict(
                key=key,
                label=R.LABELS[key],
                tradesCount=len(known),
                totalPL=decimal_text(total) if known or not covered["unknown"] else None,
                winRatePct=100 * sum(economics[r["row_id"]] > 0 for r in known) / len(known)
                if known
                else None,
                topTrade=extreme(ordered[-1] if ordered else None),
                worstLoss=extreme(ordered[0] if ordered else None),
                coverage=covered,
            )
        )
        rankings[key] = {}
        for mode in ("winners", "losers"):
            ranked = [
                (symbol, value)
                for symbol, value in symbols.items()
                if (value > 0 if mode == "winners" else value < 0)
            ]
            ranked.sort(
                key=lambda item: (item[1].copy_negate() if mode == "winners" else item[1], item[0])
            )
            rankings[key][mode] = [
                dict(rank=index + 1, symbol=symbol, pl=decimal_text(value))
                for index, (symbol, value) in enumerate(ranked[:12])
            ]
    rows.sort(
        key=lambda row: (
            H.instant(row["exit_time"] or row["recorded_exit_time"])
            if row["exit_time"] or row["recorded_exit_time"]
            else -(10**30),
            row["row_id"],
        ),
        reverse=True,
    )
    source_rows.sort(
        key=lambda row: (
            H.instant(row["updated_at"]) if row["updated_at"] else -(10**30),
            row["signal_id"],
        ),
        reverse=True,
    )
    covered = coverage(rows)
    result = envelope(now)
    result.update(
        ok=True,
        status="partial" if covered["unknown"] else "ready" if rows else "empty",
        cohort_id=cohort_id,
        population_complete=True,
        coverage=covered,
        rows=rows,
        stats=stats,
        rankings=rankings,
        strategy_account_binding="verified",
        policy_validation_status="VALIDATED_POLICY",
        source_as_of=max(
            (r["exit_time"] for r in rows if r["inclusion"] == "INCLUDED"),
            key=H.instant,
            default=None,
        ),
    )
    result["strategy_history"] = dict(
        schema_version=H.SCHEMA,
        ok=True,
        status="partial"
        if any(r["status"] == "incomplete" for r in source_rows)
        else "ready"
        if source_rows
        else "empty",
        source="postgres_strategy_execution",
        trade_view="STRATEGY_EXECUTION_HISTORY",
        evidence_basis=H.BASIS,
        freshness="stored_observations_not_live",
        broker_account_match="verified",
        accounting_complete=False,
        fees_status="UNKNOWN_UNALLOCATED",
        net_status="UNKNOWN",
        observed_at=now.isoformat(),
        source_as_of=max(
            (r["updated_at"] for r in source_rows if r["updated_at"]), key=H.instant, default=None
        ),
        limit=25,
        has_more=len(source_rows) > 25,
        # v1 counts lifecycle rows omitted for legacy overlap. This embedded
        # view omits none: every overlap retains its explicit combined reason.
        legacy_overlap_count=0,
        account_scope="configured_paper_broker",
        performance_status="NOT_QUALIFIED_STRATEGY_HISTORY",
        rows=source_rows[:25],
    )
    return result


def envelope(now):
    return dict(
        schema_version=SCHEMA,
        ok=False,
        status="unavailable",
        reason=None,
        cohort_id=None,
        source="postgres",
        view_scope=SCOPE,
        performance_status="COMBINED_GROSS_ONLY",
        strategy_account_binding="unavailable",
        legacy_account_binding="unverified",
        policy_validation_status="UNAVAILABLE",
        population_complete=False,
        observed_at=now.isoformat(),
        source_as_of=None,
        source_freshness="STORED_EVIDENCE_NOT_LIVE_BROKER_COMPLETENESS",
        qualified_net_status="NOT_ESTABLISHED",
        net_eligible=False,
        accounting_complete=False,
        recovery_complete=False,
        entry_ready=False,
        fees=None,
        net_pnl=None,
        fees_status="UNKNOWN",
        net_status="UNKNOWN",
        coverage=None,
        rows=[],
        stats=[],
        rankings={},
        strategy_history=None,
    )


def context_unavailable():
    result = envelope(datetime.now(timezone.utc))
    result.update(reason="paper_context_unavailable", failure_stage="paper_context")
    return result


def snapshot(connection_factory, client, identity, *, now=None, read=H.bounded_read):
    now = now or datetime.now(timezone.utc)
    result = envelope(now)
    stage = "paper_context"
    try:
        H.need(
            type(identity) is tuple
            and len(identity) == 2
            and all(type(v) is str and v for v in identity)
            and client is not None,
            "paper_context_unavailable",
        )
        stage = "broker_account_read"
        account = read(client.get_account, seconds=2).id
        H.need(type(account) in (str, UUID), "invalid_account_identity")
        account = str(account)
        H.need(str(UUID(account)) == account, "invalid_account_identity")
        stage = "database_cohort_read"
        cohort = read(lambda: read_cohort(connection_factory, identity, account), seconds=10)
        policy, records, execution, signals, orders, claims, legacy_fills, cohort_id = cohort
        stage = "cohort_projection"
        return project(
            policy,
            records,
            execution,
            signals,
            orders,
            claims,
            identity,
            account,
            now,
            cohort_id,
            legacy_fills=legacy_fills,
        )
    except Exception as error:
        code = str(error)
        if code in {
            "EXECUTION_SQL_OVERSIZE_OR_NULL",
            "EXECUTION_SHARED_BUDGET",
            "EXECUTION_SATURATION_OR_OVERSIZE",
        }:
            code = "population_overflow"
        result.update(
            reason=code
            if code
            in {
                "source_busy",
                "source_timeout",
                "population_overflow",
                "cohort_changed",
                "paper_context_unavailable",
                "invalid_account_identity",
                "database_unavailable",
            }
            else "combined_gross_unavailable",
            failure_stage=stage,
        )
        return result
