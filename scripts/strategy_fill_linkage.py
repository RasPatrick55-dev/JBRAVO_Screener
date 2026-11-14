"""Checkpoint-backed PAPER fill linkage. No network, DDL, or writes on import.

Independent of order execution/monitoring. Does not calculate lot basis or P&L.
The host ingestor remains fetch_account_activities; this module consumes only
its independently verified ledger/checkpoint evidence.
"""

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from uuid import UUID

VERSION = "jbravo.strategy-fill-linkage.v1"
MAX_ROWS = 200000


def require(ok, code):
    if not ok:
        raise ValueError("FILL_LINK_" + code)


def number(value, *, positive=False):
    require(not isinstance(value, (bool, float)), "EXACT_DECIMAL_REQUIRED")
    try:
        result = Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        raise ValueError("FILL_LINK_NUMBER_INVALID") from None
    require(
        result.is_finite() and (result > 0 if positive else result >= 0),
        "NUMBER_INVALID",
    )
    return result


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def uuid(value):
    try:
        require(isinstance(value, str) and str(UUID(value)) == value, "UUID_INVALID")
    except (ValueError, AttributeError):
        raise ValueError("FILL_LINK_UUID_INVALID") from None
    return value


def build_plan(
    *,
    account_id,
    signals,
    orders,
    fills,
    checkpoint,
    expected_source,
    expected_recipe,
    checkpoint_rows,
    prior_proofs=(),
):
    """Pure validated plan. checkpoint_rows must be the full committed manifest.

    All raw fields use the deployed checkpoint module's canonicalization. A
    missing fill is reported, never manufactured from filled_avg_price. The
    caller must re-read inputs under locks before writing a stale plan.
    """
    from scripts import activity_ingestion_checkpoint as cp

    uuid(account_id)
    require(
        expected_source == cp.source_fingerprint(cp.ENDPOINT, account_id),
        "ACCOUNT_SOURCE_MISMATCH",
    )
    require(
        len(signals) <= MAX_ROWS and len(orders) <= MAX_ROWS and len(fills) <= MAX_ROWS,
        "ROW_LIMIT",
    )
    require(len(prior_proofs) <= 512, "PROOF_LIMIT")
    manifested = {}
    provenance = {}
    proofs = list(prior_proofs) + [(checkpoint, checkpoint_rows)]
    for proof, rows in proofs:
        require(
            proof["source_fingerprint"] == expected_source
            and proof["recipe_sha256"] == expected_recipe
            and proof["query_axis"] == "ALPACA_CREATED_AT",
            "CHECKPOINT_SOURCE_MISMATCH",
        )
        uuid(proof["checkpoint_id"])
        require(
            cp.utc(proof["query_after"]) < cp.utc(proof["query_until"]),
            "CHECKPOINT_WINDOW_INVALID",
        )
        manifest = cp.manifestation(rows)
        require(
            manifest == proof["persisted_manifest"]
            and len(manifest) == proof["fetched_count"]
            and cp.digest(manifest) == proof["manifest_sha256"],
            "CHECKPOINT_LEDGER_MISMATCH",
        )
        for item in manifest:
            key = (item["activity_id"], item["activity_type"])
            require(
                key not in manifested or manifested[key] == item["sha256"],
                "CHECKPOINT_CONFLICT",
            )
            manifested[key] = item["sha256"]
            provenance.setdefault(key, proof["checkpoint_id"])
    by_signal = {}
    for s in signals:
        require(
            s["account_id"] == account_id and s["signal_id"] not in by_signal,
            "SIGNAL_IDENTITY",
        )
        p = s["payload"]
        h = p["handoff"]
        require(
            p["proposal"]["signal_id"] == s["signal_id"]
            and p["proposal"]["symbol"] == s["symbol"]
            and h["paper_only"] is True,
            "SIGNAL_LINEAGE",
        )
        by_signal[s["signal_id"]] = s
    by_order = {}
    for o in orders:
        require(
            o["account_id"] == account_id and o["signal_id"] in by_signal,
            "ORDER_SIGNAL_MISSING",
        )
        if o["operation"] == "DELETE":
            continue  # cancellation operation is not another economic order
        require(o["operation"] in {"POST", "PATCH"}, "ORDER_OPERATION")
        if o["observation"] is None:
            continue  # claimed/unknown/rejected is not an invented broker order
        d = o["observation"]
        s = by_signal[o["signal_id"]]
        h = s["payload"]["handoff"]
        require(
            o["state"] == "OBSERVED"
            and o["role"] in {"entry", "stop", "partial", "exit"},
            "ORDER_ROLE",
        )
        require(
            o["lineage"]["signal_id"] == o["signal_id"]
            and o["lineage"]["handoff_id"] == h["handoff_id"]
            and o["lineage"]["strategy_contract"] == h["strategy_contract"],
            "ORDER_LINEAGE",
        )
        require(
            d["client_order_id"] == o["client_order_id"] and d["symbol"] == s["symbol"],
            "ORDER_IDENTITY",
        )
        require(d["side"] == ("buy" if o["role"] == "entry" else "sell"), "ORDER_SIDE")
        oid = uuid(d["id"])
        filled = number(d["filled_qty"])
        qty = number(d["qty"], positive=True)
        require(filled <= qty and oid not in by_order, "ORDER_DUPLICATE_OR_QUANTITY")
        by_order[oid] = dict(
            account_id=account_id,
            broker_order_id=oid,
            client_order_id=o["client_order_id"],
            signal_id=o["signal_id"],
            role=o["role"],
            symbol=d["symbol"],
            side=d["side"],
            filled_qty=str(filled),
            order_qty=str(qty),
            status=d["status"],
            lineage_sha256=digest(o["lineage"]),
        )
    links = []
    unmatched = []
    seen = set()
    seen_ids = set()
    sums = {k: Decimal(0) for k in by_order}
    for f in fills:
        require(
            f["activity_type"] == "FILL" and f["activity_pk"] not in seen,
            "FILL_DUPLICATE_OR_TYPE",
        )
        seen.add(f["activity_pk"])
        require(f["activity_id"] not in seen_ids, "FILL_DUPLICATE_IDENTITY")
        seen_ids.add(f["activity_id"])
        canonical = cp.row_value({k: f[k] for k in cp.FIELDS})
        require(
            manifested.get((f["activity_id"], "FILL")) == cp.digest(canonical),
            "FILL_NOT_CHECKPOINTED",
        )
        qty = number(f["qty"], positive=True)
        price = number(f["price"], positive=True)
        order = by_order.get(f["order_id"])
        if order is None:
            unmatched.append(
                dict(activity_pk=f["activity_pk"], reason="ORDER_NOT_STRATEGY_OWNED")
            )
            continue
        require(
            f["symbol"] == order["symbol"] and f["side"] == order["side"],
            "FILL_ORDER_MISMATCH",
        )
        sums[f["order_id"]] += qty
        links.append(
            dict(
                account_id=account_id,
                activity_pk=f["activity_pk"],
                activity_id=f["activity_id"],
                broker_order_id=f["order_id"],
                checkpoint_id=provenance[(f["activity_id"], "FILL")],
                fill_sha256=cp.digest(canonical),
                quantity=str(qty),
                price=str(price),
                transaction_time=canonical["transaction_time"],
            )
        )
    coverage = []
    for oid, order in sorted(by_order.items()):
        observed = number(order["filled_qty"])
        linked = sums[oid]
        # A stale executor snapshot is not permission to discard actual FILL evidence.
        # Link bounded, authenticated fills, but expose disagreement explicitly.
        require(linked <= number(order["order_qty"]), "FILL_EXCEEDS_ORDER_QUANTITY")
        coverage.append(
            dict(
                broker_order_id=oid,
                observed_qty=str(observed),
                linked_qty=str(linked),
                status=(
                    "MATCHED"
                    if observed == linked
                    else "ORDER_SNAPSHOT_BEHIND"
                    if linked > observed
                    else "MISSING_FILL_ACTIVITIES"
                ),
            )
        )
    plan = dict(
        schema=VERSION,
        source_fingerprint=expected_source,
        checkpoint_id=checkpoint["checkpoint_id"],
        checkpoint_manifest_sha256=checkpoint["manifest_sha256"],
        orders=[by_order[k] for k in sorted(by_order)],
        fills=sorted(links, key=lambda x: x["activity_pk"]),
        unmatched=sorted(unmatched, key=lambda x: x["activity_pk"]),
        coverage=coverage,
        accounting_status="NOT_IMPLEMENTED_NO_LOT_ALLOCATION",
        trading_operations=0,
    )
    plan["plan_sha256"] = digest(plan)
    return plan


def immutable_order(row):
    return {
        k: row[k]
        for k in (
            "account_id",
            "broker_order_id",
            "client_order_id",
            "signal_id",
            "role",
            "symbol",
            "side",
            "lineage_sha256",
        )
    }


def persist_verified_plan(connection, plan, *, recheck, policy_lock):
    """One transaction. No schema creation, legacy updates or broker calls.

    recheck(cursor) MUST re-read checkpoint + all source inputs under locks and
    rebuild this plan. Any mismatch aborts before inserts. Committed but lost
    acknowledgement is reconciled by immutable receipt, never treated as failure.
    """
    require(
        connection.autocommit is False
        and connection.readonly is False
        and connection.get_transaction_status() == 0,
        "IDLE_WRITE_TRANSACTION_REQUIRED",
    )
    require(
        plan.get("schema") == VERSION and plan.get("trading_operations") == 0,
        "PLAN_INVALID",
    )
    require(
        plan.get("plan_sha256")
        == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "PLAN_CHANGED",
    )
    try:
        policy_lock(connection)
        with connection.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='15000ms'")
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext('jbravo:strategy-fill-linkage'))"
            )
            require(recheck(cur) == plan, "INPUT_CHANGED")
            for row in plan["orders"]:
                item = immutable_order(row)
                cur.execute(
                    "INSERT INTO public.strategy_broker_orders_app "
                    "(account_id,broker_order_id,client_order_id,signal_id,role,symbol,side,lineage_sha256) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    tuple(item.values()),
                )
                cur.execute(
                    "SELECT account_id,broker_order_id,client_order_id,signal_id,role,symbol,side,lineage_sha256 "
                    "FROM public.strategy_broker_orders_app WHERE account_id=%s AND broker_order_id=%s",
                    (row["account_id"], row["broker_order_id"]),
                )
                require(cur.fetchall() == [tuple(item.values())], "ORDER_LINK_CONFLICT")
            for row in plan["fills"]:
                # checkpoint_id is provenance of the first insert, not identity:
                # a later overlapping checkpoint must not change the binding.
                fields = (
                    "account_id",
                    "activity_pk",
                    "activity_id",
                    "broker_order_id",
                    "fill_sha256",
                )
                cur.execute(
                    "INSERT INTO public.strategy_fill_links_app "
                    "(account_id,activity_pk,activity_id,broker_order_id,fill_sha256,checkpoint_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    tuple(row[k] for k in fields) + (row["checkpoint_id"],),
                )
                cur.execute(
                    "SELECT account_id,activity_pk,activity_id,broker_order_id,fill_sha256 "
                    "FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_pk=%s",
                    (row["account_id"], row["activity_pk"]),
                )
                require(
                    cur.fetchall() == [tuple(row[k] for k in fields)],
                    "FILL_LINK_CONFLICT",
                )
            cur.execute(
                "INSERT INTO public.strategy_fill_link_runs_app (plan_sha256,checkpoint_id,receipt) "
                "VALUES (%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                (plan["plan_sha256"], plan["checkpoint_id"], encoded(plan)),
            )
            cur.execute(
                "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
                (plan["plan_sha256"],),
            )
            require(cur.fetchall() == [(plan,)], "RECEIPT_CONFLICT")
    except Exception:
        connection.rollback()
        raise
    try:
        connection.commit()
    except Exception:
        # Do not retry. A fresh connection must inspect the immutable receipt.
        raise CommitUncertain(plan["plan_sha256"]) from None
    return {
        "status": "COMMITTED",
        "plan_sha256": plan["plan_sha256"],
        "orders": len(plan["orders"]),
        "fills": len(plan["fills"]),
        "accounting_complete": False,
    }


class CommitUncertain(RuntimeError):
    """Commit may have succeeded. Reconcile on a new connection; never replay."""


def recover_receipt(connection, plan):
    """Read-only resolution on a fresh connection, including link postimages.

    ABSENT is not permission to retry; the original connection/transaction must
    first be known ended and source state rebuilt. Partial/conflicting state is
    an integrity error. Never reconnect or perform writes here.
    """
    require(
        connection.readonly is True and connection.get_transaction_status() == 0,
        "FRESH_READONLY_REQUIRED",
    )
    try:
        with connection.cursor() as cur:
            cur.execute(
                "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
                (plan["plan_sha256"],),
            )
            found = cur.fetchall()
            if not found:
                return {"status": "ABSENT_UNRESOLVED", "retry_authorized": False}
            require(found == [(plan,)], "RECEIPT_CONFLICT")
            for row in plan["orders"]:
                item = immutable_order(row)
                cur.execute(
                    "SELECT account_id,broker_order_id,client_order_id,signal_id,role,symbol,side,lineage_sha256 "
                    "FROM public.strategy_broker_orders_app WHERE account_id=%s AND broker_order_id=%s",
                    (row["account_id"], row["broker_order_id"]),
                )
                require(cur.fetchall() == [tuple(item.values())], "ORDER_LINK_CONFLICT")
            for row in plan["fills"]:
                fields = (
                    "account_id",
                    "activity_pk",
                    "activity_id",
                    "broker_order_id",
                    "fill_sha256",
                )
                cur.execute(
                    "SELECT account_id,activity_pk,activity_id,broker_order_id,fill_sha256 "
                    "FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_pk=%s",
                    (row["account_id"], row["activity_pk"]),
                )
                require(
                    cur.fetchall() == [tuple(row[k] for k in fields)],
                    "FILL_LINK_CONFLICT",
                )
            return {"status": "COMMITTED_VERIFIED", "plan_sha256": plan["plan_sha256"]}
    finally:
        connection.rollback()
