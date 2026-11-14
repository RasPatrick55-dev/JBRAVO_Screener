"""Data-only existing-order observation and fill-proven closure. No broker clients.

No claims, orders, cancellations, position inference, prices or strategy decisions.
Every projection update preserves before/after images in an immutable receipt.
"""

from decimal import Decimal
from datetime import datetime, timezone
import re
from scripts import strategy_fill_linkage as core
from scripts import activity_ingestion_checkpoint as cp
from scripts.strategy_fill_linkage_store import rows

TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}
KEEP = (
    "id",
    "client_order_id",
    "symbol",
    "side",
    "type",
    "time_in_force",
    "extended_hours",
    "status",
    "qty",
    "filled_qty",
    "filled_avg_price",
    "limit_price",
    "stop_price",
    "submitted_at",
    "updated_at",
    "filled_at",
    "canceled_at",
    "expired_at",
    "replaced_by",
    "replaces",
)


def broker_time_key(value):
    """Exact UTC ordering of broker ISO timestamps, including 1-9 digits.

    Python3.10 fromisoformat rejects nanoseconds. Parsing the whole second
    separately preserves all broker fractional digits instead of truncating.
    """
    if isinstance(value, datetime):
        core.require(value.utcoffset() is not None, "BROKER_TIME_INVALID")
        utc = value.astimezone(timezone.utc)
        return utc.replace(microsecond=0), utc.microsecond * 1000
    core.require(isinstance(value, str), "BROKER_TIME_INVALID")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
        value,
    )
    core.require(match is not None, "BROKER_TIME_INVALID")
    base, fraction, zone = match.groups()
    try:
        utc = datetime.fromisoformat(
            base + ("+00:00" if zone == "Z" else zone)
        ).astimezone(timezone.utc)
    except ValueError:
        raise ValueError("FILL_LINK_BROKER_TIME_INVALID") from None
    return utc, int((fraction or "").ljust(9, "0"))


TIMESTAMP_FIELDS = frozenset({
    "submitted_at", "updated_at", "filled_at", "canceled_at", "expired_at",
})


def timestamp_precision(value):
    if isinstance(value, datetime):
        return 6
    match = re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})",
        value if isinstance(value, str) else "",
    )
    core.require(match is not None, "BROKER_TIME_INVALID")
    return len(match[1] or "")


def lower_precision_overlap_only(old, new):
    """Retain old evidence when only reported timestamp precision was lost.

    All non-time fields must match exactly. Every changed timestamp must lose
    precision and overlap its old reported point under rounding or truncation.
    This never authorizes an observation UPDATE or a timestamp rewrite.
    """
    if {k:v for k,v in old.items() if k not in TIMESTAMP_FIELDS} != {
        k:v for k,v in new.items() if k not in TIMESTAMP_FIELDS}:
        return False
    changed = False
    for key in TIMESTAMP_FIELDS:
        if (key in old) != (key in new):
            return False
        a,b=old.get(key),new.get(key)
        if a==b:
            continue
        if a is None or b is None:
            return False
        old_digits,new_digits=timestamp_precision(a),timestamp_precision(b)
        if new_digits >= old_digits:
            return False
        old_base,old_fraction=broker_time_key(a)
        new_base,new_fraction=broker_time_key(b)
        delta=old_base-new_base
        difference=(delta.days*86400+delta.seconds)*10**9+old_fraction-new_fraction
        if not -10**(9-new_digits) < difference < 10**(9-new_digits):
            return False
        changed=True
    return changed


def precision_interval_evidence(old, new):
    evidence=[]
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    for key in sorted(TIMESTAMP_FIELDS):
        if old.get(key)==new.get(key):
            continue
        old_base,old_fraction=broker_time_key(old[key])
        new_base,new_fraction=broker_time_key(new[key])
        a,b=old_base-epoch,new_base-epoch
        old_ns=(a.days*86400+a.seconds)*10**9+old_fraction
        new_ns=(b.days*86400+b.seconds)*10**9+new_fraction
        digits=timestamp_precision(new[key])
        resolution=10**(9-digits)
        evidence.append(dict(field=key,old_raw=old[key],broker_raw=new[key],
            old_precision_digits=timestamp_precision(old[key]),new_precision_digits=digits,
            old_reported_ns=old_ns,new_reported_ns=new_ns,
            lower_exclusive_ns=new_ns-resolution,upper_exclusive_ns=new_ns+resolution,
            basis="ONE_REPORTED_RESOLUTION_EITHER_SIDE_ROUNDING_OR_TRUNCATION",
            chronology_certified=False))
    return evidence


def read_state(cur, account_id, lock=False):
    suffix = " FOR UPDATE" if lock else ""
    if lock:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext('jbravo:strategy-execution'),hashtext(%s))",
            (account_id,),
        )
    cur.execute(
        "SELECT account_id,signal_id,symbol,state,revision,payload FROM public.strategy_execution_signals_app WHERE account_id=%s ORDER BY signal_id LIMIT 20001"
        + suffix,
        (account_id,),
    )
    signals = rows(cur)
    cur.execute(
        "SELECT account_id,client_order_id,signal_id,role,operation,state,observation FROM public.strategy_execution_orders_app WHERE account_id=%s ORDER BY client_order_id LIMIT 20001"
        + suffix,
        (account_id,),
    )
    orders = rows(cur)
    core.require(len(signals) <= 20000 and len(orders) <= 20000, "STATE_LIMIT")
    return {"signals": signals, "orders": orders}


def snapshot_plan(state, snapshots, *, account_id, observed_at):
    core.uuid(account_id)
    cp.utc(observed_at)
    protected = {
        x["signal_id"] for x in state["signals"] if x["payload"].get("execution_owner")
    }
    changes, unresolved, requested = [], [], set()
    protected_differences = []
    protected_evidence = []
    precision_evidence = []
    for order in state["orders"]:
        core.require(order["account_id"] == account_id, "SNAPSHOT_ACCOUNT")
        if order["state"] in {"CLAIMED", "UNKNOWN"}:
            unresolved.append(order["client_order_id"])
        if order["operation"] == "DELETE" or order["state"] == "REJECTED":
            continue
        old = order["observation"]
        if old is None:
            if order["client_order_id"] not in unresolved:
                unresolved.append(order["client_order_id"])
            continue
        oid = core.uuid(old["id"])
        requested.add(oid)
        core.require(oid in snapshots, "SNAPSHOT_MISSING")
        new = {k: snapshots[oid][k] for k in KEEP if k in snapshots[oid]}
        core.require(len(core.encoded(new)) <= 8192, "SNAPSHOT_OVERSIZE")
        for key in (
            "id",
            "client_order_id",
            "symbol",
            "side",
            "type",
            "time_in_force",
            "extended_hours",
            "qty",
            "replaces",
            "replaced_by",
        ):
            core.require(new.get(key) == old.get(key), "SNAPSHOT_IDENTITY_CHANGED")
        core.require(
            new["client_order_id"] == order["client_order_id"],
            "SNAPSHOT_CLAIM_MISMATCH",
        )
        qty = core.number(new["qty"], positive=True)
        core.require(
            core.number(old["filled_qty"]) <= core.number(new["filled_qty"]) <= qty,
            "SNAPSHOT_QUANTITY_REGRESSED",
        )
        old_time = broker_time_key(old["updated_at"])
        new_time = broker_time_key(new["updated_at"])
        observation_time = broker_time_key(observed_at)
        is_protected = order["signal_id"] in protected
        # Protected observations are never changed or used for closure. Keep
        # broker discrepancies as evidence without blocking unrelated owners.
        # A lower-precision repeat is evidence only, never a timestamp rewrite.
        precision_only = (not is_protected and old != new
                          and lower_precision_overlap_only(old, new))
        core.require(
            new_time <= observation_time and (is_protected or old_time <= new_time or precision_only),
            "SNAPSHOT_TIME_REGRESSED",
        )
        core.require(
            old["status"] not in TERMINAL or new["status"] == old["status"],
            "SNAPSHOT_TERMINAL_CHANGED",
        )
        if precision_only:
            precision_evidence.append({
                "client_order_id": order["client_order_id"],
                "reason": "LOWER_PRECISION_OVERLAP_NO_UPDATE",
                "before": old,
                "broker_snapshot": new,
                "timestamp_fields": sorted(k for k in TIMESTAMP_FIELDS if old.get(k)!=new.get(k)),
                "precision_intervals": precision_interval_evidence(old,new),
                "chronology_certified": False,
            })
        elif old != new and is_protected:
            protected_differences.append(order["client_order_id"])
            protected_evidence.append(
                {
                    "client_order_id": order["client_order_id"],
                    "reason": "PROTECTED_EXTERNAL_OWNERSHIP_NO_UPDATE",
                    "timestamp_regressed": new_time < old_time,
                    "before": old,
                    "broker_snapshot": new,
                }
            )
        elif old != new:
            changes.append(
                {
                    "client_order_id": order["client_order_id"],
                    "before": old,
                    "after": new,
                }
            )
    core.require(set(snapshots) == requested, "SNAPSHOT_SCOPE")
    p = {
        "schema": "jbravo.data-projection.v1",
        "kind": "SNAPSHOT_REFRESH",
        "account_id": account_id,
        "observed_at": cp.utc(observed_at).isoformat(),
        "source_state": state,
        "orders": changes,
        "signals": [],
        "unresolved_claims": sorted(unresolved),
        "protected_observation_differences": sorted(protected_differences),
        "protected_observation_evidence": protected_evidence,
        "precision_preserved_observation_evidence": precision_evidence,
        "trading_operations": 0,
    }
    p["sha256"] = core.digest(p)
    return p


def closure_plan(state, linkage, *, account_id):
    core.require(
        linkage["plan_sha256"]
        == core.digest({k: v for k, v in linkage.items() if k != "plan_sha256"}),
        "LINKAGE_CHANGED",
    )
    core.require(
        all(o["account_id"] == account_id for o in state["orders"]),
        "CLOSURE_ORDER_ACCOUNT",
    )
    by_id = {o["broker_order_id"]: o for o in linkage["orders"]}
    coverage = {x["broker_order_id"]: x["status"] for x in linkage["coverage"]}
    changes, held = [], []
    for signal in state["signals"]:
        core.require(signal["account_id"] == account_id, "CLOSURE_ACCOUNT")
        if signal["state"] in {"CLOSED", "EXPIRED"}:
            continue
        if signal["payload"].get("execution_owner"):
            held.append(
                {
                    "signal_id": signal["signal_id"],
                    "reason": "PROTECTED_EXTERNAL_OWNERSHIP",
                }
            )
            continue
        if any(
            o["signal_id"] == signal["signal_id"]
            and o["state"] in {"CLAIMED", "UNKNOWN"}
            for o in state["orders"]
        ):
            held.append(
                {"signal_id": signal["signal_id"], "reason": "UNRESOLVED_CLAIM"}
            )
            continue
        owned = [
            o
            for o in state["orders"]
            if o["signal_id"] == signal["signal_id"]
            and o["operation"] != "DELETE"
            and o["state"] != "REJECTED"
        ]
        if not owned or signal["state"] not in {
            "OPEN",
            "ENTRY_PENDING",
            "EXIT_PENDING",
        }:
            held.append(
                {"signal_id": signal["signal_id"], "reason": "UNRESOLVED_STATE"}
            )
            continue
        if any(o["observation"] is None or o["state"] != "OBSERVED" for o in owned):
            held.append(
                {"signal_id": signal["signal_id"], "reason": "UNRESOLVED_CLAIM"}
            )
            continue
        observed = [o["observation"] for o in owned]
        ids = {o["id"] for o in observed}
        core.require(len(ids) == len(observed), "CLOSURE_DUPLICATE_ORDER")
        if any(
            o["status"] not in TERMINAL or coverage.get(o["id"]) != "MATCHED"
            for o in observed
        ):
            held.append(
                {"signal_id": signal["signal_id"], "reason": "LIVE_ORDER_OR_FILL_GAP"}
            )
            continue
        for o in observed:
            core.require(
                o["id"] in by_id
                and by_id[o["id"]]["account_id"] == account_id
                and by_id[o["id"]]["signal_id"] == signal["signal_id"]
                and core.number(by_id[o["id"]]["filled_qty"])
                == core.number(o["filled_qty"]),
                "CLOSURE_LINKAGE_MISMATCH",
            )
        entries = [o for o in owned if o["role"] == "entry"]
        core.require(len(entries) == 1, "CLOSURE_ENTRY_COUNT")
        buys, sells = Decimal(0), Decimal(0)
        for f in linkage["fills"]:
            if f["broker_order_id"] in ids:
                o = by_id[f["broker_order_id"]]
                if o["side"] == "buy":
                    buys += core.number(f["quantity"], positive=True)
                else:
                    sells += core.number(f["quantity"], positive=True)
        core.require(sells <= buys, "CLOSURE_OVERSELL")
        if buys == 0:
            held.append(
                {"signal_id": signal["signal_id"], "reason": "NO_ACTUAL_ENTRY_FILLS"}
            )
            continue
        if buys != sells:
            held.append(
                {"signal_id": signal["signal_id"], "reason": "OPEN_OR_PARTIAL_QUANTITY"}
            )
            continue
        changes.append(
            {
                "signal_id": signal["signal_id"],
                "before_state": signal["state"],
                "before_revision": signal["revision"],
                "after_state": "CLOSED",
                "bought_qty": str(buys),
                "sold_qty": str(sells),
            }
        )
    p = {
        "schema": "jbravo.data-projection.v1",
        "kind": "FILL_PROVEN_CLOSURE",
        "account_id": account_id,
        "source_state": state,
        "linkage_plan_sha256": linkage["plan_sha256"],
        "orders": [],
        "signals": changes,
        "held": held,
        "accounting_complete": False,
        "trading_operations": 0,
    }
    p["sha256"] = core.digest(p)
    return p


def persist(connection, plan, *, policy_lock, recheck):
    core.require(
        connection.autocommit is False
        and connection.readonly is False
        and connection.get_transaction_status() == 0,
        "IDLE_WRITE_TRANSACTION_REQUIRED",
    )
    core.require(
        plan.get("schema") == "jbravo.data-projection.v1"
        and plan.get("trading_operations") == 0,
        "PROJECTION_SCHEMA",
    )
    core.require(
        plan["sha256"] == core.digest({k: v for k, v in plan.items() if k != "sha256"}),
        "PROJECTION_CHANGED",
    )
    try:
        policy_lock(connection)
        with connection.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='15000ms'")
            core.require(recheck(cur) == plan, "PROJECTION_INPUT_CHANGED")
            for o in plan["orders"]:
                cur.execute(
                    "UPDATE public.strategy_execution_orders_app SET observation=%s::jsonb,updated_at=clock_timestamp() WHERE account_id=%s AND client_order_id=%s AND observation=%s::jsonb AND state='OBSERVED'",
                    (
                        core.encoded(o["after"]),
                        plan["account_id"],
                        o["client_order_id"],
                        core.encoded(o["before"]),
                    ),
                )
                core.require(cur.rowcount == 1, "SNAPSHOT_CAS_FAILED")
            for s in plan["signals"]:
                cur.execute(
                    "UPDATE public.strategy_execution_signals_app SET state=%s,revision=revision+1,updated_at=clock_timestamp() WHERE account_id=%s AND signal_id=%s AND state=%s AND revision=%s",
                    (
                        s["after_state"],
                        plan["account_id"],
                        s["signal_id"],
                        s["before_state"],
                        s["before_revision"],
                    ),
                )
                core.require(cur.rowcount == 1, "CLOSURE_CAS_FAILED")
            cur.execute(
                "INSERT INTO public.strategy_data_projection_runs_app(plan_sha256,account_id,kind,receipt) VALUES (%s,%s,%s,%s::jsonb)",
                (plan["sha256"], plan["account_id"], plan["kind"], core.encoded(plan)),
            )
            cur.execute(
                "SELECT receipt FROM public.strategy_data_projection_runs_app WHERE plan_sha256=%s",
                (plan["sha256"],),
            )
            core.require(cur.fetchall() == [(plan,)], "PROJECTION_RECEIPT_MISMATCH")
    except Exception:
        connection.rollback()
        raise
    try:
        connection.commit()
    except Exception:
        raise core.CommitUncertain(plan["sha256"]) from None
    return {
        "status": "COMMITTED",
        "orders": len(plan["orders"]),
        "signals": len(plan["signals"]),
    }


def recover(connection, plan):
    core.require(connection.readonly is True, "READONLY_REQUIRED")
    with connection.cursor() as cur:
        cur.execute(
            "SELECT receipt FROM public.strategy_data_projection_runs_app WHERE plan_sha256=%s",
            (plan["sha256"],),
        )
        got = cur.fetchall()
        if not got:
            return {"status": "ABSENT_UNRESOLVED", "retry_authorized": False}
        core.require(got == [(plan,)], "PROJECTION_RECEIPT_CHANGED")
        for o in plan["orders"]:
            cur.execute(
                "SELECT observation FROM public.strategy_execution_orders_app WHERE account_id=%s AND client_order_id=%s",
                (plan["account_id"], o["client_order_id"]),
            )
            core.require(
                cur.fetchall() == [(o["after"],)], "PROJECTION_POSTIMAGE_CHANGED"
            )
        for s in plan["signals"]:
            cur.execute(
                "SELECT state,revision FROM public.strategy_execution_signals_app WHERE account_id=%s AND signal_id=%s",
                (plan["account_id"], s["signal_id"]),
            )
            core.require(
                cur.fetchall() == [(s["after_state"], s["before_revision"] + 1)],
                "PROJECTION_POSTIMAGE_CHANGED",
            )
    return {"status": "COMMITTED_VERIFIED", "retry_authorized": False}
