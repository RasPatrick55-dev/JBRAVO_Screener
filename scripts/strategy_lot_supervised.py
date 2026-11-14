"""Prepared-only initial THC allocation boundary. No clients or import-time I/O.

The source-release worker supplies the existing connector and pins this module
and its dependencies. There is no timer, broker call, deployment or CLI here.
"""

from pathlib import Path
import hashlib

from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot


def validate(envelope, expected_sha256):
    lot.require(
        envelope["schema"] == "jbravo.thc-initial-lot-release.v1", "SUPERVISED_SCHEMA"
    )
    lot.require(
        envelope["sha256"]
        == expected_sha256
        == core.digest({k: v for k, v in envelope.items() if k != "sha256"}),
        "SUPERVISED_NOT_APPROVED",
    )
    p = envelope["packet"]
    lot.require(
        p["run_sha256"]
        == core.digest({k: v for k, v in p.items() if k != "run_sha256"}),
        "PACKET_CHANGED",
    )
    lot.require(
        len(p["lots"]) == 3
        and len(p["events"]) == 2
        and p["prior_events"] == []
        and p["held"] == [],
        "INITIAL_SCOPE",
    )
    lot.require(
        {x["symbol"] for x in p["lots"]} == {"THC"}
        and {x["signal_id"] for x in p["lots"]} == {envelope["signal_id"]},
        "THC_ONLY",
    )
    lot.require(
        all(
            e["kind"] == "ALLOCATE"
            and e["reverses"] is None
            and e.get("replaces_reversal") is None
            for e in p["events"]
        ),
        "INITIAL_EDGES_ONLY",
    )
    lot.require(
        p["accounting_complete"] is False
        and p["publication_allowed"] is False
        and p["gross_pnl"] is None
        and p["net_pnl"] is None,
        "NO_ACCOUNTING_CERTIFICATION",
    )
    lot.require(
        hashlib.sha256(Path(lot.__file__).read_bytes()).hexdigest()
        == envelope["module_sha256"],
        "INSTALLED_MODULE_CHANGED",
    )
    lot.require(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == envelope["helper_sha256"],
        "INSTALLED_HELPER_CHANGED",
    )
    lot.require(
        envelope["publication_allowed"] is False
        and envelope["accounting_complete"] is False
        and envelope["gross_pnl"] is None
        and envelope["net_pnl"] is None,
        "RELEASE_NOT_ACCOUNTING_CERTIFIED",
    )


def rows(q):
    names = [d.name for d in q.description]
    return [dict(zip(names, r)) for r in q.fetchall()]


def recheck(q, envelope):
    """Called by lot.persist AFTER policy/account lease, in READ COMMITTED."""
    p = envelope["packet"]
    account = p["account_id"]
    signal = envelope["signal_id"]
    linked = envelope["source_linkage_receipt"]
    q.execute(
        "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
        (linked["plan_sha256"],),
    )
    lot.require(q.fetchall() == [(linked,)], "SOURCE_LINKAGE_RECEIPT_CHANGED")
    q.execute(
        "SELECT payload FROM public.strategy_execution_signals_app WHERE account_id=%s AND signal_id=%s AND symbol='THC' FOR SHARE",
        (account, signal),
    )
    found = q.fetchall()
    lot.require(
        len(found) == 1 and not found[0][0].get("execution_owner"),
        "PROTECTED_OR_MISSING_SIGNAL",
    )
    payload = found[0][0]
    lot.require(
        payload["handoff"]["paper_only"] is True
        and payload["proposal"]["signal_id"] == signal
        and payload["proposal"]["symbol"] == "THC",
        "SIGNAL_LINEAGE_CHANGED",
    )
    q.execute(
        "SELECT client_order_id,role,operation,state,lineage,observation FROM public.strategy_execution_orders_app WHERE account_id=%s AND signal_id=%s ORDER BY client_order_id FOR SHARE",
        (account, signal),
    )
    orders = rows(q)
    expected = {
        o["client_order_id"]: o for o in linked["orders"] if o["signal_id"] == signal
    }
    lot.require(
        len(orders) == 2 and {x["client_order_id"] for x in orders} == set(expected),
        "NEW_OR_MISSING_THC_CLAIM",
    )
    for current in orders:
        old = expected[current["client_order_id"]]
        raw = current["observation"]
        lot.require(
            core.digest(current["lineage"]) == old["lineage_sha256"]
            and current["lineage"]["signal_id"] == signal
            and current["lineage"]["handoff_id"] == payload["handoff"]["handoff_id"]
            and current["lineage"]["strategy_contract"]
            == payload["handoff"]["strategy_contract"],
            "ORDER_LINEAGE_CHANGED",
        )
        lot.require(
            current["state"] == "OBSERVED"
            and current["operation"] in ("POST", "PATCH")
            and current["role"] == old["role"],
            "THC_CLAIM_CHANGED",
        )
        lot.require(
            raw
            and raw["id"] == old["broker_order_id"]
            and raw["client_order_id"] == old["client_order_id"]
            and raw["symbol"] == "THC"
            and raw["side"] == old["side"]
            and raw["status"] == "filled"
            and core.number(raw["filled_qty"]) == 14,
            "THC_ORDER_CHANGED",
        )
        lot.require(
            lot.timestamp(raw["updated_at"])["end_ns"]
            < lot.timestamp(envelope["review_cutoff"])["start_ns"],
            "LATER_ORDER_EVIDENCE",
        )
    ids = [o["broker_order_id"] for o in expected.values()]
    q.execute(
        "SELECT activity_pk,"
        + ",".join(cp.FIELDS)
        + " FROM public.alpaca_activities WHERE activity_type='FILL' AND order_id=ANY(%s) ORDER BY activity_pk FOR SHARE",
        (ids,),
    )
    activity = rows(q)
    lot.require(
        {r["activity_pk"] for r in activity} == {x["activity_pk"] for x in p["lots"]},
        "NEW_OR_MISSING_THC_FILL",
    )
    by_pk = {r["activity_pk"]: r for r in activity}
    for item in p["lots"]:
        canonical = cp.row_value({k: by_pk[item["activity_pk"]][k] for k in cp.FIELDS})
        lot.require(
            canonical == item["source_row"]
            and cp.digest(canonical) == item["fill_sha256"],
            "THC_FILL_CHANGED",
        )
        q.execute(
            "SELECT fill_sha256,checkpoint_id::text FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_id=%s AND activity_pk=%s",
            (account, item["activity_id"], item["activity_pk"]),
        )
        lot.require(
            q.fetchall() == [(item["fill_sha256"], item["checkpoint_id"])],
            "THC_LINK_CHANGED",
        )
    proof = envelope["checkpoint"]
    q.execute(
        "SELECT checkpoint_id::text,source_fingerprint,recipe_sha256,query_axis,query_after,query_until,persisted_manifest,fetched_count,manifest_sha256 FROM public.activity_ingestion_checkpoints WHERE checkpoint_id=%s FOR SHARE",
        (proof["checkpoint_id"],),
    )
    current = rows(q)
    lot.require(len(current) == 1, "CHECKPOINT_MISSING")
    current = current[0]
    for key in (
        "checkpoint_id",
        "source_fingerprint",
        "recipe_sha256",
        "query_axis",
        "persisted_manifest",
        "fetched_count",
        "manifest_sha256",
    ):
        lot.require(current[key] == proof[key], "CHECKPOINT_CHANGED")
    for key in ("query_after", "query_until"):
        lot.require(
            cp.utc(current[key]) == cp.utc(proof[key]), "CHECKPOINT_WINDOW_CHANGED"
        )
    keys = [(x["activity_id"], x["activity_type"]) for x in proof["persisted_manifest"]]
    q.execute(
        "SELECT "
        + ",".join(cp.FIELDS)
        + " FROM public.alpaca_activities WHERE (activity_id,activity_type) IN ("
        + ",".join("(%s,%s)" for _ in keys)
        + ") ORDER BY activity_pk FOR SHARE",
        tuple(x for pair in keys for x in pair),
    )
    lot.require(
        cp.manifestation(rows(q)) == proof["persisted_manifest"],
        "FULL_CHECKPOINT_LEDGER_CHANGED",
    )
    q.execute(
        "SELECT event FROM public.strategy_lot_events_app WHERE account_id=%s ORDER BY event_id",
        (account,),
    )
    lot.require(q.fetchall() == [], "INITIAL_ACCOUNT_ALREADY_ALLOCATED")
    rebuilt = lot.plan(
        p["lots"],
        account_id=account,
        history_review=p["history_review"],
        recipe=p["recipe"],
    )
    lot.require(rebuilt == p, "INITIAL_PLAN_CHANGED")
    return rebuilt


def apply_once(
    *,
    connect,
    policy_lock,
    envelope,
    expected_sha256,
    journal_directory,
    verify_release_gate,
):
    """Caller must attest pinned installed sources/schema/native fixture proof.

    verify_release_gate must raise on any missing/failed exact release prerequisite.
    It does not grant authorization; the caller's reviewed release owns that.
    """
    validate(envelope, expected_sha256)
    lot.require(verify_release_gate(envelope) is True, "NATIVE_RELEASE_GATE_REQUIRED")

    def guarded_policy(connection):
        policy_lock(connection)
        with connection.cursor() as q:
            q.execute(
                "SELECT deployment_id,database_id,database_name,policy_revision,current_database() FROM public.trade_disposition_policy WHERE singleton=1"
            )
            lot.require(
                q.fetchall() == [tuple(envelope["database_policy"])],
                "DATABASE_POLICY_CHANGED",
            )

    return lot.apply_guarded(
        connect=connect,
        packet=envelope["packet"],
        recheck=lambda q: recheck(q, envelope),
        policy_lock=guarded_policy,
        journal_directory=journal_directory,
    )


def verify_readonly(connection, envelope, expected_sha256):
    validate(envelope, expected_sha256)
    # Fresh read-only connection, repeatable-read receipt/lot/capacity audit.
    return lot.recover(connection, envelope["packet"])


def verify_initial_state(q, envelope):
    """Current initial-release scope, not just an intact historical receipt."""
    packet = envelope["packet"]
    account = packet["account_id"]
    q.execute(
        "SELECT receipt FROM public.strategy_lot_runs_app WHERE account_id=%s",
        (account,),
    )
    lot.require(q.fetchall() == [(packet,)], "INITIAL_RECEIPT_SCOPE_CHANGED")
    q.execute(
        "SELECT event FROM public.strategy_lot_events_app WHERE account_id=%s ORDER BY event_id",
        (account,),
    )
    lot.require(
        [r[0] for r in q.fetchall()]
        == sorted(packet["events"], key=lambda e: e["event_id"]),
        "INITIAL_EVENT_SCOPE_CHANGED",
    )
    q.execute(
        "SELECT b.activity_id,b.capacity,COALESCE(sum(CASE WHEN e.kind='ALLOCATE' THEN e.quantity ELSE -e.quantity END),0) FROM public.strategy_lot_budgets_app b LEFT JOIN public.strategy_lot_events_app e ON e.account_id=b.account_id AND (e.entry_activity_id=b.activity_id OR e.exit_activity_id=b.activity_id) WHERE b.account_id=%s GROUP BY b.activity_id,b.capacity ORDER BY b.activity_id",
        (account,),
    )
    expected = sorted(
        (x["activity_id"], core.number(x["quantity"]), core.number(x["quantity"]))
        for x in packet["lots"]
    )
    lot.require(q.fetchall() == expected, "INITIAL_EFFECTIVE_QUANTITIES_CHANGED")
