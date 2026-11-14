"""Locked PostgreSQL input reader for strategy_fill_linkage (no connection creation)."""

from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as core


def rows(cur):
    names = [d.name for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def load_plan(
    cur, *, account_id, checkpoint_id, expected_source, expected_recipe, lock=False
):
    core.uuid(account_id)
    core.uuid(checkpoint_id)
    suffix = " FOR SHARE" if lock else ""
    if lock:
        # Same account lock used by strategy execution, acquired before row locks.
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext('jbravo:strategy-execution'),hashtext(%s))",
            (account_id,),
        )
    cur.execute(
        "SELECT account_id,signal_id,symbol,payload FROM public.strategy_execution_signals_app "
        "WHERE account_id=%s ORDER BY signal_id LIMIT 20001" + suffix,
        (account_id,),
    )
    signals = rows(cur)
    core.require(len(signals) <= 20000, "SIGNAL_LIMIT")
    cur.execute(
        "SELECT account_id,signal_id,client_order_id,operation,role,state,lineage,observation "
        "FROM public.strategy_execution_orders_app WHERE account_id=%s ORDER BY client_order_id LIMIT 20001"
        + suffix,
        (account_id,),
    )
    orders = rows(cur)
    core.require(len(orders) <= 20000, "ORDER_LIMIT")
    cur.execute(
        "SELECT DISTINCT checkpoint_id::text FROM public.strategy_fill_links_app WHERE account_id=%s AND checkpoint_id<>%s "
        "ORDER BY checkpoint_id::text LIMIT 513",
        (account_id, checkpoint_id),
    )
    prior_ids = [r[0] for r in cur.fetchall() if r[0] != checkpoint_id]
    core.require(len(prior_ids) <= 512, "PROOF_LIMIT")
    proofs = []
    fills = {}
    count = 0
    for cid in prior_ids + [checkpoint_id]:
        cur.execute(
            "SELECT checkpoint_id::text,source_fingerprint,recipe_sha256,query_axis,query_after,query_until,"
            "persisted_manifest,fetched_count,manifest_sha256 FROM public.activity_ingestion_checkpoints "
            "WHERE checkpoint_id=%s" + suffix,
            (cid,),
        )
        found = rows(cur)
        core.require(len(found) == 1, "CHECKPOINT_MISSING")
        proof = found[0]
        manifest = proof["persisted_manifest"]
        count += len(manifest)
        core.require(count <= core.MAX_ROWS, "PROOF_ROW_LIMIT")
        ledger = []
        for offset in range(0, len(manifest), 200):
            part = manifest[offset : offset + 200]
            keys = [(x["activity_id"], x["activity_type"]) for x in part]
            sql = (
                "SELECT activity_pk,"
                + ",".join(cp.FIELDS)
                + " FROM public.alpaca_activities WHERE (activity_id,activity_type) IN ("
            )
            sql += ",".join("(%s,%s)" for _ in keys) + ") ORDER BY activity_pk" + suffix
            cur.execute(sql, tuple(x for key in keys for x in key))
            for row in rows(cur):
                ledger.append({k: row[k] for k in cp.FIELDS})
                if row["activity_type"] == "FILL":
                    if row["activity_pk"] in fills:
                        core.require(fills[row["activity_pk"]] == row, "LEDGER_CHANGED")
                    fills[row["activity_pk"]] = row
        proofs.append((proof, ledger))
    latest, latest_rows = proofs[-1]
    return core.build_plan(
        account_id=account_id,
        signals=signals,
        orders=orders,
        fills=list(fills.values()),
        checkpoint=latest,
        checkpoint_rows=latest_rows,
        prior_proofs=proofs[:-1],
        expected_source=expected_source,
        expected_recipe=expected_recipe,
    )
