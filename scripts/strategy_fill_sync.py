"""Independent data-only orchestration port. No production clients or CLI.

Deployment supplies reviewed adapters to the existing durable activity ingestor
and this package's locked reader/writer. Nothing here can submit a broker order.
"""

from scripts.strategy_fill_linkage import CommitUncertain, require


def sync_once(*, ingest, read_plan, persist, reconcile_uncertain):
    # The existing ingestor owns its atomic activities/checkpoint/cursor txn.
    # Failure prevents linkage; it does not falsely label filled orders missing.
    checkpoint_id = ingest()
    plan = read_plan(checkpoint_id)
    try:
        result = persist(plan)
    except CommitUncertain:
        result = reconcile_uncertain(plan)
        if result["status"] != "COMMITTED_VERIFIED":
            return {
                "status": "COMMIT_UNRESOLVED",
                "retry_authorized": False,
                "trading_operations": 0,
                "accounting_complete": False,
            }
    require(
        result.get("status") in {"COMMITTED", "COMMITTED_VERIFIED"},
        "PERSISTENCE_NOT_CONFIRMED",
    )
    missing = [x for x in plan["coverage"] if x["status"] != "MATCHED"]
    return {
        "status": "LINKED_WITH_GAPS" if missing or plan["unmatched"] else "LINKED",
        "checkpoint_id": checkpoint_id,
        "plan_sha256": plan["plan_sha256"],
        "missing_fill_orders": sum(
            x["status"] == "MISSING_FILL_ACTIVITIES" for x in missing
        ),
        "stale_order_snapshots": sum(
            x["status"] == "ORDER_SNAPSHOT_BEHIND" for x in missing
        ),
        "unmatched_fills": len(plan["unmatched"]),
        "trading_operations": 0,
        "accounting_complete": False,
        "persistence": result,
    }
