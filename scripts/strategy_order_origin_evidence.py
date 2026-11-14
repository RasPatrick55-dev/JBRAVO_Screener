"""Additive immutable origin receipts; transaction primitives only, no clients.

Only reviewed fetch importers may supply raw evidence. No hash proves broker
provenance. No commit/rollback/retry/DDL or file IO occurs in this module.
"""

import re
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot

VERSION = "jbravo.order-origin.v1"


def build(*, account_id, source_run_id, source_plan_sha256, raw_order, fetched_at):
    core.uuid(account_id)
    core.uuid(raw_order["id"])
    core.require(
        re.fullmatch("[0-9a-f]{32}", source_run_id) is not None, "ORIGIN_RUN_ID"
    )
    core.require(
        re.fullmatch("[0-9a-f]{64}", source_plan_sha256) is not None,
        "ORIGIN_SOURCE_PLAN",
    )
    created = lot.timestamp(raw_order["created_at"])
    updated = lot.timestamp(raw_order["updated_at"])
    fetched = lot.timestamp(fetched_at)
    core.require(
        created["start_ns"] <= updated["start_ns"] <= fetched["start_ns"],
        "ORIGIN_TIME_INCONSISTENT",
    )
    p = {
        "schema": VERSION,
        "account_id": account_id,
        "broker_order_id": raw_order["id"],
        "source_run_id": source_run_id,
        "source_plan_sha256": source_plan_sha256,
        "method": "GET",
        "endpoint": "/v2/orders/" + raw_order["id"],
        "fetched_at": fetched_at,
        "raw_order": raw_order,
        "accounting_complete": False,
        "publication_allowed": False,
    }
    p["receipt_sha256"] = core.digest(p)
    return p


def insert_in_transaction(q, receipts):
    """Caller owns policy/account lock, durable intent and commit reconciliation.

    This primitive never commits or retries. Rebuild/fetch provenance and compare
    the exact packet before calling. SQL independently checks current relationships.
    """
    for p in sorted(receipts, key=lambda x: x["receipt_sha256"]):
        core.require(
            p["receipt_sha256"]
            == core.digest({k: v for k, v in p.items() if k != "receipt_sha256"}),
            "ORIGIN_DIGEST",
        )
        q.execute(
            "SELECT receipt FROM public.strategy_order_origin_receipts_app WHERE receipt_sha256=%s",
            (p["receipt_sha256"],),
        )
        existing = q.fetchall()
        if existing:
            core.require(existing == [(p,)], "ORIGIN_RECEIPT_CONFLICT")
            continue
        q.execute(
            "INSERT INTO public.strategy_order_origin_receipts_app(receipt_sha256,account_id,broker_order_id,source_plan_sha256,receipt) VALUES(%s,%s,%s,%s,%s::jsonb)",
            (
                p["receipt_sha256"],
                p["account_id"],
                p["broker_order_id"],
                p["source_plan_sha256"],
                core.encoded(p),
            ),
        )


def verify_receipts(q, receipts):
    for p in receipts:
        q.execute(
            "SELECT receipt FROM public.strategy_order_origin_receipts_app WHERE receipt_sha256=%s",
            (p["receipt_sha256"],),
        )
        if q.fetchall() != [(p,)]:
            return {
                "status": "ORIGIN_ABSENT_OR_CONFLICT_UNRESOLVED",
                "retry_authorized": False,
            }
    return {
        "status": "ORIGIN_RECEIPTS_VERIFIED",
        "receipt_sha256s": sorted(p["receipt_sha256"] for p in receipts),
        "retry_authorized": False,
    }


def validate_fetch_receipt(p):
    """Require actual composite sync/fetch evidence, not legacy minimal v1 hashes."""
    from datetime import datetime
    from email.utils import parsedate_to_datetime
    from scripts import strategy_lot_hourly_phase as phase
    core.require(p["receipt_sha256"] == core.digest({k:v for k,v in p.items()
                 if k != "receipt_sha256"}), "ORIGIN_DIGEST")
    base=build(account_id=p["account_id"],source_run_id=p["source_run_id"],
               source_plan_sha256=p["source_plan_sha256"],raw_order=p["raw_order"],
               fetched_at=p["fetched_at"])
    core.require(all(p.get(k)==v for k,v in base.items() if k!="receipt_sha256"),
                 "ORIGIN_FETCH_IDENTITY")
    required = {"sync_completion_json", "sync_receipt_json", "sync_completion_sha256",
                "sync_slot", "sync_checkpoint", "sync_receipts", "fetch_timing"}
    core.require(required <= set(p), "ORIGIN_COMPOSITE_EVIDENCE_REQUIRED")
    c, linkage = phase.validate_sync(
        completion_bytes=p["sync_completion_json"].encode('utf-8'),
        expected_completion_sha256=p["sync_completion_sha256"],
        receipt_bytes={k:v.encode('utf-8') for k,v in p["sync_receipt_json"].items()},
        expected_slot=p["sync_slot"])
    core.require(c["native_run_id"]==p["source_run_id"] and
                 linkage["linkage"]["plan_sha256"]==p["source_plan_sha256"] and
                 c["checkpoint"]==p["sync_checkpoint"] and
                 c["native_receipts"]==p["sync_receipts"], "ORIGIN_SYNC_BINDING")
    t=p["fetch_timing"]
    before=datetime.fromisoformat(t["started_at"])
    after=datetime.fromisoformat(t["completed_at"])
    server=parsedate_to_datetime(t["server_date"])
    elapsed=t["elapsed_seconds"]
    core.require(before.tzinfo is not None and after.tzinfo is not None and server.tzinfo is not None
                 and t["maximum_clock_skew_seconds"]==5 and p["fetched_at"]==t["completed_at"]
                 and 0 <= elapsed <= 35 and 0 <= (after-before).total_seconds() <=35
                 and abs((after-before).total_seconds()-elapsed)<=2
                 and before.timestamp()-5 <= server.timestamp() <= after.timestamp()+5,
                 "ORIGIN_FETCH_CLOCK_EVIDENCE")

    if "bounded_history" in p:
        from scripts import activity_ingestion_checkpoint as cp
        from datetime import timedelta
        h=p['bounded_history']
        core.require(h['proof_type']=='BOUNDED_BROKER_GET_ORIGIN_RECEIPT'
            and h['checkpoint_id'] is None and h['history_sha256']==core.digest(
                {k:v for k,v in h.items() if k!='history_sha256'})
            and h['source_fingerprint']==cp.source_fingerprint(cp.ENDPOINT,p['account_id'])
            and h['recipe_sha256']==cp.recipe_fingerprint()
            and h['query_axis']=='ALPACA_CREATED_AT'
            and h['max_pages']==1000 and h['page_size']==100
            and len(h['rows'])<=core.MAX_ROWS and cp.manifestation(h['rows'])==h['persisted_manifest']
            and cp.digest(h['persisted_manifest'])==h['manifest_sha256']
            and h['fetched_count']==len(h['rows']), 'ORIGIN_HISTORY_EVIDENCE')
        core.require(timedelta(0)<cp.utc(h['query_until'])-cp.utc(h['query_after'])<=timedelta(days=45)
            and cp.utc(h['query_until'])<=cp.utc(h['observed_complete_at']), 'ORIGIN_HISTORY_WINDOW')
