"""Explicit data-only PAPER adapter. Defaults to preparation; no broker mutation.

Run as a module in the reviewed deployment. No imports of executor/monitor and
no schema creation. --apply requires a previously reviewed immutable packet.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as core
from scripts import strategy_fill_linkage_store as store


class PaperGetTransport:
    """Restrict existing ingestor transport in this process, without file changes."""

    def __init__(self, request_get):
        self._get = request_get

    def get(self, url, *, headers, params=None, timeout=30, allow_redirects=False):
        allowed = url in {
            cp.ENDPOINT + "/v2/account",
            cp.ENDPOINT + "/v2/account/activities",
        }
        if isinstance(url, str) and url.startswith(cp.ENDPOINT + "/v2/orders/"):
            tail = url[len(cp.ENDPOINT + "/v2/orders/") :]
            allowed = bool(
                re.fullmatch(
                    "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", tail
                )
            )
        core.require(
            allowed and allow_redirects is False and timeout == 30,
            "READONLY_PAPER_TRANSPORT_REQUIRED",
        )
        response = self._get(
            url, headers=headers, params=params, timeout=30, allow_redirects=False
        )
        core.require(response.status_code == 200, "PAPER_GET_NOT_SUCCESSFUL")
        return response


def configure_readonly_transport(ingest):
    if not isinstance(ingest.requests, PaperGetTransport):
        ingest.requests = PaperGetTransport(ingest.requests.get)


def prepare(
    *, account_id, source, before, required_after, cutoff, activities, normalize
):
    core.uuid(account_id)
    core.require(
        source == cp.source_fingerprint(cp.ENDPOINT, account_id), "SOURCE_MISMATCH"
    )
    after, until = cp.window(before, required_after, cutoff, 14)
    values = []
    for activity in activities:
        row, _ = normalize(activity)
        core.require(row["activity_id"] and row["activity_type"], "ACTIVITY_IDENTITY")
        core.require(cp.utc(row["transaction_time"]) <= until, "ACTIVITY_FUTURE")
        values.append(cp.row_value(row))
    manifest = cp.manifestation(values)
    packet = {
        "schema": "jbravo.paper-fill-sync-packet.v1",
        "account_id": account_id,
        "source": source,
        "workflow_id": uuid4().hex,
        "after": after.isoformat(),
        "until": until.isoformat(),
        "required_after": cp.utc(required_after).isoformat(),
        "before": cp.utc(before).isoformat() if before else None,
        "rows": values,
        "manifest": manifest,
        "recipe_sha256": cp.recipe_fingerprint(),
        "trading_operations": 0,
    }
    packet["sha256"] = core.digest(packet)
    return packet


def validate(packet, account_id):
    core.require(
        packet.get("schema") == "jbravo.paper-fill-sync-packet.v1", "PACKET_SCHEMA"
    )
    core.require(packet.get("trading_operations") == 0, "PACKET_SCOPE")
    core.require(
        packet["sha256"]
        == core.digest({k: v for k, v in packet.items() if k != "sha256"}),
        "PACKET_CHANGED",
    )
    core.require(packet["account_id"] == account_id, "ACCOUNT_CHANGED")
    core.require(
        packet["source"] == cp.source_fingerprint(cp.ENDPOINT, account_id),
        "SOURCE_MISMATCH",
    )
    core.require(packet["recipe_sha256"] == cp.recipe_fingerprint(), "RECIPE_CHANGED")
    core.require(
        cp.manifestation(packet["rows"]) == packet["manifest"], "MANIFEST_CHANGED"
    )
    cp.workflow(packet["workflow_id"])
    core.require(
        cp.utc(packet["after"])
        < cp.utc(packet["required_after"])
        <= cp.utc(packet["until"])
        <= datetime.now(timezone.utc),
        "WINDOW_INVALID",
    )


def apply_packet(
    packet, *, account_id, connect, policy_lock, verify_broker, expected_sha256
):
    """Manual apply boundary: approved digest is supplied independently."""
    core.require(packet.get("sha256") == expected_sha256, "PACKET_NOT_APPROVED")
    return _apply_verified_packet(
        packet,
        account_id=account_id,
        connect=connect,
        policy_lock=policy_lock,
        verify_broker=verify_broker,
    )


def _apply_verified_packet(packet, *, account_id, connect, policy_lock, verify_broker):
    """Shared write implementation. Call only behind manual or reviewed auto scope."""
    validate(packet, account_id)
    core.require(
        cp.manifestation(verify_broker(packet)) == packet["manifest"],
        "BROKER_EVIDENCE_CHANGED",
    )
    rows = []
    for value in packet["rows"]:
        row = dict(value)
        row["raw"] = cp.encoded(row["raw"])
        row["transaction_time"] = cp.utc(row["transaction_time"])
        rows.append(row)
    conn = connect(False)
    try:
        try:
            cp.persist(
                conn,
                workflow_id=packet["workflow_id"],
                source=packet["source"],
                after=cp.utc(packet["after"]),
                until=cp.utc(packet["until"]),
                before=cp.utc(packet["before"]) if packet["before"] else None,
                rows=rows,
                policy_lock=policy_lock,
            )
        except Exception:
            # Do not retry a possible commit. Independent consume below must
            # prove exact identity, interval, recipe and ledger contents.
            pass
    finally:
        conn.close()
    conn = connect(True)
    try:
        proof = cp.consume(
            conn,
            workflow_id=packet["workflow_id"],
            source=packet["source"],
            required_after=packet["required_after"],
            required_until=packet["until"],
            now=datetime.now(timezone.utc),
        )
        core.require(
            proof["manifest_sha256"] == cp.digest(packet["manifest"]),
            "CHECKPOINT_PACKET_MISMATCH",
        )
        with conn.cursor() as cur:
            plan = store.load_plan(
                cur,
                account_id=account_id,
                checkpoint_id=proof["checkpoint_id"],
                expected_source=packet["source"],
                expected_recipe=packet["recipe_sha256"],
            )
    finally:
        conn.rollback()
        conn.close()
    conn = connect(False)
    try:
        try:
            result = core.persist_verified_plan(
                conn,
                plan,
                policy_lock=policy_lock,
                recheck=lambda cur: store.load_plan(
                    cur,
                    account_id=account_id,
                    checkpoint_id=proof["checkpoint_id"],
                    expected_source=packet["source"],
                    expected_recipe=packet["recipe_sha256"],
                    lock=True,
                ),
            )
        except core.CommitUncertain:
            conn.close()
            conn = connect(True)
            result = core.recover_receipt(conn, plan)
    finally:
        conn.close()
    confirmed = result.get("status") in {"COMMITTED", "COMMITTED_VERIFIED"}
    gaps = bool(plan["unmatched"]) or any(
        x["status"] != "MATCHED" for x in plan["coverage"]
    )
    return {
        "status": "COMMIT_UNRESOLVED"
        if not confirmed
        else "LINKED_WITH_GAPS"
        if gaps
        else "LINKED",
        "retry_authorized": False,
        "checkpoint": proof,
        "linkage": result,
        "coverage": plan["coverage"],
        "unmatched_fills": len(plan["unmatched"]),
        "accounting_complete": False,
        "trading_operations": 0,
    }


def main(argv=None):
    os.umask(0o077)  # Prepared packets contain private broker evidence.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--required-after", help="Reviewed earliest owned-order boundary; UTC ISO time"
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--expected-sha256",
        help="Independently approved packet digest, required for apply",
    )
    args = parser.parse_args(argv)
    if args.apply:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        core.require(
            args.expected_sha256 is not None
            and packet.get("sha256") == args.expected_sha256,
            "PACKET_NOT_APPROVED",
        )
    # Lazy imports: offline tests cannot accidentally create clients or load secrets.
    from dotenv import load_dotenv
    from scripts import db
    from scripts import fetch_account_activities as ingest

    load_dotenv()
    ingest.validate_env()
    configure_readonly_transport(ingest)
    endpoint = os.environ.get("APCA_API_BASE_URL", ingest.DEFAULT_BASE_URL).rstrip("/")
    core.require(endpoint == cp.ENDPOINT, "PAPER_REQUIRED")
    response = ingest.requests.get(
        endpoint + "/v2/account", headers=ingest._build_headers(), timeout=30
    )
    core.require(response.ok, "ACCOUNT_UNAVAILABLE")
    account_id = str(core.uuid(response.json()["id"]))

    def connect(readonly):
        conn = db.get_db_conn()
        core.require(conn is not None, "DATABASE_UNAVAILABLE")
        conn.rollback()
        conn.set_session(
            readonly=readonly, autocommit=False, isolation_level="READ COMMITTED"
        )
        return conn

    if args.apply:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        validate(packet, account_id)
        # Re-fetch the same fixed interval so edited artifacts cannot fabricate fills.
        fetched = ingest.fetch_activities(
            endpoint, packet["after"], packet["until"], 100, 1000
        )
        canonical = [cp.row_value(ingest._normalize_activity(x)[0]) for x in fetched]
        core.require(
            cp.manifestation(canonical) == packet["manifest"], "BROKER_EVIDENCE_CHANGED"
        )
        result = apply_packet(
            packet,
            account_id=account_id,
            connect=connect,
            policy_lock=db._policy_write_lock,
            verify_broker=lambda _: canonical,
            expected_sha256=args.expected_sha256,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "LINKED" else 2
    else:
        core.require(args.required_after is not None, "REQUIRED_AFTER_MISSING")
        core.require(not args.packet.exists(), "PACKET_EXISTS")
        cutoff = datetime.now(timezone.utc)
        conn = connect(True)
        try:
            before = ingest.load_watermark(conn, strict=True)
        finally:
            conn.rollback()
            conn.close()
        after, until = cp.window(before, args.required_after, cutoff, 14)
        activities = ingest.fetch_activities(
            endpoint, after.isoformat(), until.isoformat(), 100, 1000
        )
        packet = prepare(
            account_id=account_id,
            source=cp.source_fingerprint(endpoint, account_id),
            before=before,
            required_after=args.required_after,
            cutoff=cutoff,
            activities=activities,
            normalize=ingest._normalize_activity,
        )
        with args.packet.open("x", encoding="utf-8") as file:
            json.dump(packet, file, sort_keys=True)
        print(
            json.dumps(
                {
                    "status": "PREPARED_ONLY",
                    "sha256": packet["sha256"],
                    "rows": len(packet["rows"]),
                    "after": packet["after"],
                    "until": packet["until"],
                    "trading_operations": 0,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
