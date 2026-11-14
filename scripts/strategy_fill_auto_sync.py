"""Bounded automatic DATA-ONLY sync; requires independently approved config hash.

No executor/monitor import and no broker mutation capability. Default CLI checks
configuration only. --run applies the reviewed standing data-maintenance scope.
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as core
from scripts import strategy_fill_linkage_store as linkage_store
from scripts import strategy_fill_sync_adapter as adapter
from scripts import strategy_data_projection as projection

PIN_FILES = {
    "scripts/strategy_fill_auto_sync.py",
    "scripts/strategy_data_projection.py",
    "scripts/strategy_fill_sync_adapter.py",
    "scripts/strategy_fill_linkage.py",
    "scripts/strategy_fill_linkage_store.py",
    "scripts/activity_ingestion_checkpoint.py",
    "scripts/fetch_account_activities.py",
    "scripts/db.py",
}


def verify_config(config, expected_sha256, root):
    core.require(core.digest(config) == expected_sha256, "CONFIG_NOT_APPROVED")
    core.require(
        config["schema"] == "jbravo.data-sync-authorization.v1"
        and config["scope"] == "AUTOMATIC_PAPER_DATA_INGEST_LINK_OBSERVE_CLOSE_ONLY",
        "CONFIG_SCOPE",
    )
    core.uuid(config["account_id"])
    core.require(config["endpoint"] == cp.ENDPOINT, "PAPER_REQUIRED")
    core.require(
        type(config["max_window_days"]) is int and 1 <= config["max_window_days"] <= 45,
        "WINDOW_LIMIT",
    )
    core.require(
        type(config["max_pages"]) is int and 1 <= config["max_pages"] <= 2000,
        "PAGE_LIMIT",
    )
    cp.utc(config["initial_required_after"])
    core.require(set(config["source_pins"]) == PIN_FILES, "CONFIG_SOURCE_SCOPE")
    root = Path(root).resolve()
    for name, expected in config["source_pins"].items():
        path = (root / name).resolve()
        core.require(root in path.parents and path.is_file(), "SOURCE_PATH_INVALID")
        core.require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected, "SOURCE_CHANGED"
        )
    return config


def read(connect, account_id):
    conn = connect(True)
    try:
        with conn.cursor() as cur:
            return projection.read_state(cur, account_id)
    finally:
        conn.rollback()
        conn.close()


def write_projection(plan, *, connect, policy_lock, recheck):
    if not plan["orders"] and not plan["signals"]:
        return {"status": "NO_CHANGE"}
    conn = connect(False)
    try:
        try:
            return projection.persist(
                conn, plan, policy_lock=policy_lock, recheck=recheck
            )
        except core.CommitUncertain:
            conn.close()
            conn = connect(True)
            recovered = projection.recover(conn, plan)
            core.require(
                recovered["status"] == "COMMITTED_VERIFIED",
                "PROJECTION_COMMIT_UNRESOLVED",
            )
            return recovered
    finally:
        conn.close()


def run(
    config,
    *,
    connect,
    policy_lock,
    get_account,
    get_order,
    fetch_activities,
    normalize,
    journal,
    now=lambda: datetime.now(timezone.utc),
):
    """Caller must verify immutable config before invoking this orchestration port."""
    account = config["account_id"]
    core.require(get_account()["id"] == account, "ACCOUNT_CHANGED")
    state = read(connect, account)
    ids = {
        o["observation"]["id"]
        for o in state["orders"]
        if o["observation"] is not None
        and o["operation"] != "DELETE"
        and o["state"] != "REJECTED"
    }
    snapshots = {oid: get_order(core.uuid(oid)) for oid in sorted(ids)}
    observed_at = now()
    snapshot = projection.snapshot_plan(
        state, snapshots, account_id=account, observed_at=observed_at
    )
    journal("snapshot_prepared", snapshot)
    result = write_projection(
        snapshot,
        connect=connect,
        policy_lock=policy_lock,
        recheck=lambda cur: projection.snapshot_plan(
            projection.read_state(cur, account, lock=True),
            snapshots,
            account_id=account,
            observed_at=observed_at,
        ),
    )
    journal("snapshot_result", result)
    conn = connect(True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT last_after FROM public.reconcile_state WHERE id=2")
            cursors = cur.fetchall()
            core.require(len(cursors) <= 1, "CURSOR_POPULATION")
            before = cursors[0][0] if cursors else None
            # Include origins of orders with no bridge yet; never infer their
            # first fill from the latest cursor. Linked orders retain old proofs.
            cur.execute(
                "SELECT min(o.claimed_at) FROM public.strategy_execution_orders_app o LEFT JOIN public.strategy_broker_orders_app b ON b.account_id=o.account_id AND b.client_order_id=o.client_order_id WHERE o.account_id=%s AND o.operation<>'DELETE' AND o.state<>'REJECTED' AND (b.broker_order_id IS NULL OR COALESCE((SELECT sum(a.qty) FROM public.strategy_fill_links_app l JOIN public.alpaca_activities a ON a.activity_pk=l.activity_pk WHERE l.account_id=o.account_id AND l.broker_order_id=b.broker_order_id),0) < COALESCE(NULLIF(o.observation->>'filled_qty','')::numeric,0))",
                (account,),
            )
            missing_origin = cur.fetchone()[0]
    finally:
        conn.rollback()
        conn.close()
    cutoff = now()
    required = (
        min(cp.utc(before), cp.utc(missing_origin))
        if before and missing_origin
        else cp.utc(before or missing_origin or config["initial_required_after"])
    )
    after, until = cp.window(before, required, cutoff, 14)
    core.require(
        timedelta(0) < until - after <= timedelta(days=config["max_window_days"]),
        "AUTOMATIC_WINDOW_EXCEEDED",
    )
    raw = fetch_activities(after.isoformat(), until.isoformat(), config["max_pages"])
    packet = adapter.prepare(
        account_id=account,
        source=cp.source_fingerprint(cp.ENDPOINT, account),
        before=before,
        required_after=required,
        cutoff=cutoff,
        activities=raw,
        normalize=normalize,
    )
    journal("activity_packet", packet)
    # This is the separately reviewed automatic scope, not an invented manual approval.
    result = adapter._apply_verified_packet(
        packet,
        account_id=account,
        connect=connect,
        policy_lock=policy_lock,
        verify_broker=lambda p: [
            cp.row_value(normalize(x)[0])
            for x in fetch_activities(p["after"], p["until"], config["max_pages"])
        ],
    )
    journal("linkage_result", result)
    core.require(result["status"] != "COMMIT_UNRESOLVED", "LINKAGE_COMMIT_UNRESOLVED")
    checkpoint = result["checkpoint"]["checkpoint_id"]

    def make_closure(cur, lock=False):
        state = projection.read_state(cur, account, lock=lock)
        linkage = linkage_store.load_plan(
            cur,
            account_id=account,
            checkpoint_id=checkpoint,
            expected_source=packet["source"],
            expected_recipe=packet["recipe_sha256"],
            lock=lock,
        )
        cur.execute(
            "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
            (linkage["plan_sha256"],),
        )
        core.require(cur.fetchall() == [(linkage,)], "CLOSURE_LINK_RECEIPT_MISSING")
        for fill in linkage["fills"]:
            cur.execute(
                "SELECT broker_order_id,fill_sha256 FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_pk=%s",
                (account, fill["activity_pk"]),
            )
            core.require(
                cur.fetchall() == [(fill["broker_order_id"], fill["fill_sha256"])],
                "CLOSURE_DURABLE_LINK_MISSING",
            )
        return projection.closure_plan(state, linkage, account_id=account)

    conn = connect(True)
    try:
        with conn.cursor() as cur:
            closure = make_closure(cur)
    finally:
        conn.rollback()
        conn.close()
    journal("closure_prepared", closure)
    closed = write_projection(
        closure,
        connect=connect,
        policy_lock=policy_lock,
        recheck=lambda cur: make_closure(cur, lock=True),
    )
    journal("closure_result", closed)
    summary = {
        "status": result["status"],
        "snapshot_changes": len(snapshot["orders"]),
        "closed_signals": len(closure["signals"]),
        "held_signals": closure["held"],
        "unresolved_claims": len(snapshot["unresolved_claims"]),
        "protected_observation_differences": len(
            snapshot["protected_observation_differences"]
        ),
        "accounting_complete": False,
        "trading_operations": 0,
    }
    if snapshot["unresolved_claims"] or snapshot["protected_observation_differences"]:
        summary["status"] = "LINKED_WITH_GAPS"
    journal("summary", summary)
    return summary


def main(argv=None):
    os.umask(0o077)  # New per-run activity/config receipts are private by default.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = verify_config(
        json.loads(args.config.read_text()), args.expected_config_sha256, root
    )
    if not args.run:
        print(json.dumps({"status": "CONFIG_VERIFIED_ONLY", "trading_operations": 0}))
        return 0
    from dotenv import load_dotenv
    from scripts import db
    from scripts import fetch_account_activities as ingest

    load_dotenv()
    ingest.validate_env()
    adapter.configure_readonly_transport(ingest)

    def connect(readonly):
        c = db.get_db_conn()
        core.require(c is not None, "DATABASE_UNAVAILABLE")
        c.rollback()
        c.set_session(
            readonly=readonly, autocommit=False, isolation_level="READ COMMITTED"
        )
        return c

    def policy_lock(c):
        db._policy_write_lock(c)
        with c.cursor() as cur:
            cur.execute(
                "SELECT deployment_id,database_id,database_name,policy_revision,current_database() FROM public.trade_disposition_policy WHERE singleton=1"
            )
            core.require(
                cur.fetchall() == [tuple(config["database_policy"])],
                "DATABASE_POLICY_CHANGED",
            )

    def get(path):
        response = ingest.requests.get(
            cp.ENDPOINT + path, headers=ingest._build_headers(), timeout=30
        )
        core.require(response.ok, "BROKER_GET_FAILED")
        return response.json()

    reports = root / "reports" / "strategy_fill_sync"
    reports.mkdir(parents=True, exist_ok=True)
    # A killed process or unresolved prior stage is not permission to start a new write.
    pending = [
        p.name
        for p in reports.iterdir()
        if p.is_dir()
        and re.fullmatch("[0-9a-f]{32}", p.name)
        and not (p / "summary.json").is_file()
    ]
    if pending or (reports / "REVIEW_REQUIRED.json").exists():
        print(
            json.dumps(
                {"status": "PRIOR_RUN_REQUIRES_RECONCILIATION", "trading_operations": 0}
            )
        )
        return 2
    lock = reports / ".run-lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print(
            json.dumps({"status": "RUN_ACTIVE_OR_INTERRUPTED", "trading_operations": 0})
        )
        return 2
    out = reports / uuid4().hex
    out.mkdir(parents=True, exist_ok=False)

    def journal(name, value):
        with (out / (name + ".json")).open("x", encoding="utf-8") as file:
            json.dump(value, file, sort_keys=True)

    journal(
        "authorization",
        {"config_sha256": args.expected_config_sha256, "scope": config["scope"]},
    )
    try:
        result = run(
            config,
            connect=connect,
            policy_lock=policy_lock,
            get_account=lambda: get("/v2/account"),
            get_order=lambda oid: get("/v2/orders/" + oid),
            fetch_activities=lambda after, until, pages: ingest.fetch_activities(
                cp.ENDPOINT, after, until, 100, pages
            ),
            normalize=ingest._normalize_activity,
            journal=journal,
        )
    except Exception as error:
        code = (
            str(error)
            if re.fullmatch(r"(?:FILL_LINK|ACT_CHECKPOINT)_[A-Z_]{1,80}", str(error))
            else "OPERATION_FAILED_OR_UNCERTAIN"
        )
        (reports / "REVIEW_REQUIRED.json").write_text(
            json.dumps({"run_id": out.name, "code": code, "retry_authorized": False})
        )
        journal(
            "failure",
            {
                "status": "STOPPED_REVIEW_STAGE_RECEIPTS",
                "retry_authorized": False,
                "trading_operations": 0,
            },
        )
        print(
            json.dumps({"status": "STOPPED_REVIEW_STAGE_RECEIPTS", "run_id": out.name})
        )
        return 2
    lock.rmdir()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "LINKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
