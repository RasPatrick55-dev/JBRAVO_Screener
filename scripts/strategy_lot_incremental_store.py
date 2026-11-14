"""Audited DB input reader, no connections, writes, broker calls or runtime entry.

Requires proposed origin-evidence migration. Missing origins/history remain holds.
The eventual writer must use lock=True inside V2 account/policy lease and compare
fresh output; previews never substitute for write-time recheck.
"""

from datetime import datetime, timezone, timedelta
from scripts import strategy_fill_linkage as core
from scripts import strategy_fill_linkage_store as store
from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_lot_accounting_v2 as lot
from scripts import strategy_order_origin_evidence as origins


def load_inputs(
    q, *, account_id, checkpoint_id, expected_source, expected_recipe, lock=False
):
    linked = store.load_plan(
        q,
        account_id=account_id,
        checkpoint_id=checkpoint_id,
        expected_source=expected_source,
        expected_recipe=expected_recipe,
        lock=lock,
    )
    q.execute(
        "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
        (linked["plan_sha256"],),
    )
    core.require(q.fetchall() == [(linked,)], "COMMITTED_LINKAGE_REQUIRED")
    suffix = " FOR SHARE" if lock else ""
    q.execute(
        "SELECT account_id,signal_id,symbol,payload FROM public.strategy_execution_signals_app WHERE account_id=%s ORDER BY signal_id"
        + suffix,
        (account_id,),
    )
    signals = store.rows(q)
    q.execute(
        "SELECT account_id,signal_id,client_order_id,operation,role,state,lineage,observation FROM public.strategy_execution_orders_app WHERE account_id=%s ORDER BY client_order_id"
        + suffix,
        (account_id,),
    )
    orders = store.rows(q)
    # Complete current owned-order ledger, not merely the latest checkpoint slice.
    owned_ids = sorted({o["observation"]["id"] for o in orders if o.get("observation")})
    owned_ledger = []
    for oid in owned_ids:
        q.execute("SELECT " + ",".join(cp.FIELDS) +
                  " FROM public.alpaca_activities WHERE activity_type='FILL' AND order_id=%s ORDER BY activity_pk"
                  + suffix, (oid,))
        owned_ledger.extend(store.rows(q))
    source = []
    canonical = {}
    for f in linked["fills"]:
        q.execute(
            "SELECT activity_pk,"
            + ",".join(cp.FIELDS)
            + " FROM public.alpaca_activities WHERE activity_pk=%s"
            + suffix,
            (f["activity_pk"],),
        )
        source.extend(store.rows(q))
        q.execute(
            "SELECT fill_sha256,checkpoint_id::text FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_id=%s",
            (account_id, f["activity_id"]),
        )
        row = q.fetchall()
        core.require(
            len(row) == 1 and row[0][0] == f["fill_sha256"], "DURABLE_FILL_LINK_CHANGED"
        )
        canonical[f["activity_id"]] = row[0][1]
    q.execute(
        "SELECT e.event,r.run_sha256,r.receipt FROM public.strategy_lot_events_app e JOIN public.strategy_lot_runs_app r USING(run_sha256) WHERE e.account_id=%s ORDER BY e.event_id",
        (account_id,),
    )
    prior = []
    for e, h, p in q.fetchall():
        core.require(
            h
            == p["run_sha256"]
            == core.digest({k: v for k, v in p.items() if k != "run_sha256"})
            and e in p["events"],
            "PRIOR_RECEIPT_CHANGED",
        )
        prior.append(e)
    # Historical budgets cannot disappear when the current ingestion window narrows.
    prior_lots = []
    referenced = sorted({e[k] for e in prior for k in
                         ("entry_activity_id", "exit_activity_id")})
    for aid in referenced:
        q.execute("SELECT l.evidence,b.capacity,r.receipt FROM public.strategy_fill_lots_app l "
                  "JOIN public.strategy_lot_budgets_app b USING(account_id,activity_id) "
                  "JOIN public.strategy_fill_link_runs_app r ON r.plan_sha256=l.evidence->>'source_plan_sha256' "
                  "WHERE l.account_id=%s AND l.activity_id=%s" + suffix, (account_id, aid))
        found = q.fetchall()
        core.require(len(found) == 1, "PRIOR_LOT_PROVENANCE_MISSING")
        evidence, capacity, original = found[0]
        old_rows = []
        old_canonical = {}
        for f in original["fills"]:
            q.execute("SELECT activity_pk," + ",".join(cp.FIELDS) +
                      " FROM public.alpaca_activities WHERE activity_pk=%s" + suffix,
                      (f["activity_pk"],))
            old_rows.extend(store.rows(q))
            q.execute("SELECT fill_sha256,checkpoint_id::text FROM public.strategy_fill_links_app "
                      "WHERE account_id=%s AND activity_id=%s", (account_id, f["activity_id"]))
            rows = q.fetchall()
            core.require(len(rows) == 1 and rows[0][0] == f["fill_sha256"],
                         "PRIOR_FILL_LINK_CHANGED")
            old_canonical[f["activity_id"]] = rows[0][1]
        rebuilt = {x["activity_id"]: x for x in lot.source_lots(
            original, old_rows, canonical_checkpoints=old_canonical)}
        core.require(rebuilt.get(aid) == evidence and evidence["account_id"] == account_id
                     and capacity == core.number(evidence["quantity"], positive=True),
                     "PRIOR_LOT_CHANGED")
        prior_lots.append(evidence)
    q.execute(
        "SELECT r.receipt,r.receipt_sha256,l.receipt AS linkage_receipt FROM public.strategy_order_origin_receipts_app r JOIN public.strategy_fill_link_runs_app l ON l.plan_sha256=r.source_plan_sha256 WHERE r.account_id=%s ORDER BY r.broker_order_id,r.receipt_sha256",
        (account_id,),
    )
    evidence = {}
    evidence_receipts = {}
    origin_histories = {}
    history_options = {}
    evidence_ranks = {}
    creation = {}
    current = {
        o["observation"]["id"]: o["observation"] for o in orders if o.get("observation")
    }
    for p, h, linked_receipt in q.fetchall():
        core.require(
            p["schema"] == origins.VERSION
            and p["account_id"] == account_id
            and h
            == p["receipt_sha256"]
            == core.digest({k: v for k, v in p.items() if k != "receipt_sha256"}),
            "ORIGIN_RECEIPT_CHANGED",
        )
        # Legacy minimal receipts cannot qualify an automatic origin.
        try:
            origins.validate_fetch_receipt(p)
        except (ValueError, KeyError, TypeError):
            continue
        oid = p["broker_order_id"]
        raw = p["raw_order"]
        core.require(
            any(
                x["account_id"] == account_id and x["broker_order_id"] == oid
                for x in linked_receipt["orders"]
            ),
            "ORIGIN_LINKAGE_MISSING",
        )
        core.require(
            oid not in creation or creation[oid] == raw["created_at"],
            "CONFLICTING_ORIGIN_EVIDENCE",
        )
        creation[oid] = raw["created_at"]
        if oid in current and all(raw.get(k) == v for k, v in current[oid].items()):
            history=p.get("bounded_history")
            if history:
                history_options.setdefault(oid,{})[history["history_sha256"]]=(history,h,raw)
            rank=(1 if history else 0,
                  lot.timestamp(history["query_until"] if history else p["fetched_at"])["start_ns"], h)
            if oid not in evidence_ranks or rank > evidence_ranks[oid]:
                evidence_ranks[oid]=rank
                evidence[oid] = raw
                evidence_receipts[oid] = h
                if history:
                    origin_histories[oid] = history
    histories = {}
    for s in signals:
        claims = [
            o["observation"]
            for o in orders
            if o["signal_id"] == s["signal_id"] and o.get("observation")
        ]
        if not claims or any(o["id"] not in evidence for o in claims):
            continue
        earliest = min(
            lot.timestamp(evidence[o["id"]]["created_at"])["lower_ns"] for o in claims
        )
        latest = max(lot.timestamp(o["updated_at"])["end_ns"] for o in claims)
        # Filter by this cohort's origin/terminal bounds before limiting rows.
        # Match timestamp(cp.utc(...).isoformat()) uncertainty exactly: whole
        # seconds have +/-1s, fractional DB timestamps +/-1us. Strict ns
        # inequalities are rounded inward to integer DB microseconds before LIMIT.
        epoch = datetime(1970,1,1,tzinfo=timezone.utc)
        q.execute("SELECT checkpoint_id::text,source_fingerprint,recipe_sha256,query_axis,query_after,query_until,persisted_manifest,fetched_count,manifest_sha256 "
                  "FROM public.activity_ingestion_checkpoints WHERE source_fingerprint=%s AND recipe_sha256=%s "
                  "AND query_axis='ALPACA_CREATED_AT' "
                  "AND query_after + CASE WHEN extract(microseconds FROM query_after)::numeric %% 1000000 = 0 THEN interval '1 second' ELSE interval '1 microsecond' END <= %s "
                  "AND query_until - CASE WHEN extract(microseconds FROM query_until)::numeric %% 1000000 = 0 THEN interval '1 second' ELSE interval '1 microsecond' END >= %s "
                  "ORDER BY query_until DESC,checkpoint_id LIMIT 1" + suffix,
                  (expected_source,expected_recipe,
                   epoch+timedelta(microseconds=(earliest-1)//1000),
                   epoch+timedelta(microseconds=latest//1000+1)))
        proofs = store.rows(q)
        for p in proofs:
            if p["query_axis"] != "ALPACA_CREATED_AT":
                continue
            if not (
                lot.timestamp(cp.utc(p["query_after"]).isoformat())["end_ns"] < earliest
                and lot.timestamp(cp.utc(p["query_until"]).isoformat())["lower_ns"]
                > latest
            ):
                continue
            ledger = []
            manifest = p["persisted_manifest"]
            core.require(len(manifest) <= core.MAX_ROWS, "HISTORY_ROW_LIMIT")
            for offset in range(0, len(manifest), 200):
                part = manifest[offset : offset + 200]
                keys = [(x["activity_id"], x["activity_type"]) for x in part]
                q.execute(
                    "SELECT "
                    + ",".join(cp.FIELDS)
                    + " FROM public.alpaca_activities WHERE (activity_id,activity_type) IN ("
                    + ",".join("(%s,%s)" for _ in keys)
                    + ") ORDER BY activity_pk"
                    + suffix,
                    tuple(x for pair in keys for x in pair),
                )
                ledger.extend(store.rows(q))
            histories[s["signal_id"]] = (p, ledger)
            break
        if s["signal_id"] not in histories:
            common=set.intersection(*(set(history_options.get(o["id"],{})) for o in claims))
            covering=[]
            for digest in common:
                h=history_options[claims[0]["id"]][digest][0]
                if (lot.timestamp(h["query_after"])["end_ns"] < earliest and
                    lot.timestamp(h["query_until"])["lower_ns"] > latest and
                    h["recipe_sha256"]==expected_recipe):
                    covering.append((lot.timestamp(h["query_until"])["start_ns"],digest,h))
            if covering:
                _,digest,h=max(covering,key=lambda item:(item[0],item[1]))
                histories[s["signal_id"]]=(h,h["rows"])
                for o in claims:
                    evidence_receipts[o["id"]]=history_options[o["id"]][digest][1]
                    evidence[o["id"]]=history_options[o["id"]][digest][2]
    # Missing retained full-origin proof has an explicit incomplete default.
    default = {
        "source_fingerprint": "MISSING",
        "recipe_sha256": expected_recipe,
        "query_axis": "MISSING",
        "persisted_manifest": [],
        "manifest_sha256": core.digest([]),
        "fetched_count": 0,
    }
    return dict(
        account_id=account_id,
        linked=linked,
        source_rows=source,
        canonical_checkpoints=canonical,
        signals=signals,
        orders=orders,
        checkpoint=default,
        checkpoint_rows=[],
        expected_recipe=expected_recipe,
        origins=evidence,
        origin_receipt_sha256s=evidence_receipts,
        prior_events=prior,
        prior_lots=prior_lots,
        owned_ledger_rows=owned_ledger,
        history_by_signal=histories,
    )
