"""LOCAL PREVIEW ONLY: bounded automatic lot eligibility, no clients or writers.

Inputs must later come from an audited loader of committed linkage/checkpoints,
current owner state, immutable raw order-origin evidence and committed events.
A self-consistent input dictionary is NOT authorization or provenance by itself.
"""

from collections import defaultdict
from decimal import Decimal, localcontext
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot
from scripts import activity_ingestion_checkpoint as cp

VERSION = "jbravo.auto-lot-preview.v1"


def prior_usage(lots, events, account):
    ids = {x["activity_id"]: x for x in lots}
    core.require(len(ids) == len(lots), "DUPLICATE_LOT")
    by_event = {e["event_id"]: e for e in events}
    core.require(len(by_event) == len(events), "DUPLICATE_EVENT")
    reversed_ids = set()
    for e in events:
        core.require(
            e["account_id"] == account and lot.event_id(e) == e["event_id"],
            "PRIOR_EVENT_INVALID",
        )
        a, b = e["entry_activity_id"], e["exit_activity_id"]
        core.require(a in ids and b in ids, "PRIOR_HISTORY_MISSING")
        core.require(
            ids[a]["signal_id"] == ids[b]["signal_id"]
            and ids[a]["side"] == "buy"
            and ids[b]["side"] == "sell",
            "PRIOR_CROSS_COHORT",
        )
        if e["kind"] == "REVERSE":
            original = by_event.get(e["reverses"])
            core.require(
                original is not None
                and original["kind"] == "ALLOCATE"
                and e["reverses"] not in reversed_ids,
                "INVALID_REVERSAL",
            )
            core.require(
                all(
                    original[k] == e[k]
                    for k in [
                        "account_id",
                        "entry_activity_id",
                        "exit_activity_id",
                        "quantity",
                    ]
                ),
                "REVERSAL_MISMATCH",
            )
            reversed_ids.add(e["reverses"])
        else:
            core.require(e["kind"] == "ALLOCATE", "PRIOR_KIND_INVALID")
    used = {k: Decimal(0) for k in ids}
    for e in events:
        if e["kind"] == "ALLOCATE" and e["event_id"] not in reversed_ids:
            for k in ["entry_activity_id", "exit_activity_id"]:
                used[e[k]] += core.number(e["quantity"], positive=True)
    core.require(
        all(
            0 <= used[k] <= core.number(ids[k]["quantity"], positive=True) for k in ids
        ),
        "PRIOR_BUDGET_INVALID",
    )
    return used


def preview(
    *,
    account_id,
    linked,
    source_rows,
    canonical_checkpoints,
    signals,
    orders,
    checkpoint,
    checkpoint_rows,
    expected_recipe,
    origins,
    prior_events,
    history_by_signal=None,
    prior_lots=(),
    owned_ledger_rows=None,
    origin_receipt_sha256s=None,
):
    """Compute proposals only. No-op is never persisted, particularly existing THC.

    Minimal automatic scope: one uniquely owned filled entry and one filled exit,
    balanced exact FILL quantities, one full-origin checkpoint, precise chronology.
    Other cohorts hold independently. Runtime dispatch is deliberately unavailable.
    """
    with localcontext() as ctx:
        ctx.prec = 100
        return _preview(
            account_id,
            linked,
            source_rows,
            canonical_checkpoints,
            signals,
            orders,
            checkpoint,
            checkpoint_rows,
            expected_recipe,
            origins,
            prior_events,
            history_by_signal or {},
            prior_lots,
            source_rows if owned_ledger_rows is None else owned_ledger_rows,
            origin_receipt_sha256s or {},
        )


def _preview(
    account,
    linked,
    source_rows,
    canonical,
    signals,
    orders,
    checkpoint,
    checkpoint_rows,
    expected_recipe,
    origins,
    prior,
    histories,
    prior_lots,
    owned_ledger,
    origin_receipts,
):
    core.require(
        linked["plan_sha256"]
        == core.digest({k: v for k, v in linked.items() if k != "plan_sha256"}),
        "LINKAGE_DIGEST",
    )
    lots = lot.source_lots(linked, source_rows, canonical_checkpoints=canonical)
    core.require(all(x["account_id"] == account for x in lots), "ACCOUNT")
    all_lots = {x["activity_id"]: x for x in prior_lots}
    core.require(len(all_lots) == len(prior_lots), "DUPLICATE_PRIOR_LOT")
    for x in lots:
        previous = all_lots.get(x["activity_id"])
        if previous is not None:
            core.require(all(previous[k] == x[k] for k in x if k != "source_plan_sha256"),
                         "PRIOR_LOT_CHANGED")
        all_lots[x["activity_id"]] = x
    core.require(all(x["account_id"] == account for x in all_lots.values()), "ACCOUNT")
    used = prior_usage(list(all_lots.values()), prior, account)
    groups = defaultdict(list)
    for x in lots:
        groups[x["signal_id"]].append(x)
    signal_map = {s["signal_id"]: s for s in signals if s["account_id"] == account}
    core.require(
        len(signal_map) == sum(s["account_id"] == account for s in signals),
        "DUPLICATE_SIGNAL",
    )
    for sid in signal_map:
        groups.setdefault(sid, [])
    held = []
    candidates = []
    noops = []
    default_history = (checkpoint, checkpoint_rows)
    for sid, fills in sorted(groups.items()):
        checkpoint, checkpoint_rows = histories.get(sid, default_history)

        def hold(reason):
            held.append({"signal_id": sid, "reason": reason})

        if not fills:
            hold("NO_PROVEN_FILLS")
            continue
        # Existing receipts do not hide later ledger corrections on consumed orders.
        cohort_order_ids = {f["broker_order_id"] for f in fills}
        observed_rows = [r for r in owned_ledger if r["activity_type"] == "FILL"
                         and r["order_id"] in cohort_order_ids]
        observed_hashes = {r["activity_id"]: cp.digest(cp.row_value({k:r[k] for k in cp.FIELDS}))
                           for r in observed_rows}
        if len(observed_hashes) != len(observed_rows) or observed_hashes != {
                f["activity_id"]:f["fill_sha256"] for f in fills}:
            hold("CURRENT_LEDGER_COHORT_SET_MISMATCH")
            continue
        # Existing audited allocations are not rewritten merely to refresh evidence.
        cohort_prior = [
            e
            for e in prior
            if e["entry_activity_id"] in {f["activity_id"] for f in fills}
        ]
        if any(e["kind"] == "REVERSE" for e in cohort_prior):
            hold("CORRECTION_REVIEW_REQUIRED")
            continue
        if all(used[f["activity_id"]] == core.number(f["quantity"]) for f in fills):
            noops.append(
                {
                    "signal_id": sid,
                    "reason": "ALREADY_FULLY_ALLOCATED",
                    "existing_event_ids": sorted(e["event_id"] for e in cohort_prior),
                }
            )
            continue
        signal = signal_map.get(sid)
        if signal is None:
            hold("CURRENT_SIGNAL_MISSING")
            continue
        if signal["payload"].get("execution_owner"):
            hold("PROTECTED_EXTERNAL_OWNERSHIP")
            continue
        claims = [
            o for o in orders if o["account_id"] == account and o["signal_id"] == sid
        ]
        if any(o["state"] != "OBSERVED" or not o.get("observation") for o in claims):
            hold("UNRESOLVED_ORDER_CLAIM")
            continue
        if len(claims) != 2 or {o["role"] for o in claims} != {"entry", "stop"}:
            hold("COMPLEX_ORDER_COHORT_REVIEW_REQUIRED")
            continue
        raw = [o["observation"] for o in claims]
        if any(o.get("replaces") or o.get("replaced_by") for o in raw):
            hold("REPLACEMENT_CHAIN_REVIEW_REQUIRED")
            continue
        if any(o.get("status") != "filled" for o in raw):
            hold("NONTERMINAL_OR_PARTIAL_ORDER")
            continue
        if {o.get("side") for o in raw} != {"buy", "sell"}:
            hold("ORDER_SIDE_AMBIGUOUS")
            continue
        if any(o.get("symbol") != signal["symbol"] for o in raw):
            hold("SYMBOL_MISMATCH")
            continue
        byorder = {o["id"]: o for o in raw}
        linked_orders = [o for o in linked["orders"] if o["signal_id"] == sid]
        if set(byorder) != {o["broker_order_id"] for o in linked_orders}:
            hold("ORDER_UNIVERSE_CHANGED")
            continue
        if any(
            o["observation"]["client_order_id"] != o["client_order_id"] for o in claims
        ):
            hold("ORDER_OWNERSHIP_MISMATCH")
            continue
        by_client = {o["client_order_id"]: o for o in linked_orders}
        if any(
            o["client_order_id"] not in by_client
            or core.digest(o["lineage"])
            != by_client[o["client_order_id"]]["lineage_sha256"]
            or o["lineage"].get("handoff_id")
            != signal["payload"]["handoff"]["handoff_id"]
            or o["lineage"].get("strategy_contract")
            != signal["payload"]["handoff"]["strategy_contract"]
            or o["lineage"].get("signal_id") != sid
            for o in claims
        ):
            hold("OWNERSHIP_LINEAGE_CHANGED")
            continue
        quantities = {
            oid: sum(
                (
                    core.number(f["quantity"], positive=True)
                    for f in fills
                    if f["broker_order_id"] == oid
                ),
                Decimal(0),
            )
            for oid in byorder
        }
        if any(
            quantities[oid] != core.number(o["filled_qty"])
            or core.number(o["filled_qty"]) != core.number(o["qty"])
            for oid, o in byorder.items()
        ):
            hold("INCOMPLETE_ORDER_FILLS")
            continue
        buy = sum(
            (core.number(f["quantity"]) for f in fills if f["side"] == "buy"),
            Decimal(0),
        )
        sell = sum(
            (core.number(f["quantity"]) for f in fills if f["side"] == "sell"),
            Decimal(0),
        )
        if not sell:
            hold("NO_PROVEN_EXIT")
            continue
        if buy != sell:
            hold("EXIT_ENTRY_QUANTITY_MISMATCH")
            continue
        if any(
            oid not in origins or not origins[oid].get("created_at") for oid in byorder
        ):
            hold("IMMUTABLE_ORDER_ORIGIN_REQUIRED")
            continue
        if any(
            any(origins[oid].get(k) != v for k, v in o.items())
            for oid, o in byorder.items()
        ):
            hold("ORIGIN_SNAPSHOT_MISMATCH")
            continue
        if any(
            lot.timestamp(origins[f["broker_order_id"]]["created_at"])["end_ns"]
            >= f["ordering"]["lower_ns"]
            for f in fills
        ):
            hold("ORDER_ORIGIN_AFTER_OR_OVERLAPS_FILL")
            continue
        # Only one full-origin fetch supported in this first increment; no gap inference.
        proofok = (
            checkpoint["source_fingerprint"]
            == cp.source_fingerprint(cp.ENDPOINT, account)
            and checkpoint["recipe_sha256"] == expected_recipe
            and checkpoint["query_axis"] == "ALPACA_CREATED_AT"
            and cp.manifestation(checkpoint_rows) == checkpoint["persisted_manifest"]
            and cp.digest(checkpoint["persisted_manifest"])
            == checkpoint["manifest_sha256"]
            and checkpoint["fetched_count"] == len(checkpoint["persisted_manifest"])
        )
        if not proofok:
            hold("CHECKPOINT_PROVENANCE_INCOMPLETE")
            continue
        after = lot.timestamp(cp.utc(checkpoint["query_after"]).isoformat())["end_ns"]
        until = lot.timestamp(cp.utc(checkpoint["query_until"]).isoformat())["lower_ns"]
        if any(
            not (
                after < lot.timestamp(origins[oid]["created_at"])["lower_ns"]
                and until > lot.timestamp(o["updated_at"])["end_ns"]
            )
            for oid, o in byorder.items()
        ):
            hold("FULL_ORIGIN_HISTORY_NOT_COVERED")
            continue
        manifest = {
            (x["activity_id"], x["activity_type"]): x["sha256"]
            for x in checkpoint["persisted_manifest"]
        }
        if any(
            manifest.get((f["activity_id"], "FILL")) != f["fill_sha256"] for f in fills
        ):
            hold("COHORT_FILL_NOT_IN_FULL_FETCH")
            continue
        scoped = [r for r in checkpoint_rows
                  if r["activity_type"] == "FILL" and r["order_id"] in byorder]
        expected = {f["activity_id"]: f["fill_sha256"] for f in fills}
        actual = {r["activity_id"]: cp.digest(cp.row_value({k: r[k] for k in cp.FIELDS}))
                  for r in scoped}
        if len(actual) != len(scoped) or actual != expected:
            hold("FULL_FETCH_COHORT_SET_MISMATCH")
            continue
        current_scoped = [r for r in owned_ledger
                          if r["activity_type"] == "FILL" and r["order_id"] in byorder]
        current_hashes = {r["activity_id"]: cp.digest(cp.row_value({k:r[k] for k in cp.FIELDS}))
                          for r in current_scoped}
        if len(current_hashes) != len(current_scoped) or current_hashes != expected:
            hold("CURRENT_LEDGER_COHORT_SET_MISMATCH")
            continue
        qualification = {
            "schema": VERSION,
            "signal_id": sid,
            "source_plan_sha256": linked["plan_sha256"],
            "checkpoint_id": checkpoint["checkpoint_id"],
            "history_proof_type": checkpoint.get("proof_type", "COMMITTED_INGESTION_CHECKPOINT"),
            "history_evidence_sha256": checkpoint.get("history_sha256"),
            "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
            "origin_receipt_sha256s": {oid: origin_receipts.get(oid) for oid in sorted(byorder)},
            "origin_snapshots_sha256": core.digest(
                {oid: origins[oid] for oid in sorted(byorder)}
            ),
            "source_rows_manifest_sha256": core.digest(fills),
        }
        history = {
            "manifest_sha256": core.digest(fills),
            "review_id": core.digest(qualification),
            "qualification": qualification,
            "basis": "AUTOMATED_BOUNDED_COHORT_CANDIDATE_NOT_RUNTIME_AUTHORITY",
        }
        packet = lot.plan(
            fills, account_id=account, prior_events=cohort_prior, history_review=history
        )
        if packet["held"]:
            held.extend(
                {"signal_id": sid, "reason": h["reason"]} for h in packet["held"]
            )
            continue
        if packet["events"]:
            candidates.append(
                {"signal_id": sid, "packet": packet, "qualification": qualification}
            )
        else:
            noops.append({"signal_id": sid, "reason": "NO_RESIDUAL_ALLOCATION"})
    return {
        "schema": VERSION,
        "account_id": account,
        "candidates": candidates,
        "held": held,
        "noops": noops,
        "execution_allowed": False,
        "runtime_evidence_loader_required": True,
        "net_pnl": None,
        "gross_pnl": None,
        "fees_status": "UNKNOWN",
        "currency_status": "UNKNOWN",
        "accounting_complete": False,
        "publication_allowed": False,
        "trading_operations": 0,
    }
