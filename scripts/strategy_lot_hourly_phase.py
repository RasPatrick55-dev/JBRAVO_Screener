"""Pure hourly handoff validation. No scheduler, network, subprocess or write API.

The caller must obtain completion and receipts from the actual fresh sync run,
not exit-code success or a previous slot, and use an audited readonly DB loader.
"""

import hashlib
import json
import re
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_incremental_preview as auto

REQUIRED = {
    "activity_packet.json",
    "authorization.json",
    "snapshot_prepared.json",
    "snapshot_result.json",
    "linkage_result.json",
    "closure_prepared.json",
    "closure_result.json",
    "summary.json",
}


def validate_sync(
    *,
    completion_bytes,
    expected_completion_sha256,
    receipt_bytes,
    expected_slot,
):
    core.require(
        hashlib.sha256(completion_bytes).hexdigest() == expected_completion_sha256,
        "SYNC_COMPLETION_PIN",
    )
    c = json.loads(completion_bytes)
    core.require(
        c["status"] in {"COMPLETED", "COMPLETED_EXPECTED_HOLDS"}, "NO_COMPLETED_SYNC"
    )
    core.require(
        re.fullmatch("[0-9a-f]{32}", c["native_run_id"]) is not None
        and c["slot"] == expected_slot,
        "SYNC_RUN_OR_SLOT_MISMATCH",
    )
    core.require(
        c["trading_operations"] == 0 and c["accounting_complete"] is False, "SYNC_SCOPE"
    )
    core.require(
        set(c["native_receipts"]) == REQUIRED and set(receipt_bytes) == REQUIRED,
        "SYNC_RECEIPT_SET",
    )
    for name, pin in c["native_receipts"].items():
        raw = receipt_bytes[name]
        core.require(
            len(raw) == pin["bytes"]
            and hashlib.sha256(raw).hexdigest() == pin["sha256"],
            "SYNC_RECEIPT_PIN",
        )
    linkage = json.loads(receipt_bytes["linkage_result.json"])
    core.require(
        linkage["status"] == "LINKED"
        and linkage["linkage"]["status"] in {"COMMITTED", "COMMITTED_VERIFIED"}
        and linkage["trading_operations"] == 0,
        "COMMITTED_LINKAGE_REQUIRED",
    )
    core.require(linkage["checkpoint"] == c["checkpoint"], "SYNC_CHECKPOINT_MISMATCH")
    return c, linkage


def after_sync_preview(*, completion_bytes, expected_completion_sha256,
                       receipt_bytes, expected_slot, load_inputs):
    c, linkage = validate_sync(completion_bytes=completion_bytes,
        expected_completion_sha256=expected_completion_sha256,
        receipt_bytes=receipt_bytes, expected_slot=expected_slot)
    inputs = load_inputs(
        run_id=c["native_run_id"],
        checkpoint_id=c["checkpoint"]["checkpoint_id"],
        plan_sha256=linkage["linkage"]["plan_sha256"],
    )
    core.require(
        inputs["linked"]["plan_sha256"] == linkage["linkage"]["plan_sha256"],
        "LOADED_LINKAGE_MISMATCH",
    )
    result = auto.preview(**inputs)
    result.update(
        sync_run_id=c["native_run_id"],
        sync_slot=c["slot"],
        sync_completion_sha256=expected_completion_sha256,
        phase="LOT_PREVIEW_ONLY_NOT_PERSISTED",
    )
    return result
