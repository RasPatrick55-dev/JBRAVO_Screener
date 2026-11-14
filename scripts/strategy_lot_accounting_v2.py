"""Candidate-only append-only fill allocation. No broker clients or runtime DDL.

This is relationship accounting, not certified accounting or a tax-lot engine.
History qualification is a separate reviewed input, never inferred from a cursor.
"""

from datetime import datetime, timezone
from decimal import Decimal, localcontext
from functools import wraps
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re

from scripts import activity_ingestion_checkpoint as cp
from scripts import strategy_fill_linkage as core

VERSION = "jbravo.lot-accounting.v2"
TIME = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$"
)


def require(ok, code):
    if not ok:
        raise ValueError("LOT_" + code)


def timestamp(value):
    """Integer uncertainty interval; quantization may round OR truncate.

    start_ns is the reported value, not asserted true execution chronology.
    One source-resolution unit on either side bounds both conventions.
    """
    require(isinstance(value, str), "RAW_TIME_REQUIRED")
    m = TIME.fullmatch(value)
    require(m is not None, "TIME_FORMAT")
    fraction = m[2] or ""
    zone = m[3]
    if zone != "Z":
        require(int(zone[1:3]) <= 23 and int(zone[4:6]) <= 59, "TIME_OFFSET")
    dt = datetime.fromisoformat(m[1] + ("+00:00" if zone == "Z" else zone))
    delta = dt.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = delta.days * 86400 + delta.seconds
    start = seconds * 10**9 + int(fraction.ljust(9, "0") or "0")
    return {
        "raw": value,
        "start_ns": start,
        "lower_ns": start - 10 ** (9 - len(fraction)),
        "end_ns": start + 10 ** (9 - len(fraction)),
        "fraction_digits": len(fraction),
    }


def source_lots(link_plan, source_rows, *, canonical_checkpoints=None):
    require(
        link_plan["plan_sha256"]
        == core.digest({k: v for k, v in link_plan.items() if k != "plan_sha256"}),
        "SOURCE_CHANGED",
    )
    require(link_plan["schema"] == core.VERSION, "SOURCE_VERSION")
    rows = {r["activity_pk"]: r for r in source_rows}
    require(len(rows) == len(source_rows), "DUPLICATE_SOURCE")
    orders = {o["broker_order_id"]: o for o in link_plan["orders"]}
    result = []
    for f in link_plan["fills"]:
        row = rows[f["activity_pk"]]
        canonical = cp.row_value({k: row[k] for k in cp.FIELDS})
        require(cp.digest(canonical) == f["fill_sha256"], "SOURCE_HASH")
        o = orders[f["broker_order_id"]]
        raw_time = canonical["raw"].get("transaction_time")
        # The DB canonical timestamp proves only microseconds. Raw broker JSON,
        # when present, preserves original precision and must agree at that scale.
        if raw_time is not None:
            order_time = timestamp(raw_time)
            require(
                order_time["start_ns"] // 1000
                == timestamp(canonical["transaction_time"])["start_ns"] // 1000,
                "RAW_TIME_MISMATCH",
            )
            time_source = "PERSISTED_ACTIVITY_RAW"
        else:
            order_time = timestamp(canonical["transaction_time"])
            order_time["fraction_digits"] = 6
            order_time["end_ns"] = order_time["start_ns"] + 1000
            order_time["lower_ns"] = order_time["start_ns"] - 1000
            time_source = "DB_MICROSECOND_ONLY"
        require(row["side"] == o["side"] and row["symbol"] == o["symbol"], "IDENTITY")
        result.append(
            dict(
                account_id=o["account_id"],
                activity_id=f["activity_id"],
                activity_pk=f["activity_pk"],
                signal_id=o["signal_id"],
                symbol=o["symbol"],
                side=o["side"],
                quantity=f["quantity"],
                price=f["price"],
                fill_sha256=f["fill_sha256"],
                source_plan_sha256=link_plan["plan_sha256"],
                broker_order_id=f["broker_order_id"],
                checkpoint_id=(
                    canonical_checkpoints[f["activity_id"]]
                    if canonical_checkpoints is not None
                    else f["checkpoint_id"]
                ),
                source_row=canonical,
                ordering=order_time,
                time_source=time_source,
                currency=None,
                fees=None,
            )
        )
    require(
        len({(r["account_id"], r["activity_id"]) for r in result}) == len(result),
        "DUPLICATE_FILL",
    )
    return sorted(result, key=lambda x: (x["account_id"], x["activity_id"]))


def event_id(event):
    return core.digest({k: v for k, v in event.items() if k != "event_id"})


def exact_context(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with localcontext() as ctx:
            ctx.prec = 128
            return fn(*args, **kwargs)

    return wrapped


@exact_context
def plan(lots, *, account_id, prior_events=(), history_review=None, recipe=VERSION):
    """Conservative incremental FIFO; holds late evidence and ambiguous intervals.

    history_review must identify this exact bounded input manifest. It is an
    external qualification record, NOT an assertion this module can certify.
    Prior events must be loaded from committed receipts, never local guesses.
    """
    require(all(item["account_id"] == account_id for item in lots), "ACCOUNT")
    require(len(lots) <= core.MAX_ROWS, "LOT_LIMIT")
    for lot in lots:
        n = core.number(lot["quantity"], positive=True)
        require(
            len(n.as_tuple().digits) <= 38 and -18 <= n.as_tuple().exponent <= 18,
            "QUANTITY_PRECISION",
        )
    manifest = core.digest(lots)
    prior = list(prior_events)
    require(len({e["event_id"] for e in prior}) == len(prior), "DUPLICATE_EVENT")
    by_id = {x["activity_id"]: x for x in lots}
    require(len(by_id) == len(lots), "DUPLICATE_LOT")
    used = {key: Decimal(0) for key in by_id}
    reversed_ids = {e["reverses"] for e in prior if e["kind"] == "REVERSE"}
    effective = [
        e
        for e in prior
        if e["kind"] == "ALLOCATE" and e["event_id"] not in reversed_ids
    ]
    for e in effective:
        require(e["account_id"] == account_id, "PRIOR_ACCOUNT")
        for key in (e["entry_activity_id"], e["exit_activity_id"]):
            require(key in by_id, "PRIOR_HISTORY_MISSING")
            used[key] += core.number(e["quantity"], positive=True)
    proposed, held = [], []
    if (
        not history_review
        or history_review.get("manifest_sha256") != manifest
        or not history_review.get("review_id")
    ):
        held.append({"reason": "BOUNDED_HISTORY_REVIEW_REQUIRED"})
    groups = {}
    for lot in lots:
        require(
            used[lot["activity_id"]] <= core.number(lot["quantity"], positive=True),
            "PRIOR_OVERALLOCATED",
        )
        groups.setdefault((lot["signal_id"], lot["symbol"]), []).append(lot)
    if not held:
        for group, fills in sorted(groups.items()):
            if any(
                e["kind"] == "REVERSE"
                and by_id[e["entry_activity_id"]]["signal_id"] == group[0]
                and by_id[e["entry_activity_id"]]["symbol"] == group[1]
                for e in prior
            ):
                held.append(
                    {
                        "signal_id": group[0],
                        "symbol": group[1],
                        "reason": "CORRECTION_REVIEW_REQUIRED",
                    }
                )
                continue
            fills.sort(key=lambda x: (x["ordering"]["start_ns"], x["activity_id"]))
            # Reject overlaps even for equal-side fills: deterministic tie breaking
            # cannot certify chronology and can change economic lot selection.
            if any(
                a["ordering"]["end_ns"] > b["ordering"]["lower_ns"]
                for a, b in zip(fills, fills[1:])
            ):
                held.append(
                    {
                        "signal_id": group[0],
                        "symbol": group[1],
                        "reason": "ORDERING_PRECISION_OVERLAP",
                    }
                )
                continue
            exits = [
                by_id[e["exit_activity_id"]]["ordering"]["end_ns"]
                for e in effective
                if by_id[e["entry_activity_id"]]["signal_id"] == group[0]
                and by_id[e["entry_activity_id"]]["symbol"] == group[1]
            ]
            known = {
                key
                for e in prior
                for key in (e["entry_activity_id"], e["exit_activity_id"])
            }
            if exits and any(
                f["activity_id"] not in known and f["ordering"]["start_ns"] < max(exits)
                for f in fills
            ):
                held.append(
                    {
                        "signal_id": group[0],
                        "symbol": group[1],
                        "reason": "LATE_EVIDENCE_REVIEW_REQUIRED",
                    }
                )
                continue
            buys, pending = [], []
            balances = dict(used)
            for fill in fills:
                key = fill["activity_id"]
                if fill["side"] == "buy":
                    buys.append(fill)
                    continue
                require(fill["side"] == "sell", "UNSUPPORTED_SIDE")
                remaining = core.number(fill["quantity"], positive=True) - balances[key]
                for buy in buys:
                    bid = buy["activity_id"]
                    qty = min(
                        remaining,
                        core.number(buy["quantity"], positive=True) - balances[bid],
                    )
                    if qty <= 0:
                        continue
                    event = dict(
                        kind="ALLOCATE",
                        account_id=account_id,
                        entry_activity_id=bid,
                        exit_activity_id=key,
                        quantity=str(qty),
                        recipe=recipe,
                        reverses=None,
                        reason=None,
                        evidence_sha256=manifest,
                        replaces_reversal=None,
                    )
                    event["event_id"] = event_id(event)
                    pending.append(event)
                    balances[bid] += qty
                    balances[key] += qty
                    remaining -= qty
                if remaining:
                    held.append(
                        {
                            "signal_id": group[0],
                            "symbol": group[1],
                            "reason": "EXIT_EXCEEDS_PROVEN_ENTRY",
                        }
                    )
                    pending = []
                    break
            proposed.extend(pending)
    packet = dict(
        schema=VERSION,
        recipe=recipe,
        account_id=account_id,
        lots=lots,
        manifest_sha256=manifest,
        prior_events=prior,
        events=proposed,
        held=held,
        history_review=history_review,
        net_pnl=None,
        gross_pnl=None,
        fees_status="UNKNOWN",
        currency_status="UNKNOWN",
        accounting_complete=False,
        publication_allowed=False,
        trading_operations=0,
    )
    packet["run_sha256"] = core.digest(packet)
    return packet


def reversal(original, *, reason, evidence_sha256):
    """Full-edge reversal only. No inferred redistribution or silent correction."""
    require(
        original["kind"] == "ALLOCATE"
        and bool(reason)
        and re.fullmatch("[0-9a-f]{64}", evidence_sha256),
        "REVERSAL_EVIDENCE",
    )
    e = {k: v for k, v in original.items() if k != "event_id"}
    e.update(
        kind="REVERSE",
        reverses=original["event_id"],
        reason=reason,
        evidence_sha256=evidence_sha256,
        replaces_reversal=None,
    )
    e["event_id"] = event_id(e)
    return e


def load_packet(
    cursor,
    *,
    account_id,
    checkpoint_id,
    expected_source,
    expected_recipe,
    history_review=None,
    recipe=VERSION,
    lock=False,
):
    """Rebuild from checkpoint-verified ledger AND committed linkage receipts."""
    from scripts.strategy_fill_linkage_store import load_plan, rows

    linked = load_plan(
        cursor,
        account_id=account_id,
        checkpoint_id=checkpoint_id,
        expected_source=expected_source,
        expected_recipe=expected_recipe,
        lock=lock,
    )
    cursor.execute(
        "SELECT receipt FROM public.strategy_fill_link_runs_app WHERE plan_sha256=%s",
        (linked["plan_sha256"],),
    )
    require(cursor.fetchall() == [(linked,)], "COMMITTED_LINKAGE_REQUIRED")
    source = []
    canonical_checkpoints = {}
    for fill in linked["fills"]:
        cursor.execute(
            "SELECT activity_pk,"
            + ",".join(cp.FIELDS)
            + " FROM public.alpaca_activities WHERE activity_pk=%s"
            + (" FOR SHARE" if lock else ""),
            (fill["activity_pk"],),
        )
        source.extend(rows(cursor))
        cursor.execute(
            "SELECT fill_sha256,checkpoint_id::text FROM public.strategy_fill_links_app WHERE account_id=%s AND activity_id=%s AND activity_pk=%s",
            (account_id, fill["activity_id"], fill["activity_pk"]),
        )
        committed = cursor.fetchall()
        require(
            len(committed) == 1 and committed[0][0] == fill["fill_sha256"],
            "COMMITTED_LINK_MISMATCH",
        )
        canonical_checkpoints[fill["activity_id"]] = committed[0][1]
    cursor.execute(
        "SELECT e.event FROM public.strategy_lot_events_app e JOIN public.strategy_lot_runs_app r USING(run_sha256) WHERE e.account_id=%s ORDER BY e.event_id",
        (account_id,),
    )
    prior = [r[0] for r in cursor.fetchall()]
    return plan(
        source_lots(linked, source, canonical_checkpoints=canonical_checkpoints),
        account_id=account_id,
        prior_events=prior,
        history_review=history_review,
        recipe=recipe,
    )


class CommitUncertain(RuntimeError):
    pass


def persist(connection, packet, *, recheck, policy_lock):
    """Transaction primitive. Runtime must use apply_guarded for durable latching.

    Caller provides fresh lock-backed source/provenance and prior-event reader.
    """
    require(
        not connection.autocommit
        and connection.readonly is False
        and connection.get_transaction_status() == 0,
        "IDLE_WRITE_REQUIRED",
    )
    require(
        packet["run_sha256"]
        == core.digest({k: v for k, v in packet.items() if k != "run_sha256"}),
        "PACKET_CHANGED",
    )
    try:
        with connection.cursor() as isolation_cursor:
            isolation_cursor.execute("SELECT current_setting('transaction_isolation')")
            require(
                isolation_cursor.fetchone() == ("read committed",),
                "READ_COMMITTED_REQUIRED",
            )
        policy_lock(connection)
        with connection.cursor() as q:
            q.execute("SET LOCAL lock_timeout='3000ms'")
            q.execute(
                "SELECT pg_advisory_xact_lock(hashtext('jbravo:strategy-execution'),hashtext(%s))",
                (packet["account_id"],),
            )
            q.execute(
                "SELECT receipt FROM public.strategy_lot_runs_app WHERE run_sha256=%s",
                (packet["run_sha256"],),
            )
            existing = q.fetchone()
            if existing:
                require(existing[0] == packet, "REPLAY_CONFLICT")
            else:
                require(recheck(q) == packet, "STALE_PLAN")
                q.execute(
                    "INSERT INTO public.strategy_lot_runs_app(run_sha256,account_id,receipt) VALUES(%s,%s,%s::jsonb)",
                    (packet["run_sha256"], packet["account_id"], core.encoded(packet)),
                )
                for lot in packet["lots"]:
                    q.execute(
                        "INSERT INTO public.strategy_fill_lots_app(account_id,activity_id,evidence) VALUES(%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                        (packet["account_id"], lot["activity_id"], core.encoded(lot)),
                    )
                    q.execute(
                        "SELECT evidence FROM public.strategy_fill_lots_app WHERE account_id=%s AND activity_id=%s",
                        (packet["account_id"], lot["activity_id"]),
                    )
                    saved = q.fetchone()[0]
                    # Cumulative checkpoints/recipes may change provenance, not economics.
                    require(
                        {k: v for k, v in saved.items() if k != "source_plan_sha256"}
                        == {k: v for k, v in lot.items() if k != "source_plan_sha256"},
                        "LOT_CONFLICT",
                    )
                for e in packet["events"]:
                    require(e["event_id"] == event_id(e), "EVENT_CHANGED")
                    q.execute(
                        "INSERT INTO public.strategy_lot_events_app(event_id,run_sha256,account_id,entry_activity_id,exit_activity_id,kind,quantity,reverses,replaces_reversal,event) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
                        (
                            e["event_id"],
                            packet["run_sha256"],
                            e["account_id"],
                            e["entry_activity_id"],
                            e["exit_activity_id"],
                            e["kind"],
                            e["quantity"],
                            e["reverses"],
                            e.get("replaces_reversal"),
                            core.encoded(e),
                        ),
                    )
    except Exception:
        connection.rollback()
        raise
    try:
        connection.commit()
    except Exception:
        raise CommitUncertain(packet["run_sha256"]) from None
    return {
        "status": "COMMITTED_UNVERIFIED",
        "run_sha256": packet["run_sha256"],
        "accounting_complete": False,
    }


def recover(connection, packet):
    """Fresh read-only connection; ABSENT stays latched, never authorizes retry."""
    require(
        connection.readonly is True and connection.get_transaction_status() == 0,
        "FRESH_READ_ONLY_REQUIRED",
    )
    try:
        with connection.cursor() as q:
            q.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            q.execute(
                "SELECT receipt FROM public.strategy_lot_runs_app WHERE run_sha256=%s",
                (packet["run_sha256"],),
            )
            found = q.fetchone()
            if not found:
                return {"status": "ABSENT_UNRESOLVED", "retry_authorized": False}
            require(found[0] == packet, "RECEIPT_CONFLICT")
            q.execute(
                "SELECT event FROM public.strategy_lot_events_app WHERE run_sha256=%s ORDER BY event_id",
                (packet["run_sha256"],),
            )
            require(
                [r[0] for r in q.fetchall()]
                == sorted(packet["events"], key=lambda e: e["event_id"]),
                "EVENT_POSTIMAGE",
            )
            for lot in packet["lots"]:
                q.execute(
                    "SELECT evidence FROM public.strategy_fill_lots_app WHERE account_id=%s AND activity_id=%s",
                    (packet["account_id"], lot["activity_id"]),
                )
                saved = q.fetchone()
                require(
                    saved is not None
                    and {k: v for k, v in saved[0].items() if k != "source_plan_sha256"}
                    == {k: v for k, v in lot.items() if k != "source_plan_sha256"},
                    "LOT_POSTIMAGE",
                )
                q.execute(
                    "SELECT capacity FROM public.strategy_lot_budgets_app WHERE account_id=%s AND activity_id=%s",
                    (packet["account_id"], lot["activity_id"]),
                )
                require(
                    q.fetchall() == [(core.number(lot["quantity"], positive=True),)],
                    "CAPACITY_POSTIMAGE",
                )
            q.execute(
                "SELECT b.capacity,COALESCE(sum(CASE WHEN e.kind='ALLOCATE' THEN e.quantity ELSE -e.quantity END),0) FROM public.strategy_lot_budgets_app b LEFT JOIN public.strategy_lot_events_app e ON e.account_id=b.account_id AND (e.entry_activity_id=b.activity_id OR e.exit_activity_id=b.activity_id) WHERE b.account_id=%s GROUP BY b.activity_id,b.capacity",
                (packet["account_id"],),
            )
            require(
                all(0 <= total <= capacity for capacity, total in q.fetchall()),
                "GLOBAL_BUDGET_POSTIMAGE",
            )
            return {
                "status": "COMMITTED_VERIFIED",
                "run_sha256": packet["run_sha256"],
                "accounting_complete": False,
            }
    finally:
        connection.rollback()


def apply_guarded(*, connect, packet, recheck, policy_lock, journal_directory):
    """Persist an account-wide reconciliation latch BEFORE starting a DB write.

    Directory must already exist in trusted durable private storage. An existing
    latch always gets a fresh read-only recovery; ABSENT or unavailable DB never
    permits automatic replay. Even a known pre-commit failure is conservatively
    held for reviewed cleanup. Only independently verified commit clears it.
    Requires POSIX directory fsync for durability. Windows is fail-closed until
    separately qualified; fixture tests may mock only that durability primitive.
    No filesystem action is performed on import.
    """
    core.uuid(packet["account_id"])
    require(
        packet["run_sha256"]
        == core.digest({k: v for k, v in packet.items() if k != "run_sha256"}),
        "PACKET_CHANGED",
    )
    path = Path(journal_directory) / (packet["account_id"] + ".lot-reconciliation.json")
    with _account_lock(path.with_suffix(".lock")):
        return _apply_locked(path, connect, packet, recheck, policy_lock)


@contextmanager
def _account_lock(path):
    # This stable lock file is NEVER unlinked. Keep the OS lock across recovery,
    # writes and latch cleanup, not merely across the initial O_EXCL creation.
    with open(path, "a+b") as f:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _sync_directory(directory):
    require(os.name != "nt", "DIRECTORY_DURABILITY_UNQUALIFIED")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino


def _remove_latch(path, identity):
    require(_identity(path) == identity, "LATCH_IDENTITY_CHANGED")
    path.unlink()
    _sync_directory(path.parent)


def _apply_locked(path, connect, packet, recheck, policy_lock):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        identity = _identity(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        require(saved["account_id"] == packet["account_id"], "LATCH_ACCOUNT")
        c = connect(readonly=True)
        try:
            result = recover(c, saved)
        finally:
            c.close()
        if result["status"] == "COMMITTED_VERIFIED":
            _remove_latch(path, identity)
        # Never start another write in the same invocation, even after resolving
        # a different previous run. Inputs must be independently rebuilt next.
        return dict(
            result, reconciled_run_sha256=saved["run_sha256"], retry_authorized=False
        )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(core.encoded(packet))
        f.flush()
        os.fsync(f.fileno())
    identity = _identity(path)
    _sync_directory(path.parent)
    c = connect(readonly=False)
    try:
        try:
            persist(c, packet, recheck=recheck, policy_lock=policy_lock)
        except CommitUncertain:
            pass
    finally:
        c.close()
    c = connect(readonly=True)
    try:
        result = recover(c, packet)
    finally:
        c.close()
    if result["status"] == "COMMITTED_VERIFIED":
        _remove_latch(path, identity)
    return dict(result, retry_authorized=False)
