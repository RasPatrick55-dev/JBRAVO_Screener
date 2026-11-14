"""GET-only raw order evidence acquisition. No DB/file writes or trading clients.

Call from a pinned private bootstrap with the existing ingestor headers/GET
transport. Durable origin insertion remains a separately reconciled phase.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from scripts import strategy_fill_linkage as core
from scripts import activity_ingestion_checkpoint as cp
from scripts.strategy_fill_sync_adapter import PaperGetTransport
from scripts import strategy_lot_hourly_phase as phase
from scripts import strategy_order_origin_evidence as origin


def fetch_origins(*, account_id, orders, linked, completion_bytes,
                  expected_completion_sha256, receipt_bytes, expected_slot,
                  request_get, headers, now=lambda: datetime.now(timezone.utc),
                  monotonic=time.monotonic):
    completion, linkage = phase.validate_sync(
        completion_bytes=completion_bytes,
        expected_completion_sha256=expected_completion_sha256,
        receipt_bytes=receipt_bytes, expected_slot=expected_slot)
    core.uuid(account_id)
    core.require(linked['plan_sha256'] == linkage['linkage']['plan_sha256']
                 == core.digest({k:v for k,v in linked.items() if k != 'plan_sha256'}),
                 'ORIGIN_SYNC_LINKAGE_MISMATCH')
    transport = PaperGetTransport(request_get)

    def get(path):
        before, tick = now(), monotonic()
        response = transport.get(cp.ENDPOINT + path, headers=headers,
                                 timeout=30, allow_redirects=False)
        after, elapsed = now(), monotonic() - tick
        core.require(before.tzinfo is not None and after.tzinfo is not None,
                     'ORIGIN_CLOCK_ZONE')
        wall = (after - before).total_seconds()
        core.require(0 <= elapsed <= 35 and 0 <= wall <= 35 and abs(wall-elapsed) <= 2,
                     'ORIGIN_CLOCK_DISCONTINUITY')
        date = parsedate_to_datetime(response.headers['Date'])
        core.require(date.tzinfo is not None and
                     before.timestamp()-5 <= date.timestamp() <= after.timestamp()+5,
                     'ORIGIN_SERVER_CLOCK_MISMATCH')
        return response.json(), dict(started_at=before.isoformat(),
                    completed_at=after.isoformat(), server_date=response.headers['Date'],
                    elapsed_seconds=elapsed, maximum_clock_skew_seconds=5)

    account, _ = get('/v2/account')
    core.require(account['id'] == account_id, 'ORIGIN_ACCOUNT_MISMATCH')
    receipts, held = [], []
    bound = {o['broker_order_id']:o for o in linked['orders']}
    seen = set()
    for claim in sorted(orders, key=lambda o:o['client_order_id']):
        obs = claim.get('observation')
        if claim['account_id'] != account_id or claim['state'] != 'OBSERVED' or not obs:
            continue
        oid = str(core.uuid(obs['id']))
        core.require(oid not in seen, 'ORIGIN_DUPLICATE_ORDER')
        seen.add(oid)
        expected = bound.get(oid)
        core.require(expected is not None and expected['account_id'] == account_id
            and expected['signal_id'] == claim['signal_id']
            and expected['client_order_id'] == claim['client_order_id'],
            'ORIGIN_OWNERSHIP_MISMATCH')
        raw, timing = get('/v2/orders/' + oid)
        if any(raw.get(k) != v for k,v in obs.items()):
            held.append(dict(broker_order_id=oid, reason='ORIGIN_SNAPSHOT_CHANGED_SINCE_SYNC'))
            continue
        if not raw.get('created_at'):
            held.append(dict(broker_order_id=oid, reason='IMMUTABLE_ORDER_ORIGIN_REQUIRED'))
            continue
        p = origin.build(account_id=account_id, source_run_id=completion['native_run_id'],
                         source_plan_sha256=linked['plan_sha256'], raw_order=raw,
                         fetched_at=timing['completed_at'])
        p.update(sync_completion_json=completion_bytes.decode("utf-8"),
                 sync_receipt_json={k:v.decode("utf-8") for k,v in receipt_bytes.items()},
                 fetch_timing=timing, sync_completion_sha256=expected_completion_sha256,
                 sync_slot=expected_slot, sync_receipts=completion['native_receipts'],
                 sync_checkpoint=completion['checkpoint'])
        p['receipt_sha256'] = core.digest({k:v for k,v in p.items() if k != 'receipt_sha256'})
        origin.validate_fetch_receipt(p)
        receipts.append(p)
    return dict(status='ORIGIN_EVIDENCE_FETCHED_NOT_PERSISTED', receipts=receipts,
                held=held, trading_operations=0, accounting_complete=False,
                publication_allowed=False)


def attach_bounded_history(receipts, *, fetch_activities, normalize,
                           now=lambda: datetime.now(timezone.utc)):
    """Read full bounded account activity history; never ingest or advance cursor.

    Bootstrap supplies existing pinned ingestor.fetch_activities and normalizer
    with PaperGetTransport installed. New/missing ledger rows remain holds.
    """
    from datetime import timedelta
    from scripts import strategy_lot_accounting_v2 as lot
    core.require(bool(receipts), "HISTORY_RECEIPTS_REQUIRED")
    account=receipts[0]['account_id']
    for p in receipts:
        origin.validate_fetch_receipt(p)
        core.require(p['account_id']==account, 'HISTORY_ACCOUNT')
    earliest=min(lot.timestamp(p['raw_order']['created_at'])['lower_ns'] for p in receipts)
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    after=epoch+timedelta(seconds=earliest//10**9-300)
    until=now()
    core.require(until.tzinfo is not None and timedelta(0)<until-after<=timedelta(days=45),
                 'HISTORY_WINDOW_LIMIT')
    raw=fetch_activities(cp.ENDPOINT,after.isoformat(),until.isoformat(),100,1000)
    core.require(len(raw)<=core.MAX_ROWS, 'HISTORY_ROW_LIMIT')
    rows=[cp.row_value(normalize(x)[0]) for x in raw]
    manifest=cp.manifestation(rows)
    history=dict(proof_type='BOUNDED_BROKER_GET_ORIGIN_RECEIPT',checkpoint_id=None,
        source_fingerprint=cp.source_fingerprint(cp.ENDPOINT,account),
        recipe_sha256=cp.recipe_fingerprint(),query_axis='ALPACA_CREATED_AT',
        query_after=after.isoformat(),query_until=until.isoformat(),
        fetched_count=len(rows),persisted_manifest=manifest,
        manifest_sha256=cp.digest(manifest),rows=rows,max_pages=1000,page_size=100,
        observed_complete_at=now().isoformat())
    history['history_sha256']=core.digest(history)
    result=[]
    for p in receipts:
        p=dict(p,bounded_history=history)
        p['receipt_sha256']=core.digest({k:v for k,v in p.items() if k!='receipt_sha256'})
        origin.validate_fetch_receipt(p)
        result.append(p)
    return result


def attach_history_by_cohort(receipts, *, orders, signals, fetch_activities,
                             normalize, now=lambda: datetime.now(timezone.utc)):
    """Qualify each terminal owned cohort independently; a history cap is a hold."""
    by_order={p['broker_order_id']:p for p in receipts}
    core.require(len(by_order)==len(receipts),'ORIGIN_DUPLICATE_ORDER')
    results,held=[],[]
    for signal in sorted(signals,key=lambda s:s['signal_id']):
        claims=[o for o in orders if o['account_id']==signal['account_id']
                and o['signal_id']==signal['signal_id']]
        selected=[by_order[o['observation']['id']] for o in claims
                  if o.get('observation') and o['observation']['id'] in by_order]
        if not selected:
            continue
        sid=signal['signal_id']
        if signal['payload'].get('execution_owner'):
            held.append(dict(signal_id=sid,reason='PROTECTED_EXTERNAL_OWNERSHIP'));continue
        if (len(claims)!=2 or {o['role'] for o in claims}!={'entry','stop'}
            or len(selected)!=2 or any(o['state']!='OBSERVED' or
                o.get('observation',{}).get('status')!='filled' for o in claims)):
            held.append(dict(signal_id=sid,reason='HISTORY_COHORT_NOT_ELIGIBLE'));continue
        try:
            results.extend(attach_bounded_history(selected,fetch_activities=fetch_activities,
                                                   normalize=normalize,now=now))
        except Exception as exc:
            reason='HISTORY_WINDOW_LIMIT' if str(exc).endswith('HISTORY_WINDOW_LIMIT') else 'BOUNDED_HISTORY_FETCH_FAILED'
            held.append(dict(signal_id=sid,reason=reason))
    return dict(receipts=results,held=held,trading_operations=0,
                accounting_complete=False,publication_allowed=False)
