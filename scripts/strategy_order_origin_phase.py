"""Durable origin-only transaction phase. Reuses V2 lock/latch primitives.

No retries: any existing intent receives read-only receipt reconciliation only.
No lot writes or trading operations. Private bootstrap owns fixed evidence fetch,
current relationship recheck, policy lock, connector and trusted journal path.
"""
import json
import os
from pathlib import Path
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot
from scripts import strategy_order_origin_evidence as origin


def validate(packet):
    core.uuid(packet['account_id'])
    core.require(packet['schema']=='jbravo.order-origin-phase.v1' and packet['receipts']
        and packet['phase_sha256']==core.digest({k:v for k,v in packet.items() if k!='phase_sha256'}),
        'ORIGIN_PHASE_CHANGED')
    ids=[]
    for p in packet['receipts']:
        core.require(p['account_id']==packet['account_id'] and p['receipt_sha256']==
            core.digest({k:v for k,v in p.items() if k!='receipt_sha256'}), 'ORIGIN_PHASE_RECEIPT')
        origin.validate_fetch_receipt(p)
        ids.append(p['broker_order_id'])
    core.require(len(set(ids))==len(ids), 'ORIGIN_PHASE_DUPLICATE')


def packet(account_id, receipts):
    p=dict(schema='jbravo.order-origin-phase.v1',account_id=account_id,
           receipts=sorted(receipts,key=lambda x:x['receipt_sha256']))
    p['phase_sha256']=core.digest(p)
    validate(p)
    return p


def recover(connect, p):
    validate(p)
    c=connect(readonly=True)
    try:
        core.require(c.readonly is True and c.get_transaction_status()==0,
                     'FRESH_READ_ONLY_REQUIRED')
        with c.cursor() as q:
            q.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            result=origin.verify_receipts(q,p['receipts'])
        return dict(result,phase_sha256=p['phase_sha256'])
    finally:
        c.rollback(); c.close()


def apply_guarded(*, connect, packet, recheck, policy_lock, journal_directory):
    validate(packet)
    path=Path(journal_directory)/(packet['account_id']+'.origin-reconciliation.json')
    with lot._account_lock(path.with_suffix('.lock')):
        if path.exists():
            identity=lot._identity(path)
            saved=json.loads(path.read_text(encoding='utf-8'))
            core.require(saved['account_id']==packet['account_id'],'ORIGIN_LATCH_ACCOUNT')
            result=recover(connect,saved)
            if result['status']=='ORIGIN_RECEIPTS_VERIFIED':
                lot._remove_latch(path,identity)
            return dict(result,reconciled_phase_sha256=saved['phase_sha256'],retry_authorized=False)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(core.encoded(packet)); f.flush(); os.fsync(f.fileno())
        identity=lot._identity(path)
        lot._sync_directory(path.parent)
        c=connect(readonly=False)
        try:
            core.require(c.readonly is False and c.get_transaction_status()==0,
                         'FRESH_WRITE_REQUIRED')
            try:
                policy_lock(c)
                with c.cursor() as q:
                    q.execute("SELECT current_setting('transaction_isolation')")
                    core.require(q.fetchone()==('read committed',),'READ_COMMITTED_REQUIRED')
                    q.execute("SET LOCAL lock_timeout='3000ms'")
                    q.execute("SET LOCAL statement_timeout='30000ms'")
                    q.execute("SELECT pg_advisory_xact_lock(hashtext('jbravo:strategy-execution'),hashtext(%s))",
                              (packet['account_id'],))
                    core.require(recheck(q)==packet,'ORIGIN_STALE_PACKET')
                    origin.insert_in_transaction(q,packet['receipts'])
            except Exception:
                c.rollback()
                raise
            try:
                c.commit()
            except Exception:
                pass # Fresh exact receipt proof below; never replay a possible commit.
        finally:
            c.close()
        result=recover(connect,packet)
        if result['status']=='ORIGIN_RECEIPTS_VERIFIED':
            lot._remove_latch(path,identity)
        return dict(result,retry_authorized=False)
