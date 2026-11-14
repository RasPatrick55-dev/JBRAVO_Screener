"""One data-only automatic allocation step using the existing V2 durable writer.

Private reviewed bootstrap supplies the real DB connector/policy lock, audited
loader and protected journal directory. No CLI, credentials, broker or scheduler.
At most one cohort per step; caller may rebuild and step again after verification.
"""
import json
from pathlib import Path
from scripts import strategy_fill_linkage as core
from scripts import strategy_lot_accounting_v2 as lot
from scripts import strategy_lot_incremental_preview as preview


def checked_preview(inputs):
    result = preview.preview(**inputs)
    for candidate in result['candidates']:
        pins = candidate['qualification']['origin_receipt_sha256s']
        core.require(pins and all(isinstance(h,str) and len(h)==64 for h in pins.values()),
                     'DURABLE_ORIGIN_RECEIPTS_REQUIRED')
    return result


def reconcile_pending(*, account_id, connect, journal_directory):
    core.uuid(account_id)
    path = Path(journal_directory) / (account_id + '.lot-reconciliation.json')
    with lot._account_lock(path.with_suffix('.lock')):
        if not path.exists():
            return None
        identity = lot._identity(path)
        saved = json.loads(path.read_text(encoding='utf-8'))
        core.require(saved['account_id']==account_id and saved['run_sha256']==
                     core.digest({k:v for k,v in saved.items() if k!='run_sha256'}),
                     'PENDING_LOT_CHANGED')
        c = connect(readonly=True)
        try:
            result = lot.recover(c,saved)
        finally:
            c.close()
        if result.get('status')=='COMMITTED_VERIFIED':
            lot._remove_latch(path,identity)
        return dict(status='PRIOR_LOT_RECONCILIATION_ONLY', result=result,
                    retry_authorized=False, accounting_complete=False,
                    publication_allowed=False, gross_pnl=None, net_pnl=None,
                    trading_operations=0)


def step(*, account_id, connect, load_inputs, policy_lock, journal_directory):
    """Loader is fixed to the already verified sync run; never accepts user JSON.

    load_inputs(cursor, lock=bool) must be the audited incremental store adapter.
    Recheck runs under V2's account lease after policy checks and rebuilds all
    global budgets/current ownership/history. An uncertain result never retries.
    """
    pending = reconcile_pending(account_id=account_id, connect=connect,
                                journal_directory=journal_directory)
    if pending is not None:
        return pending
    c = connect(readonly=True)
    try:
        core.require(c.readonly is True and c.get_transaction_status()==0,
                     'FRESH_READ_ONLY_REQUIRED')
        with c.cursor() as q:
            q.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            initial = checked_preview(load_inputs(q, lock=False))
            core.require(initial["account_id"]==account_id, "AUTO_ACCOUNT_CHANGED")
    finally:
        c.rollback()
        c.close()
    if not initial['candidates']:
        return dict(status='NO_ELIGIBLE_RESIDUAL_ALLOCATION', held=initial['held'],
                    noops=initial['noops'], retry_authorized=False,
                    accounting_complete=False, publication_allowed=False,
                    gross_pnl=None, net_pnl=None, trading_operations=0)
    selected = initial['candidates'][0]
    packet = selected['packet']

    def recheck(q):
        fresh = checked_preview(load_inputs(q, lock=True))
        matches = [x for x in fresh['candidates'] if x['signal_id']==selected['signal_id']]
        core.require(len(matches)==1 and matches[0]==selected, 'AUTO_COHORT_CHANGED')
        return matches[0]['packet']

    result = lot.apply_guarded(connect=connect, packet=packet, recheck=recheck,
                              policy_lock=policy_lock, journal_directory=journal_directory)
    success = (result.get('status')=='COMMITTED_VERIFIED'
               and result.get('run_sha256')==packet['run_sha256']
               and result.get('reconciled_run_sha256',packet['run_sha256'])==packet['run_sha256'])
    return dict(status='AUTO_LOT_COMMITTED_VERIFIED' if success else 'AUTO_LOT_UNRESOLVED',
                run_sha256=packet['run_sha256'], result=result,
                retry_authorized=False, accounting_complete=False,
                publication_allowed=False, gross_pnl=None, net_pnl=None,
                trading_operations=0)
