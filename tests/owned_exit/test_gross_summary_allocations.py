"""Exercise the actual summary caller after canonical multi-exit validation."""
import sys,time,unittest
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from dataclasses import replace
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from test_owned_exit_full_consumer import Native
from scripts import trade_disposition as D,trade_allocation as A
from dashboards import execution_gross_projection as E,recorded_trade_analytics as R

NOW=datetime(2026,6,1,tzinfo=timezone.utc)

def cohort():
    native=Native();cursor=E.BoundedCursor(native,time.monotonic()+8)
    policy=D.read_policy(cursor,native.identity,scope='VALIDATED_PERFORMANCE')
    rows,execution=E.read_inputs(cursor,policy);cursor.recheck_canonical()
    assert not execution.unknown
    return native,policy,rows,execution

def summary(native,policy,rows,execution,basis):
    return R.project(policy,rows,identity=native.identity,now=NOW,
                     execution=execution if basis=='execution_gross' else None)

class SummaryAllocationRegression(unittest.TestCase):
    def test_both_actual_summary_paths_accept_two_new_groups_and_preserve_crc(self):
        native,policy,rows,execution=cohort();before=deepcopy(rows)
        for basis in ('recorded_gross','execution_gross'):
            with self.subTest(basis=basis):
                result=summary(native,policy,rows,execution,basis)
                all_row=next(row for row in result['rows'] if row['key']=='all')
                self.assertEqual((all_row['tradesCount'],all_row['totalPL']),(5,'41.62'))
                self.assertEqual(result['coverage']['unknown'],0)
                self.assertFalse(result['net_eligible']);self.assertIsNone(result['fees']);self.assertIsNone(result['net_pnl'])
                self.assertFalse(result['entry_ready']);self.assertFalse(result['accounting_complete'])
        self.assertEqual(rows,before)

    def test_every_full_row_change_is_rejected_before_gross_projection(self):
        native,policy,rows,execution=cohort()
        for key in ('symbol','qty','entry_price','exit_price','exit_reason','updated_at'):
            for basis in ('recorded_gross','execution_gross'):
                changed=deepcopy(rows);row=next(r for r in changed if r['trade_id']==1003)
                row[key]='changed' if key not in ('qty','entry_price','exit_price') else 999
                # Model a new queried cohort after policy validation; the
                # allocation's earlier audited full-row binding must still hold.
                current_projection=replace(execution,records_hash=D.fingerprint({'rows':changed}))
                with self.subTest(key=key,basis=basis),self.assertRaisesRegex(A.AllocationError,'^TRADE_ALLOCATION_ROW_CHANGED$'):
                    summary(native,policy,changed,current_projection,basis)

    def test_existing_execution_projection_drift_guard_remains_first(self):
        native,policy,rows,execution=cohort();changed=deepcopy(rows)
        changed[0]['exit_reason']='changed'
        with self.assertRaisesRegex(ValueError,'^EXECUTION_BINDING_CHANGED$'):
            summary(native,policy,changed,execution,'execution_gross')

    def test_missing_new_binding_refuses_both_paths(self):
        native,policy,rows,execution=cohort()
        allocations=tuple(replace(a,row_fingerprint=None) if a.trade_id==1003 else a for a in policy.allocations)
        policy=replace(policy,allocations=allocations)
        for basis in ('recorded_gross','execution_gross'):
            with self.subTest(basis=basis),self.assertRaisesRegex(A.AllocationError,'^TRADE_ALLOCATION_ROW_BINDING_UNAVAILABLE$'):
                summary(native,policy,rows,execution,basis)

    def test_new_binding_is_the_exact_audited_row_fingerprint(self):
        native,policy,rows,execution=cohort()
        for a in policy.allocations:
            if a.trade_id==110:self.assertIsNone(a.row_fingerprint)
            else:self.assertEqual(a.row_fingerprint,D.fingerprint(next(r for r in rows if r['trade_id']==a.trade_id)))

    def test_legacy_crc_does_not_accept_a_replacement_fingerprint(self):
        native,policy,rows,execution=cohort();changed=deepcopy(next(r for r in rows if r['trade_id']==110))
        changed['exit_reason']='changed'
        legacy=next(a for a in policy.allocations if a.trade_id==110)
        forged=replace(legacy,row_fingerprint=D.fingerprint(changed))
        with self.assertRaisesRegex(A.AllocationError,'^TRADE_ALLOCATION_ROW_CHANGED$'):
            A.validate_bound_trade(forged,changed)

    def test_wrong_trade_or_inactive_allocation_cannot_validate(self):
        native,policy,rows,execution=cohort();allocation=next(a for a in policy.allocations if a.trade_id==1003)
        for a in (replace(allocation,trade_id=9999),replace(allocation,state='REVOKED')):
            with self.assertRaisesRegex(A.AllocationError,'^TRADE_ALLOCATION_ROW_CHANGED$'):
                A.validate_bound_trade(a,next(r for r in rows if r['trade_id']==1003))

    def test_binding_is_internal_not_a_persisted_metadata_change(self):
        native,policy,rows,execution=cohort()
        for a in policy.allocations:
            self.assertNotIn('row_fingerprint',a.metadata_dict())
            self.assertEqual(a.metadata_dict(),replace(a,row_fingerprint=None).metadata_dict())

if __name__=='__main__':unittest.main()
