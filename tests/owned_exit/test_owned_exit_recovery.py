import sys,unittest
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import owned_exit_transaction as X
from test_owned_exit_transaction import ready,Connection
import test_owned_exit_transaction as fixtures

class Recovery(unittest.TestCase):
    def test_changed_immutable_body_or_replication_role_cannot_confirm_commit(self):
        for scenario in ('body','session'):
            c,r=ready();applied=Connection(c);fixtures.Transaction().invoke(applied,c,r)
            recovery=Connection(c);recovery.state=deepcopy(applied.state);original=recovery.native.execute
            def execute(sql,p=None):
                original(sql,p)
                if scenario=='session' and sql=="SELECT pg_catalog.current_setting('session_replication_role')":recovery.native.pending=[('replica',)]
                if scenario=='body' and sql.startswith('SELECT p.proname,CASE WHEN'):
                    rows=[]
                    for row in recovery.native.pending:
                        if row[0]=='trade_allocation_immutable':row=tuple(row[:1])+('BEGIN RETURN NULL; END',)+tuple(row[2:])
                        rows.append(row)
                    recovery.native.pending=rows
            recovery.native.execute=execute
            result=X.reconcile_commit(recovery,c,r,verify_sources=lambda:True)
            with self.subTest(scenario=scenario):
                self.assertEqual(result['status'],'STOPPED');self.assertIsNone(result['committed'])
                self.assertFalse(result['replay_authorized']);self.assertLessEqual(result['usage']['total_bound'],32)
    def test_current_absence_is_not_replay_or_retroactive_no_commit_proof(self):
        c,r=ready();conn=Connection(c)
        result=X.reconcile_commit(conn,c,r,verify_sources=lambda:True)
        self.assertEqual(result['status'],'AUDIT_ABSENT_CURRENT_SNAPSHOT_NO_REPLAY');self.assertIsNone(result['committed']);self.assertFalse(result['replay_authorized'])
        self.assertFalse(any(q.startswith(('UPDATE ','INSERT ','CREATE ','ALTER ','REVOKE ')) for q in conn.calls))
    def test_lost_commit_ack_can_be_confirmed_without_replaying_update(self):
        c,r=ready();applied=Connection(c,uncertain=True);res=fixtures.Transaction().invoke(applied,c,r)
        self.assertEqual(res['status'],'COMMIT_UNCERTAIN')
        recovery=Connection(c);recovery.state=deepcopy(applied.state)
        result=X.reconcile_commit(recovery,c,r,verify_sources=lambda:True)
        self.assertEqual(result['status'],'COMMIT_CONFIRMED_CURRENT_ROWS_MATCH',result);self.assertTrue(result['committed']);self.assertFalse(result['replay_authorized'])
        self.assertNotIn(X.UPDATE_CASE1,recovery.calls)
    def test_matching_audit_still_proves_commit_if_later_native_row_changed(self):
        c,r=ready();applied=Connection(c);fixtures.Transaction().invoke(applied,c,r)
        recovery=Connection(c);recovery.state=deepcopy(applied.state);recovery.state['rows'][1001]['row']['exit_reason']='later synthetic change'
        result=X.reconcile_commit(recovery,c,r,verify_sources=lambda:True)
        self.assertEqual(result['status'],'COMMIT_CONFIRMED_CURRENT_ROWS_CHANGED');self.assertTrue(result['committed']);self.assertFalse(result['current_native_rows_match'])
    def test_conflicting_approval_digest_refuses_without_commit_claim(self):
        c,r=ready();applied=Connection(c);fixtures.Transaction().invoke(applied,c,r)
        recovery=Connection(c);recovery.state=deepcopy(applied.state);record=list(recovery.state['audit'][r['approval_ref']]);record[1]='f'*64;recovery.state['audit'][r['approval_ref']]=tuple(record)
        result=X.reconcile_commit(recovery,c,r,verify_sources=lambda:True)
        self.assertEqual(result['status'],'STOPPED');self.assertIsNone(result['committed']);self.assertFalse(result['replay_authorized'])

if __name__=='__main__':unittest.main()
