import unittest
from copy import deepcopy
import test_owned_exit_transaction as T
import test_owned_exit_integrated_apply as I
from scripts import owned_exit_transaction as X
import owned_exit_preview_fixture as P
D=T.D

def admin(c,r,conn):
    c['catalog']['policy_owner']=False
    c['self_sha256']=D.fingerprint({k:v for k,v in c.items() if k!='self_sha256'})
    r.update(preview_digest=c['self_sha256'],execution_context='EXISTING_UNSWITCHED_ADMIN',allow_new_object_owner_transfer=True)
    conn.actor_context=(False,True,True,True,True,True)
    return c,r,conn

class AdminRoute(unittest.TestCase):
    def test_same_existing_unswitched_admin_transfers_only_two_new_objects(self):
        c,r=T.ready();c,r,conn=admin(c,r,T.Connection(c));res=T.Transaction().invoke(conn,c,r)
        self.assertEqual(res['status'],'COMMITTED',res)
        self.assertEqual((res['execution_context'],res['new_object_owner_transfers']),('EXISTING_UNSWITCHED_ADMIN',2))
        self.assertEqual([q for q in conn.calls if ' OWNER TO ' in q],[X.NEW_AUDIT_OWNER_SQL,X.NEW_GUARD_OWNER_SQL])
        self.assertFalse(any(q.startswith(('SET ROLE','SET LOCAL ROLE','GRANT ','ALTER ROLE','CREATE ROLE','ALTER DEFAULT PRIVILEGES')) for q in conn.calls))
        self.assertEqual(len(conn.state['audit']),1);self.assertEqual(len(conn.state['alloc']),2)

    def test_admin_permission_or_any_live_actor_predicate_failure_refuses_before_ddl(self):
        for scenario in ('approval','current_owner','superuser','expected_owner','cases_owner','schema_create','audit_present','untyped'):
            c,r=T.ready();c,r,conn=admin(c,r,T.Connection(c));row=list(conn.actor_context)
            if scenario=='approval':r['allow_new_object_owner_transfer']=False
            elif scenario=='current_owner':row[0]=True
            elif scenario=='untyped':row[1]='true'
            else:row[{'superuser':1,'expected_owner':2,'cases_owner':3,'schema_create':4,'audit_present':5}[scenario]]=False
            conn.actor_context=tuple(row);res=T.Transaction().invoke(conn,c,r)
            with self.subTest(scenario=scenario):
                self.assertEqual(res['status'],'ROLLED_BACK');self.assertEqual(conn.state,conn.original)
                self.assertFalse(any(q.startswith(('CREATE ','ALTER ','REVOKE ')) for q in conn.calls));self.assertNotIn(X.UPDATE_CASE1,conn.calls)

    def test_new_object_owner_transfer_failure_rolls_back_schema_before_native_update(self):
        for target in (X.NEW_AUDIT_OWNER_SQL,X.NEW_GUARD_OWNER_SQL):
            c,r=T.ready();c,r,conn=admin(c,r,T.Connection(c,fail_at=target));res=T.Transaction().invoke(conn,c,r)
            self.assertEqual(res['status'],'ROLLED_BACK');self.assertEqual(conn.state,conn.original);self.assertNotIn(X.UPDATE_CASE1,conn.calls)

    def test_owner_mode_remains_explicit_and_performs_no_owner_transfer(self):
        c,r=T.ready();conn=T.Connection(c);res=T.Transaction().invoke(conn,c,r)
        self.assertEqual(res['status'],'COMMITTED');self.assertEqual(res['new_object_owner_transfers'],0)
        self.assertFalse(any(' OWNER TO ' in q for q in conn.calls))

    def test_actual_full_admin_actor_real_readers_fits_original_ceilings(self):
        c,r,n=I.ready();c,r,conn=admin(c,r,I.Connection(c,n))
        res=X.apply_correction(conn,c,r,verify_sources=lambda:True,rebind=P.rebind)
        self.assertEqual(res['status'],'COMMITTED',res);self.assertEqual(res['new_object_owner_transfers'],2)
        self.assertLessEqual(res['usage']['total_bound'],256)
        self.assertEqual(res['committed_updated_rows'],1);self.assertEqual(res['committed_allocation_rows'],2)
        print('SYNTHETIC_FULL_ADMIN_APPLY_BUDGET',res['usage'])

    def test_recovery_retains_admin_mode_and_does_not_switch_or_repeat_owner_transfer(self):
        c,r=T.ready();c,r,conn=admin(c,r,T.Connection(c,uncertain=True));res=T.Transaction().invoke(conn,c,r)
        self.assertEqual(res['status'],'COMMIT_UNCERTAIN')
        recovery=T.Connection(c);recovery.state=deepcopy(conn.state)
        result=X.reconcile_commit(recovery,c,r,verify_sources=lambda:True)
        self.assertEqual(result['status'],'COMMIT_CONFIRMED_CURRENT_ROWS_MATCH',result)
        self.assertFalse(any(q.startswith(('ALTER ','CREATE ','UPDATE ','INSERT ','SET ROLE','SET LOCAL ROLE')) for q in recovery.calls))

if __name__=='__main__':unittest.main()
