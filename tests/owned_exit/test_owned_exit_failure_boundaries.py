import unittest
from copy import deepcopy
from unittest.mock import patch
import test_owned_exit_transaction as T
from scripts import owned_exit_transaction as X,owned_exit_repair as O

class FailureBoundaries(unittest.TestCase):
    def test_scoped_preview_still_refuses_target_excluded_in_another_scope(self):
        c,r=T.ready();D=O.policy_module()
        base=T.Policy(c['bundle'],c['bundle']['policy_revision'])
        active=D.ActiveCase('synthetic-case',1003,'synthetic-other-scope','UNRESOLVED',('ACTIVE_RECONCILIATION',))
        policy=D.Policy(base.deployment_id,base.database_id,base.revision,(active,),base.scope,(),base.allocation_metadata_json)
        self.assertNotIn(1003,policy.excluded_ids);self.assertIn(1003,policy.all_excluded_ids)
        b=deepcopy(c['bundle']);b['policy_stable_hash']=O.stable_policy(policy.metadata())
        c=T.P.capsule(b,c['catalog']);r['preview_digest']=c['self_sha256'];conn=T.Connection(c)
        with patch.object(D,'read_policy',return_value=policy):
            result=X.apply_correction(conn,c,r,verify_sources=lambda:True,rebind=T.P.rebind)
        self.assertEqual(result['reason'],'OWNED_EXIT_TARGET_EXCLUDED_IN_ANY_SCOPE')
        self.assertFalse(result['schema_attempted']);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
    def test_failed_sql_attempt_is_charged(self):
        class Raw:
            def execute(self,sql,params=None):
                if sql=='SELECT failed':raise RuntimeError('not exported')
        budget=X.Budget(Raw())
        with self.assertRaises(RuntimeError):budget.execute('SELECT failed')
        self.assertEqual(budget.statements,2)

    def test_closed_sql_failure_never_exports_message(self):
        class SqlFailure(Exception):pgcode='42501'
        self.assertEqual(X.closed_failure(SqlFailure('private message')),'SQL_PERMISSION_REFUSED')
        class InvalidCode(Exception):pgcode=[]
        self.assertEqual(X.closed_failure(InvalidCode('private message')),'APPLY_REFUSED')
        self.assertEqual(X.closed_failure(O.RepairError('CASE1_CAS')),'CASE1_CAS')

    def test_consumer_overflow_rolls_back_schema_and_data(self):
        c,r=T.ready();conn=T.Connection(c)
        with patch.object(X,'qualify_consumers',side_effect=O.RepairError('CONSUMER_CASE_WIRE_BOUND')):
            result=T.Transaction().invoke(conn,c,r)
        self.assertEqual(result['phase'],'CONSUMER_QUALIFICATION')
        self.assertEqual(result['status'],'ROLLED_BACK')
        self.assertEqual(result['reason'],'CONSUMER_CASE_WIRE_BOUND')
        self.assertEqual(conn.state,conn.original)

    def test_new_parent_event_refuses_before_schema_and_updates(self):
        c,r=T.ready();conn=T.Connection(c)
        extra=deepcopy(conn.b['events'][0]);extra['snapshot']['row']['event_id']=999999
        conn.b['events'].append(extra)
        result=T.Transaction().invoke(conn,c,r)
        self.assertEqual(result['status'],'ROLLED_BACK')
        self.assertFalse(result['schema_attempted'])
        self.assertEqual(conn.state,conn.original)

    def test_source_change_before_commit_rolls_back_everything(self):
        c,r=T.ready();conn=T.Connection(c)
        actor=X.apply_correction
        def wrapped(*a,**kw):
            checks=iter((True,False));kw['verify_sources']=lambda:next(checks)
            return actor(*a,**kw)
        with patch.object(X,'apply_correction',side_effect=wrapped):
            result=T.Transaction().invoke(conn,c,r)
        self.assertEqual(result['status'],'ROLLED_BACK')
        self.assertEqual(result['reason'],'OWNED_EXIT_SOURCE_CHANGED_BEFORE_COMMIT')
        self.assertEqual(conn.state,conn.original)

if __name__=='__main__':unittest.main()
