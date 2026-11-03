import sys,unittest
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import trade_allocation as A,trade_disposition as D,owned_exit_repair as O,owned_exit_transaction as X,gross_paper_accounting as G
from dashboards import execution_read_transport as T,execution_gross_projection as E
from test_owned_exit_core import fixture,NOW

def records():
    b=fixture();p=O.compile_plan(b,now=NOW)
    ids={'3':'00000000-0000-4000-8000-000000000003','6':'00000000-0000-4000-8000-000000000006'}
    ms=O.allocation_manifests(b,p,approval_ref='FIXTURE',preview_digest='c'*64,allocation_ids=ids);rs=[];history=[]
    for index,m in enumerate(ms):
        rev=10+index;ref='FIXTURE/'+str(m['case'])
        q=dict(action='APPLY',allocation_id=m['allocation_id'],trade_id=m['trade_id'],identity=b['identity'],manifest_sha256=D.fingerprint(m),expected_allocation_revision=0,expected_policy_revision=rev-1,approval_ref=ref)
        r=dict(allocation_id=m['allocation_id'],trade_id=m['trade_id'],state='ACTIVE',allocation_revision=1,policy_revision=rev,approval_ref=ref)
        rs.append((m['allocation_id'],'ACTIVE',1,D.canonical(q),D.canonical(m),D.fingerprint(q),D.canonical(q),D.canonical(r),rev,'FIXTURE'))
        history.append((ref,m['allocation_id'],D.fingerprint(q),D.canonical(q),True,D.canonical(r),rev))
    return b,ms,rs,history

class Cursor:
    def __init__(self,b,ms,history):self.b=b;self.ms=ms;self.history=history;self.pending=[];self.calls=[];self.bad=None
    def execute(self,sql,p=None):
        self.calls.append(sql)
        if sql==O.STATE_SQL:
            calls=list(self.calls);combined=[]
            for kind,query in [('TRADE',O.CURRENT_TRADES_SQL),('FILL',O.CURRENT_FILLS_SQL),('CLAIM',O.CLAIMS_SQL),('HISTORY',O.HISTORY_SQL)]:
                self.execute(query)
                for row in self.pending:
                    values=list(row)+(['777','778'] if kind=='HISTORY' else [])
                    combined.append((kind,D.canonical({'row':values})[7:-1]))
            self.calls=calls;self.pending=combined
        elif sql==O.CURRENT_TRADES_SQL:self.pending=[(D.canonical(s['row']),s['version']) for s in self.b['owners']]
        elif sql==O.CURRENT_FILLS_SQL:self.pending=[(D.canonical(s['row']),s['version']) for m in self.ms for s in O.manifest_fills(m)]
        elif sql==O.CLAIMS_SQL:
            self.pending=[('ORDER',D.canonical(dict(order_id=o,allocation_id=m['allocation_id']))) for m in self.ms for o in m['orders']]
            self.pending += [('FILL',D.canonical(dict(activity_id=s['row']['activity_id'],activity_type='FILL',order_id=s['row']['order_id'],allocation_id=m['allocation_id'],quantity=s['row']['qty']))) for m in self.ms for s in O.manifest_fills(m)]
        elif sql==O.HISTORY_SQL:self.pending=list(self.history)
        else:raise AssertionError('unexpected query')
        if self.bad==sql:self.pending=self.pending[:-1]
    def fetchmany(self,n):r=self.pending[:n];self.pending=self.pending[n:];return r

class Readers(unittest.TestCase):
    def test_actual_reader_preserves_two_complete_multi_exit_groups(self):
        b,ms,rs,h=records();c=Cursor(b,ms,h)
        active,reserved,cases=O.read_extensions(c,rs,tuple(b['identity']),11,(),allocation_type=A.Allocation,attest=lambda cursor:None)
        self.assertEqual([x.entry_quantity for x in active],[11,53]);self.assertEqual([x.gross_result for x in active],[19,78])
        self.assertEqual(len(reserved),6);self.assertEqual(c.calls,[O.STATE_SQL]);self.assertTrue(all(x.clock_fills for x in active))
        overlay=A.result_read('VALIDATED',active,tuple(b['identity']),11,reserved,cases)
        policy=D.Policy(*b['identity'],11,(),scope='VALIDATED_PERFORMANCE',allocations=tuple(active),allocation_metadata_json=overlay._metadata_json)
        rows=[m['row_snapshot']['row'] for m in ms]
        report=G.build_snapshot(policy,rows,expected_identity=tuple(b['identity']),expected_revision=11)
        self.assertEqual(report['gross_pnl'],'97');self.assertIsNone(report['net_pnl']);self.assertFalse(report['net_eligible'])
        projection=E.reconstruct(policy,rows,[],[s['row'] for s in b['owners']])
        self.assertEqual(set(projection.completion),{1003,1006})
    def test_missing_fill_claim_history_or_changed_owner_refuses(self):
        for scenario in ('fill','claim','history','owner'):
            b,ms,rs,h=records();c=Cursor(b,ms,h)
            if scenario=='fill':c.bad=O.CURRENT_FILLS_SQL
            if scenario=='claim':c.bad=O.CLAIMS_SQL
            if scenario=='history':c.bad=O.HISTORY_SQL
            if scenario=='owner':b['owners'][0]['row']['exit_order_id']=ms[0]['orders'][2]
            with self.subTest(scenario=scenario),self.assertRaises(O.RepairError):O.read_extensions(c,rs,tuple(b['identity']),11,(),allocation_type=A.Allocation,attest=lambda cursor:None)
    def test_existing_crc110_dataclass_and_metadata_are_unchanged(self):
        a=A.Allocation('public-fixture',110,'ACTIVE',1,O.number(30),O.number(30),O.number('-63.38'),'fixture',(),tuple(A.ORDERS),tuple(f[1] for f in A.FILLS))
        self.assertEqual(a.clock_fills,());self.assertNotIn('clock_fills',a.metadata_dict());self.assertFalse(a.financial_eligible)
    def test_exact_new_sql_and_case_row_plan_arity_and_no_raised_transport_caps(self):
        table=T.plan_table();self.assertEqual(T.MAX_SQL,128)
        sql=next(p['sql'] for p in T.CANONICAL_PLANS if p['sql'].startswith('SELECT c.allocation_id::text,c.state,c.allocation_revision,c.request::text,c.manifest::text,'))
        self.assertEqual(table[sql]['arity'],10);self.assertEqual(table[sql]['wire_arity'],12)
        for query in (X.ATTEST_SQL,O.STATE_SQL):self.assertIn(query,table)
        # The baseline repeats the identical shared-lock control in D and A.
        for query in (X.ATTEST_SQL,O.STATE_SQL):self.assertEqual(sum(p['sql']==query for p in T.CANONICAL_PLANS),1)
    def test_empty_extension_reader_does_no_extra_queries(self):
        b,ms,rs,h=records();c=Cursor(b,ms,h)
        self.assertEqual(O.read_extensions(c,[],tuple(b['identity']),11,(),allocation_type=A.Allocation,attest=lambda cursor:None),([],{},[]));self.assertEqual(c.calls,[])

if __name__=='__main__':unittest.main()
