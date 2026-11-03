import sys,unittest
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal,localcontext
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import owned_exit_repair as O
NOW=datetime(2026,1,1,14,tzinfo=timezone.utc)
def uid(label):return str(uuid5(NAMESPACE_URL,'synthetic-owned-exit/'+label))
def snap(row,version='100'):return dict(row=row,version=version)
def fixture():
    rows={};fills=[];events=[];pk=0
    orders={case:dict(entry=uid('entry'+str(case)),direct=uid('direct'+str(case)),child=uid('child'+str(case))) for case in O.CASES}
    orders[1]['direct']=orders[2]['direct']
    for case in O.CASES:
        rows[case]=dict(trade_id=1000+case,status='CLOSED',symbol='SYN'+str(1 if case==2 else case),qty=O.QUANTITIES[case],
            entry_order_id=orders[case]['entry'],exit_order_id=orders[case]['direct'],entry_time='2026-01-01T10:00:00Z',
            exit_time='2026-01-01T12:00:00Z',entry_price=Decimal(10),exit_price=Decimal(11),realized_pnl=O.QUANTITIES[case],
            exit_reason='original reason',updated_at='2026-01-01T12:10:00Z')
    def fill(case,order,side,qty,index,price):
        nonlocal pk;pk+=1;stamp='2026-01-01T10:00:00Z' if side=='buy' else '2026-01-01T12:00:00Z'
        raw=dict(id='SYNTHETIC_FILL_'+str(pk),activity_type='FILL',order_id=order,symbol=rows[case]['symbol'],side=side,qty=str(qty),price=str(price),transaction_time=stamp)
        row=dict(activity_pk=pk,activity_id=raw['id'],activity_type='FILL',order_id=order,symbol=raw['symbol'],side=side,qty=Decimal(qty),price=Decimal(price),transaction_time=stamp,raw=raw,amount=None,description=None,created_at=stamp)
        fills.append(snap(row));return raw
    for case in O.CASES:
        fill(case,orders[case]['entry'],'buy',str(O.QUANTITIES[case]),0,'10')
        if case!=1:fill(case,orders[case]['direct'],'sell',str(O.DIRECT[case]),0,'11')
        if case in O.CHILD:
            qs=['1','1'] if case==1 else [str(O.CHILD[case])]
            for index,q in enumerate(qs):fill(case,orders[case]['child'],'sell',q,index,'12')
            metadata=dict(event_type='TRAIL_SUBMIT',parent_order_id=orders[case]['entry'],submitted_at='2026-01-01T11:00:00.123456789Z')
            raw=dict(order_id=orders[case]['child'],event_type='TRAIL_SUBMIT',event_time=metadata['submitted_at'],raw=metadata)
            events.append(dict(case=case,snapshot=snap(dict(event_id=case,event_type='TRAIL_SUBMIT',symbol=rows[case]['symbol'],qty=O.QUANTITIES[case],order_id=orders[case]['child'],event_time='2026-01-01T11:00:00.123456Z',raw=raw))))
    proof=dict(paper_identity_verified=True,method='INDEPENDENT_ACCOUNT_NUMBER_EXACT_MATCH',observed_at='2026-01-01T13:59:59Z',orders=[])
    for case in O.CHILD:
        selected=[deepcopy(x['row']['raw']) for x in fills if x['row']['order_id']==orders[case]['child']]
        proof['orders'].append(dict(case=case,order_id=orders[case]['child'],symbol=rows[case]['symbol'],side='sell',qty=str(O.QUANTITIES[case]),status='filled' if case==1 else 'canceled',filled_qty=str(O.CHILD[case]),fills=selected,complete=True))
    return dict(schema=O.PREVIEW_SCHEMA,identity=[uid('deployment'),uid('database')],policy_revision=7,policy_stable_hash='a'*64,
        source_pins={'scripts/db.py':'b'*64},cases=[dict(case=case,snapshot=snap(rows[case])) for case in O.CASES],
        owners=[snap(rows[case]) for case in O.CASES],events=events,fills=fills,reserved_order_owners={},
        relevant_fill_claims=[],relevant_order_claims=[],excluded_trade_ids=[],broker_proof=proof)

class Core(unittest.TestCase):
    def test_exact_four_case_plan_and_modified_field_contract(self):
        b=fixture();before=deepcopy(b);p=O.compile_plan(b,now=NOW)
        self.assertEqual(b,before);self.assertTrue(p['broker_bound'])
        self.assertEqual(set(p['case1_after_fields']),{'exit_order_id','exit_price','exit_time','realized_pnl'})
        self.assertEqual(p['case1_after_fields']['exit_price'],12);self.assertEqual(p['case1_after_fields']['realized_pnl'],4)
        self.assertEqual([(a['case'],a['entry_quantity'],a['gross_result']) for a in p['allocations']],[(3,Decimal(11),Decimal(19)),(6,Decimal(53),Decimal(78))])
        self.assertFalse(p['financial_eligible']);self.assertIsNone(p['fees']);self.assertIsNone(p['net'])
        self.assertTrue(all(r['projection_accepted_as_encoding'] for r in p['clock_reports'].values()))
    def test_missing_broker_continuity_is_not_invented_from_counts(self):
        b=fixture();b['broker_proof']=None
        with self.assertRaisesRegex(O.RepairError,'BROKER_CONTINUITY_REQUIRED'):O.compile_plan(b,now=NOW)
        self.assertFalse(O.compile_plan(b,now=NOW,allow_unbound=True)['broker_bound'])
    def test_wrong_paper_identity_refuses(self):
        b=fixture();b['broker_proof']['paper_identity_verified']=False
        with self.assertRaisesRegex(O.RepairError,'PAPER_IDENTITY'):O.compile_plan(b,now=NOW)
    def test_broker_id_price_time_and_qty_are_bound(self):
        for field,value in [('id','OTHER'),('price','12.01'),('qty','1.1'),('transaction_time','2026-01-01T12:00:00.000000001Z')]:
            b=fixture();b['broker_proof']['orders'][0]['fills'][0][field]=value
            with self.subTest(field=field),self.assertRaises(O.RepairError):O.compile_plan(b,now=NOW)
    def test_broker_submission_and_partial_status_are_explicit(self):
        b=fixture();b['broker_proof']['orders'][1]['qty']='8'
        with self.assertRaisesRegex(O.RepairError,'BROKER_SUBMISSION_CONTEXT'):O.compile_plan(b,now=NOW)
    def test_incomplete_page_refuses(self):
        b=fixture();b['broker_proof']['orders'][0]['complete']=False
        with self.assertRaisesRegex(O.RepairError,'BROKER_TERMINAL_STATUS'):O.compile_plan(b,now=NOW)
    def test_global_other_owner_and_duplicate_owner_refuse(self):
        b=fixture();other=deepcopy(b['owners'][0]);other['row']['trade_id']=99999;other['row']['exit_order_id']=b['events'][0]['snapshot']['row']['order_id'];b['owners'].append(other)
        with self.assertRaisesRegex(O.RepairError,'GLOBAL_OWNER_CONFLICT'):O.compile_plan(b,now=NOW)
        b=fixture();b['owners'].append(deepcopy(b['owners'][0]))
        with self.assertRaisesRegex(O.RepairError,'OWNER_DUPLICATE'):O.compile_plan(b,now=NOW)
    def test_reused_fill_and_existing_global_claim_refuse(self):
        b=fixture();b['fills'].append(deepcopy(b['fills'][0]))
        with self.assertRaisesRegex(O.RepairError,'FILL_DUPLICATE'):O.compile_plan(b,now=NOW)
        b=fixture();b['relevant_fill_claims']=[{'allocation_id':uid('old')}]
        with self.assertRaisesRegex(O.RepairError,'ALREADY_CLAIMED'):O.compile_plan(b,now=NOW)
    def test_reservation_and_exclusion_refuse(self):
        b=fixture();b['reserved_order_owners'][b['cases'][0]['snapshot']['row']['entry_order_id']]=110
        with self.assertRaisesRegex(O.RepairError,'ORDER_RESERVED'):O.compile_plan(b,now=NOW)
        b=fixture();b['excluded_trade_ids']=[1003]
        with self.assertRaisesRegex(O.RepairError,'TRADE_EXCLUDED'):O.compile_plan(b,now=NOW)
    def test_missing_component_and_changed_native_fact_refuse(self):
        b=fixture();b['fills'].pop()
        with self.assertRaises(O.RepairError):O.compile_plan(b,now=NOW)
        b=fixture();b['fills'][0]['row']['raw']['price']='99'
        with self.assertRaisesRegex(O.RepairError,'FILL_ECONOMICS'):O.compile_plan(b,now=NOW)
    def test_case4_or5_cannot_enter_scope(self):
        b=fixture();b['cases'][0]['case']=4
        with self.assertRaisesRegex(O.RepairError,'CASE_SCOPE'):O.compile_plan(b,now=NOW)
    def test_clock_projection_is_not_equality_and_cutoff_is_exact(self):
        b=fixture();b['events'][0]['snapshot']['row']['raw']['raw']['submitted_at']='2026-01-01T11:00:00.123456788Z'
        with self.assertRaises(Exception):O.compile_plan(b,now=NOW)
    def test_unnecessary_account_or_credential_fields_never_enter_private_artifact(self):
        b=fixture();b['fills'][0]['row']['raw']['account_id']='SYNTHETIC_NOT_NEEDED'
        with self.assertRaisesRegex(O.RepairError,'UNNECESSARY_PRIVATE_FIELD'):O.compile_plan(b,now=NOW)
    def test_decimal_context_does_not_round_plan(self):
        with localcontext() as context:
            context.prec=6;p=O.compile_plan(fixture(),now=NOW)
        self.assertEqual(p['case1_after_fields']['realized_pnl'],4)

if __name__=='__main__':unittest.main()
