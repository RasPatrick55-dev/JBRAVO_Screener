"""Pure supplied-cursor tests; no connection, schema application or activation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from types import SimpleNamespace
from pathlib import Path
import hashlib
import re
import unittest
from unittest.mock import patch
from scripts import trade_allocation as A
from scripts import trade_disposition as D

IDENTITY = ('fixture-paper', '0dbad265-9d1c-4de9-9eaa-1b234a533a05')
CASE = 'c765d757-e67f-4a04-85b0-a3e77327d28f'
SCHEMA_SOURCE = Path(__file__).parents[1] / 'scripts' / 'sql' / 'trade_allocation_schema.sql'
FUNCTION_BODIES = dict(re.findall(r'CREATE FUNCTION public\.(\w+)\(\) RETURNS pg_catalog.trigger\s+'
    r'LANGUAGE plpgsql SET search_path = pg_catalog AS \$body\$(.*?)\$body\$;',
    SCHEMA_SOURCE.read_text(encoding='utf-8'), re.S))
TRADE = D.document('{"qty":30.0,"status":"CLOSED","symbol":"CRC","trade_id":110,'
    '"exit_time":"2026-04-17T19:05:44.628183+00:00","created_at":"2026-04-16T11:00:17.548511+00:00",'
    '"entry_time":"2026-04-16T11:00:12.026948+00:00","exit_price":null,'
    '"updated_at":"2026-04-17T19:05:44.607608+00:00","entry_price":65.0,"exit_reason":"POSITION_CLOSED",'
    '"realized_pnl":null,"exit_order_id":null,"entry_order_id":"7ad70913-5833-4f98-a3cc-d9a590da02a6"}')

def fills():
    result = []
    for index, (pk, native, side, order, qty, price, time) in enumerate(A.FILLS):
        snapshot = {'activity_pk':pk,'activity_id':native,'activity_type':'FILL','transaction_time':time,
                    'symbol':'CRC','side':side,'qty':Decimal(qty),'price':Decimal(price),
                    'amount':None,'order_id':order,'description':None,'raw':{},'created_at':A.CREATED_AT[index]}
        result.append({'snapshot':snapshot,'row_fingerprint':D.fingerprint(snapshot)})
    return result

def manifest():
    # Synthetic full raw snapshots: fixture data, not invented production bindings.
    return {'allocation_id':CASE,'trade_id':110,'row_fingerprint':A.ROW_FINGERPRINT,
            'row_version':A.ROW_VERSION,'deployment_id':IDENTITY[0],'database_id':IDENTITY[1],
            'classification':'NATIVE_FILL_COHORT','resolution_scope':'RECONCILIATION_ONLY',
            'financial_eligible':False,'fees':None,'net':None,'source_pins':dict(A.SOURCE_PINS),
            'rationale':'Synthetic isolated test approval only','row_snapshot':deepcopy(TRADE),
            'fills':fills(),'evidence':[{'reference':'fixture-only','sha256':'a'*64}]}

def request(document, action='APPLY', revision=0, policy=1, ref='fixture-approval'):
    return {'action':action,'allocation_id':CASE,'trade_id':110,'row_fingerprint':A.ROW_FINGERPRINT,
            'row_version':A.ROW_VERSION,'deployment_id':IDENTITY[0],'database_id':IDENTITY[1],
            'expected_policy_revision':policy,'expected_allocation_revision':revision,
            'manifest_sha256':D.fingerprint(document),'actor':'fixture','approved_by':'fixture-owner',
            'approved_at':'2026-10-19T00:00:00+00:00','approval_ref':ref}

def encoded_fills(document):
    return [(D.canonical(item['snapshot']),) for item in document['fills']]

def claim_rows():
    return [(item[1],'FILL',item[3],item[4],CASE) for item in A.FILLS]

class Cursor:
    def __init__(self, ones=(), many=()):
        self.ones=list(ones);self.many=list(many);self.calls=[];self.rowcount=1
        self.current_sql=''
        self.original_metadata=[(name,'r',False,False,False,True) for name in ('trades','alpaca_activities')]
        self.replication_role='origin'
        self.audit_history=[]
        self.function_metadata=[(name,FUNCTION_BODIES[name],False,True,True,
            'plpgsql',True,'f',False,False,'v','u',0) for name in sorted(A.FUNCTION_PINS)]
        self.trigger_metadata=[(table,name,function,'public',bits,'O',True,0,0,True,'6' if marker else '', '6')
            for (table,name),(function,bits,marker) in sorted(A.TRIGGER_CONTRACT.items())]
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def execute(self,sql,parameters=None): self.calls.append((sql,parameters));self.current_sql=sql
    def fetchone(self):
        if self.current_sql=="SELECT pg_catalog.current_setting('session_replication_role')":
            return (self.replication_role,)
        return self.ones.pop(0)
    def fetchmany(self,n):
        if "c.relname IN ('trades','alpaca_activities')" in self.current_sql:
            return self.original_metadata
        if 'FROM pg_catalog.pg_proc p JOIN' in self.current_sql:
            return self.function_metadata
        if 'FROM pg_catalog.pg_trigger t JOIN' in self.current_sql:
            return self.trigger_metadata
        if 'FROM public.trade_allocation_audit a JOIN' in self.current_sql:
            return self.audit_history
        return self.many.pop(0)

class Connection:
    autocommit=False
    def __init__(self,cursor,status=0): self.c=cursor;self.status=status;self.commits=0;self.rollbacks=0
    def get_transaction_status(self): return self.status
    def __enter__(self): return self
    def __exit__(self,kind,*unused):
        if kind is None: self.commits+=1
        else: self.rollbacks+=1
        return False
    def cursor(self): return self.c

def active_cursor(document, approved, *, trade=None, state='ACTIVE', revision=1, policy_revision=2):
    metadata=[(name,'r',False,False,False,True) for name in A.TABLES]
    saved=(CASE,state,revision,D.canonical(approved),D.canonical(document),D.fingerprint(approved),
           D.canonical(approved),D.canonical(A.receipt(approved)),policy_revision)
    cursor=Cursor(ones=[(*IDENTITY,'fixture',policy_revision,'fixture'),tuple(A.TABLES),(True,True),(1,)],
       many=[metadata,[saved],[(D.canonical(trade or TRADE),)],encoded_fills(document),
             claim_rows(),[(order,CASE) for order in A.ORDERS]])
    history=[approved] if revision==1 else [request(document),approved]
    cursor.audit_history=[(D.fingerprint(item),D.canonical(item),True,D.canonical(A.receipt(item)),
                           item['expected_policy_revision']+1,True) for item in history]
    return cursor

class AllocationTests(unittest.TestCase):
    def test_exact_cohort_economics_and_unknown_financials_are_immutable(self):
        document=manifest();approved=request(document)
        A.validate_request(approved,document,IDENTITY)
        self.assertEqual(len(A.validate_fills(document['fills'])),5)
        overlay=A.read_active(active_cursor(document,approved),IDENTITY,2,frozenset({63}))
        allocation=overlay.projection[110]
        self.assertEqual(allocation.gross_result,Decimal('-63.38'))
        self.assertEqual(allocation.entry_quantity,allocation.exit_quantity)
        self.assertIsNone(allocation.fees);self.assertIsNone(allocation.net)
        self.assertFalse(allocation.financial_eligible)
        self.assertEqual(allocation.order_ids,A.ORDERS)
        self.assertEqual(allocation.activity_ids,tuple(item[1] for item in A.FILLS))
        self.assertEqual(overlay.metadata_dict()['allocations'][0]['order_ids'],list(A.ORDERS))
        with self.assertRaises(FrozenInstanceError): allocation.gross_result=Decimal(0)
        with self.assertRaises(TypeError): overlay.projection[110]=None
        with self.assertRaises(TypeError): overlay.metadata['allocations'][0]['fees']=0
        exported=overlay.metadata_dict();exported['allocations'][0]['gross_result']='0'
        self.assertEqual(overlay.metadata_dict()['allocations'][0]['gross_result'],'-63.38')

    def test_decimal_ambient_precision_does_not_change_conservation(self):
        document=manifest()
        with localcontext() as context:
            context.prec=1
            A.validate_fills(document['fills'])
            self.assertEqual(context.prec,1)

    def test_original_complete_row_fingerprint_and_version_are_frozen(self):
        self.assertEqual(D.fingerprint(TRADE),A.ROW_FINGERPRINT)
        for key,value in [('qty',Decimal(31)),('updated_at','2026-04-17T19:05:45+00:00'),
                          ('exit_price',Decimal('62.887')),('new_column','drift')]:
            changed=deepcopy(TRADE);changed[key]=value
            with self.assertRaises(A.AllocationError): A.validate_trade(changed)

    def test_duplicate_overallocated_wrong_order_or_time_native_facts_refuse(self):
        for key,value in [('qty',Decimal(16)),('order_id',A.ENTRY_ORDER),('price',Decimal('63.27')),
                          ('transaction_time','2026-04-17T18:12:34+00:00')]:
            data=fills();data[-1]['snapshot'][key]=value
            data[-1]['row_fingerprint']=D.fingerprint(data[-1]['snapshot'])
            with self.assertRaises(A.AllocationError): A.validate_fills(data)
        data=fills();data[-1]=deepcopy(data[-2])
        with self.assertRaises(A.AllocationError): A.validate_fills(data)

    def test_current_full_fill_raw_drift_is_not_hidden_by_matching_projection(self):
        document=manifest();current=encoded_fills(document)
        changed=deepcopy(document['fills'][0]['snapshot']);changed['raw']={'new':'fact'}
        current[0]=(D.canonical(changed),)
        cursor=Cursor(many=[[(D.canonical(TRADE),)],current])
        with self.assertRaises(A.AllocationError): A.validate_current(cursor,document)
        # A native opaque ID is not unique across activity types. Legitimate
        # non-FILL/fee observations remain observations and never supply net PnL.
        for native,order in ((A.FILLS[0][1],'unrelated-order'),('fee-native',A.EXIT_ORDERS[0])):
            fee={'activity_pk':20000,'activity_id':native,'activity_type':'FEE','order_id':order,'amount':Decimal('-1')}
            cursor=Cursor(many=[[(D.canonical(TRADE),)],encoded_fills(document)+[(D.canonical(fee),)]])
            A.validate_current(cursor,document)
        fee={'activity_pk':A.FILLS[0][0],'activity_id':A.FILLS[0][1],'activity_type':'FEE','order_id':'other'}
        cursor=Cursor(many=[[(D.canonical(TRADE),)],encoded_fills(document)[1:]+[(D.canonical(fee),)]])
        with self.assertRaises(A.AllocationError): A.validate_current(cursor,document)

    def test_all_physical_closed_or_excluded_owners_conflict(self):
        document=manifest()
        for status in ('CLOSED','OPEN'):
            owner={'trade_id':63,'status':status,'entry_order_id':A.EXIT_ORDERS[0],'exit_order_id':None}
            cursor=Cursor(many=[[(D.canonical(TRADE),),(D.canonical(owner),)]])
            with self.assertRaises(A.AllocationError): A.validate_current(cursor,document)
        with self.assertRaises(A.AllocationError): A.rows(Cursor(many=[[None]*1002]),1001)

    def test_approval_target_scope_and_float_documents_fail_closed(self):
        document=manifest();approved=request(document)
        for key,value in [('trade_id',99),('manifest_sha256','0'*64),('extra','unsupported')]:
            changed=dict(approved);changed[key]=value
            with self.assertRaises((A.AllocationError,D.TradePolicyError)): A.validate_request(changed,document,IDENTITY)
        changed=dict(approved);changed['approved_at']=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
        with self.assertRaises(A.AllocationError): A.validate_request(changed,document,IDENTITY)
        for value in ({'qty':0.1},'{"a":1,"a":2}','{"n":NaN}'):
            with self.assertRaises((A.AllocationError,D.TradePolicyError)): A.bounded_document(value)

    def test_only_all_tables_absent_is_migration_pending_and_permission_propagates(self):
        cursor=Cursor(ones=[(*IDENTITY,'fixture',1,'fixture'),(None,None,None,None),None])
        result=A.read_active(cursor,IDENTITY,1,frozenset())
        self.assertEqual(result.metadata_dict()['status'],'MIGRATION_PENDING')
        self.assertEqual(result.allocations,())
        with self.assertRaises(A.AllocationError): A.schema_present(Cursor(ones=[('one',None,None,None),(True,True),(1,)]))
        # Installed marker survives every optional table disappearing.
        with self.assertRaises(A.AllocationError): A.schema_present(Cursor(ones=[(None,None,None,None),(True,True),(1,)]))
        class Denied(Cursor):
            def execute(self,*args): raise PermissionError('fixture')
        with self.assertRaises(PermissionError): A.schema_present(Denied())

    def test_rls_inheritance_or_invisible_sidecars_are_not_empty_allocations(self):
        for index,value in ((1,'v'),(2,True),(3,True),(4,True),(5,False)):
            metadata=[(name,'r',False,False,False,True) for name in A.TABLES]
            row=list(metadata[0]);row[index]=value;metadata[0]=tuple(row)
            with self.assertRaises(A.AllocationError): A.schema_present(Cursor(ones=[tuple(A.TABLES),(True,True),(1,)],many=[metadata]))
            cursor=Cursor();original=list(cursor.original_metadata[0]);original[index]=value
            cursor.original_metadata[0]=tuple(original)
            with self.assertRaises(A.AllocationError): A.validate_current(cursor,manifest())

    def test_installed_contract_attests_real_sql_bodies_and_enabled_exact_triggers(self):
        self.assertEqual(set(FUNCTION_BODIES),set(A.FUNCTION_PINS))
        for name,body in FUNCTION_BODIES.items():
            self.assertEqual(hashlib.sha256(body.encode('utf-8')).hexdigest(),A.FUNCTION_PINS[name])
        A.attest_contract(Cursor())
        for index,value in ((1,'BEGIN RETURN NULL; END'),(2,True),(3,False),(4,False),(5,'sql')):
            cursor=Cursor();row=list(cursor.function_metadata[0]);row[index]=value;cursor.function_metadata[0]=tuple(row)
            with self.assertRaises(A.AllocationError): A.attest_contract(cursor)
        for index,value in ((2,'other'),(3,'pg_temp'),(4,0),(5,'D'),(6,False),(7,1),(8,1),(9,False),(10,'9')):
            cursor=Cursor();row=list(cursor.trigger_metadata[0]);row[index]=value;cursor.trigger_metadata[0]=tuple(row)
            with self.assertRaises(A.AllocationError): A.attest_contract(cursor)
        cursor=Cursor();cursor.trigger_metadata.pop()
        with self.assertRaises(A.AllocationError): A.attest_contract(cursor)
        cursor=Cursor();cursor.replication_role='replica'
        with self.assertRaises(A.AllocationError): A.attest_contract(cursor)

    def test_active_exclusion_audit_or_row_drift_refuses_whole_overlay(self):
        document=manifest();approved=request(document)
        with self.assertRaises(A.AllocationError): A.read_active(active_cursor(document,approved),IDENTITY,2,{110})
        changed=deepcopy(TRADE);changed['symbol']='OTHER'
        with self.assertRaises(A.AllocationError): A.read_active(active_cursor(document,approved,trade=changed),IDENTITY,2,set())
        cursor=active_cursor(document,approved);case=list(cursor.many[1][0]);case[5]='0'*64;cursor.many[1]=[tuple(case)]
        with self.assertRaises(A.AllocationError): A.read_active(cursor,IDENTITY,2,set())
        cursor=active_cursor(document,approved);cursor.audit_history=[]
        with self.assertRaises(A.AllocationError): A.read_active(cursor,IDENTITY,2,set())

    def test_claim_reuse_or_oversubscription_refuses(self):
        claims=claim_rows();claims[0]=(*claims[0][:4],'00000000-0000-0000-0000-000000000001')
        with self.assertRaises(A.AllocationError): A.check_claims(Cursor(many=[claims]),CASE)
        claims=claim_rows();claims[0]=(claims[0][0],claims[0][1],claims[0][2],'19',CASE)
        with self.assertRaises(A.AllocationError): A.check_claims(Cursor(many=[claims]),CASE)

    def test_actual_mutator_cas_lock_order_and_only_additive_writes(self):
        document=manifest();approved=request(document)
        cursor=Cursor(ones=[None,None,None],many=[[(D.canonical(TRADE),)],encoded_fills(document),[]])
        conn=Connection(cursor)
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=1,active=())) as policy_read, \
             patch.object(A,'schema_present',return_value=True):
            result=A.change_allocation(conn,approved,document,IDENTITY)
        self.assertEqual(result['policy_revision'],2);self.assertEqual(conn.commits,1)
        self.assertTrue(policy_read.call_args.kwargs['exclusive'])
        sql=[text for text,args in cursor.calls]
        self.assertLess(next(i for i,text in enumerate(sql) if text.startswith('LOCK TABLE')),
                        next(i for i,text in enumerate(sql) if 'FROM public.trades t' in text))
        self.assertFalse(any(text.startswith(('UPDATE public.trades','UPDATE public.alpaca_activities','INSERT INTO public.order_events')) for text in sql))
        self.assertEqual(sum(text.startswith('INSERT INTO public.trade_allocation_fill_claims') for text in sql),5)

    def test_replay_vs_conflict_and_policy_cas_rolls_back(self):
        document=manifest();approved=request(document)
        cursor=Cursor(ones=[None,(D.fingerprint(approved),D.canonical(A.receipt(approved))),('ACTIVE',1)],
                      many=[[(D.canonical(TRADE),)],encoded_fills(document),claim_rows(),[(order,CASE) for order in A.ORDERS]])
        conn=Connection(cursor)
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=2,active=())),patch.object(A,'schema_present',return_value=True):
            self.assertEqual(A.change_allocation(conn,approved,document,IDENTITY),A.receipt(approved))
        self.assertFalse(any(sql.startswith(('INSERT','UPDATE')) for sql,args in cursor.calls))
        cursor=Cursor(ones=[None,(D.fingerprint(approved),D.canonical(A.receipt(approved))),('REVOKED',2)])
        conn=Connection(cursor)
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=3,active=())),patch.object(A,'schema_present',return_value=True):
            with self.assertRaisesRegex(A.AllocationError,'REPLAY_SUPERSEDED'):
                A.change_allocation(conn,approved,document,IDENTITY)
        self.assertEqual(conn.rollbacks,1)
        for audit in (('0'*64,D.canonical(A.receipt(approved))),None):
            conn=Connection(Cursor(ones=[None,audit]))
            with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=2,active=())),patch.object(A,'schema_present',return_value=True):
                with self.assertRaises(A.AllocationError): A.change_allocation(conn,approved,document,IDENTITY)
            self.assertEqual(conn.rollbacks,1)

    def test_revoke_is_audited_and_preserves_all_native_claims(self):
        document=manifest();approved=request(document,'REVOKE',1,2,'fixture-revoke')
        cursor=Cursor(ones=[None,None,('ACTIVE',1,D.canonical(document))],
                      many=[[(D.canonical(TRADE),)],encoded_fills(document),claim_rows(),[(order,CASE) for order in A.ORDERS]])
        conn=Connection(cursor)
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=2,active=())),patch.object(A,'schema_present',return_value=True):
            result=A.change_allocation(conn,approved,document,IDENTITY)
        self.assertEqual(result['state'],'REVOKED');self.assertEqual(result['policy_revision'],3)
        self.assertTrue(any(sql.startswith('INSERT INTO public.trade_allocation_audit') for sql,args in cursor.calls))
        self.assertFalse(any(sql.startswith('DELETE') for sql,args in cursor.calls))
        revoked=A.read_active(active_cursor(document,approved,state='REVOKED',revision=2,policy_revision=3),IDENTITY,3,{63})
        self.assertEqual(revoked.allocations,())
        self.assertEqual(revoked.metadata_dict()['active_count'],0)
        self.assertEqual(dict(revoked.reserved_order_owners),{order:110 for order in A.ORDERS})
        self.assertEqual(revoked.metadata_dict()['reserved_order_owners'],{order:110 for order in A.ORDERS})
        with self.assertRaises(TypeError): revoked.reserved_order_owners[A.EXIT_ORDERS[0]]=111

    def test_connection_must_be_idle_and_non_autocommit(self):
        document=manifest();approved=request(document)
        for status,autocommit in ((1,False),(0,True)):
            conn=Connection(Cursor(),status);conn.autocommit=autocommit
            with self.assertRaises(A.AllocationError): A.change_allocation(conn,approved,document,IDENTITY)
            self.assertEqual(conn.c.calls,[])

    def test_revoked_owner_row_may_change_without_freeing_claims_but_reactivation_is_strict(self):
        document=manifest();revocation=request(document,'REVOKE',1,2,'fixture-revoke')
        changed=deepcopy(TRADE);changed['exit_reason']='OWNER_REVIEWED_AFTER_REVOKE'
        changed['updated_at']='2026-10-19T22:00:00+00:00'
        overlay=A.read_active(active_cursor(document,revocation,trade=changed,state='REVOKED',revision=2,policy_revision=3),IDENTITY,3,{63})
        self.assertEqual(overlay.allocations,())
        self.assertEqual(dict(overlay.reserved_order_owners),{order:110 for order in A.ORDERS})
        # An old revoke replay remains idempotent after a legitimate scalar change.
        cursor=Cursor(ones=[None,(D.fingerprint(revocation),D.canonical(A.receipt(revocation))),('REVOKED',2)],
                      many=[[(D.canonical(changed),)],encoded_fills(document),claim_rows(),[(order,CASE) for order in A.ORDERS]])
        conn=Connection(cursor)
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=3,active=())),patch.object(A,'schema_present',return_value=True):
            self.assertEqual(A.change_allocation(conn,revocation,document,IDENTITY),A.receipt(revocation))
        self.assertFalse(any(sql.startswith(('INSERT','UPDATE')) for sql,args in cursor.calls))
        # Active validation and a fresh reactivation still need the exact current row.
        active=request(document)
        with self.assertRaises(A.AllocationError):
            A.read_active(active_cursor(document,active,trade=changed),IDENTITY,2,{63})
        apply=request(document,'APPLY',2,3,'fixture-reactivation')
        conn=Connection(Cursor(ones=[None,None],many=[[(D.canonical(changed),)]]))
        with patch.object(D,'read_policy',return_value=SimpleNamespace(revision=3,active=())),patch.object(A,'schema_present',return_value=True):
            with self.assertRaises(A.AllocationError): A.change_allocation(conn,apply,document,IDENTITY)
        self.assertEqual(conn.rollbacks,1)
        self.assertFalse(any(sql.startswith(('INSERT','UPDATE')) for sql,args in conn.c.calls))

if __name__=='__main__':
    unittest.main()
