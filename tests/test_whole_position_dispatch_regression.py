"""Routing tests only; real ledger validity remains delegated to unchanged W reader."""
import ast,json,sys,types,pathlib
from types import SimpleNamespace as N
from unittest.mock import patch
R=pathlib.Path(__file__).parent
source=R.parent/'scripts/trade_allocation.py'
identity=('deployment','database')
def record(kind,state='ACTIVE'):
 return ('allocation',state,1,'{}',json.dumps({'classification':kind}),'digest','{}','{}',1,'group-approval')
def run(records,whole,owned=None):
 a=ast.parse(source.read_text());nodes=[n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('AllocationError','fail','require','read_active')]
 D=N(identity=lambda *x:None,revision=lambda x:None,document=json.loads)
 g={'policy_module':lambda:D,'schema_present':lambda c:True,'rows':lambda c,n:records,'Allocation':object,
 'result_read':lambda status,active,i,p,reserved,cases:dict(status=status,active=active,reserved=reserved,cases=cases)}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),g)
 class Cursor:
  def execute(self,*args):pass
  def fetchone(self):return ('deployment','database','db',1,'db')
 package=types.ModuleType('scripts');package.whole_position_allocation=N(read_extensions=whole)
 package.owned_exit_repair=N(read_extensions=owned or (lambda *a,**k:([],{},[])))
 package.owned_exit_transaction=N(attest_extension=object())
 with patch.dict(sys.modules,{'scripts':package}):return g['read_active'](Cursor(),identity,1,frozenset())
def test_whole_position_dispatch_guards():
    seen=[]
    def success(cursor,records,i,p,e,*,allocation_type):
     seen.extend(records);assert i==identity and p==1 and e==frozenset()
     return ([N(trade_id=131)],{'whole_order':131},[{'state':'ACTIVE'}])
    row=record('WHOLE_POSITION_MULTI_ORDER_V1');r=run([row],success)
    assert seen==[row] and r['reserved']=={'whole_order':131} and len(r['active'])==1
    # Real-validator errors are not swallowed or converted to empty ownership.
    def fail(*a,**k):raise ValueError('SYNTHETIC_ATTESTATION_FAILURE')
    try:run([row],fail)
    except ValueError as e:assert str(e)=='SYNTHETIC_ATTESTATION_FAILURE'
    else:raise AssertionError('validation failure swallowed')
    # Cross-format claim and trade-owner collisions stay fail-closed.
    for whole_result in (([N(trade_id=131)],{'same_order':131},[]),([N(trade_id=132)],{'other_order':132},[])):
     try:run([record('OWNED_EXIT_COHORT_V1'),row],lambda *a,**k:whole_result,lambda *a,**k:([N(trade_id=132)],{'same_order':132},[]))
     except RuntimeError as e:assert str(e)=='TRADE_ALLOCATION_RESERVED_OWNER_CONFLICT'
     else:raise AssertionError('collision accepted')
    # Revoked groups retain reserved orders without manufacturing active positions.
    rev=record('WHOLE_POSITION_MULTI_ORDER_V1','REVOKED')
    def revoked(c,rows,*a,**k):
     assert rows==[rev];return ([],{'reserved_order':131},[{'state':'REVOKED'}])
    r=run([rev],revoked);assert r['active']==[] and r['reserved']=={'reserved_order':131} and r['cases']==[{'state':'REVOKED'}]
    # Unknown nonlegacy groups are not broadened into accepted ownership.
    try:run([record('UNKNOWN_FORMAT')],success)
    except RuntimeError as e:assert str(e)=='TRADE_ALLOCATION_LEGACY_GROUP_REFUSED'
    else:raise AssertionError('unknown group accepted')

if __name__ == "__main__":
    test_whole_position_dispatch_guards()
    print("PASS: six synthetic dispatch guard scenarios")
