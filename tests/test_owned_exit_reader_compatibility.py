"""Offline synthetic guard metadata and reader-delegation regression tests."""
import ast,copy,hashlib,json,pathlib,sys,types
from unittest.mock import patch
from types import SimpleNamespace as N
ROOT=pathlib.Path(__file__).resolve().parents[1]/'scripts'

def load_attestation():
    tree=ast.parse((ROOT/'owned_exit_transaction.py').read_text())
    names={'CASE_GUARD_BODY','FUNCTION_SHA','MULTI_CASE_GUARD_BODY','MULTI_FUNCTION_SHA','ATTEST_SQL'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='attest_extension']
    def need(ok,code):
        if not ok:raise ValueError(code)
    g={'hashlib':hashlib,'O':N(policy_module=lambda:N(document=json.loads),need=need,read_rows=lambda c,n:c.rows)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_attestation','exec'),g)
    return g

def rows(body):
    return [('FUNCTION',dict(body=body,owner=True,definer=False,config=True,language='plpgsql',trigger=True,args=0)),
            ('AUDIT',dict(owner=True,ordinary=True,private=True,columns=True,primary_key=True)),
            ('FK',dict(valid=True)),
            ('TRIGGER',dict(name='trade_owned_exit_case_guard',enabled=True,bits=23,function='trade_owned_exit_case_guard',public_function=True)),
            ('TRIGGER',dict(name='jbravo_owned_exit_audit_immutable',enabled=True,bits=58,function='trade_allocation_immutable',public_function=True))]

def invoke(g,data):
    c=N(rows=[(k,json.dumps(v)) for k,v in data],execute=lambda sql:None)
    return g['attest_extension'](c)

def refused(g,data,reason):
    try:invoke(g,data)
    except ValueError as e:assert e.args==(reason,)
    else:raise AssertionError('invalid attestation accepted')

def test_exact_guard_versions_and_metadata():
    g=load_attestation()
    assert g['FUNCTION_SHA']!=g['MULTI_FUNCTION_SHA']
    for body,digest in [('CASE_GUARD_BODY','FUNCTION_SHA'),('MULTI_CASE_GUARD_BODY','MULTI_FUNCTION_SHA')]:
        good=rows(g[body]);assert invoke(g,good)==g[digest]
        for field,bad in [('body',g[body]+' '),('owner',False),('definer',True),('config',False),('language','sql'),('trigger',False),('args',1)]:
            candidate=copy.deepcopy(good);candidate[0][1][field]=bad
            refused(g,candidate,'EXTENSION_FUNCTION')
        for field in ('owner','ordinary','private','columns','primary_key'):
            candidate=copy.deepcopy(good);candidate[1][1][field]=False
            refused(g,candidate,'EXTENSION_SCHEMA_OR_ACL')
        candidate=copy.deepcopy(good);candidate[2][1]['valid']=False
        refused(g,candidate,'EXTENSION_SCHEMA_OR_ACL')
        for index in (3,4):
            for field,bad in [('enabled',False),('bits',0),('function','wrong_guard'),('public_function',False)]:
                candidate=copy.deepcopy(good);candidate[index][1][field]=bad
                refused(g,candidate,'EXTENSION_TRIGGER')
        refused(g,good[:-1],'EXTENSION_NOT_INSTALLED')

def test_owned_reader_scope_and_validator_delegation():
    tree=ast.parse((ROOT/'owned_exit_repair.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_extensions']
    seen=[];validator=object();result=object()
    def need(ok,code):
        if not ok:raise ValueError(code)
    def delegate(*args,**kwargs):seen.append((args,kwargs));return result
    g={'need':need,'validate_group':validator,'_read_extensions':delegate}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_reader','exec'),g)
    args=(object(),[object(),object()],('synthetic','identity'),1,frozenset())
    assert g['read_extensions'](*args,allocation_type=int,attest=str) is result
    assert seen==[(args,dict(allocation_type=int,attest=str,validator=validator))]
    try:g['read_extensions'](None,[1,2,3],None,None,None,allocation_type=int,attest=str)
    except ValueError as e:assert e.args==('ALLOCATION_SCOPE',)
    else:raise AssertionError('owned reader scope widened')


def test_whole_position_requires_exact_multi_guard():
    tree=ast.parse((ROOT/'whole_position_allocation.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_extensions']
    seen=[];validator=object();expected=object()
    def need(ok,code):
        if not ok:raise ValueError(code)
    def delegate(*args,**kwargs):
        kwargs['attest'](args[0]);seen.append((args,kwargs));return expected
    g={'__name__':'scripts.whole_position_allocation','__package__':'scripts','O':N(need=need,_read_extensions=delegate),'validate_group':validator}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_whole_reader','exec'),g)
    package=types.ModuleType('scripts');x=N(MULTI_FUNCTION_SHA='exact-multi',attest_extension=lambda c:'old-guard');package.owned_exit_transaction=x
    args=(object(),[object()],('synthetic','identity'),1,frozenset())
    with patch.dict(sys.modules,{'scripts':package,'scripts.owned_exit_transaction':x}):
        try:g['read_extensions'](*args,allocation_type=int)
        except ValueError as e:assert e.args==('MULTI_GUARD_REQUIRED',)
        else:raise AssertionError('old guard accepted for whole-position')
        assert not seen
        x.attest_extension=lambda c:'exact-multi'
        assert g['read_extensions'](*args,allocation_type=int) is expected
        assert seen[0][0]==args and seen[0][1]['validator'] is validator
        try:g['read_extensions'](None,[1,2],None,None,None,allocation_type=int)
        except ValueError as e:assert e.args==('MULTI_ALLOCATION_SCOPE',)
        else:raise AssertionError('whole reader scope widened')

if __name__=='__main__':
    test_exact_guard_versions_and_metadata();test_owned_reader_scope_and_validator_delegation();test_whole_position_requires_exact_multi_guard()
    print('PASS synthetic exact guard hashes, adverse metadata, schema, triggers, scope and delegation; no DB/network')
