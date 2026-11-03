"""Real D/A readers over synthetic DB transport, never provider/DB access."""
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
pytestmark=pytest.mark.alpaca_optional
from dashboards import execution_gross_projection as E, execution_read_transport as T, recorded_trade_analytics as R
from scripts import trade_disposition as D, trade_allocation as A
from test_execution_gross_view import Connection, Cursor, inputs
from test_public_dashboard_integration import IDENTITY, NOW


def bounded(c):return E.BoundedCursor(c.cursor(),time.monotonic()+8)


def canonical_case(identifier=9004):
    row,_,_=inputs(identifier,'EXCLUDED')
    case_id='00000000-0000-4000-8000-000000009004'
    manifest=dict(case_id=case_id,trade_id=identifier,row_fingerprint=D.fingerprint(row),row_version=row['updated_at'],
        deployment_id=IDENTITY[0],database_id=IDENTITY[1],classification='UNRESOLVED',provenance_proved=False,
        operational_scopes=['VALIDATED_PERFORMANCE'],rationale='Fictional test only',
        evidence=[dict(reference='synthetic-fixture',sha256='0'*64)])
    request=dict(action='APPLY',**{k:manifest[k] for k in ('case_id','trade_id','row_fingerprint','row_version','deployment_id','database_id')},
        expected_policy_revision=1,expected_case_revision=0,manifest_sha256=D.fingerprint(manifest),
        actor='fictional',approved_by='fictional',approved_at=NOW.isoformat(),approval_ref='SYNTHETIC-NOT-AUTHORITY')
    result=dict(case_id=case_id,trade_id=identifier,state='ACTIVE',case_revision=1,policy_revision=2,approval_ref=request['approval_ref'])
    return (D.canonical(request),D.canonical(row),D.canonical(manifest),1,D.fingerprint(request),2,
            D.canonical(row),D.canonical(request),D.canonical(result))


def test_reproduce_original_isolation_failure_without_mocking_canonical_reader():
    c=Connection();raw=c.cursor();raw.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
    with pytest.raises(D.TradePolicyError,match='ISOLATION_INVALID'):
        D.read_policy(E.BoundedCursor(raw,time.monotonic()+8),IDENTITY,scope='VALIDATED_PERFORMANCE')
    assert not c.source_reads


def test_unrelated_api_audit_commit_must_not_invalidate_unchanged_financial_inputs():
    c=Connection();c.last_snapshot='100:201:';c.unrelated_audit_commit=True
    r=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert r['ok'] and r['rows'][-1]['totalPL']=='4'
    assert r['net_eligible'] is False
    assert c.audit_revision==1 and c.database_snapshot=='100:201:'
    assert not any('txid_current_snapshot' in sql or 'trades_api_requests' in sql for sql,_ in c.calls)


def test_read_through_real_policy_and_allocation_reader_with_valid_exclusion():
    c=Connection();c.cases=[canonical_case()]
    excluded,fills,owner=inputs(9004,'EXCLUDED');c.records.append(excluded);c.fills+=fills;c.owners.append(owner)
    r=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert r['ok'] and r['rows'][-1]['totalPL']=='4'
    assert r['coverage']['excluded']==1 and r['coverage']['included']==1
    assert r['trade_policy']['policy']['allocation_overlay']['status']=='MIGRATION_PENDING'
    assert r['read_consistency']=='READ_COMMITTED_VERSIONED_COHORT_AND_CANONICAL_RECHECK'
    assert r['net_eligible'] is r['accounting_complete'] is False
    assert c.calls[0][0]=='SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY'
    assert 0<len(c.calls)<=128 and max(c.fetch_sizes)<=16
    assert c.policy_reads==4 and c.source_reads==2 and c.snapshots==0
    c.rollback.assert_called_once();c.close.assert_called_once()


@pytest.mark.parametrize('mutation',['row_drift','fill_drift','owner_drift','version_drift','revision_drift','canonical_version_drift'])
def test_drift_and_value_restored_row_version_fail_closed_through_real_readers(mutation):
    c=Connection();setattr(c,mutation,True)
    r=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert r['ok'] is False and r['rows']==[] and r['coverage'] is None
    c.rollback.assert_called_once();c.close.assert_called_once()


@pytest.mark.parametrize('field',['records','fills','owners'])
def test_restored_economic_values_with_changed_relevant_row_version_refuse(field):
    c=Connection();c.version_drift_table=field
    # Native business values are identical on both reads; only xmin differs.
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


@pytest.mark.parametrize('scope',[' FROM public.trade_disposition_policy ',' FROM public.trade_disposition_cases c '])
def test_restored_policy_or_case_values_with_changed_xmin_refuse(scope):
    c=Connection();c.cases=[canonical_case()];c.canonical_version_scope=scope
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


@pytest.mark.parametrize('bad',['repeatable read','serializable'])
def test_canonical_isolation_cannot_be_bypassed(bad):
    c=Connection();c.force_isolation=bad
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


@pytest.mark.parametrize('table',['trades','alpaca_activities'])
@pytest.mark.parametrize('partition',[False,True])
def test_parent_or_child_inheritance_refused_even_without_active_allocations(table,partition):
    c=Connection();c.visibility=[(name,'r',False,False,partition if name==table else False,name==table,True)
        for name in ('alpaca_activities','trades')]
    result=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert not result['ok'] and result['coverage'] is None
    assert 'EXISTS(SELECT 1 FROM pg_catalog.pg_inherits' in E.VISIBILITY_SQL
    assert 'i.inhrelid=c.oid OR i.inhparent=c.oid' in E.VISIBILITY_SQL


def test_partial_allocation_schema_and_wrong_database_fail_real_canonical_checks():
    plans=T.plan_table();regclass=next(q for q in plans if 'to_regclass' in q)
    c=Connection();c.overrides={regclass:[('unexpected_table',None,None,None)]}
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']
    c=Connection();c.current_db='wrong_db'
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


def test_multi_field_canonical_json_oversize_is_redacted_before_native_transport():
    c=Connection();row=list(canonical_case());row[0]=row[1]='CANARY_'+('x'*17000);c.cases=[tuple(row)]
    result=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert not result['ok'] and result['coverage'] is None
    assert any(wire[0] and wire[1] is None for wire in c.wire)
    assert all('CANARY_' not in str(wire) for wire in c.wire)


@pytest.mark.parametrize('extra',[-1,0,1])
def test_exact_whole_row_transport_boundary_including_envelope(extra):
    c=Connection();sql='SHOW transaction_isolation'
    # JSON array of one string has four ASCII framing bytes.
    c.overrides={sql:[('x'*(T.BODY_BYTES-4+extra),)]};x=bounded(c)
    if extra<=0:
        x.execute(sql);assert len(x.fetchone()[0])==T.BODY_BYTES-4+extra
        assert x.bytes<=32768
    else:
        with pytest.raises(ValueError,match='SQL_OVERSIZE'):x.execute(sql)
        assert c.wire[-1][1] is None


def test_utf8_aggregate_rows_and_aggregate_bytes_capped_before_fetch():
    sql='SHOW transaction_isolation'
    for rows,bytes_,native in [(4095,0,[('one',),('two',)]),(0,T.MAX_BYTES-131,[('one',)])]:
        c=Connection();c.overrides={sql:native};x=bounded(c);x.rows=rows;x.bytes=bytes_
        with pytest.raises(ValueError,match='SQL_OVERSIZE'):x.execute(sql)
        assert all(row[1] is None for row in c.wire)
    c=Connection();c.overrides={sql:[('\u00e9'*17000,)]}
    with pytest.raises(ValueError,match='SQL_OVERSIZE'):bounded(c).execute(sql)


def test_entire_canonical_query_charged_even_when_only_one_row_consumed():
    c=Connection();sql='SHOW transaction_isolation';c.overrides={sql:[('a',),('b',)]};x=bounded(c)
    x.execute(sql);assert x.fetchone()==('a',) and x.rows==2
    expected=sum(len(json.dumps([v],separators=(',',':')).encode())+128 for v in ('a','b'))
    assert x.bytes==expected
    x.execute(sql);assert x.rows==4 and x.bytes==2*expected


def test_deadline_and_query_budget_include_canonical_controls_and_timeout_controls():
    c=Connection();x=bounded(c);x.statements=127
    with pytest.raises(ValueError,match='SQL_CAP'):x.execute('SHOW transaction_isolation')
    assert not c.calls
    x=E.BoundedCursor(c.cursor(),0,clock=lambda:1)
    with pytest.raises(ValueError,match='DEADLINE'):x.execute('SHOW transaction_isolation')
    x=bounded(c)
    with pytest.raises(ValueError,match='UNREVIEWED'):x.execute('SELECT secret FROM arbitrary_table')


def test_malformed_wire_metadata_cannot_disable_caps():
    c=Connection();x=bounded(c);x.execute('SHOW transaction_isolation')
    first=x.pending[0];x.pending[0]=(False,first[1],first[2],first[3]+1)
    with pytest.raises(ValueError,match='COUNT_CHANGED'):x.fetchone()


def test_view_connection_uses_same_resolver_with_only_bounded_timeout_override():
    fake=SimpleNamespace(_resolve_db_config=Mock(return_value=dict(host='synthetic-host',port=1,dbname='synthetic',user='fictional',sslmode='require')),
        psycopg2=SimpleNamespace(connect=Mock(return_value='synthetic-connection')))
    assert E.connect_current_db(fake)=='synthetic-connection'
    fake.psycopg2.connect.assert_called_once_with(host='synthetic-host',port=1,dbname='synthetic',user='fictional',sslmode='require',connect_timeout=2)
    fake._resolve_db_config.return_value=None
    assert E.connect_current_db(fake) is None
    fake._resolve_db_config.side_effect=ValueError('CANARY_PRIVATE_VALUE')
    assert E.connect_current_db(fake) is None


def test_actual_flask_routes_and_connection_helper_through_canonical_readers():
    import ast
    from flask import Flask,request,jsonify
    tree=ast.parse(Path('dashboards/dashboard_app.py').read_text(encoding='utf-8'))
    names={'api_trades_stats','api_trades_leaderboard','_execution_gross_db_conn'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==3
    connections=[]
    def connect(**kwargs):
        assert kwargs['connect_timeout']==2
        c=Connection();connections.append(c);return c
    fake=SimpleNamespace(_resolve_db_config=lambda:dict(host='synthetic',port=1,dbname='fictional',user='fictional'),psycopg2=SimpleNamespace(connect=connect))
    ns=dict(request=request,_json_no_store=jsonify,db=fake,_dashboard_read_policy_identity=lambda:IDENTITY,
        _parse_trades_range=lambda value,default='all':value or default,_parse_positive_int=lambda value,**kw:kw['default'])
    app=Flask('canonical-mocked-transport-routes')
    for node in nodes:
        node.decorator_list=[];exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'exact-route','exec'),ns)
    for suffix in ('stats','leaderboard'):app.add_url_rule('/'+suffix,suffix,ns['api_trades_'+suffix])
    for suffix in ('stats','leaderboard'):
        result=app.test_client().get('/'+suffix+'?basis=execution_gross&range=all')
        assert result.status_code==200 and result.json['ok'] and result.json['view_scope']==E.SCOPE
        assert result.json['net_eligible'] is False
    assert len(connections)==2 and all(c.source_reads==2 for c in connections)
    for c in connections:c.rollback.assert_called_once();c.close.assert_called_once()


def test_only_fixed_read_plans_and_sql_redaction_for_every_canonical_data_query():
    assert len(T.CANONICAL_PLANS)==25
    for plan in T.CANONICAL_PLANS:
        if plan['kind']=='control':continue
        sql=T.envelope(plan['sql'],plan,4096,16*1024*1024)
        assert 'CASE WHEN body IS NOT NULL' in sql and 'ELSE NULL END AS body' in sql
        assert 'json_build_array(' in sql and 'count(*) OVER ()' in sql
        assert 'total_rows<=4096 AND total_bytes<=16777216' in sql
        assert 'octet_length(body)<=32640' in sql


def installed(c):
    """Actual canonical schema/trigger logic with fixed source body fixtures."""
    p=T.plan_table();c.overrides={}
    for sql in p:
        if 'to_regclass' in sql:c.overrides[sql]=[tuple('public.'+n for n in A.TABLES)]
        elif 'a.attnotnull' in sql:c.overrides[sql]=[(True,True)]
        elif sql.startswith('SELECT allocation_schema_version'):c.overrides[sql]=[(1,)]
        elif "c.relname=ANY(%s)" in sql and 'pg_class c JOIN' in sql:
            c.overrides[sql]=[(name,'r',False,False,False,True) for name in sorted(A.TABLES)]
        elif "current_setting('session_replication_role')" in sql:c.overrides[sql]=[('origin',)]
        elif 'FROM pg_catalog.pg_proc p JOIN' in sql:
            bodies=json.loads(Path('tests/fixtures/allocation_trigger_bodies.json').read_text())['function_bodies']
            assert all(hashlib.sha256(bodies[name].encode()).hexdigest()==pin for name,pin in A.FUNCTION_PINS.items())
            c.overrides[sql]=[(name,bodies[name],False,True,True,'plpgsql',True,'f',False,False,'v','u',0) for name in sorted(A.FUNCTION_PINS)]
        elif 'FROM pg_catalog.pg_trigger t JOIN' in sql:
            c.overrides[sql]=[(table,name,fun,'public',bits,'O',True,0,0,True,'6' if marker else '','6')
                for (table,name),(fun,bits,marker) in sorted(A.TRIGGER_CONTRACT.items())]
        elif 'FROM public.trade_allocation_cases c LEFT JOIN' in sql:c.overrides[sql]=[]


def test_real_installed_allocation_schema_attestation_and_changed_trigger_refusal():
    c=Connection();installed(c)
    result=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert result['ok'] and result['trade_policy']['policy']['allocation_overlay']['status']=='VALIDATED'
    assert len(c.calls)<=128
    sql=next(q for q in c.overrides if 'FROM pg_catalog.pg_proc p JOIN' in q)
    c=Connection();installed(c);rows=deepcopy(c.overrides[sql]);row=list(rows[0]);row[1]+='\n-- modified';rows[0]=tuple(row);c.overrides[sql]=rows
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


def active_crc_connection(monkeypatch):
    # Only the immutable target pins become fictional. No canonical reader,
    # attestation, approval, history, ownership or conservation logic is mocked.
    c=Connection();installed(c)
    row,_,owner=inputs(110,'CRC','30');row.update(entry_price='65',exit_price=None,realized_pnl=None,exit_order_id=None)
    owner['exit_order_id']=None
    orders=('00000000-0000-4000-8000-000000000001','00000000-0000-4000-8000-000000000002','00000000-0000-4000-8000-000000000003')
    row['entry_order_id']=owner['entry_order_id']=orders[0]
    times=['2001-02-03T09:00:00+00:00','2001-02-03T09:01:00+00:00','2001-02-03T10:00:00+00:00','2001-02-03T10:01:00+00:00','2001-02-03T10:02:00+00:00']
    facts=[(1,'fictional-f1','buy',orders[0],'18','65',times[0]),(2,'fictional-f2','buy',orders[0],'12','65',times[1]),
        (3,'fictional-f3','sell',orders[1],'1','63',times[2]),(4,'fictional-f4','sell',orders[1],'14','62.48',times[3]),
        (5,'fictional-f5','sell',orders[2],'15','63.26',times[4])]
    for key,value in dict(FILLS=tuple(facts),ORDERS=orders,CREATED_AT=tuple(times),ENTRY_ORDER=orders[0],ROW_FINGERPRINT=D.fingerprint(row),ROW_VERSION=row['updated_at']).items():monkeypatch.setattr(A,key,value)
    fills=[]
    for pk,native,side,order,qty,price,stamp in facts:
        snap=dict(activity_pk=pk,activity_id=native,activity_type='FILL',transaction_time=stamp,symbol='CRC',side=side,qty=qty,price=price,amount=None,order_id=order,description=None,raw={},created_at=stamp)
        assert set(snap)==A.ACTIVITY_COLUMNS
        fills.append(dict(snapshot=snap,row_fingerprint=D.fingerprint(snap)))
    case='00000000-0000-4000-8000-000000000110'
    manifest=dict(allocation_id=case,trade_id=110,row_fingerprint=A.ROW_FINGERPRINT,row_version=A.ROW_VERSION,
        deployment_id=IDENTITY[0],database_id=IDENTITY[1],classification='NATIVE_FILL_COHORT',resolution_scope='RECONCILIATION_ONLY',
        financial_eligible=False,fees=None,net=None,source_pins=dict(A.SOURCE_PINS),rationale='Synthetic test only',row_snapshot=row,fills=fills,
        evidence=[dict(reference='fictional',sha256='0'*64)])
    request=dict(action='APPLY',**{k:manifest[k] for k in ('allocation_id','trade_id','row_fingerprint','row_version','deployment_id','database_id')},expected_policy_revision=1,expected_allocation_revision=0,manifest_sha256=D.fingerprint(manifest),actor='fictional',approved_by='fictional',approved_at=NOW.isoformat(),approval_ref='SYNTHETIC-NOT-AUTHORITY')
    c.records=[row];c.owners=[owner];c.fills=[]
    for sql in T.plan_table():
        if 'FROM public.trade_allocation_cases c LEFT JOIN' in sql:c.overrides[sql]=[(case,'ACTIVE',1,D.canonical(request),D.canonical(manifest),D.fingerprint(request),D.canonical(request),D.canonical(A.receipt(request)),2)]
        elif 'FROM public.trade_allocation_audit a JOIN' in sql:c.overrides[sql]=[(D.fingerprint(request),D.canonical(request),True,D.canonical(A.receipt(request)),2,True)]
        elif "c.relname IN ('trades','alpaca_activities')" in sql and sql!=E.VISIBILITY_SQL and 'pg_inherits' in sql:c.overrides[sql]=[(n,'r',False,False,False,True) for n in ('alpaca_activities','trades')]
        elif 'FROM public.trades t WHERE t.trade_id=110' in sql:c.overrides[sql]=[(D.canonical(row),)]
        elif 'FROM public.alpaca_activities a WHERE' in sql:c.overrides[sql]=[(D.canonical(f['snapshot']),) for f in fills]
        elif 'FROM public.trade_allocation_fill_claims WHERE' in sql:c.overrides[sql]=[(n,'FILL',o,q,case) for pk,n,s,o,q,p,t in facts]
        elif 'FROM public.trade_allocation_order_claims WHERE' in sql:c.overrides[sql]=[(o,case) for o in orders]
    return c


def test_real_active_crc_reader_five_fills_history_claims_and_shared_budget(monkeypatch):
    c=active_crc_connection(monkeypatch)
    result=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert result['ok'] and result['rows'][-1]['totalPL']=='-63.38'
    assert result['coverage']['included']==1 and result['net_eligible'] is False and result['net_pnl'] is None
    assert c.source_reads==2 and len(c.calls)<=128 and max(c.fetch_sizes)<=16
    # Canonical changed audit/fill provenance must remain unavailable.
    sql=next(q for q in c.overrides if 'FROM public.trade_allocation_fill_claims WHERE' in q)
    c.overrides[sql]=[];c.calls=[];c.wire=[];c.fetch_sizes=[];c.source_reads=c.policy_reads=c.snapshots=0
    assert not R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')['ok']


@pytest.mark.parametrize('scope',[
 ' FROM public.trade_allocation_cases c ',
 ' FROM public.trade_allocation_audit a ',
 ' FROM public.trades t ',
 ' FROM public.alpaca_activities a ',
 ' FROM public.trade_allocation_fill_claims ',
 ' FROM public.trade_allocation_order_claims '])
def test_active_allocation_canonical_to_projection_restored_value_drift_refuses(monkeypatch,scope):
    c=active_crc_connection(monkeypatch);c.canonical_version_scope=scope
    r=R.read_view(lambda:c,IDENTITY,now=NOW,basis='execution_gross')
    assert not r['ok'] and r['coverage'] is None


def test_canonical_replay_charges_budget_and_cannot_skip_empty_query_insertions():
    c=Connection();x=bounded(c);q=next(q for q in T.plan_table() if 'FROM public.trade_disposition_cases c LEFT JOIN' in q)
    x.execute(q);assert x.fetchall()==[]
    c.cases=[canonical_case()]
    with pytest.raises(ValueError,match='CANONICAL_INPUT_DRIFT'):x.recheck_canonical()
    c=Connection();x=bounded(c);x.execute('SHOW transaction_isolation');x.statements=127
    with pytest.raises(ValueError,match='SQL_CAP'):x.recheck_canonical()
