"""Review-only producer/entry contract proof; no publication, broker or database."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import sys

import pytest
from pricing_adapter_prototype import pricing_contract

pytestmark=pytest.mark.alpaca_optional
BASE=Path(__file__).resolve().parents[1]
NATIVE=BASE/'scripts'
OBSERVED='2026-11-10T01:30:00+00:00'
PUBLISHED='2026-11-10T03:30:00+00:00'
SESSION=dict(signal_session='2026-11-09',next_session='2026-11-10')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m


@pytest.fixture
def contract(monkeypatch):
    def deny(*a,**kw):raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket,'connect',deny)
    rules=load('prototype_native_rules',NATIVE/'strategy_execution.py')
    selection=load('prototype_native_selection',NATIVE/'setup_selection.py')
    return pricing_contract(rules,selection),rules,selection


def bar(t,p=6.7):return dict(t=t,o=p,h=p,l=p,c=p,v=100)


def ref(contract,symbol,*,ah=False,regular=True):
    c,rules,_=contract
    a=rules.after_hours_reference([bar('2026-11-09T22:00:00Z')] if ah else [],symbol=symbol,
        signal_session=SESSION['signal_session'],observed_at=OBSERVED,feed='sip',adjustment='raw',currency='USD',complete=True)
    if ah:return c.pipeline_price_reference(a)
    r=c.regular_close_reference([bar('2026-11-09T20:59:00Z')] if regular else [],symbol=symbol,
        signal_session=SESSION['signal_session'],observed_at=OBSERVED,feed='sip',adjustment='raw',currency='USD',complete=True)
    activity=c.after_hours_activity([],symbol=symbol,signal_session=SESSION['signal_session'],observed_at=OBSERVED,complete=True)
    return c.pipeline_price_reference(a,r,activity)


def parent(contract,symbols):
    _,rules,selection=contract
    selected=[dict(rank=i+1,proposal=dict(symbol=s,signal_id=s+'_signal',strategy_id='test',**SESSION)) for i,s in enumerate(symbols)]
    return dict(primary_run_id='primary',run_id=selection.selection_id('primary'),session=SESSION.copy(),
        source_bindings={'unchanged-native':'digest'},selection=dict(selected_setups=selected,selected_signal_ids=[x['proposal']['signal_id'] for x in selected]))


def handoffs():return {'test':dict(primary_run_id='primary',execution_policy=dict(entry_open_at='2026-11-10T09:00:00+00:00'))}


def test_fallback_prices_validate_with_unchanged_native_consumer(contract):
    c,rules,selection=contract;p=parent(contract,['GOOD','FALLBACK','NONE']);before=deepcopy(p)
    refs={'GOOD':ref(contract,'GOOD',ah=True),'FALLBACK':ref(contract,'FALLBACK'),'NONE':ref(contract,'NONE',regular=False)}
    original=deepcopy(refs);native_before=(rules.validate_pipeline_limit_order,selection.execution_queue,selection.PIPELINE_LIMIT_POLICY)
    review=c.prepare_review(p,refs,published_at=PUBLISHED)
    result=c.entry_preview(p,review,handoffs())
    assert [r['signal_id'] for r in result]==['GOOD_signal','FALLBACK_signal']
    assert {r['ticket']['limit_price'] for r in result}=={'6.73'}
    assert result[1]['ticket']['price_provenance']['reference_kind']=='REGULAR_SESSION_FALLBACK'
    assert all(r['order_submission_authorized'] is False for r in result)
    assert p==before and refs==original
    assert native_before==(rules.validate_pipeline_limit_order,selection.execution_queue,selection.PIPELINE_LIMIT_POLICY)


@pytest.mark.parametrize('field',['parent_selection_sha256','native_source_bindings_sha256','parent_selection_run_id'])
def test_mismatched_parent_or_source_binding_is_rejected(contract,field):
    c,_,_=contract;p=parent(contract,['X']);r=c.prepare_review(p,{'X':ref(contract,'X')},published_at=PUBLISHED)
    r[field]='changed'
    with pytest.raises(ValueError,match='review_changed'):c.entry_preview(p,r,handoffs())


def test_tampered_price_disposition_and_authority_rejected(contract):
    c,_,_=contract;p=parent(contract,['X']);r=c.prepare_review(p,{'X':ref(contract,'X')},published_at=PUBLISHED)
    for key,value in [('review_only',False),('order_submission_authorized',True)]:
        candidate=deepcopy(r);candidate[key]=value
        with pytest.raises(ValueError):c.entry_preview(p,candidate,handoffs())
    r['pricing']['execution_orders']['X_signal']['limit_price']='99.99'
    with pytest.raises(ValueError):c.entry_preview(p,r,handoffs())


def test_all_missing_excludes_without_native_order_request(contract):
    c,_,_=contract;p=parent(contract,['X']);r=c.prepare_review(p,{'X':ref(contract,'X',regular=False)},published_at=PUBLISHED)
    assert r['pricing']['pricing_status']=='NO_EXECUTABLE_PRICE_REFERENCES'
    assert c.entry_preview(p,r,handoffs())==[]


def test_prior_day_or_nonminute_fallback_rejected(contract):
    c,_,_=contract
    for stamp in ('2026-11-08T20:59:00Z','2026-11-09T20:59:01Z'):
        with pytest.raises(ValueError):c.regular_close_reference([bar(stamp)],symbol='X',signal_session=SESSION['signal_session'],observed_at=OBSERVED,feed='sip',adjustment='raw',currency='USD',complete=True)


def test_incomplete_reference_cannot_be_empty_fallback(contract):
    c,_,_=contract
    with pytest.raises(ValueError):c.regular_close_reference([],symbol='X',signal_session=SESSION['signal_session'],observed_at=OBSERVED,feed='sip',adjustment='raw',currency='USD',complete=False)


def test_canonical_runtime_does_not_admit_unreviewed_lifecycle_compatibility(contract):
    runtime=load('prototype_native_runtime',NATIVE/'strategy_execution_runtime.py')
    # Main has no host-only compatibility-history mechanism. Do not import it
    # merely to make a relocated historical-artifact test pass.
    assert not hasattr(runtime,'append_executor_release_compatibility')
    with pytest.raises(ValueError,match='sources_not_bound'):
        runtime.source_guard({'source_bindings':{'unreviewed.py':{}}},lifecycle=True)


def test_old_primary_publisher_remains_legacy_and_rejects_empty_afterhours(contract):
    _,rules,selection=contract;p=parent(contract,['X'])
    raw=rules.after_hours_reference([],symbol='X',signal_session=SESSION['signal_session'],observed_at=OBSERVED,feed='sip',adjustment='raw',currency='USD',complete=True)
    with pytest.raises(ValueError,match='reference_unavailable'):
        selection.build_pipeline_limit_orders(p['selection'],p,{'X':raw},published_at=PUBLISHED)
    assert selection.PIPELINE_LIMIT_POLICY=='RAW_SIP_AFTER_HOURS_CLOSE_PLUS_0_5_PCT_V1'
