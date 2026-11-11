from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import inspect
import pytest
from tests.test_isolated_pricing_integration import fixture,contract
from tests.test_pricing_adapter_prototype import BASE,load
from isolated_pricing_integration import dashboard_adapter
pytestmark=pytest.mark.alpaca_optional
WEB=BASE/'scripts/strategy_dashboard.py'


def test_actual_web_signatures_and_historical_reader_preserved(fixture):
    f=fixture;pre=load('actual_web_preimage',WEB)
    bound=dashboard_adapter(pre,f.adapter,row_read=None,parent_context=None,allow_legacy_display=True)
    assert inspect.signature(bound.snapshot)==inspect.signature(pre.snapshot)
    assert inspect.signature(bound.api_payload)==inspect.signature(pre.api_payload)
    assert bound.snapshot.__code__ is pre.snapshot.__code__
    assert bound.snapshot.__globals__['primary_receipt'] is pre.primary_receipt
    assert bound.api_payload.__globals__['snapshot'] is bound.snapshot
    assert pre.api_payload.__globals__['snapshot'] is pre.snapshot


def test_validated_legacy_display_keeps_original_readiness(fixture,monkeypatch):
    f=fixture;pre=load('legacy_web_preimage',WEB)
    legacy={'record':{'execution_orders':{}},'primary':{}}
    monkeypatch.setattr(pre,'display_selection',lambda *a,**k:legacy)
    monkeypatch.setattr(pre,'handoff_readiness',lambda *a,**k:{'entry_ready':False,'entry_block_reason':'expired'})
    bound=dashboard_adapter(pre,f.adapter,row_read=lambda *a:pytest.fail('legacy supplemental query'),parent_context=None,allow_legacy_display=True)
    result=bound.display_selection(None,None,None)
    assert result is legacy
    assert bound.handoff_readiness(result,[])=={'entry_ready':False,'entry_block_reason':'expired'}


def test_ranking_only_web_requires_supplement(fixture,monkeypatch):
    f=fixture;pre=load('ranking_web_preimage',WEB);verified=f.read();verified['primary']=f.primary
    monkeypatch.setattr(pre,'display_selection',lambda *a,**k:verified)
    bound=dashboard_adapter(pre,f.adapter,row_read=lambda *a:None,parent_context=lambda p:{},allow_legacy_display=True)
    with pytest.raises(ValueError,match='not_published'):bound.display_selection(None,None,None)


def test_web_binding_rejects_changed_preimage(fixture,tmp_path):
    from isolated_web_reader_binding import bind_deployed_web
    f=fixture;p=tmp_path/'web.py';p.write_bytes(b'raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='source_changed'):
        bind_deployed_web(preimage_path=p,preimage_pin={'bytes':1,'sha256':'0'*64},adapter_root=tmp_path,adapter_pins={},rules=f.rules,selection=f.selection,clock=lambda:None)
