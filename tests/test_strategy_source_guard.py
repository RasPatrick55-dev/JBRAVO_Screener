"""Source-guard regressions are local only; they cannot contact a broker."""
from copy import deepcopy
import socket

import pytest

from scripts import pipeline_postflight as health
from scripts import strategy_execution_runtime as runtime
from scripts.strategy_library import get_strategy
from scripts.strategy_pipeline import EXTRA_SOURCES

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args,**kwargs):raise AssertionError('Source-guard tests must stay offline')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


@pytest.fixture
def bound(tmp_path):
    pins={}
    for name in (*health.SOURCE_FILES,*EXTRA_SOURCES):
        raw=('inert source fixture '+name).encode()
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        pins[name]=health._binding(raw)
    definition=get_strategy('trend_pullback_reclaim_qualifying_close_v2').describe()
    return dict(source_bindings=pins,strategy_contract=dict(strategy_id=definition['id'],definition_sha256=definition['definition_sha256']))


@pytest.mark.parametrize('lifecycle',[False,True])
def test_current_complete_inventory_remains_valid(bound,tmp_path,lifecycle):
    assert runtime.source_guard(bound,base_dir=tmp_path,lifecycle=lifecycle)


@pytest.mark.parametrize('change',['added_dashboard','removed_dashboard','changed_dashboard'])
def test_historical_lifecycle_ignores_only_non_lifecycle_inventory_drift(bound,tmp_path,change):
    pins=bound['source_bindings']
    if change=='added_dashboard':pins.pop('dashboards/operational_status.py')
    elif change=='removed_dashboard':pins['dashboards/retired.py']={'bytes':10,'sha256':'0'*64}
    else:pins['dashboards/operational_status.py']={'bytes':10,'sha256':'0'*64}
    before=deepcopy(bound)
    assert runtime.source_guard(bound,base_dir=tmp_path,lifecycle=True)
    assert bound==before
    with pytest.raises(ValueError,match='strategy_execution_source'):
        runtime.source_guard(bound,base_dir=tmp_path)


@pytest.mark.parametrize('name',runtime.LIFECYCLE_COMPONENTS)
def test_every_lifecycle_binding_remains_required(bound,tmp_path,name):
    bound['source_bindings'].pop(name)
    with pytest.raises(ValueError,match='strategy_execution_sources_not_bound'):
        runtime.source_guard(bound,base_dir=tmp_path,lifecycle=True)


@pytest.mark.parametrize('name',runtime.LIFECYCLE_COMPONENTS)
def test_every_lifecycle_source_must_still_match(bound,tmp_path,name):
    (tmp_path/name).write_bytes(b'changed')
    with pytest.raises(ValueError,match='strategy_execution_source_changed'):
        runtime.source_guard(bound,base_dir=tmp_path,lifecycle=True)


@pytest.mark.parametrize('lifecycle',[False,True])
def test_strategy_definition_must_still_match(bound,tmp_path,lifecycle):
    bound['strategy_contract']['definition_sha256']='0'*64
    with pytest.raises(ValueError,match='strategy_execution_rule_version_changed'):
        runtime.source_guard(bound,base_dir=tmp_path,lifecycle=lifecycle)


@pytest.mark.parametrize('lifecycle',[False,True])
@pytest.mark.parametrize('bad_pins',[None,[],['scripts/strategy_execution.py']])
def test_malformed_binding_inventory_fails_closed(bound,tmp_path,lifecycle,bad_pins):
    bound['source_bindings']=bad_pins
    with pytest.raises(ValueError,match='strategy_execution_sources_not_bound'):
        runtime.source_guard(bound,base_dir=tmp_path,lifecycle=lifecycle)


def test_unneeded_historical_paths_are_never_opened(bound,tmp_path,monkeypatch):
    # A historical extra key is lineage, not permission to read an arbitrary path.
    bound['source_bindings']['../../outside-root']={'bytes':0,'sha256':'0'*64}
    read=health._read;observed=[]
    def capture(path,limit):
        observed.append(path.relative_to(tmp_path).as_posix())
        return read(path,limit)
    monkeypatch.setattr(health,'_read',capture)
    assert runtime.source_guard(bound,base_dir=tmp_path,lifecycle=True)
    assert observed==list(runtime.LIFECYCLE_COMPONENTS)
