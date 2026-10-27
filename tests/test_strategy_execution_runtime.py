"""Runtime boundary checks use only local frozen fixtures and inert providers."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import json
import socket

import pytest

from scripts import strategy_execution_runtime as runtime
from scripts import strategy_execution as rules
from scripts.strategy_backtest import digest
from tests.test_strategy_execution import positive,NOW

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Runtime tests cannot reach a provider')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def config(**changes):
    return SimpleNamespace(**(dict(paper_research_entries=True,source='db',reconcile_auto=True,
        reconcile_only=False,diagnostic=False,allocation_pct=.05,max_positions=4,max_new_positions=3,
        disable_open_position_cap=False,min_order_usd=0,allow_bump_to_one=False,allow_fractional=False,
        submit_at_ny='04:00',extended_hours=True,limit_buffer_pct=.5,cancel_after_min=35,
        chase_enabled=False,dry_run=False)|changes))


def handoff(inspection=True):
    record,rows=positive()
    return rules.prepare_handoff(record,rows,now=NOW+timedelta(seconds=10),inspection=inspection)


def test_monitor_default_is_inert_even_without_environment_or_config(monkeypatch,tmp_path):
    monkeypatch.delenv('JBR_STRATEGY_EXECUTION',raising=False)
    monkeypatch.setattr(runtime,'load_config',lambda *a:pytest.fail('Default monitor read configuration'))
    assert runtime.monitor_cycle(base_dir=tmp_path)==dict(owned_symbols=[],native_covered=0)


@pytest.mark.parametrize('dry_run',[False,True])
def test_no_signal_and_dry_run_do_not_construct_broker_or_write_order_state(monkeypatch,tmp_path,dry_run):
    value=handoff()
    if not dry_run:value['proposals']=[]
    monkeypatch.setattr(runtime,'environment',lambda:{})
    monkeypatch.setattr(runtime,'source_lease',lambda *a:nullcontext())
    monkeypatch.setattr(runtime,'load_config',lambda *a:{})
    monkeypatch.setattr(runtime,'source_guard',lambda *a,**kw:True)
    monkeypatch.setattr(runtime,'readonly_handoff',lambda *a,**kw:deepcopy(value))
    from scripts import strategy_execution_broker as broker
    monkeypatch.setattr(broker,'PaperBroker',lambda *a,**kw:pytest.fail('No-signal run constructed broker'))
    monkeypatch.setattr(runtime,'store_connection',lambda *a,**kw:pytest.fail('No-signal run opened writable store'))
    result=runtime.run_entry(config(dry_run=dry_run),base_dir=tmp_path)
    assert result['status']=='skipped' and result['broker_operations']==0 and result['orders_submitted']==0
    assert result['candidate_count']==(1 if dry_run else 0)
    saved=json.loads((tmp_path/'reports/strategy_execution/entry/latest.json').read_bytes())
    assert saved['handoff_id']==value['handoff_id']


@pytest.mark.parametrize('field,value', [('allocation_pct',.06),('max_positions',5),('max_new_positions',4),
    ('min_order_usd',300),('allow_bump_to_one',True),('allow_fractional',True),('source','csv'),
    ('submit_at_ny','09:30'),('limit_buffer_pct',1),('cancel_after_min',60),('chase_enabled',True),
    ('disable_open_position_cap',True),('paper_research_entries',False),('extended_hours',False),
    ('reconcile_only',True),('diagnostic',True),('reconcile_auto',False)])
def test_changed_task_risk_or_reconciliation_settings_refuse_before_network(monkeypatch,field,value):
    monkeypatch.setattr(runtime,'environment',lambda:pytest.fail('Invalid settings accessed credentials'))
    with pytest.raises(ValueError):runtime.run_entry(config(**{field:value}))


def test_monitor_receipt_requires_actual_account_freshness_and_identical_sources(tmp_path):
    value=handoff();now=NOW+timedelta(seconds=10)
    receipt=dict(status='ok',entry_ready=True,paper_only=True,account_id='actual-paper-account',
        finished_at=now.isoformat(),source_bindings=value['source_bindings'])
    def save(data):runtime.write_report(data,role='monitor',base_dir=tmp_path)
    save(receipt);runtime.monitor_ready(value,'actual-paper-account',now=now,base_dir=tmp_path)
    for change in (dict(account_id='other-account'),dict(status='blocked'),dict(entry_ready=False),dict(paper_only=False),
        dict(finished_at=(now-timedelta(seconds=121)).isoformat()),dict(source_bindings={'wrong':{}})):
        save(receipt|change)
        with pytest.raises(ValueError,match='monitor_not_fresh'):
            runtime.monitor_ready(value,'actual-paper-account',now=now,base_dir=tmp_path)


def test_source_contract_includes_executor_monitor_store_and_configuration():
    from scripts import pipeline_postflight as health,strategy_pipeline as pipeline
    bound=(*health.SOURCE_FILES,*pipeline.EXTRA_SOURCES)
    assert len(bound)==len(set(bound))==56
    assert set(runtime.COMPONENTS)<=set(bound)


def test_market_monitor_prepares_held_symbols_even_when_current_batch_has_no_buys(monkeypatch,tmp_path):
    from scripts import research_store,strategy_pipeline,daily_strategy_scan
    from contextlib import contextmanager
    class Connection:
        @contextmanager
        def cursor(self):
            yield SimpleNamespace(execute=lambda *a:None,fetchone=lambda:('spec','hash',{}))
        def rollback(self):pass
        def close(self):pass
    monkeypatch.setattr(research_store,'connect',lambda **kw:Connection())
    monkeypatch.setattr(daily_strategy_scan,'load_config',lambda:{})
    monkeypatch.setattr(strategy_pipeline,'validate_archive',lambda *a,**kw:dict(output_root=str(tmp_path)))
    prepared=[]
    dataset=SimpleNamespace(market_wide=True,raw={'HELD':None,'UNRELATED':None},
        prepare=lambda symbol,*a:prepared.append(symbol) or symbol)
    monkeypatch.setattr(strategy_pipeline,'verified_dataset',lambda *a,**kw:dataset)
    path=tmp_path/'reports/pipeline_postflight';path.mkdir(parents=True)
    (path/'latest.json').write_text(json.dumps(dict(batch=dict(archive_id='archive'))))
    path=tmp_path/'scan';path.mkdir()
    (path/'specification.json').write_text(json.dumps(dict(folds=[{}],universe=dict(symbols=['HELD','UNRELATED']))))
    value=dict(batch_sha256='batch',strategy_contract=dict(strategy_id='trend_pullback_reclaim_qualifying_close_v2',
        definition_sha256='definition'),proposals=[],session={})
    runtime._FEATURE_CACHE.clear()
    result=runtime.features({},value,base_dir=tmp_path,symbols={'HELD'})
    assert result=={'HELD':'HELD'} and prepared==['HELD']
    runtime.features({},value,base_dir=tmp_path,symbols={'HELD'})
    assert prepared==['HELD']
    runtime._FEATURE_CACHE.clear()
    dataset.raw={}
    with pytest.raises(ValueError,match='completed_data_unavailable'):
        runtime.features({},value,base_dir=tmp_path,symbols={'HELD'})


def test_inspection_handoff_cannot_be_used_for_late_buy():
    value=handoff()
    from tests.test_strategy_execution import reference
    with pytest.raises(ValueError,match='handoff_changed'):
        rules.build_entry_order(value,signal_id=value['proposals'][0]['signal_id'],account_id='paper-account',
            cash=100000,reference=reference(),occupied_slots=0,now='2026-10-26T13:45:00+00:00',
            clock_next_regular_open='2026-10-27T13:30:00+00:00',clock_timestamp='2026-10-26T13:45:00+00:00')


def test_paper_process_environment_cannot_be_changed_by_private_configuration(monkeypatch):
    from scripts import daily_strategy_scan
    monkeypatch.setattr(daily_strategy_scan,'configured_environment',lambda:{'APCA_API_BASE_URL':'https://paper-api.alpaca.markets','JBR_EXEC_PAPER':'true'})
    monkeypatch.setenv('APCA_API_BASE_URL','https://api.alpaca.markets')
    with pytest.raises(ValueError,match='paper_process_context'):runtime.environment()


def test_monitor_failure_replaces_prior_healthy_receipt_and_closes_adapter(monkeypatch,tmp_path):
    monkeypatch.setenv('JBR_STRATEGY_EXECUTION','true')
    monkeypatch.setattr(runtime,'environment',lambda:{})
    monkeypatch.setattr(runtime,'load_config',lambda *a:{})
    monkeypatch.setattr(runtime,'source_lease',lambda *a:nullcontext())
    from scripts import strategy_execution_broker
    class BrokenBroker:
        account_id=None
        closed=False
        def account(self):raise ValueError('unavailable')
        def close(self):self.closed=True
    broker=BrokenBroker()
    monkeypatch.setattr(strategy_execution_broker,'PaperBroker',lambda *a,**kw:broker)
    runtime.write_report(dict(status='ok'),role='monitor',base_dir=tmp_path)
    with pytest.raises(ValueError):runtime.monitor_cycle(base_dir=tmp_path)
    receipt=json.loads((tmp_path/'reports/strategy_execution/monitor/latest.json').read_bytes())
    assert receipt['status']=='blocked' and broker.closed


def test_unrelated_source_change_blocks_entry_without_stopping_frozen_exit_code(monkeypatch,tmp_path):
    from scripts import pipeline_postflight as health,strategy_pipeline as pipeline,strategy_library
    value=handoff();pins={}
    for name in (*health.SOURCE_FILES,*pipeline.EXTRA_SOURCES):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'frozen-source')
        pins[name]=health._binding(b'frozen-source')
    value['source_bindings']=pins
    monkeypatch.setattr(strategy_library,'get_strategy',lambda *a:SimpleNamespace(describe=lambda:
        dict(definition_sha256=value['strategy_contract']['definition_sha256'])))
    unrelated=next(p for p in pins if p not in runtime.LIFECYCLE_COMPONENTS)
    (tmp_path/unrelated).write_bytes(b'producer-changed')
    assert runtime.source_guard(value,base_dir=tmp_path,lifecycle=True)
    with pytest.raises(ValueError,match='source_changed'):runtime.source_guard(value,base_dir=tmp_path)
    (tmp_path/'scripts/strategy_execution_engine.py').write_bytes(b'exit-code-changed')
    with pytest.raises(ValueError,match='source_changed'):
        runtime.source_guard(value,base_dir=tmp_path,lifecycle=True)
