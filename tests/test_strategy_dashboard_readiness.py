from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from scripts import strategy_dashboard as dashboard, strategy_pipeline as pipeline
from scripts import pipeline_postflight as health, setup_selection as selection
from dashboards import operational_status as status
from test_strategy_pipeline import report

pytestmark = pytest.mark.alpaca_optional


def completed(tmp_path, monkeypatch):
    primary = report()
    primary['candidate_run_ts_utc'] = primary['started_at']
    primary['source_bindings'] = {}
    folder = tmp_path / 'reports/pipeline_postflight'
    folder.mkdir(parents=True)
    raw = (json.dumps(primary, sort_keys=True) + '\n').encode()
    (folder / 'latest.json').write_bytes(raw)
    (folder / ('run-' + primary['run_id'] + '.json')).write_bytes(raw)
    (folder / 'active-primary.json').write_text(json.dumps(dict(id=primary['run_id'], started_at=primary['started_at'])))
    (folder / 'completed-primary.json').write_text(json.dumps(dict(id=primary['run_id'], report_binding=health._binding(raw))))
    selected = dict(record=dict(run_id=selection.selection_id(primary['run_id']),
        selection=dict(selected_setups=[], selected_signal_ids=[])), result_sha256='b'*64, handoffs={})
    control = tmp_path / 'reports/strategy_companions/unified_setups/latest.json'
    control.parent.mkdir(parents=True)
    control.write_text(json.dumps(dict(status='COMPLETED_UNIFIED_SETUP_SELECTION',
        run_id=selected['record']['run_id'], primary_run_id=primary['run_id'],
        parent_report_sha256=selection.digest(primary), result_sha256=selected['result_sha256'])))
    monkeypatch.setattr(pipeline, 'enabled', lambda *a: True)
    return primary, selected


def test_display_keeps_verified_postgres_batch_after_entry_expires(tmp_path, monkeypatch):
    primary, selected = completed(tmp_path, monkeypatch)
    now = datetime.fromisoformat(primary['finished_at']) + timedelta(hours=18)
    with pytest.raises(ValueError, match='stale_or_future'):
        pipeline.assess(primary, now=now, max_age_seconds=43200, inspection=True)
    connection = object()
    seen = []
    def read(conn, parent, **kw):
        assert conn is connection and parent['primary_run_id'] == primary['run_id']
        seen.append(True)
        return selected
    monkeypatch.setattr(selection, 'read_selection', read)
    batch = (datetime.fromisoformat(primary['session']['signal_session']).date(),
             datetime.fromisoformat(primary['candidate_run_ts_utc']), 0, primary['batch'])
    verified = dashboard.display_selection(connection, batch, primary['run_id'], base_dir=tmp_path, now=now)
    result = dashboard.handoff_readiness(verified, [], base_dir=tmp_path, now=now)
    assert seen and result['selection_ready'] is True
    assert result['entry_ready'] is False
    assert result['entry_block_reason'] == 'execution_premarket_preparation_expired'
    assert result['order_submission_authorized'] is False


@pytest.mark.parametrize('change', ['history', 'completion', 'invocation', 'future'])
def test_historical_display_still_rejects_broken_run_proof(tmp_path, monkeypatch, change):
    primary, _ = completed(tmp_path, monkeypatch)
    folder = tmp_path / 'reports/pipeline_postflight'
    now = datetime.fromisoformat(primary['finished_at']) + timedelta(hours=18)
    if change == 'history': (folder / ('run-' + primary['run_id'] + '.json')).write_text('{}')
    if change == 'completion': (folder / 'completed-primary.json').write_text('{}')
    if change == 'invocation': (folder / 'active-primary.json').write_text(json.dumps(dict(id='c'*32, started_at=primary['started_at'])))
    if change == 'future': now = datetime.fromisoformat(primary['started_at'])
    with pytest.raises((ValueError, KeyError)):
        dashboard.primary_receipt(tmp_path, now=now)


def test_db_selection_cannot_be_replaced_by_a_success_control_file(tmp_path, monkeypatch):
    primary, _ = completed(tmp_path, monkeypatch)
    def missing(*a, **kw): raise ValueError('unified_selection_not_published')
    monkeypatch.setattr(selection, 'read_selection', missing)
    batch = (datetime.fromisoformat(primary['session']['signal_session']).date(),
             datetime.fromisoformat(primary['candidate_run_ts_utc']), 0, primary['batch'])
    with pytest.raises(ValueError, match='not_published'):
        dashboard.display_selection(object(), batch, primary['run_id'], base_dir=tmp_path,
            now=datetime.fromisoformat(primary['finished_at'])+timedelta(hours=1))


def test_ready_handoff_has_separate_broker_gate_and_rejects_current_source_change(tmp_path, monkeypatch):
    from scripts import strategy_execution as execution
    primary, selected = completed(tmp_path, monkeypatch)
    selected.update(primary=primary)
    selected['record']['selection']['selected_setups'] = [dict(proposal={})]
    now = datetime.fromisoformat(primary['finished_at'])+timedelta(seconds=1)
    monkeypatch.setattr(pipeline, 'check_report', lambda *a, **kw: {})
    monkeypatch.setattr(execution, 'prepare_handoff', lambda *a, **kw: {})
    ready = dashboard.handoff_readiness(selected, [], base_dir=tmp_path, now=now)
    assert ready['entry_ready'] is True and ready['broker_preflight_required'] is True
    assert ready['order_submission_authorized'] is False
    def changed(*a, **kw): raise ValueError('strategy_postflight_source_changed')
    monkeypatch.setattr(pipeline, 'check_report', changed)
    assert dashboard.handoff_readiness(selected, [], base_dir=tmp_path, now=now)['entry_block_reason'] == 'strategy_postflight_source_changed'


def test_process_card_uses_current_completed_primary_and_rejects_new_unfinished_invocation(tmp_path, monkeypatch):
    primary, _ = completed(tmp_path, monkeypatch)
    now = datetime.fromisoformat(primary['finished_at'])+timedelta(hours=18)
    current = status.pipeline_task(tmp_path, now=now)
    assert current['status'] == 'COMPLETED' and current['stage_rcs'] == dict(screener=0, backtest=0, metrics=0)
    assert current['finished_utc'] == primary['finished_at']
    folder = tmp_path / 'reports/pipeline_postflight'
    (folder / 'active-primary.json').write_text(json.dumps(dict(id='c'*32, started_at=primary['finished_at'])))
    assert status.pipeline_task(tmp_path, now=now)['status'] == 'UNAVAILABLE'


def test_arbitrary_exception_values_do_not_escape_to_api():
    assert dashboard.failure_code(ValueError('password=secret')) == 'strategy_dashboard_read_failed'
    assert dashboard.failure_code(RuntimeError('unified_selection_not_published')) == 'strategy_dashboard_read_failed'
    assert dashboard.failure_code(ValueError('unified_selection_not_published')) == 'unified_selection_not_published'


@pytest.mark.parametrize('change, expected', [('none', 'COMPLETED'), ('failed', 'FAILED'),
    ('history', 'UNAVAILABLE'), ('future', 'UNAVAILABLE'), ('stale', 'STALE'), ('false_green', 'UNAVAILABLE')])
def test_execution_card_uses_bound_premarket_outcome(tmp_path, change, expected):
    from scripts import executor_health as executor
    now = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)
    counts = {k: 0 for k in executor.COUNTS}
    record = dict(schema_version=1, role='premarket', run_id='d'*32,
        started_at='2026-11-06T09:00:00+00:00', finished_at='2026-11-06T09:35:00+00:00',
        process_rc=0, health='no_orders', metrics=dict(counts, api_failures=0, status='ok'),
        order_observations=[], observations_truncated=False, pending_terminal_confirmations=0,
        unobserved_submissions=0, research_qualified=False, operating_acceptance=False)
    if change == 'failed': record.update(process_rc=1, health='failed')
    if change == 'false_green': record['health'] = 'ok'
    if change == 'future': now -= timedelta(days=1)
    if change == 'stale': now += timedelta(days=2)
    folder = tmp_path / 'reports/executor/premarket'
    folder.mkdir(parents=True)
    raw = json.dumps(record).encode()
    (folder / 'latest.json').write_bytes(raw)
    (folder / ('run-' + record['run_id'] + '.json')).write_bytes(b'{}' if change == 'history' else raw)
    result = status.execute_task(tmp_path, now=now)
    assert result['status'] == expected
    if expected != 'UNAVAILABLE':
        assert result['finished_utc'] == record['finished_at']


@pytest.mark.parametrize('name, reader', [('api_pipeline_task', 'pipeline_task'),
    ('api_execute_task', 'execute_task'), ('api_health', 'health_snapshot')])
def test_actual_routes_use_current_readers_instead_of_legacy_logs(monkeypatch, tmp_path, name, reader):
    from test_strategy_dashboard import extract
    monkeypatch.setattr(pipeline, 'enabled', lambda *a: True)
    expected = dict(source='current_reader', status='UNAVAILABLE')
    monkeypatch.setattr(status, reader, lambda base, **kwargs: expected if base == str(tmp_path) else None)
    route = extract('dashboards/dashboard_app.py', name)
    route.__globals__['BASE_DIR'] = str(tmp_path)
    route.__globals__['_fetch_pythonanywhere_execute_task_log'] = lambda: None
    route.__globals__['_parse_task_log'] = lambda *a, **kw: pytest.fail('legacy log consulted')
    route.__globals__['load_screener_health'] = lambda: pytest.fail('legacy health consulted')
    assert route() == expected
