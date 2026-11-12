"""Retained web snapshot/API qualification with synthetic external ports only.

Receipt, market, account and strategy-research verification ports are synthetic;
actual retained snapshot, display reader, readiness and API rendering run here.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from tests.test_shared_pricing_wiring import wiring, publish
from tests.test_isolated_pricing_integration import fixture
from tests.test_pricing_adapter_prototype import contract, ref, load, PUBLISHED
from tests.test_actual_web_binding import WEB

pytestmark = pytest.mark.alpaca_optional
NOW = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)


@pytest.fixture
def web_case(wiring, monkeypatch):
    f, w, database, statuses, checks, runtime = wiring
    from scripts import db, setup_selection as native, strategy_pipeline as pipeline
    from scripts import pipeline_postflight as health, strategy_pipeline_store as store
    from scripts import research_strategies, strategy_execution as execution, strategy_metrics
    f.selected['model_sha256'] = 'model-pin'
    metrics = {'result_id': 'metrics', 'rows': []}
    backtest = {'result_id': 'backtest', 'results': []}
    for setup in f.selected['selected_setups']:
        p = setup['proposal']
        setup['matching_strategies'] = ['test']
        p.update(signal_close_raw=6.5, setup_ranking={'score': 80, 'signal_volume_raw': 100},
                 strategy_version='1', next_open_at='2026-11-10T09:00:00+00:00')
        metrics['rows'].append(dict(signal_id=p['signal_id'], symbol=p['symbol'], trades=1))
        backtest['results'].append(dict(current_signal_id=p['signal_id'], symbol=p['symbol'],
            closed_signals=1, winning_signals=1, losing_signals=0, win_rate_pct=100,
            closed_trade_net_pnl=1.0, expectancy_net_per_closed_signal=1.0,
            max_closed_signal_drawdown_cash=0.0))
    f.primary.update(candidate_run_ts_utc=PUBLISHED, finished_at=PUBLISHED,
                     backtest=backtest, metrics=metrics)
    f.primary['batch'].update(backtest=backtest, metrics=metrics)
    stamp = datetime.fromisoformat(PUBLISHED)
    batch = (datetime(2026, 11, 9).date(), stamp, 2, f.primary['batch'])
    frame = pd.DataFrame([{'symbol': 'GOOD'}, {'symbol': 'NONE'}])
    state = {'receipt': None, 'raw': None, 'batch': batch, 'connections': [], 'sessions': []}

    class Connection:
        def __init__(self):
            self.inner = database.connect(write=False)
            self.primary_result = False
            state['connections'].append(self)
        def set_session(self, **kwargs):
            state['sessions'].append(kwargs)
        def cursor(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, args=None):
            self.primary_result = sql.startswith('SELECT run_id FROM public.screener_strategy_batches_app')
            if not self.primary_result:
                self.inner.execute(sql, args)
        def fetchone(self):
            return ('primary',) if self.primary_result else self.inner.fetchone()
        def rollback(self):
            return self.inner.rollback()
        def close(self):
            return self.inner.close()

    web = load('shared_full_web_api', WEB)
    monkeypatch.setattr(db, 'get_db_conn', Connection)
    monkeypatch.setattr(pipeline, 'latest_batch', lambda cursor: state['batch'])
    monkeypatch.setattr(store, 'candidate_frame', lambda *a, **k: (frame, stamp))
    monkeypatch.setattr(pipeline, 'validate_backtest', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'selected_batch', lambda value: value)
    monkeypatch.setattr(strategy_metrics, 'validate_metrics', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'check_report', lambda *a, **k: None)
    monkeypatch.setattr(execution, 'prepare_handoff', lambda *a, **k: None)
    monkeypatch.setattr(web, 'primary_receipt', lambda *a, **k: (f.primary, state['raw']))
    monkeypatch.setattr(native, 'read_selection', w.read_selection)
    monkeypatch.setattr(native, 'parent_context', f.selection.parent_context)
    monkeypatch.setattr(native, 'dashboard_rows', w.dashboard_rows)
    monkeypatch.setattr(native, 'dashboard_reports', w.dashboard_reports)
    monkeypatch.setattr(research_strategies, 'CHALLENGER', 'test')
    monkeypatch.setattr(health, '_read', lambda path, *a: json.dumps(state['receipt'])
                        if 'unified_setups' in str(path) else state['raw'])

    def commit():
        state['receipt'] = publish(wiring)
        state['raw'] = json.dumps(f.primary)
        assert state['receipt']['status'] == 'COMPLETED_UNIFIED_SETUP_SELECTION'
    return SimpleNamespace(f=f, w=w, database=database, web=web, state=state, commit=commit)


@pytest.mark.parametrize('kind', ['candidates', 'metrics', 'backtest'])
def test_full_retained_api_preserves_ranked_population(web_case, kind):
    c = web_case
    c.commit()
    result = c.web.api_payload(kind, {}, now=NOW)
    assert result['ok'] is True, result
    assert result['status'] == 'COMPLETE'
    assert [row['symbol'] for row in result['rows']] == ['GOOD', 'NONE']
    assert result['rows_final'] == 2
    assert result['entry_ready'] is True
    assert result['computation_complete'] is True
    assert result['order_submission_authorized'] is False
    assert result['setup_selection']['selected_signal_ids'] == ['GOOD_signal']
    if kind == 'candidates':
        good, excluded = result['rows']
        assert good['execution_eligible'] is True and good['execution_limit_price'] == '6.73'
        assert excluded['execution_eligible'] is False and excluded['execution_limit_price'] is None
        assert excluded['pricing_disposition']['status'] == 'EXCLUDED_NO_ELIGIBLE_REFERENCE'
        assert all(row['price'] == 6.5 for row in result['rows'])
    else:
        assert result['metric_basis'] == 'STRATEGY_RESEARCH_SIMULATION'
    assert c.state['sessions'] == [dict(readonly=True, isolation_level='REPEATABLE READ', autocommit=False)]
    assert all(conn.inner.closed and conn.inner.rollbacks == 1 for conn in c.state['connections'])


@pytest.mark.parametrize('kind', ['candidates', 'metrics', 'backtest'])
def test_all_unpriceable_is_visible_but_never_entry_ready(web_case, kind):
    c = web_case
    c.f.prices['GOOD'] = ref((c.f.adapter.contract, c.f.rules, c.f.selection), 'GOOD', regular=False)
    c.commit()
    result = c.web.api_payload(kind, {}, now=NOW)
    assert result['ok'] is True, result
    assert result['rows_final'] == 2
    assert result['entry_ready'] is False
    assert 'unified_selection_no_buy_signals' in result['entry_block_reasons']
    assert result['setup_selection']['selected_signal_ids'] == []
    assert result['order_submission_authorized'] is False
    if kind == 'candidates':
        assert all(row['execution_limit_price'] is None and row['execution_eligible'] is False
                   for row in result['rows'])


@pytest.mark.parametrize('kind', ['candidates', 'metrics', 'backtest'])
def test_requested_batch_change_clears_completion_and_rows(web_case, kind):
    c = web_case
    c.commit()
    result = c.web.api_payload(kind, {'run_ts_utc': '2026-11-09T03:30:00+00:00'}, now=NOW)
    assert result['ok'] is False and result['status'] == 'ERROR'
    assert result['rows'] == [] and result['rows_final'] == 0
    assert result['completion_status'] == 'unverified'
    assert result['computation_complete'] is False and result['completion_evidence'] is None


def test_changed_control_receipt_fails_closed_in_actual_snapshot(web_case):
    c = web_case
    c.commit()
    c.state['receipt']['result_sha256'] = '0' * 64
    result = c.web.snapshot(now=NOW)
    assert result['ok'] is False and result['entry_ready'] is False
    assert result['rows'] == [] and result['computation_complete'] is False
    assert all(conn.inner.closed for conn in c.state['connections'])


def test_requested_batch_change_must_clear_entry_readiness(web_case):
    c = web_case
    c.commit()
    result = c.web.api_payload('candidates', {'run_ts_utc': '2026-11-09T03:30:00+00:00'}, now=NOW)
    assert result['ok'] is False
    assert result['entry_ready'] is False

