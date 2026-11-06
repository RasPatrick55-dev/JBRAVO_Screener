"""Current immutable process receipts, separate from financial/entry acceptance."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _base(source):
    return dict(ok=False, status='UNAVAILABLE', source=source, started_utc=None,
        finished_utc=None, duration_seconds=None, rc=None, stage_rcs={},
        evidence_kind='operational_diagnostic_artifacts',
        financial_readiness='not_established_by_process_receipts')


def _times(record, now):
    from scripts.executor_health import utc
    start, end = utc(record['started_at']), utc(record['finished_at'])
    if not start <= end <= now:
        raise ValueError('invalid_process_receipt_time')
    return dict(started_utc=start.isoformat(), finished_utc=end.isoformat(),
        duration_seconds=(end-start).total_seconds(),
        age_seconds=(now-end).total_seconds())


def pipeline_task(base_dir=ROOT, *, now=None):
    from scripts import pipeline_postflight as health, setup_selection as selection
    from scripts.strategy_dashboard import primary_receipt, failure_code
    now = now or datetime.now(timezone.utc)
    result = _base('primary_receipt')
    try:
        failure = health.current_primary_failure(base_dir)
        if failure is not None:
            return dict(result, ok=True, status='FAILED', run_id=failure['run_id'],
                rc=failure['pipeline_rc'], **_times(failure, now))
        primary, _ = primary_receipt(base_dir, now=now)
        control = json.loads(health._read(Path(base_dir) /
            'reports/strategy_companions/unified_setups/latest.json', 65536))
        if not (control.get('status') == 'COMPLETED_UNIFIED_SETUP_SELECTION' and
                control.get('run_id') == selection.selection_id(primary['run_id']) and
                control.get('primary_run_id') == primary['run_id'] and
                control.get('parent_report_sha256') == selection.digest(primary)):
            raise ValueError('strategy_postflight_unified_selection_unavailable')
        times = _times(primary, now)
        return dict(result, ok=True, status='COMPLETED' if times['age_seconds'] <= 86400 else 'STALE',
            run_id=primary['run_id'], rc=primary['pipeline_rc'], **times,
            stage_rcs=dict(screener=0, backtest=0, metrics=0))
    except Exception as error:
        return dict(result, failure_code=failure_code(error))


def execute_task(base_dir=ROOT, *, now=None):
    """Inspect the actual premarket receipt; never prefer an older task log."""
    from scripts import executor_health as health
    now = now or datetime.now(timezone.utc)
    result = _base('executor_receipt')
    try:
        folder = Path(base_dir) / 'reports/executor/premarket'
        raw = health.read(folder / 'latest.json')
        record = health.decode(raw)
        if not (record.get('schema_version') == 1 and record.get('role') == 'premarket' and
                re.fullmatch('[0-9a-f]{32}', record.get('run_id', '')) and
                health.read(folder / ('run-' + record['run_id'] + '.json')) == raw and
                type(record.get('process_rc')) is int and
                record.get('research_qualified') is False and record.get('operating_acceptance') is False):
            raise ValueError('invalid_executor_receipt')
        outcome, pending, unknown = health.assess_outcome(record['process_rc'], record['metrics'],
            record['order_observations'], record['observations_truncated'], record.get('preflight'), 'premarket')
        if (record['health'], record['pending_terminal_confirmations'], record['unobserved_submissions']) != (outcome, pending, unknown):
            raise ValueError('inconsistent_executor_receipt')
        times = _times(record, now)
        state = 'FAILED' if record['process_rc'] != 0 or outcome == 'failed' else (
            'PENDING' if outcome == 'pending_confirmation' else 'COMPLETED')
        if times['age_seconds'] > 86400:
            state = 'STALE'
        return dict(result, ok=True, status=state, run_id=record['run_id'],
            rc=record['process_rc'], process_health=outcome, **times)
    except Exception:
        return dict(result, failure_code='executor_receipt_missing_or_invalid')


def health_snapshot(base_dir=ROOT):
    from scripts.strategy_dashboard import snapshot
    from dashboards.data_io import load_connection_health, connection_check_snapshot
    task = pipeline_task(base_dir)
    candidates = snapshot()
    connection = load_connection_health(Path(base_dir) / 'data')
    return dict(last_run_utc=task['finished_utc'], pipeline_rc=task['rc'],
        source='primary_receipt', latest_source='postgres:unified_strategy_screen',
        process_status=task['status'], pipeline_task=task,
        rows_final=candidates.get('rows_final'), selection_ready=candidates.get('selection_ready', False),
        entry_ready=candidates.get('entry_ready', False),
        entry_block_reason=candidates.get('entry_block_reason') or candidates.get('failure_code'),
        source_detail=candidates.get('source_detail'),
        freshness=dict(age_seconds=task.get('age_seconds'), freshness_level=(
            'green' if task['status'] == 'COMPLETED' else 'gray')),
        trading_ok=connection.get('trading_ok'), trading_status=connection.get('trading_status'),
        data_ok=connection.get('data_ok'), data_status=connection.get('data_status'),
        feed=connection.get('feed'), connection_check=connection_check_snapshot(connection),
        evidence_kind='operational_diagnostic_artifacts',
        financial_readiness='not_established_by_health_checks')
