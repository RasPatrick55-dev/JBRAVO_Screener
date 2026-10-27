"""Additive bull-flag producer/publication; the existing strategy stays primary.

No provider call, connection or order on import. Immutable PostgreSQL records
are authoritative; control files identify exact rows, never substitute for them.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

from scripts.bull_flag_strategy import BULL_FLAG
from scripts.research_dataset import canonical_hash, strict_json
from scripts.strategy_backtest import require, digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'config/strategy_companions.json'


def configuration(base_dir=ROOT):
    from scripts.pipeline_postflight import _read
    path = Path(base_dir)/CONFIG
    if not path.exists():
        return None
    value = strict_json(_read(path, 4096))
    expected = dict(schema='jbravo.strategy-companions.v1', enabled=True,
        strategy_id=BULL_FLAG, priority='PULLBACK_RECLAIM_FIRST', max_scan_seconds=1800,
        max_new_entries_total=3, max_positions_total=4, allocation_pct_remaining_cash=.05,
        duplicate_symbol_policy='ONE_OWNER_HELD_OR_PENDING', paper_only=True, research_qualified=False)
    if value == dict(expected, enabled=False):
        return None
    require(value == expected, 'companion_configuration_changed')
    return value


def archive_id(session):
    from scripts.daily_strategy_scan import PIPELINE
    return canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1',
        pipeline=PIPELINE, strategy_id=BULL_FLAG, signal_session=session))


def primary_id(run_id):
    return canonical_hash(dict(schema='jbravo.companion-primary-id.v1', primary_run_id=run_id,
                               strategy_id=BULL_FLAG))


def _status(value, base_dir=ROOT):
    from scripts.executor_health import ordinary
    folder = Path(base_dir)/'reports/strategy_companions'/BULL_FLAG
    ordinary(folder)
    folder.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    require(len(body) <= 65536, 'companion_control_oversize')
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False, prefix='.companion-') as stream:
            name = stream.name
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, folder/'latest.json'); name = None
    finally:
        if name is not None:
            os.unlink(name)


def produce(parent_root, parent_spec, cfg, *, now, archive, total_deadline, base_dir=ROOT):
    """Reuse the complete parent capture, with no new provider request.

    Failure after the primary archive is complete never reverses that archive.
    A failed companion claim is preserved and cannot be replayed in place.
    """
    policy = configuration(base_dir)
    if policy is None:
        return dict(status='DISABLED')
    from scripts.research_testing import save_json
    from scripts.strategy_scanner import run_scan
    root = Path(parent_root)/'companions'/BULL_FLAG
    try:
        root.mkdir(parents=True, exist_ok=False)
        spec = deepcopy(parent_spec)
        spec['strategies'] = [BULL_FLAG]
        spec['max_seconds'] = min(spec['max_seconds'], policy['max_scan_seconds'])
        save_json(root/'study.json', spec)
        result = run_scan(spec, as_of=spec['folds'][0]['test_end'], output=root/'scan',
            strategy_ids=[BULL_FLAG], fold_id=cfg['episode_id'], cost_case_id=cfg['cost_case']['id'],
            now=now, total_deadline=min(total_deadline, time.monotonic()+policy['max_scan_seconds']))
        proof = archive(result, root, dict(cfg, strategy_id=BULL_FLAG))
        value = dict(status='COMPLETED_COMPANION_SCREEN', strategy_id=BULL_FLAG,
            archive=proof, candidates=len(result['top_candidates']), provider_gets=0,
            signal_session=result['as_of'], parent_root=str(parent_root))
        save_json(root/'completion.json', value)
        return value
    except Exception as error:
        failure = dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED', strategy_id=BULL_FLAG,
                       error_type=type(error).__name__, provider_gets=0)
        if root.is_dir() and not (root/'failure.json').exists():
            save_json(root/'failure.json', failure)
        return failure


def _row(connection, run_id):
    with connection.cursor() as cursor:
        cursor.execute('SELECT specification_sha256,result_sha256,result '
            'FROM jbravo_research.research_runs WHERE run_id=%s', (run_id,))
        return cursor.fetchone()


def _publish(record, spec_sha, *, env):
    from scripts.research_store import connect
    body = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    expected = digest(record)
    connection = connect(write=True, env=env)
    try:
        with connection.cursor() as cursor:
            cursor.execute('INSERT INTO jbravo_research.research_runs '
                '(run_id,specification_sha256,result_sha256,result) VALUES (%s,%s,%s,%s::jsonb)',
                (record['run_id'], spec_sha, expected, body))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    connection = connect(env=env)
    try:
        observed = _row(connection, record['run_id'])
        require(observed is not None and observed[0].strip() == spec_sha and
                observed[1].strip() == expected and observed[2] == record,
                'companion_committed_readback_failed')
    finally:
        connection.rollback(); connection.close()


def prepare_primary(primary, *, env, base_dir=ROOT):
    """Separate exact-session Backtest/Metrics and immutable DB publication.

    No canonical candidate rows/views, primary report, existing strategy policy
    or historical trade is modified. Missing/failed companions remain unavailable.
    """
    if configuration(base_dir) is None:
        return dict(status='DISABLED')
    from scripts import strategy_pipeline as pipeline
    from scripts.daily_strategy_scan import load_config
    from scripts.research_store import connect
    from scripts.strategy_backtest import run_candidate_backtest
    from scripts.strategy_metrics import build_metrics
    try:
        cfg = dict(load_config(Path(base_dir)/'config/strategy_scan_daily.json'), strategy_id=BULL_FLAG)
        connection = connect(env=env)
        try:
            archived = _row(connection, archive_id(primary['session']['signal_session']))
            value = pipeline.validate_archive(archived, cfg, primary['session'], base_dir=base_dir,
                                               strategy_id=BULL_FLAG)
        finally:
            connection.rollback(); connection.close()
        dataset = pipeline.verified_dataset(value, archived[0], base_dir=base_dir, strategy_id=BULL_FLAG)
        spec = strict_json((Path(value['output_root'])/'study.json').read_bytes())
        batch = pipeline.selection_batch(value, primary['session'])
        backtest = run_candidate_backtest(batch, primary_run_id=primary['run_id'],
                                         dataset=dataset, specification=spec)
        metrics = build_metrics(batch, backtest, primary_run_id=primary['run_id'])
        record = dict(schema='jbravo.companion-primary.v1', status='COMPLETED_COMPANION_PRIMARY',
            run_id=primary_id(primary['run_id']), primary_run_id=primary['run_id'],
            parent_report_sha256=digest(primary), source_bindings=deepcopy(primary['source_bindings']),
            batch=batch, backtest=backtest, metrics=metrics, session=primary['session'],
            finished_at=datetime.now(timezone.utc).isoformat(), specification_sha256=archived[0].strip(),
            research_qualified=False, execution_authorized=False, order_submission_authorized=False)
        from scripts import pipeline_postflight as health
        require(all(health._binding(health._read(Path(base_dir)/name,health.MAX_SOURCE_BYTES))==pin
                    for name,pin in primary['source_bindings'].items()),'companion_source_changed_during_run')
        _publish(record, archived[0].strip(), env=env)
        value = dict(status=record['status'], run_id=record['run_id'], result_sha256=digest(record),
            primary_run_id=primary['run_id'], candidates=batch['candidate_count'],
            backtest_result_id=backtest['result_id'], metrics_result_id=metrics['result_id'])
    except Exception as error:
        value = dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED', primary_run_id=primary['run_id'],
                     error_type=type(error).__name__)
    _status(value, base_dir)
    return value


def read_handoff(connection, parent, *, now, inspection=False, base_dir=ROOT):
    """Bind one exact immutable DB result to an already verified primary handoff."""
    if configuration(base_dir) is None:
        return None
    from scripts.strategy_library import get_strategy
    from scripts.strategy_pipeline import validate_backtest
    from scripts.strategy_metrics import validate_metrics
    from scripts import strategy_execution as rules
    archived = _row(connection, primary_id(parent['primary_run_id']))
    if archived is None:
        return None
    spec_sha, result_sha, record = archived
    require(isinstance(record, dict) and result_sha.strip() == digest(record), 'companion_db_hash_changed')
    require(record.get('schema') == 'jbravo.companion-primary.v1' and
        record.get('status') == 'COMPLETED_COMPANION_PRIMARY' and
        record['run_id'] == primary_id(parent['primary_run_id']) and
        record['primary_run_id'] == parent['primary_run_id'] and
        record['parent_report_sha256'] == parent['primary_report_sha256'] and
        record['source_bindings'] == parent['source_bindings'] and record['session'] == parent['session'] and
        all(record.get(k) is False for k in ('research_qualified','execution_authorized','order_submission_authorized')),
        'companion_primary_lineage_changed')
    require(spec_sha.strip() == record['specification_sha256'] and
        rules.utc(record['finished_at']) <= rules.utc(now) and
        (inspection or rules.utc(now) < rules.utc(parent['execution_policy']['entry_open_at'])),
        'companion_future_or_expired')
    batch = record['batch']
    definition = get_strategy(BULL_FLAG).describe()
    contract = dict(strategy_id=BULL_FLAG, strategy_version=definition['version'],
                    definition_sha256=definition['definition_sha256'])
    require(batch['strategy_contract'] == contract and batch['session'] == parent['session'] and
        batch['candidate_count'] == len(batch['proposals']) <= 3 and
        len({p['symbol'] for p in batch['proposals']}) == len(batch['proposals']) and
        len({p['signal_id'] for p in batch['proposals']}) == len(batch['proposals']),
        'companion_population_changed')
    validate_backtest(record['backtest'], batch, parent['primary_run_id'])
    validate_metrics(record['metrics'], record['backtest'], batch, parent['primary_run_id'])
    for proposal in batch['proposals']:
        from scripts.bull_flag_strategy import plan_from_row
        plan = plan_from_row(proposal['symbol'], {'entry_details': {'pattern_exit': proposal.get('pattern_exit')}})
        require(plan is not None and plan == proposal.get('pattern_exit') and
            plan['entry'] == proposal['signal_close_raw'] and
            plan['signal_time'] == parent['session']['signal_session']+'T00:00:00' and
            proposal['strategy_id'] == BULL_FLAG and
            proposal['strategy_version'] == '1' and proposal['signal_session'] == parent['session']['signal_session'] and
            proposal['next_session'] == parent['session']['next_session'] and
            proposal['paper_only'] is True and proposal['order_submission_authorized'] is False,
            'companion_pattern_evidence_changed')
    result = deepcopy(parent)
    result.update(primary_report_sha256=result_sha.strip(), batch_sha256=digest(batch),
        strategy_contract=contract, proposals=deepcopy(batch['proposals']),
        backtest_result_id=record['backtest']['result_id'], metrics_result_id=record['metrics']['result_id'],
        candidate_values=deepcopy(batch['proposals']), companion_parent_report_sha256=parent['primary_report_sha256'],
        inspection_mode=inspection)
    result.pop('handoff_id', None)
    result['handoff_id'] = rules.digest(result)
    return result
