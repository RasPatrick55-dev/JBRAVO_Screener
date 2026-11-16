"""Versioned research archive preparation, separate from frozen lifecycle code.

The original companion module and its position-management bindings stay unchanged.
Only exact-config archive lookup differs from its historical preparer.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
from scripts.bull_flag_companion import (ROOT, BULL_FLAG, policy_for, _row,
    require, strict_json, primary_id, digest, _publish, _status)
from scripts.strategy_archive_identity import archive_id

def prepare_primary(primary, *, env, base_dir=ROOT, strategy_id=BULL_FLAG, dataset=None):
    """Separate exact-session Backtest/Metrics and immutable DB publication.

    No canonical candidate rows/views, primary report, existing strategy policy
    or historical trade is modified. Missing/failed companions remain unavailable.
    """
    if policy_for(strategy_id, base_dir) is None:
        return dict(status='DISABLED')
    from scripts import strategy_pipeline as pipeline
    from scripts.daily_strategy_scan import load_config
    from scripts.research_store import connect
    from scripts.strategy_backtest import run_candidate_backtest
    from scripts.strategy_metrics import build_metrics
    started=time.monotonic(); phase='COMPANION_ARCHIVE_READBACK'
    try:
        cfg = dict(load_config(Path(base_dir)/'config/strategy_scan_daily.json'), strategy_id=strategy_id)
        connection = connect(env=env)
        try:
            archived = _row(connection, archive_id(cfg, primary['session']['signal_session'], strategy_id))
            value = pipeline.validate_archive(archived, cfg, primary['session'], base_dir=base_dir,
                                               strategy_id=strategy_id)
        finally:
            connection.rollback(); connection.close()
        dataset = pipeline.verified_dataset(value, archived[0], base_dir=base_dir, strategy_id=strategy_id,
            **({'dataset': dataset} if dataset is not None else {}))
        spec = strict_json((Path(value['output_root'])/'study.json').read_bytes())
        batch = pipeline.selection_batch(value, primary['session'])
        phase='COMPANION_BACKTEST_METRICS'
        backtest = run_candidate_backtest(batch, primary_run_id=primary['run_id'],
                                         dataset=dataset, specification=spec)
        metrics = build_metrics(batch, backtest, primary_run_id=primary['run_id'])
        record = dict(schema='jbravo.companion-primary.v1', status='COMPLETED_COMPANION_PRIMARY',
            run_id=primary_id(primary['run_id'], strategy_id), primary_run_id=primary['run_id'],
            parent_report_sha256=digest(primary), source_bindings=deepcopy(primary['source_bindings']),
            batch=batch, backtest=backtest, metrics=metrics, session=primary['session'],
            finished_at=datetime.now(timezone.utc).isoformat(), specification_sha256=archived[0].strip(),
            research_qualified=False, execution_authorized=False, order_submission_authorized=False)
        from scripts import pipeline_postflight as health
        require(all(health._binding(health._read(Path(base_dir)/name,health.MAX_SOURCE_BYTES))==pin
                    for name,pin in primary['source_bindings'].items()),'companion_source_changed_during_run')
        phase='COMPANION_PRIMARY_PUBLICATION'
        _publish(record, archived[0].strip(), env=env)
        value = dict(status=record['status'], run_id=record['run_id'], result_sha256=digest(record),
            primary_run_id=primary['run_id'], candidates=batch['candidate_count'],
            backtest_result_id=backtest['result_id'], metrics_result_id=metrics['result_id'])
    except Exception as error:
        from scripts.strategy_runtime_diagnostics import diagnostic
        value = dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED', primary_run_id=primary['run_id'],
                     **diagnostic(error,phase,started))
    _status(value, base_dir, strategy_id)
    return value
