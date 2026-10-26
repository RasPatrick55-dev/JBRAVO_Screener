"""Read-only PostgreSQL handoff for the strategy executor; never submits orders."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import pipeline_postflight as health
from scripts import strategy_pipeline as pipeline
from scripts.strategy_backtest import require
from scripts.strategy_execution import prepare_handoff

ROOT = Path(__file__).resolve().parents[1]


def load_handoff(conn, *, base_dir=ROOT, now=None):
    """Require source/history/completion proof plus one repeatable DB snapshot.

    The connection must be a newly opened read-only research_store connection.
    No candidate date defaults, newest-nonempty query, CSV or legacy ML join.
    """
    now = now or datetime.now(timezone.utc)
    require(conn.readonly is True and conn.autocommit is False and
        conn.isolation_level == 2, 'execution_readonly_repeatable_snapshot_required')
    result = health.check_health(base_dir, now=now, max_age_seconds=43200,
        require_session=True, require_completion=True)
    require(result.get('status') == 'ok' and result.get('route') == 'pullback_reclaim_strategy' and
        result.get('screen_ready') is True, 'execution_primary_not_healthy')
    raw = health._read(Path(base_dir)/'reports/pipeline_postflight/latest.json', health.MAX_REPORT_BYTES)
    require(health._binding(raw) == result['report_binding'], 'execution_primary_changed_during_read')
    record = health._decode(raw) if hasattr(health, '_decode') else json.loads(raw,
        object_pairs_hook=health._strict_object, parse_constant=health._invalid_constant)
    with conn.cursor() as cur:
        cur.execute('SELECT run_id,run_date,run_ts_utc,candidate_count,payload '
            'FROM public.screener_strategy_batches_app ORDER BY run_date DESC,run_ts_utc DESC LIMIT 1')
        batch = cur.fetchone()
    require(batch is not None and batch[0] == record['run_id'] and
        batch[1].isoformat() == record['session']['signal_session'] and
        batch[2] == datetime.fromisoformat(record['candidate_run_ts_utc']) and
        batch[3] == record['batch']['candidate_count'] and batch[4] == record['batch'],
        'execution_database_batch_differs_from_primary')
    from scripts.strategy_pipeline_store import candidate_frame
    frame, stamp = candidate_frame(conn, batch[1], exact_run_ts=batch[2])
    require(stamp == batch[2], 'execution_database_snapshot_changed')
    handoff = prepare_handoff(record, frame.to_dict('records'), now=now)
    require(health._read(Path(base_dir)/'reports/pipeline_postflight/latest.json', health.MAX_REPORT_BYTES) == raw,
        'execution_primary_changed_after_database_read')
    return handoff


def main():
    from scripts.daily_strategy_scan import configured_environment
    from scripts.research_store import connect
    conn = connect(env=configured_environment())
    try:
        handoff = load_handoff(conn)
    finally:
        conn.rollback()
        conn.close()
    print(json.dumps(dict(status=handoff['status'], handoff_id=handoff['handoff_id'],
        primary_run_id=handoff['primary_run_id'], strategy_contract=handoff['strategy_contract'],
        signal_ids=[p['signal_id'] for p in handoff['proposals']],
        backtest_result_id=handoff['backtest_result_id'], metrics_result_id=handoff['metrics_result_id'],
        session=handoff['session'], order_submission_authorized=False, broker_operations=0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
