"""Immutable Webull-assisted selection, alongside the existing daily ranking."""
from datetime import datetime, time, timezone
import time as clock
from pathlib import Path

from scripts import setup_selection as base
from scripts.bull_flag_companion import _publish, _row, _status
from scripts.strategy_webull_market_data import (
    LIMIT_POLICY, NY, confirm_capture, limit_order, screening_evidence,
)

POLICY = 'UNIFIED_EXTENDED_HOURS_CONFIRMATION_V1'
ARCHIVE_KEY = 'webull_after_hours_archive'


def enabled(session):
    return 'webull_hours' in session


def capture_from_archive(connection, session):
    pin = session[ARCHIVE_KEY]
    row = _row(connection, pin['id'])
    base.require(row is not None, 'webull_capture_archive_missing')
    _, result_sha, value = row
    base.require(result_sha.strip() == pin['sha256'] == base.digest(value) and
                 value['run_id'] == pin['id'] and value['signal_session'] == session['signal_session'] and
                 base.digest(value['webull_after_hours']) == pin['capture_sha256'],
                 'webull_capture_archive_changed')
    return value['webull_after_hours']


def selection_and_orders(daily, capture, parent, *, published):
    session = parent['session']
    cutoff = datetime.fromisoformat(session['webull_hours']['after_hours_close_at'])
    opening = datetime.combine(datetime.fromisoformat(session['next_session']).date(), time(4), NY)
    ranked = daily['ranked_setups']
    confirm_capture(capture, session, [s['proposal']['symbol'] for s in ranked], published=published)
    for setup in ranked:
        proposal = setup['proposal']
        base.require(proposal['signal_session'] == session['signal_session'] and
                     proposal['next_session'] == session['next_session'], 'webull_proposal_session_changed')
        setup['after_hours'] = screening_evidence(capture['symbols'][proposal['symbol']], cutoff)
    # Confirmed gainers lead. Within each confirmation group, use the existing
    # technical score, volume and deterministic ownership ordering.
    ordered = sorted(ranked, key=lambda item: (
        item['after_hours']['status'] != 'AVAILABLE', not item['after_hours'].get('confirmed', False)))
    for index, item in enumerate(ordered, 1):
        item['rank'] = index
    selected = [s for s in ordered if s['after_hours']['status'] == 'AVAILABLE'][:3]
    selection = dict(daily, policy=POLICY, ranked_setups=ordered, selected_setups=selected,
        selected_signal_ids=[s['proposal']['signal_id'] for s in selected],
        tie_break='AFTER_HOURS_CONFIRMATION_THEN_TECHNICAL_RANK', after_hours_coverage=dict(
            requested=len(ordered), available=sum(s['after_hours']['status'] == 'AVAILABLE' for s in ordered),
            missing_or_unusable=sum(s['after_hours']['status'] != 'AVAILABLE' for s in ordered)))
    orders = {}
    selection_id = base.selection_id(parent['primary_run_id'])
    for setup in selected:
        p = setup['proposal']
        orders[p['signal_id']] = limit_order(p, capture['symbols'][p['symbol']], capture,
            parent, selection_id=selection_id, published=published, opening=opening)
    return selection, orders


def expected_record(connection, parent, *, now, published, base_dir):
    handoffs, refs, daily = base.inputs(connection, parent, now=now, base_dir=base_dir)
    capture = capture_from_archive(connection, parent['session'])
    selection, orders = selection_and_orders(daily, capture, parent, published=published)
    specification = dict(policy=POLICY, model_sha256=base.model_sha256(), required_strategies=sorted(handoffs),
        strategy_result_sha256=refs, execution_limit_policy=LIMIT_POLICY, capture_sha256=base.digest(capture))
    record = dict(schema='jbravo.unified-setup-selection.v1', status='COMPLETED_UNIFIED_SETUP_SELECTION',
        run_id=base.selection_id(parent['primary_run_id']), primary_run_id=parent['primary_run_id'],
        parent_report_sha256=parent['primary_report_sha256'], source_bindings=parent['source_bindings'],
        session=parent['session'], specification=specification, selection=selection,
        finished_at=published.isoformat(), execution_orders=orders, paper_only=True, research_qualified=False,
        execution_authorized=False, order_submission_authorized=False)
    return record, handoffs


def publish(primary, *, env, base_dir=base.ROOT):
    if not enabled(primary['session']):
        return base.publish(primary, env=env, base_dir=base_dir)
    from scripts import pipeline_postflight as health
    from scripts.research_store import connect
    from scripts.strategy_runtime_diagnostics import diagnostic

    started = clock.monotonic()
    parent = base.parent_context(primary)
    phase = 'WEBULL_SELECTION_REPLAY'
    try:
        now = datetime.now(timezone.utc)
        connection = connect(env=env)
        try:
            record, _ = expected_record(connection, parent, now=now, published=now, base_dir=base_dir)
        finally:
            connection.rollback()
            connection.close()
        base.require(all(health._binding(health._read(Path(base_dir) / name, health.MAX_SOURCE_BYTES)) == pin
                         for name, pin in record['source_bindings'].items()),
                     'unified_selection_sources_changed')
        phase = 'WEBULL_SELECTION_PUBLICATION'
        _publish(record, base.digest(record['specification']), env=env)
        status = dict(status=record['status'], run_id=record['run_id'], result_sha256=base.digest(record),
            primary_run_id=primary['run_id'], parent_report_sha256=parent['primary_report_sha256'],
            selected_signal_ids=record['selection']['selected_signal_ids'],
            after_hours_coverage=record['selection']['after_hours_coverage'])
    except Exception as error:
        status = dict(status='UNIFIED_SETUP_SELECTION_UNAVAILABLE', primary_run_id=primary['run_id'],
            parent_report_sha256=parent['primary_report_sha256'], **diagnostic(error, phase, started))
    _status(status, base_dir, 'unified_setups')
    return status


def read_selection(connection, parent, *, now, base_dir=base.ROOT):
    if not enabled(parent['session']):
        return base.read_selection(connection, parent, now=now, base_dir=base_dir)
    row = _row(connection, base.selection_id(parent['primary_run_id']))
    base.require(row is not None, 'unified_selection_not_published')
    spec_sha, result_sha, record = row
    published = datetime.fromisoformat(record['finished_at'])
    base.require(published <= now and result_sha.strip() == base.digest(record),
                 'webull_selection_receipt_changed')
    expected, handoffs = expected_record(connection, parent, now=now, published=published, base_dir=base_dir)
    base.require(record == expected and spec_sha.strip() == base.digest(expected['specification']),
                 'webull_selection_replay_changed')
    return dict(record=record, result_sha256=result_sha.strip(), handoffs=handoffs)
