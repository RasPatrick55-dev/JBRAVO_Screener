"""Inert executor readiness, per-role receipts and bounded sanitized log inspection.

No credentials, application imports, database calls, subprocesses or broker calls.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat
import uuid

from scripts import pipeline_postflight as pipeline

ROOT = Path(__file__).resolve().parents[1]
MAX_AGE_SECONDS = 43200  # Same UTC day, at most 12 hours; overnight preparation.
MAX_BYTES = 64 * 1024
MAX_CANDIDATES = 10000
MAX_OBSERVATIONS = 64
ROLES = ('premarket', 'reconciliation', 'diagnostic', 'dry_run', 'unresolved')
SOURCES = ('scripts/execute_trades.py', 'scripts/executor_health.py',
           'scripts/db_queries.py', 'scripts/pipeline_postflight.py')
TERMINAL = {'filled', 'canceled', 'expired', 'rejected'}
OBSERVATION_STATES = TERMINAL | {'submitted', 'resting', 'cancel_requested',
                                'cancel_failed', 'cancel_unavailable'}
COUNTS = ('symbols_in', 'orders_submitted', 'orders_filled', 'orders_canceled',
          'trailing_attached', 'api_failures')
LOG_TAGS = ('EXEC_START', 'EXECUTE END', 'EXECUTE_SUMMARY', 'EXECUTE_SKIP',
            'DB_CANDIDATES_LOADED', 'MARKET_DAY_CLOSED', 'WAIT_UNTIL', 'POLL_DETACH',
            'ORDER_RESTING', 'ORDERS_RESTING', 'BUY_FILL', 'BUY_CANCELLED',
            'RECONCILE_AUTO_BEGIN', 'RECONCILE_AUTO_END', 'EXECUTOR_PREFLIGHT')
LOG_FIELDS = ('dry_run', 'diagnostic', 'time_window', 'resolved', 'ny_now', 'hhmm',
              'in_window', 'candidates', 'submit_at_ny', 'count', 'max_timestamp',
              'latest_run_ts', 'submitted', 'trailing', 'orders_submitted',
              'orders_filled', 'reason', 'stage', 'remaining', 'remaining_qty',
              'remaining_seconds', 'waited_secs', 'cancel_after_min', 'until_utc', 'rc')


def binding(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def ordinary(path):
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('redirected_path')


def read(path, limit=MAX_BYTES):
    ordinary(path)
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError('empty_or_oversize_input')
    return data


def decode(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=pipeline._strict_object,
                      parse_constant=pipeline._invalid_constant)


def utc(value):
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('ambiguous_timestamp')
        return value.astimezone(timezone.utc)
    return pipeline._utc(value)


def candidate_values(rows, *, ordered=False):
    """Bind complete loaded values, types and presence; never save candidate values.

    Missing floats are tagged, not coerced to zero. Date/time representations and
    Decimal quantities retain their supplied precision. Unsupported values fail.
    """
    def encode(value):
        if value is None:
            return ['null']
        if type(value) is bool:
            return ['bool', value]
        if isinstance(value, datetime):
            return ['datetime', value.isoformat()]
        if isinstance(value, date):
            return ['date', value.isoformat()]
        if isinstance(value, str):
            return ['string', value]
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError('invalid_candidate_value')
            return ['decimal', str(value)]
        if isinstance(value, Integral):
            return ['integer', str(value)]
        if isinstance(value, Real):
            number = float(value)
            if math.isnan(number):
                return ['missing_float']
            if not math.isfinite(number):
                raise ValueError('invalid_candidate_value')
            return ['float', number.hex()]
        if isinstance(value, (list, tuple)):
            return ['sequence', [encode(item) for item in value]]
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError('invalid_candidate_field')
            return ['object', [[k, encode(value[k])] for k in sorted(value)]]
        # pandas nullable scalars are missing, rather than arbitrary str coercion.
        if type(value).__module__ == 'pandas._libs.missing' and type(value).__name__ == 'NAType':
            return ['missing_nullable']
        raise ValueError('unsupported_candidate_value')

    if not isinstance(rows, list) or len(rows) > MAX_CANDIDATES or any(not isinstance(r, dict) for r in rows):
        raise ValueError('invalid_candidate_population')
    population = rows if ordered else sorted(rows, key=lambda row: row.get('symbol', ''))
    data = json.dumps([encode(row) for row in population], separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('candidate_value_binding_oversize')
    return binding(data)


def preflight(base_dir=ROOT, *, now=None, rows=None, prior_binding=None,
              prior_value_binding=None,
              max_age_seconds=MAX_AGE_SECONDS):
    """Require healthy bound primary evidence, then the exact declared DB batch.

    Batch linkage proves timestamp/count/unique membership, not independent
    historical data quality or an atomic database snapshot through submission.
    """
    now = utc(now or datetime.now(timezone.utc))
    root = Path(base_dir)
    result = {'status': 'blocked', 'reasons': [], 'checked_at': now.isoformat(),
              'max_age_seconds': max_age_seconds, 'research_qualified': False,
              'operating_acceptance': False}
    try:
        health = pipeline.check_health(root, now=now, expected_day=now.date(),
                                       max_age_seconds=max_age_seconds)
        if health['status'] != 'ok':
            result['reasons'] = ['pipeline_' + r for r in
                                 health.get('failures', []) + health.get('qualifications', [])]
            if not result['reasons']:
                result['reasons'] = ['pipeline_not_healthy']
            return result
        data = read(root / 'reports/pipeline_postflight/latest.json')
        identity = binding(data)
        if identity != health['report_binding']:
            raise ValueError('pipeline_changed_during_check')
        if prior_binding is not None and identity != prior_binding:
            raise ValueError('pipeline_changed_after_candidate_load')
        record = decode(data)
        result.update(pipeline_run_id=health['run_id'], pipeline_binding=identity,
                      candidate_run_ts_utc=record['coverage']['run_ts_utc'])
        if rows is not None:
            if not isinstance(rows, list) or len(rows) > MAX_CANDIDATES:
                raise ValueError('candidate_population_invalid_or_oversize')
            if len(rows) != record['coverage']['total']:
                raise ValueError('candidate_count_mismatch')
            expected = utc(record['coverage']['run_ts_utc'])
            symbols = set()
            identities = []
            for row in rows:
                symbol = row.get('symbol') if isinstance(row, dict) else None
                if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,15}', symbol):
                    raise ValueError('candidate_identity_invalid')
                if symbol in symbols:
                    raise ValueError('candidate_identity_duplicate')
                symbols.add(symbol)
                if utc(row.get('run_ts_utc')) != expected:
                    raise ValueError('candidate_batch_mismatch')
                # run_date need not equal preparation date: previous-session labels
                # are allowed only through this same-day, bound preparation batch.
                identities.append({'symbol': symbol, 'run_ts_utc': expected.isoformat()})
            result['candidate_binding'] = binding(json.dumps(
                sorted(identities, key=lambda r: r['symbol']), sort_keys=True).encode())
            result['candidate_count'] = len(rows)
            result['candidate_value_binding'] = candidate_values(rows)
            if prior_value_binding is not None and result['candidate_value_binding'] != prior_value_binding:
                raise ValueError('candidate_values_changed')
        result['status'] = 'pass'
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # Never persist raw exceptions, paths, credentials or database rows.
        import sys
        error = sys.exc_info()[1]
        known = str(error) if type(error) is ValueError else ''
        result['reasons'] = [known if re.fullmatch(r'[a-z_]{1,80}', known)
                             else 'missing_malformed_or_unbound_preflight']
    return result


def observe(metrics, *, order_id, symbol, state, deadline=None):
    """Record only observations already made by existing broker logic."""
    if state not in OBSERVATION_STATES:
        raise ValueError('invalid_observation_state')
    observations = getattr(metrics, '_order_observations', None)
    if observations is None:
        observations = []
        metrics._order_observations = observations
    if len(observations) >= MAX_OBSERVATIONS:
        metrics._order_observations_truncated = True
        if hasattr(metrics, '_observation_config'):
            metrics._observation_config._entry_observations_truncated = True
        return
    if not order_id:
        # An accepted submission without an identifier cannot be confirmed later.
        return
    observations.append({'order_id': str(order_id)[:128], 'symbol': str(symbol)[:16],
                         'state': state, 'observed_at': datetime.now(timezone.utc).isoformat(),
                         'deadline': utc(deadline).isoformat() if deadline else None,
                         'terminal_confirmed': state in TERMINAL})


def assess_outcome(rc, metrics, observations, truncated, gate, role):
    """Recompute health; counters and cancellation acknowledgments are not proof."""
    if type(rc) is not int or type(truncated) is not bool or not isinstance(observations, list) or len(observations) > MAX_OBSERVATIONS:
        raise ValueError('invalid_receipt_outcome')
    states = {}
    for item in observations:
        if (not isinstance(item, dict) or not isinstance(item.get('order_id'), str)
                or not 1 <= len(item['order_id']) <= 128 or item.get('state') not in OBSERVATION_STATES
                or type(item.get('terminal_confirmed')) is not bool
                or item['terminal_confirmed'] != (item['state'] in TERMINAL)):
            raise ValueError('invalid_order_observation')
        utc(item['observed_at'])
        if item.get('deadline') is not None:
            utc(item['deadline'])
        states[item['order_id']] = item['state']
    valid_counts = all(type(metrics.get(k)) is int and metrics[k] >= 0 for k in COUNTS)
    unknown = max(0, metrics['orders_submitted'] - len(states)) if valid_counts else 0
    pending = sum(s not in TERMINAL for s in states.values()) + unknown
    cancel_failed = any(s in {'cancel_failed', 'cancel_unavailable'} for s in states.values())
    unbound_entry = (role == 'premarket' and metrics.get('orders_submitted') and
                     (not isinstance(gate, dict) or gate.get('status') != 'pass'
                      or not gate.get('candidate_binding')))
    health = 'failed' if (rc != 0 or metrics.get('api_failures') or cancel_failed or not valid_counts
                         or unbound_entry or metrics.get('status') not in {'ok', 'skipped'}) else 'ok'
    if health == 'ok' and (pending or truncated):
        health = 'pending_confirmation'
    if health == 'ok' and not metrics.get('orders_submitted'):
        health = 'skipped' if metrics.get('skips') or metrics.get('exit_reason') else 'no_orders'
    return health, pending, unknown


def publish_receipt(base_dir, *, role, started_at, finished_at, rc, metrics,
                    config=None, observations=(), observations_truncated=False):
    if role not in ROLES or type(rc) is not int:
        raise ValueError('invalid_receipt_identity')
    root = Path(base_dir)
    run_id = uuid.uuid4().hex
    selected = {k: metrics.get(k) for k in ('status', 'exit_reason', 'symbols_in',
                'orders_submitted', 'orders_filled', 'orders_canceled', 'trailing_attached',
                'api_failures', 'in_window')}
    skips = metrics.get('skips') or {}
    selected['skips'] = {k: v for k, v in skips.items()
                         if re.fullmatch('[A-Z_]{1,40}', k) and type(v) is int and v > 0}
    observations = list(observations)
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError('oversize_observations')
    gate = getattr(config, '_entry_preflight', None) if config else None
    health, pending, unknown = assess_outcome(rc, selected, observations, observations_truncated, gate, role)
    record = {'schema_version': 1, 'run_id': run_id, 'role': role,
              'started_at': utc(started_at).isoformat(), 'finished_at': utc(finished_at).isoformat(),
              'process_rc': rc, 'health': health, 'metrics': selected, 'preflight': gate,
              'order_observations': observations, 'observations_truncated': observations_truncated,
              'pending_terminal_confirmations': pending, 'legacy_orders_canceled_is_not_confirmation': True,
              'unobserved_submissions': unknown,
              'cancellation_policy': 'minutes_from_submission',
              'cancel_after_min': getattr(config, 'cancel_after_min', None),
              'poll_detach_seconds': getattr(config, 'poll_detach_secs', None),
              'task_attribution': {'declared_task_id': getattr(config, 'scheduler_task_id', None),
                                   'basis': 'explicit_process_configuration',
                                   'independently_verified': False},
              'candidate_snapshot': getattr(config, '_entry_db_snapshot', None),
              'raw_candidate_value_binding': getattr(config, '_entry_raw_value_binding', None),
              'filtered_candidate_value_binding': getattr(config, '_entry_filtered_binding', None),
              'execution_candidate_binding': getattr(config, '_entry_execution_binding', None),
              'effective_settings': {key: getattr(config, key, None) for key in (
                  'source', 'dry_run', 'diagnostic', 'reconcile_only', 'submit_at_ny',
                  'time_window', 'extended_hours', 'allocation_pct', 'min_order_usd',
                  'max_new_positions', 'disable_open_position_cap', 'trailing_percent')},
              'research_qualified': False, 'operating_acceptance': False,
              'source_bindings': {p: binding(read(root / p, 1024 * 1024)) for p in SOURCES}}
    directory = root / 'reports/executor' / role
    ordinary(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(data) > MAX_BYTES:
        raise ValueError('oversize_receipt')
    history = directory / ('run-' + run_id + '.json')
    with history.open('xb') as handle:
        handle.write(data)
    if read(history) != data:
        raise OSError('receipt_readback_failed')
    latest = directory / 'latest.json'
    ordinary(latest)
    temporary = directory / ('latest-' + run_id + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(data)
    if read(temporary) != data:
        raise OSError('receipt_readback_failed')
    os.replace(temporary, latest)
    if read(latest) != data:
        raise OSError('receipt_readback_failed')
    return record


def inspect_receipt(base_dir=ROOT, *, role='premarket', now=None, max_age_seconds=MAX_AGE_SECONDS,
                    expected_task_id=None):
    now = utc(now or datetime.now(timezone.utc))
    root = Path(base_dir)
    try:
        if role not in ROLES or not pipeline._number(max_age_seconds) or not 0 < max_age_seconds <= 86400:
            raise ValueError('invalid_inspection_settings')
        directory = root / 'reports/executor' / role
        data = read(directory / 'latest.json')
        record = decode(data)
        attribution = record.get('task_attribution', {})
        if not isinstance(attribution, dict) or (attribution and (
                attribution.get('basis') != 'explicit_process_configuration' or
                attribution.get('independently_verified') is not False)):
            raise ValueError('invalid_task_attribution')
        task_id = attribution.get('declared_task_id')
        if task_id is not None and (not isinstance(task_id, str) or not re.fullmatch('[1-9][0-9]{0,11}', task_id)):
            raise ValueError('invalid_task_attribution')
        if expected_task_id is not None and task_id != expected_task_id:
            return {'status': 'failed', 'reason': 'task_attribution_missing_or_mismatched'}
        if type(record.get('schema_version')) is not int or record['schema_version'] != 1 or record.get('role') != role or not re.fullmatch(
                '[0-9a-f]{32}', record.get('run_id', '')):
            raise ValueError('invalid_receipt_identity')
        if read(directory / ('run-' + record['run_id'] + '.json')) != data:
            raise ValueError('receipt_history_mismatch')
        for path in SOURCES:
            if record.get('source_bindings', {}).get(path) != binding(read(root / path, 1024 * 1024)):
                raise ValueError('receipt_source_mismatch')
        started, finished = utc(record['started_at']), utc(record['finished_at'])
        if started > finished or finished > now or finished.date() != now.date() or (
                now - finished).total_seconds() > max_age_seconds:
            raise ValueError('receipt_stale_or_invalid')
        if type(record.get('process_rc')) is not int or record.get('health') not in {
                'ok', 'failed', 'pending_confirmation', 'skipped', 'no_orders'}:
            raise ValueError('invalid_receipt_outcome')
        outcome, pending, unknown = assess_outcome(record['process_rc'], record['metrics'],
            record['order_observations'], record['observations_truncated'], record.get('preflight'), role)
        if (record['health'] != outcome or record.get('pending_terminal_confirmations') != pending
                or record.get('unobserved_submissions') != unknown
                or type(record.get('pending_terminal_confirmations')) is not int
                or type(record.get('unobserved_submissions')) is not int
                or record.get('research_qualified') is not False
                or record.get('operating_acceptance') is not False):
            raise ValueError('inconsistent_receipt_outcome')
        return {'status': record['health'], 'receipt_binding': binding(data), 'record': record}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {'status': 'failed', 'reason': 'missing_stale_malformed_or_unbound_receipt'}


def deadline_plan(inspection, *, now=None):
    """Read-only proposal for a future supervised cancel-and-confirm worker.

    Consumes validated receipts only. It cannot call a broker, mutate an order,
    submit a replacement, or claim remote cancellation. No account-wide sweep.
    """
    now = utc(now or datetime.now(timezone.utc))
    result = {'status': 'blocked', 'reasons': [], 'orders': [], 'broker_calls': 0,
              'cancellation_enforced': False, 'operating_acceptance': False}
    if inspection.get('status') not in {'ok', 'pending_confirmation', 'skipped', 'no_orders'}:
        result['reasons'] = ['validated_entry_receipt_required']
        return result
    record = inspection.get('record', {})
    if record.get('role') != 'premarket' or record.get('process_rc') != 0:
        result['reasons'] = ['successful_entry_role_required']
        return result
    if record.get('observations_truncated') or record.get('unobserved_submissions'):
        result['reasons'] = ['incomplete_order_population']
        return result
    states = {}
    try:
        for item in record['order_observations']:
            state = states.setdefault(item['order_id'], {'symbol': item['symbol'], 'deadline': None})
            if state['symbol'] != item['symbol']:
                raise ValueError('contradictory_order_identity')
            if item.get('deadline') is not None:
                deadline = utc(item['deadline'])
                if state['deadline'] is not None and state['deadline'] != deadline:
                    raise ValueError('contradictory_order_deadline')
                state['deadline'] = deadline
            if state.get('state') in TERMINAL and item['state'] not in TERMINAL:
                raise ValueError('terminal_state_regressed')
            if state.get('state') in TERMINAL and item['state'] != state['state']:
                raise ValueError('contradictory_terminal_state')
            state['state'] = item['state']
        planned = []
        for order_id, state in states.items():
            if state['state'] in TERMINAL:
                continue
            if state['deadline'] is None:
                raise ValueError('missing_cancellation_deadline')
            action = ('confirm_existing_request' if state['state'] == 'cancel_requested' else
                      'cancel_and_confirm' if now >= state['deadline'] else 'wait_until_deadline')
            planned.append({'order_id': order_id, 'symbol': state['symbol'],
                            'deadline': state['deadline'].isoformat(), 'proposed_action': action,
                            'requires_broker_identity_side_account_and_state_check': True})
        result['orders'] = planned
        result['status'] = 'action_required' if any(p['proposed_action'] != 'wait_until_deadline' for p in planned) else 'no_action_due'
    except (KeyError, TypeError, ValueError) as error:
        result['reasons'] = [str(error) if type(error) is ValueError else 'malformed_order_population']
    return result


def log_tail(base_dir=ROOT, *, max_bytes=MAX_BYTES, max_events=40):
    """Seek on the host; never download the complete log or emit raw lines."""
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES or type(max_events) is not int or not 1 <= max_events <= 100:
        raise ValueError('invalid_tail_limit')
    path = Path(base_dir) / 'logs/execute_task.log'
    ordinary(path)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        size = before.st_size
        offset = max(0, size - max_bytes)
        handle.seek(offset)
        data = handle.read(max_bytes)
        after = os.fstat(handle.fileno())
    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    lines = data.splitlines()
    if offset and lines:
        lines = lines[1:]  # Never claim a partial first line is a complete event.
    events = []
    for raw in lines:
        line = raw.decode('utf-8', 'replace')
        if re.search(r'credential|secret|api.?key|authorization|token|password|DATABASE_URL|sanitized=', line, re.I):
            continue
        tags = [t for t in LOG_TAGS if re.search(r'\b' + re.escape(t) + r'\b', line)]
        if not tags:
            continue
        timestamp = re.match(r'\[?(\d{4}-\d{2}-\d{2}[T ][0-9:.,+Z-]+)', line)
        fields = {}
        for key in LOG_FIELDS:
            match = re.search(r'\b' + key + r'=([^\s,;]+)', line)
            if match and (re.fullmatch(r'[0-9][0-9:.T+Z-]{0,63}', match[1]) or match[1] in
                          {'true', 'false', 'True', 'False', 'auto', 'premarket', 'regular',
                           'closed', 'start', 'end'} or match[1] in
                          {'TIME_WINDOW', 'CASH', 'NO_CANDIDATES', 'DATA_MISSING', 'AUTH_FAIL',
                           'RECONCILE_ONLY', 'PREFLIGHT', 'DAILY_ENTRY_LIMIT'}):
                fields[key] = match[1]
        events.append({'timestamp': timestamp[1] if timestamp else None, 'tags': tags, 'fields': fields})
    return {'file_bytes_at_open': size, 'tail_binding': binding(data), 'offset': offset,
            'events': events[-max_events:], 'matched_events': len(events),
            'sample_truncated': len(events) > max_events, 'complete_log': offset == 0 and not changed,
            'file_changed_during_read': changed,
            'raw_lines_returned': False, 'task_success_established': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect', 'logs', 'deadlines'))
    parser.add_argument('--base-dir', type=Path, default=ROOT)
    parser.add_argument('--role', choices=ROLES, default='premarket')
    parser.add_argument('--expected-task-id')
    args = parser.parse_args(argv)
    try:
        result = log_tail(args.base_dir) if args.operation == 'logs' else inspect_receipt(
            args.base_dir, role=args.role, expected_task_id=args.expected_task_id)
        if args.operation == 'deadlines':
            result = deadline_plan(result)
    except (OSError, ValueError):
        result = {'status': 'failed', 'reason': 'log_unavailable_or_unsafe'}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result.get('status') in {'failed', 'pending_confirmation', 'blocked', 'action_required'} else 0


if __name__ == '__main__':
    raise SystemExit(main())
