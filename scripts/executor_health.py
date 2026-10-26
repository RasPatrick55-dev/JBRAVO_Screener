"""Inert executor readiness, per-role receipts and bounded sanitized log inspection.

No credentials, application imports, database calls, subprocesses or broker calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
from urllib.parse import urlsplit

from scripts import pipeline_postflight as pipeline
from scripts import signal_handoff

ROOT = Path(__file__).resolve().parents[1]
MAX_AGE_SECONDS = 43200  # Same UTC day, at most 12 hours; overnight preparation.
MAX_BYTES = 64 * 1024
MAX_CANDIDATES = 10000
MAX_OBSERVATIONS = 64
ROLES = ('premarket', 'reconciliation', 'diagnostic', 'dry_run', 'unresolved')
SOURCES = ('scripts/execute_trades.py', 'scripts/executor_health.py',
           'scripts/db_queries.py', 'scripts/pipeline_postflight.py', 'scripts/signal_handoff.py',
           'scripts/activity_ingestion_checkpoint.py')
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


# Shared exact representation used by both producer and consumer.
candidate_values = signal_handoff.candidate_values


def is_paper_endpoint(value):
    """Accept only the canonical HTTPS paper trading origin, never substrings."""
    if not isinstance(value, str) or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == 'https' and parsed.hostname == 'paper-api.alpaca.markets'
                and parsed.port in (None, 443) and parsed.username is None and parsed.password is None
                and parsed.path in ('', '/') and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def preflight(base_dir=ROOT, *, now=None, rows=None, prior_binding=None,
              prior_value_binding=None,
              max_age_seconds=MAX_AGE_SECONDS, require_session=False, for_entry=False,
              paper_research_entries=False, paper_base_url=None):
    """Inspect legacy health or require a completed, frozen session handoff.

    The executor explicitly requires session/completion; legacy mode only keeps
    historical diagnostics compatible. No mode proves independent data quality
    or holds a database snapshot through submission.
    """
    now = utc(now or datetime.now(timezone.utc))
    root = Path(base_dir)
    result = {'status': 'blocked', 'reasons': [], 'checked_at': now.isoformat(),
              'max_age_seconds': max_age_seconds, 'research_qualified': False,
              'operating_acceptance': False}
    try:
        if type(paper_research_entries) is not bool:
            raise ValueError('paper_research_option_invalid')
        if paper_research_entries and (not require_session or not is_paper_endpoint(paper_base_url)):
            raise ValueError('paper_research_requires_bound_paper_session')
        health = pipeline.check_health(root, now=now, expected_day=now.date(),
                                       max_age_seconds=max_age_seconds, require_session=require_session,
                                       require_completion=require_session)
        if health.get('route') == 'pullback_reclaim_strategy':
            # Preserve the next-regular-open model. This executor's existing
            # 04:00 passive-limit policy cannot execute those proposals faithfully.
            result['reasons'] = [health['entry_block_reason']]
            result.update(strategy_screen_ready=True, entry_ready=False,
                          candidate_count=health.get('candidate_count'))
            return result
        if health.get('readiness_policy') == pipeline.PIPELINE_ONLY:
            # Explicit paper testing may waive only the intentional no-entry
            # policy. Source/history/completion/session and operational failures
            # still block; the original pipeline/research verdict is unchanged.
            if (not paper_research_entries or health.get('pipeline_ready') is not True
                    or health.get('entry_ready') is not False or health.get('status') != 'failed'
                    or health.get('failures') != ['new_entries_blocked_by_pipeline_only_policy']
                    or health.get('qualifications') != []):
                result['reasons'] = ['pipeline_new_entries_not_authorized']
                return result
            result.update(entry_authorization='paper_research_test', paper_only=True,
                          pipeline_entry_ready=False, research_status=health.get('research_status'),
                          research_reasons=health.get('research_reasons', []))
            health = {**health, 'status': 'ok'}
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
        if record.get('session') is not None:
            signal_handoff.validate_session(record['session'], started=record['started_at'],
                finished=record['finished_at'], now=now, for_entry=for_entry)
            result.update(session=record['session'], frozen_candidates=record['frozen_candidates'])
            result.update(preparation_started_at=record['started_at'],
                          preparation_finished_at=record['finished_at'])
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
                # Legacy diagnostics allow previous-session labels. Session mode
                # additionally validates run_date/bar date and producer values below.
                identities.append({'symbol': symbol, 'run_ts_utc': expected.isoformat()})
            result['candidate_binding'] = binding(json.dumps(
                sorted(identities, key=lambda r: r['symbol']), sort_keys=True).encode())
            result['candidate_count'] = len(rows)
            result['candidate_value_binding'] = candidate_values(rows)
            if record.get('session') is not None:
                actual = signal_handoff.freeze_candidates(rows, record['session'], run_ts=expected)
                if actual != record['frozen_candidates']:
                    raise ValueError('producer_candidate_values_changed')
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


_POLICY_REFUSAL_CODES = frozenset((
    'TRADE_ALLOCATION_ACTIVE_CASE_INVALID TRADE_ALLOCATION_APPROVAL_INVALID '
    'TRADE_ALLOCATION_APPROVAL_REUSED TRADE_ALLOCATION_APPROVAL_TIME '
    'TRADE_ALLOCATION_AUDIT_HISTORY_INCOMPLETE TRADE_ALLOCATION_AUDIT_HISTORY_INVALID '
    'TRADE_ALLOCATION_AUDIT_INVALID TRADE_ALLOCATION_CASE_CAS_FAILED TRADE_ALLOCATION_CASE_CONFLICT '
    'TRADE_ALLOCATION_CLAIMS_INVALID TRADE_ALLOCATION_CONSERVATION_FAILED '
    'TRADE_ALLOCATION_DECIMAL_INVALID TRADE_ALLOCATION_DOCUMENT_BOUND TRADE_ALLOCATION_EVIDENCE_INVALID '
    'TRADE_ALLOCATION_FILL_CHANGED TRADE_ALLOCATION_FILL_COUNT TRADE_ALLOCATION_FILL_DOCUMENT '
    'TRADE_ALLOCATION_FILL_DOMAIN TRADE_ALLOCATION_FILL_DUPLICATE_OR_UNKNOWN '
    'TRADE_ALLOCATION_FILL_FACT_CHANGED TRADE_ALLOCATION_FILL_FINGERPRINT '
    'TRADE_ALLOCATION_FILL_IDENTITY TRADE_ALLOCATION_FILL_IDENTITY_CONFLICT '
    'TRADE_ALLOCATION_FILL_OVERSIZE TRADE_ALLOCATION_FILL_SNAPSHOT_INCOMPLETE '
    'TRADE_ALLOCATION_GLOBAL_OWNER_CONFLICT TRADE_ALLOCATION_HISTORY_BOUND '
    'TRADE_ALLOCATION_IDENTITY_MISMATCH TRADE_ALLOCATION_MANIFEST_BINDING '
    'TRADE_ALLOCATION_MANIFEST_DIGEST TRADE_ALLOCATION_MANIFEST_SCOPE '
    'TRADE_ALLOCATION_MIGRATION_PENDING TRADE_ALLOCATION_ORIGINAL_VISIBILITY_INVALID '
    'TRADE_ALLOCATION_PARTIAL_MIGRATION TRADE_ALLOCATION_POLICY_BINDING_CHANGED '
    'TRADE_ALLOCATION_POLICY_CAS_FAILED TRADE_ALLOCATION_POPULATION_CENSORED '
    'TRADE_ALLOCATION_REPLAY_SUPERSEDED TRADE_ALLOCATION_RESERVED_OWNER_CONFLICT '
    'TRADE_ALLOCATION_ROW_CHANGED TRADE_ALLOCATION_ROW_INVALID TRADE_ALLOCATION_ROW_OVERSIZE '
    'TRADE_ALLOCATION_SCHEMA_MARKER_INVALID TRADE_ALLOCATION_SCHEMA_MARKER_MISSING '
    'TRADE_ALLOCATION_SCHEMA_METADATA TRADE_ALLOCATION_SCHEMA_VISIBILITY_INVALID '
    'TRADE_ALLOCATION_TARGET_INVALID TRADE_ALLOCATION_TRADE_EXCLUDED '
    'TRADE_ALLOCATION_TRANSACTION_NOT_IDLE TRADE_ALLOCATION_TRIGGER_ATTACHMENT_CHANGED '
    'TRADE_ALLOCATION_TRIGGER_ATTACHMENT_MISSING TRADE_ALLOCATION_TRIGGER_FUNCTION_CHANGED '
    'TRADE_ALLOCATION_TRIGGER_FUNCTION_MISSING TRADE_ALLOCATION_TRIGGER_SESSION_INVALID '
    'TRADE_ALLOCATION_UUID_INVALID TRADE_POLICY_ACCOUNTING_INCOMPLETE '
    'TRADE_POLICY_ALLOCATION_ACCOUNTING_INCOMPLETE TRADE_POLICY_ALLOCATION_CASE_CONFLICT '
    'TRADE_POLICY_ALLOCATION_IDENTITY_MISMATCH TRADE_POLICY_ALLOCATION_ORDER_CLAIMS_INVALID '
    'TRADE_POLICY_ALLOCATION_ORDER_CLAIMS_UNAVAILABLE TRADE_POLICY_ALLOCATION_ORDER_OWNER_CONFLICT '
    'TRADE_POLICY_APPROVAL_INVALID TRADE_POLICY_APPROVAL_REUSED TRADE_POLICY_APPROVAL_TIME_INVALID '
    'TRADE_POLICY_ARTIFACT_INPUT_CHANGED TRADE_POLICY_ARTIFACT_UNVALIDATED TRADE_POLICY_AUDIT_INVALID '
    'TRADE_POLICY_CASE_CAS_FAILED TRADE_POLICY_CASE_CONFLICT TRADE_POLICY_CASE_INVALID '
    'TRADE_POLICY_DB_UNAVAILABLE TRADE_POLICY_DIGEST_INVALID TRADE_POLICY_DOCUMENT_DUPLICATE_KEY '
    'TRADE_POLICY_DOCUMENT_INVALID TRADE_POLICY_EVIDENCE_INVALID TRADE_POLICY_EVIDENCE_MISMATCH '
    'TRADE_POLICY_EVIDENCE_SCOPE_INVALID TRADE_POLICY_IDENTITY_INVALID TRADE_POLICY_IDENTITY_MISMATCH '
    'TRADE_POLICY_ISOLATION_INVALID TRADE_POLICY_METRICS_STALE_OR_UNVALIDATED '
    'TRADE_POLICY_METRICS_UNAVAILABLE TRADE_POLICY_PAPER_CONTEXT_REQUIRED '
    'TRADE_POLICY_PERFORMANCE_INPUT_OR_PAYLOAD_CHANGED TRADE_POLICY_PERFORMANCE_PROVENANCE_REQUIRED '
    'TRADE_POLICY_POLICY_CAS_FAILED TRADE_POLICY_REVISION_CHANGED TRADE_POLICY_REVISION_INVALID '
    'TRADE_POLICY_ROW_CHANGED TRADE_POLICY_RUNTIME_IDENTITY_ALREADY_BOUND '
    'TRADE_POLICY_RUNTIME_IDENTITY_UNBOUND TRADE_POLICY_RUNTIME_IDENTITY_UNVERIFIED '
    'TRADE_POLICY_SCOPE_INVALID TRADE_POLICY_SCOPE_REQUIRED TRADE_POLICY_SQL_ALIAS_INVALID '
    'TRADE_POLICY_TIME_INVALID TRADE_POLICY_TRADE_EXCLUDED TRADE_POLICY_TRADE_ID_INVALID '
    'TRADE_POLICY_TRADE_MISSING TRADE_POLICY_TRADE_NOT_CLOSED TRADE_POLICY_TRANSACTION_NOT_IDLE '
    'TRADE_POLICY_UNAVAILABLE TRADE_POLICY_WRITE_TRANSACTION_REQUIRED '
).split())
_POLICY_SQLSTATE_CATEGORIES = {
    '42501': 'INSUFFICIENT_PRIVILEGE', '42P01': 'UNDEFINED_TABLE',
    '42703': 'UNDEFINED_COLUMN', '55P03': 'LOCK_UNAVAILABLE',
    '57014': 'QUERY_CANCELED', '25001': 'ACTIVE_TRANSACTION',
    '25006': 'READ_ONLY_TRANSACTION', '25P02': 'FAILED_TRANSACTION',
    '08000': 'CONNECTION_ERROR', '08003': 'CONNECTION_MISSING',
    '08006': 'CONNECTION_FAILURE', '08P01': 'PROTOCOL_VIOLATION',
    '40001': 'SERIALIZATION_FAILURE', '40P01': 'DEADLOCK'}


def policy_failure_diagnostic(error):
    """Fixed diagnostic enums only; never expose exception/configuration text."""
    result = {'refusal_code': 'UNKNOWN', 'sqlstate_category': 'UNKNOWN',
              'paper_endpoint_exact': os.environ.get('APCA_API_BASE_URL') == 'https://paper-api.alpaca.markets',
              'paper_flag_exact_true': os.environ.get('JBR_EXEC_PAPER') == 'true'}
    seen = set()
    current = error
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        try:
            kind = (type(current).__module__, type(current).__name__)
            if kind in {('scripts.trade_disposition', 'TradePolicyError'),
                        ('scripts.trade_allocation', 'AllocationError')}:
                args = current.args
                if (type(args) is tuple and len(args) == 1 and type(args[0]) is str
                        and len(args[0]) <= 96 and args[0] in _POLICY_REFUSAL_CODES
                        and result['refusal_code'] in ('UNKNOWN', 'TRADE_POLICY_UNAVAILABLE')):
                    result['refusal_code'] = args[0]
            state = getattr(current, 'pgcode', None)
            if type(state) is str and state in _POLICY_SQLSTATE_CATEGORIES and result['sqlstate_category'] == 'UNKNOWN':
                result['sqlstate_category'] = _POLICY_SQLSTATE_CATEGORIES[state]
            # db.trade_policy raises from None; its context still holds the
            # original driver/overlay exception. Inspect only fixed enums.
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        except Exception:
            # Diagnostic extraction must never replace the original held result.
            break
    return result


def _policy_failure_payload(value):
    """Copy only the four validated enum/boolean fields into run-bound receipts."""
    if type(value) is not dict or set(value) != {
            'refusal_code', 'sqlstate_category', 'paper_endpoint_exact', 'paper_flag_exact_true'}:
        return None
    code, category = value['refusal_code'], value['sqlstate_category']
    if (type(code) is not str or code not in _POLICY_REFUSAL_CODES | {'UNKNOWN'}
            or type(category) is not str or category not in set(_POLICY_SQLSTATE_CATEGORIES.values()) | {'UNKNOWN'}
            or type(value['paper_endpoint_exact']) is not bool
            or type(value['paper_flag_exact_true']) is not bool):
        return None
    return dict(value)


RECONCILIATION_FAILURE_CODES = frozenset({
    'ACTIVITY_CHECKPOINT_SOURCE_INVALID',
    'DB_UNAVAILABLE', 'POLICY_UNAVAILABLE', 'WATERMARK_READ_FAILED',
    'WATERMARK_STATE_INVALID', 'WATERMARK_TIMESTAMP_INVALID',
    'WATERMARK_TIMESTAMP_FUTURE', 'OPEN_TRADES_READ_FAILED',
    'OPEN_EXIT_LINEAGE_UNAVAILABLE', 'TRADING_CLIENT_UNAVAILABLE',
    'ORDERS_INVALID', 'ORDERS_SATURATED', 'ORDERS_READ_FAILED',
    'ORDERS_READ_INCOMPLETE', 'POSITIONS_UNAVAILABLE',
    'POSITION_CLOSE_REFUSED', 'FILL_ALLOCATION_UNBOUND_OR_AMBIGUOUS',
    'PARTIAL_EXIT_QUANTITY_UNSUPPORTED', 'OPEN_EXIT_CLOSE_REFUSED',
    'FLAT_EXIT_CHILD_UNRESOLVED', 'CLOSED_TRADES_READ_FAILED',
    'HISTORICAL_EXIT_UNBOUND', 'BOUND_EXIT_FILL_NOT_FOUND',
    'EXIT_DECORATION_REFUSED', 'UNRESOLVED_REMAINS',
    'POLICY_REVISION_CHANGED', 'POLICY_PUBLICATION_UNAVAILABLE',
})
MAX_RECONCILIATION_FAILURE_FINDINGS = 64
MAX_RECONCILIATION_FAILURE_BYTES = 16 * 1024
_RECONCILIATION_COUNT_MAX = 2**63 - 1


def new_reconciliation_failure_evidence():
    """Counts are guard observations, not unique trades, fills or financial totals."""
    return dict(schema_version=1, count_semantics='GUARD_OCCURRENCES', total_count=0,
                reason_counts={}, findings=[], omitted_count=0, unattributed_count=0,
                truncated=False, invalid=False)


def record_reconciliation_failure(evidence, reason_code, trade_id=None):
    """Record fixed labels/IDs only; diagnostic problems never replace a held result."""
    if type(evidence) is not dict:
        return
    try:
        if type(reason_code) is not str or reason_code not in RECONCILIATION_FAILURE_CODES:
            evidence['invalid'] = True
            return
        if type(trade_id) is not int or not 0 < trade_id <= _RECONCILIATION_COUNT_MAX:
            trade_id = None
        evidence['total_count'] += 1
        counts = evidence['reason_counts']
        counts[reason_code] = counts.get(reason_code, 0) + 1
        if trade_id is None:
            evidence['unattributed_count'] += 1
        for finding in evidence['findings']:
            if finding['reason_code'] == reason_code and finding['trade_id'] == trade_id:
                finding['occurrences'] += 1
                return
        if len(evidence['findings']) < MAX_RECONCILIATION_FAILURE_FINDINGS:
            evidence['findings'].append(dict(reason_code=reason_code, trade_id=trade_id,
                                             occurrences=1))
        else:
            evidence['omitted_count'] += 1
            evidence['truncated'] = True
    except Exception:
        evidence['invalid'] = True


def _validated_reconciliation_failure_payload(value):
    """Fail closed to a small INVALID marker, never copy arbitrary producer data."""
    invalid = dict(schema_version=1, status='INVALID', evidence_complete=False)
    keys = {'schema_version', 'count_semantics', 'total_count', 'reason_counts',
            'findings', 'omitted_count', 'unattributed_count', 'truncated', 'invalid'}
    count = lambda v: type(v) is int and 0 <= v <= _RECONCILIATION_COUNT_MAX
    if (type(value) is not dict or set(value) != keys
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or type(value['count_semantics']) is not str or value['count_semantics'] != 'GUARD_OCCURRENCES'
            or type(value['invalid']) is not bool or value['invalid']
            or type(value['truncated']) is not bool
            or any(not count(value[k]) for k in ('total_count', 'omitted_count', 'unattributed_count'))
            or value['unattributed_count'] > value['total_count']
            or value['truncated'] != (value['omitted_count'] > 0)
            or type(value['reason_counts']) is not dict
            or len(value['reason_counts']) > len(RECONCILIATION_FAILURE_CODES)
            or type(value['findings']) is not list
            or len(value['findings']) > MAX_RECONCILIATION_FAILURE_FINDINGS):
        return invalid
    counts = value['reason_counts']
    if (any(type(k) is not str or k not in RECONCILIATION_FAILURE_CODES
            or not count(v) or v == 0 for k, v in counts.items())
            or sum(counts.values()) != value['total_count']):
        return invalid
    copied, pairs, represented = [], set(), {}
    unattributed = 0
    for finding in value['findings']:
        if type(finding) is not dict or set(finding) != {'reason_code', 'trade_id', 'occurrences'}:
            return invalid
        code, trade_id, occurrences = (finding[k] for k in ('reason_code', 'trade_id', 'occurrences'))
        if (type(code) is not str or code not in counts or not count(occurrences) or occurrences == 0
                or trade_id is not None and (type(trade_id) is not int or not 0 < trade_id <= _RECONCILIATION_COUNT_MAX)
                or (code, trade_id) in pairs):
            return invalid
        pairs.add((code, trade_id))
        represented[code] = represented.get(code, 0) + occurrences
        if trade_id is None:
            unattributed += occurrences
        copied.append(dict(reason_code=code, trade_id=trade_id, occurrences=occurrences))
    if (sum(represented.values()) + value['omitted_count'] != value['total_count']
            or any(n > counts[code] for code, n in represented.items())
            or not unattributed <= value['unattributed_count'] <= unattributed + value['omitted_count']):
        return invalid
    result = dict(schema_version=1, status='VALID', count_semantics='GUARD_OCCURRENCES',
                  trade_id_semantics='GUARD_CONTEXT_NOT_FILL_OWNERSHIP',
                  total_count=value['total_count'], reason_counts=dict(counts), findings=copied,
                  omitted_count=value['omitted_count'], unattributed_count=value['unattributed_count'],
                  truncated=value['truncated'], evidence_complete=not value['truncated'])
    if len(json.dumps(result, sort_keys=True, separators=(',', ':')).encode('utf-8')) > MAX_RECONCILIATION_FAILURE_BYTES:
        return invalid
    return result


def reconciliation_failure_payload(value, *, failed=False):
    """Completeness concerns recorded guards only; never infer a healthy workflow."""
    try:
        result = _validated_reconciliation_failure_payload(value)
        if type(failed) is not bool:
            return dict(schema_version=1, status='INVALID', evidence_complete=False)
        if failed and result['status'] == 'VALID' and result['total_count'] == 0:
            result['status'] = 'UNCLASSIFIED'
            result['evidence_complete'] = False
        return result
    except Exception:
        return dict(schema_version=1, status='INVALID', evidence_complete=False)


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
    reconciliation = metrics.get('reconciliation')
    if rc != 0 and type(reconciliation) is dict and reconciliation.get('reason') == 'POLICY_UNAVAILABLE':
        diagnostic = _policy_failure_payload(reconciliation.get('policy_failure'))
        if diagnostic is not None:
            selected['policy_failure'] = diagnostic
    if role == 'reconciliation' and type(reconciliation) is dict and 'failure_evidence' in reconciliation:
        selected['reconciliation_failure'] = reconciliation_failure_payload(reconciliation['failure_evidence'], failed=(rc != 0))
    record = {'schema_version': 1, 'run_id': run_id, 'role': role,
              'started_at': utc(started_at).isoformat(), 'finished_at': utc(finished_at).isoformat(),
              'process_rc': rc, 'health': health, 'metrics': selected, 'preflight': gate,
              'order_observations': observations, 'observations_truncated': observations_truncated,
              'pending_terminal_confirmations': pending, 'legacy_orders_canceled_is_not_confirmation': True,
              'unobserved_submissions': unknown,
              'cancellation_policy': ('frozen_premarket_session_0435' if getattr(config,'strategy_premarket_entries',False)
                                      else 'minutes_from_submission'),
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
                  'source', 'dry_run', 'diagnostic', 'reconcile_only', 'paper_research_entries',
                  'strategy_premarket_entries', 'submit_at_ny',
                  'time_window', 'extended_hours', 'allocation_pct', 'min_order_usd',
                  'max_new_positions', 'disable_open_position_cap', 'trailing_percent',
                  'price_source', 'limit_buffer_pct', 'max_gap_pct')},
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
            if state['state'] == 'cancel_requested':
                # Confirmation is already due after acknowledgment; no deadline
                # is needed to decide whether to request cancellation again.
                action = 'confirm_existing_request'
            else:
                if state['deadline'] is None:
                    raise ValueError('missing_cancellation_deadline')
                action = 'cancel_and_confirm' if now >= state['deadline'] else 'wait_until_deadline'
            planned.append({'order_id': order_id, 'symbol': state['symbol'],
                            'deadline': state['deadline'].isoformat() if state['deadline'] is not None else None,
                            'proposed_action': action,
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
