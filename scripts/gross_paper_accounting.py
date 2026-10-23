"""Explicit PAPER gross projection. No persistence, clients or activation.

The caller must obtain Policy and eligible rows in one validated read-only
policy transaction. This view never supplies net P&L or replaces original rows.
"""
from decimal import Decimal, InvalidOperation, localcontext
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
import re
import stat
import tempfile
from scripts import trade_disposition as D
from scripts import trade_allocation as A

BASIS = 'PAPER_GROSS_RECORDED_PRICES_AND_AUDITED_FILLS_V1'
MAX_ROWS = 10000
REPORT_PATH = 'data/paper_gross_performance.json'
MAX_REPORT_BYTES = 4 * 1024 * 1024
SOURCE_PATHS = ('scripts/gross_paper_accounting.py', 'scripts/db.py',
    'scripts/trade_disposition.py', 'scripts/trade_allocation.py',
    'scripts/metrics.py', 'scripts/trade_performance.py',
    'dashboards/data_io.py', 'dashboards/dashboard_app.py',
    'scripts/primary_pipeline.py', 'scripts/run_pipeline.py', 'scripts/pipeline_postflight.py')


def _require(condition, code):
    if not condition:
        D.fail('GROSS_' + code)


def _number(value, *, positive=False, nonzero=False):
    _require(type(value) in (int, float, str, Decimal), 'NUMBER_INVALID')
    _require(type(value) is not str or len(value) <= 4096, 'NUMBER_INVALID')
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        D.fail('GROSS_NUMBER_INVALID')
    _require(number.is_finite() and (not positive or number > 0)
             and (not nonzero or number != 0), 'NUMBER_INVALID')
    return D.document({'number': number})['number']


def _money_text(number):
    text = format(number, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if number == 0 else text


def _sum_precision(numbers):
    # A common decimal scale, plus the possible carry from the operand count.
    # Derive the bound from values, including widely separated exponents.
    nonzero = [number for number in numbers if number != 0]
    if not nonzero:
        return 1
    high = max(number.adjusted() for number in nonzero)
    low = min(number.as_tuple().exponent for number in nonzero)
    return max(1, high - low + 1 + len(str(len(nonzero))))


def _gross_from_prices(exit_price, entry, qty):
    # Subtraction needs one possible carry; multiplication concatenates at
    # most the two coefficient lengths. No ambient precision is used.
    with localcontext() as context:
        context.prec = _sum_precision([exit_price, entry]) + len(qty.as_tuple().digits)
        return (exit_price - entry) * qty


def _gross_sum(numbers):
    with localcontext() as context:
        context.prec = _sum_precision(numbers)
        return sum(numbers, Decimal(0))


def build_snapshot(policy, records, *, expected_identity, expected_revision):
    """Return gross-only evidence; missing input fails rather than becoming0.

    Prices on ordinary rows are observations, not independent fill proof.
    Validated multi-fill allocations retain separate lineage and never invent
    a scalar exit price or mutate the original trade/P&L/event records.
    """
    _require(type(policy) is D.Policy and policy.scope == 'VALIDATED_PERFORMANCE', 'POLICY_REQUIRED')
    _require((policy.deployment_id, policy.database_id) == D.identity(*expected_identity), 'IDENTITY_MISMATCH')
    _require(D.revision(policy.revision) == D.revision(expected_revision), 'REVISION_CHANGED')
    _require(type(records) is list and len(records) <= MAX_ROWS, 'ROW_CAP_OR_INPUT_INVALID')
    overlay = D.document(policy.allocation_metadata_json)
    claimed = overlay.get('binding_fingerprint')
    _require(claimed == D.fingerprint({k: v for k, v in overlay.items() if k != 'binding_fingerprint'}), 'OVERLAY_BINDING_INVALID')
    _require(overlay.get('status') in ('VALIDATED', 'MIGRATION_PENDING')
             and overlay.get('deployment_id') == policy.deployment_id
             and overlay.get('database_id') == policy.database_id
             and overlay.get('policy_revision') == policy.revision, 'OVERLAY_IDENTITY_INVALID')
    _require(type(overlay.get('active_count')) is int and overlay.get('active_count') == len(policy.allocations)
             and overlay.get('allocations') == [a.metadata_dict() for a in policy.allocations], 'OVERLAY_POPULATION_INVALID')
    _require(overlay['status'] != 'MIGRATION_PENDING' or not policy.allocations, 'OVERLAY_UNAVAILABLE')
    allocations = {}
    for allocation in policy.allocations:
        _require(type(allocation) is A.Allocation and allocation.state == 'ACTIVE'
                 and allocation.financial_eligible is False and allocation.fees is None
                 and allocation.net is None, 'ALLOCATION_INVALID')
        trade_id = D.positive_id(allocation.trade_id)
        _require(trade_id not in allocations and trade_id not in policy.excluded_ids, 'ALLOCATION_COLLISION')
        _require(_number(allocation.entry_quantity, positive=True)
                 == _number(allocation.exit_quantity, positive=True), 'QUANTITY_NOT_CONSERVED')
        D.require_order_owner(policy, trade_id, *allocation.order_ids)
        allocations[trade_id] = allocation
    source_rows = D.document({'rows': records})['rows']
    _require(all(type(row) is dict for row in source_rows), 'ROW_INVALID')
    ids = [D.positive_id(row.get('trade_id')) for row in source_rows]
    _require(len(set(ids)) == len(ids) and not set(ids).intersection(policy.excluded_ids), 'ROW_IDENTITY_COLLISION')
    _require(set(allocations).issubset(ids), 'ALLOCATION_ROW_MISSING')
    source_rows = sorted(source_rows, key=lambda row: row['trade_id'])
    projected = []
    for row in source_rows:
        _require(row.get('status') == 'CLOSED', 'ROW_NOT_CLOSED')
        trade_id = row['trade_id']
        qty = _number(row.get('qty'), nonzero=True)
        entry = _number(row.get('entry_price'), positive=True)
        for field in ('entry_time', 'exit_time'):
            D.aware_time(row.get(field))
        if trade_id in allocations:
            allocation = allocations[trade_id]
            _require(qty.copy_abs() == allocation.entry_quantity, 'ALLOCATION_ROW_QUANTITY_MISMATCH')
            gross = _number(allocation.gross_result)
            lineage = {'kind': 'AUDITED_MULTI_FILL_GROSS', 'allocation_id': allocation.allocation_id,
                       'allocation_revision': allocation.revision, 'approval_ref': allocation.approval_ref,
                       'source_refs': list(allocation.source_refs), 'order_ids': list(allocation.order_ids),
                       'activity_ids': list(allocation.activity_ids)}
        else:
            exit_price = _number(row.get('exit_price'), positive=True)
            gross = _gross_from_prices(exit_price, entry, qty)
            lineage = {'kind': 'RECORDED_PRICE_GROSS', 'fill_provenance_proved': False}
        projected.append({'trade_id': trade_id, 'original_row_fingerprint': D.fingerprint(row),
            'gross_pnl': _money_text(gross), 'fees': None, 'fees_status': 'UNKNOWN',
            'net_pnl': None, 'net_status': 'UNKNOWN', 'net_eligible': False, 'lineage': lineage})
    gross_total = _gross_sum([Decimal(row['gross_pnl']) for row in projected])
    evidence = {'schema_version': 1, 'pnl_basis': BASIS, 'policy': policy.metadata(),
        'input_basis': 'public.trades:CLOSED:VALIDATED_PERFORMANCE:full-row',
        'input_fingerprint': D.fingerprint({'rows': source_rows}), 'input_count': len(source_rows),
        'projection_fingerprint': D.fingerprint({'rows': projected}), 'gross_input_complete': True,
        'accounting_complete': False, 'activity_ingestion_complete': False, 'recovery_complete': False,
        'fees_status': 'UNKNOWN', 'net_status': 'UNKNOWN', 'net_eligible': False}
    report = {'schema_version': 1, 'pnl_basis': BASIS, 'total_trades': len(projected),
              'gross_pnl': _money_text(gross_total), 'fees': None, 'net_pnl': None,
              'fees_status': 'UNKNOWN', 'net_status': 'UNKNOWN', 'net_eligible': False,
              'accounting_complete': False, 'activity_ingestion_complete': False,
              'recovery_complete': False, 'rows': projected, 'evidence': evidence}
    report['cache_revision'] = D.fingerprint({'evidence': evidence, 'gross_pnl': report['gross_pnl']})
    return report


def _regular(path, root):
    _require(path.resolve().is_relative_to(root), 'PATH_INVALID')
    for item in (path, *path.parents):
        info = item.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, 'st_file_attributes', 0) & 1024), 'PATH_INVALID')
        if item == root:
            break
    _require(stat.S_ISREG(path.stat().st_mode), 'FILE_INVALID')


def _bounded_read(path, root, cap):
    _regular(path, root)
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    _require(len(raw) <= cap, 'FILE_CAP')
    return raw


def source_bindings(root):
    """Fixed relevant source population; never read credentials or a glob."""
    root = Path(root).resolve(strict=True)
    result = {}
    for name in SOURCE_PATHS:
        raw = _bounded_read(root / name, root, 1024 * 1024)
        result[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return result


def _sources(value):
    _require(type(value) is dict and set(value) == set(SOURCE_PATHS), 'SOURCE_BINDING_INVALID')
    for binding in value.values():
        _require(type(binding) is dict and set(binding) == {'bytes', 'sha256'}
            and type(binding['bytes']) is int and 0 < binding['bytes'] <= 1024 * 1024
            and type(binding['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'SOURCE_BINDING_INVALID')
    return D.document(value)


def _now(now):
    value = datetime.now(timezone.utc) if now is None else now
    _require(type(value) is datetime, 'TIME_INVALID')
    D.aware_time(value.isoformat())
    return value.astimezone(timezone.utc)


def report_document(snapshot, sources, *, now=None):
    body = {'schema_version': 1, 'namespace': 'PAPER_GROSS_ONLY',
        'generated_at': _now(now).isoformat(), 'sources': _sources(sources),
        'report': D.document(snapshot)}
    body['binding_fingerprint'] = D.fingerprint(body)
    return body


def validate_report(document, snapshot, sources, *, now=None):
    """Revalidate against current DB inputs and code; never trust age/mtime alone."""
    body = D.document(document)
    _require(set(body) == {'schema_version', 'namespace', 'generated_at', 'sources',
        'report', 'binding_fingerprint'} and type(body['schema_version']) is int
        and body['schema_version'] == 1 and body['namespace'] == 'PAPER_GROSS_ONLY', 'ARTIFACT_SCHEMA')
    _require(body['binding_fingerprint'] == D.fingerprint({k: v for k, v in body.items()
        if k != 'binding_fingerprint'}), 'ARTIFACT_CHANGED')
    _require(body['sources'] == _sources(sources), 'SOURCE_CHANGED')
    _require(body['report'] == D.document(snapshot), 'ARTIFACT_INPUT_CHANGED')
    stamp = D.aware_time(body['generated_at'])
    checked_at = _now(now)
    _require(stamp <= checked_at, 'ARTIFACT_TIME_INVALID')
    # An old generation date is not a cache bypass: every read has just matched
    # the complete current row/policy/allocation fingerprints and source bytes.
    return {'validation_status': 'VALIDATED_PAPER_GROSS', 'report': body['report'],
        'generated_at': body['generated_at'], 'validated_at': checked_at.isoformat(),
        'cache_revision': body['binding_fingerprint']}


def _report_path(root, *, create=False):
    root = Path(root).resolve(strict=True)
    parent = root / 'data'
    if create:
        parent.mkdir(exist_ok=True)
    _require(parent.resolve().is_relative_to(root) and not parent.is_symlink()
        and not (getattr(parent.lstat(), 'st_file_attributes', 0) & 1024), 'PATH_INVALID')
    target = root / REPORT_PATH
    if target.exists() or target.is_symlink():
        _regular(target, root)
    return root, target


def read_report(root, current, *, now=None):
    """Caller holds validated shared policy transaction throughout this read."""
    root, target = _report_path(root)
    raw = _bounded_read(target, root, MAX_REPORT_BYTES)
    snapshot, sources = current()
    return validate_report(raw.decode('utf-8'), snapshot, sources, now=now)


def publish_report(root, current, *, now=None):
    """Atomic sidecar only. No database, original-row or legacy net writes."""
    root, target = _report_path(root, create=True)
    snapshot, sources = current()
    document = report_document(snapshot, sources, now=now)
    raw = D.canonical(document).encode('utf-8')
    _require(len(raw) <= MAX_REPORT_BYTES, 'FILE_CAP')
    descriptor, temporary = tempfile.mkstemp(prefix='.paper_gross_', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fresh_snapshot, fresh_sources = current()
        validate_report(document, fresh_snapshot, fresh_sources, now=now)
        os.replace(temporary, target)
        # A concurrent trade/source change cannot turn publication into a
        # successful financial result. All future readers also revalidate.
        return read_report(root, current, now=now)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pipeline_summary(snapshot):
    """Counts and exact gross totals only; no synthetic net or risk ratios."""
    rows = snapshot['rows']
    # These strings are already derived from validated, bounded source numbers.
    # Products and exact sums can legitimately exceed the source-number domain.
    values = [Decimal(row['gross_pnl']) for row in rows]
    total = _gross_sum(values)
    _require(snapshot['pnl_basis'] == BASIS and snapshot['evidence']['gross_input_complete'] is True,
        'PIPELINE_INPUT_INVALID')
    _require(snapshot['gross_pnl'] == _money_text(total) and snapshot['total_trades'] == len(rows),
        'PIPELINE_TOTAL_INVALID')
    return dict(pnl_basis=BASIS, total_trades=len(values), gross_pnl=_money_text(total),
        wins=sum(value > 0 for value in values), losses=sum(value < 0 for value in values),
        breakeven=sum(value == 0 for value in values), fees=None, net_pnl=None,
        fees_status='UNKNOWN', net_status='UNKNOWN', net_eligible=False,
        accounting_complete=False, recovery_complete=False, activity_ingestion_complete=False)


def pipeline_document(snapshot, sources, *, run_date, candidate_count, candidate_fingerprint,
                      publication_id, now=None):
    """Authoritative gross daily publication; candidate statistics retain their backtest basis."""
    _require(type(run_date) is str and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', run_date), 'RUN_DATE_INVALID')
    from datetime import date
    _require(date.fromisoformat(run_date).isoformat() == run_date, 'RUN_DATE_INVALID')
    _require(type(candidate_count) is int and 0 < candidate_count <= MAX_ROWS, 'CANDIDATE_CAP_OR_EMPTY')
    _require(type(candidate_fingerprint) is str and re.fullmatch('[0-9a-f]{64}', candidate_fingerprint), 'CANDIDATE_BINDING_INVALID')
    _require(type(publication_id) is str and re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', publication_id), 'PUBLICATION_ID_INVALID')
    body = dict(schema_version=1, namespace='PAPER_GROSS_PIPELINE_V1', run_date=run_date,
        publication_id=publication_id, published_at=_now(now).isoformat(),
        sources=_sources(sources), report=D.document(snapshot), summary=pipeline_summary(snapshot),
        candidate_count=candidate_count, candidate_fingerprint=candidate_fingerprint,
        candidate_basis='BACKTEST_RESULTS_UNCHANGED', entry_ready=False, operating_acceptance=False)
    body['binding_fingerprint'] = D.fingerprint(body)
    _require(len(D.canonical(body).encode('utf-8')) <= MAX_REPORT_BYTES, 'FILE_CAP')
    return body


def validate_pipeline_document(document, snapshot, sources, *, run_date, candidate_count,
                               candidate_fingerprint, now=None, not_before=None):
    body = D.document(document)
    expected = pipeline_document(snapshot, sources, run_date=run_date,
        candidate_count=candidate_count, candidate_fingerprint=candidate_fingerprint,
        publication_id=body.get('publication_id'), now=D.aware_time(body.get('published_at')))
    _require(body == expected, 'PIPELINE_PUBLICATION_CHANGED')
    stamp = D.aware_time(body['published_at'])
    _require(stamp <= _now(now) and (not_before is None or stamp >= not_before), 'PIPELINE_PUBLICATION_STALE')
    return body


def pipeline_completion(document):
    """Compact evidence produced only after current DB/source/candidate validation."""
    report = document['report']
    evidence = report['evidence']
    return dict(schema_version=1, pnl_basis=BASIS, run_date=document['run_date'],
        publication_id=document['publication_id'], published_at=document['published_at'],
        publication_fingerprint=document['binding_fingerprint'],
        input_fingerprint=evidence['input_fingerprint'],
        sources_fingerprint=D.fingerprint(document['sources']),
        candidate_fingerprint=document['candidate_fingerprint'],
        candidate_count=document['candidate_count'], total_trades=report['total_trades'],
        policy_revision=evidence['policy']['policy_revision'], gross_input_complete=True,
        fees_status='UNKNOWN', net_status='UNKNOWN', net_eligible=False,
        accounting_complete=False, recovery_complete=False, activity_ingestion_complete=False,
        entry_ready=False, operating_acceptance=False)
