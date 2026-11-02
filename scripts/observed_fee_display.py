"""Read-time, source-pinned CRC110 observed-cost display. SELECTs only.

No registry, migration, activation, provider calls or configuration discovery.
The owner-approved allocation rule derives a bounded observed-cost result, not
broker-supplied trade fees, canonical account accounting or validated performance.
Changing this evidence/rule/cutoff requires a new reviewed code/data release.
"""
from datetime import datetime, timezone
from pathlib import Path
from scripts import allocation_net_accounting as N, cohort_posted_costs as C
from scripts import trade_allocation as A, trade_disposition as D

EVIDENCE_SHA256 = '6856c384cd0d76d185a2b312f020f4f9e665969ad0182f2a18925b91e3f5e01b'
CALCULATION_SHA256 = '9bd45667fa557aa4a043920bae1ed515fc1817684b94e370fd99e22e17b0125b'
# Canonical policy/allocation/ownership/full-row binding from the retained
# successful read-only preview. This is not a broker-account UUID attestation.
CURRENT_BINDING_SHA256 = 'ea561075fff5dc22fd245df979d95ef48be1de3568923499de1ecb258cc3b0d5'
RULE_SHA256 = '51f51f1b83c3e1e60d6f5df3ecf6e580e35a49db5c559735c55281c2ffd56f53'
EVIDENCE_PATH = Path(__file__).with_name('crc110_observed_fees.json')


def require(value, reason):
    if not value:
        raise N.QualificationError('OBSERVED_FEE_DISPLAY_' + reason)


def wire_time(value):
    if isinstance(value, datetime):
        require(value.utcoffset() is not None, 'TIME_INVALID')
        return value.astimezone(timezone.utc).isoformat()
    return N._time(value).astimezone(timezone.utc).isoformat()


def read_for_history(cursor, physical_rows, identity, *, now=None):
    """Caller supplies READ ONLY transaction; no INSERT/UPDATE/DDL or fallback.

    The canonical policy reader holds its shared transaction advisory lock under
    READ COMMITTED. The full row must match the earlier history projection too.
    Any later change is checked again on the next read; no cached result is used.
    """
    now = now or datetime.now(timezone.utc)
    targets = [row for row in physical_rows if row.get('trade_id') == 110]
    if not targets:
        return {'status': 'NOT_REQUESTED'}, None
    require(len(targets) == 1, 'DUPLICATE_HISTORY_ROW')
    require(EVIDENCE_PATH.stat().st_size <= 65536, 'EVIDENCE_OVERSIZE')
    raw = EVIDENCE_PATH.read_bytes()
    calculation = C.derive(raw, expected_file_sha256=EVIDENCE_SHA256)
    require(D.fingerprint(calculation) == CALCULATION_SHA256, 'CALCULATION_CHANGED')

    policy = D.read_policy(cursor, identity, scope='VALIDATED_PERFORMANCE')
    cursor.execute("SELECT allocation_id::text,allocation_revision,manifest::text FROM public.trade_allocation_cases WHERE trade_id=110 AND state='ACTIVE' LIMIT 2")
    cases = cursor.fetchall()
    require(len(cases) == 1, 'ACTIVE_ALLOCATION_REQUIRED')
    case_id, revision, manifest_raw = cases[0]
    manifest = D.document(manifest_raw)
    core, _, _, gross, _, _ = N._execution(policy, manifest)
    require((case_id, revision) == (core['allocation_id'], core['allocation_revision']), 'ALLOCATION_CHANGED')
    require(D.fingerprint(core) == CURRENT_BINDING_SHA256, 'CURRENT_BINDING_CHANGED')
    cursor.execute('SELECT pg_catalog.to_jsonb(t)::text FROM public.trades t WHERE trade_id=110')
    result = cursor.fetchone()
    require(result is not None, 'TRADE_MISSING')
    full = D.document(result[0])
    A.validate_trade(full)
    require(full == D.document(manifest['row_snapshot']) and D.fingerprint(full) == core['original_row_sha256'], 'TRADE_CHANGED')
    physical = targets[0]
    for key in ('trade_id', 'symbol', 'status', 'exit_reason'):
        require(physical.get(key) == full.get(key), 'HISTORY_ROW_CHANGED')
    for key in ('qty', 'entry_price', 'exit_price', 'realized_pnl'):
        left, right = physical.get(key), full.get(key)
        require((left is None and right is None) or
                (left is not None and right is not None and N._decimal(left) == N._decimal(right)), 'HISTORY_ROW_CHANGED')
    for key in ('entry_time', 'exit_time', 'updated_at'):
        require(wire_time(physical.get(key)) == wire_time(full.get(key)), 'HISTORY_ROW_CHANGED')
    fee = N._decimal(calculation['allocated_observed_fee_cost'])
    require(gross == N._decimal(calculation['gross_pnl']) and fee >= 0, 'GROSS_OR_COST_CHANGED')
    net = N._sum([gross, fee.copy_negate()])
    require(net == N._decimal(calculation['derived_net_after_observed_posted_fees']), 'NET_CHANGED')
    digest = D.fingerprint(dict(binding=core, evidence_sha256=EVIDENCE_SHA256,
                               calculation_sha256=CALCULATION_SHA256, rule_sha256=RULE_SHA256))
    binding = dict(trade_id=110, policy_revision=policy.revision, allocation_id=case_id,
                   allocation_revision=revision, manifest_sha256=core['manifest_sha256'], fills_sha256=core['fills_sha256'],
                   row_fingerprint=D.fingerprint(full), row_revision=wire_time(full['updated_at']),
                   evidence_file_sha256=EVIDENCE_SHA256, rule_sha256=RULE_SHA256,
                   cutoff=calculation['query_creation_cutoff'], record_sha256=digest,
                   source_binding_sha256=CURRENT_BINDING_SHA256)
    wire = {key: physical.get(key) for key in ('trade_id', 'symbol', 'status', 'qty', 'entry_price', 'exit_price', 'realized_pnl')}
    wire.update({key: wire_time(physical[key]) for key in ('entry_time', 'exit_time', 'updated_at')})
    wire['row_fingerprint'] = binding['row_fingerprint']
    data = dict(status='APPROVED_SCOPED_OBSERVED_DERIVATION', source='postgres_allocation_pinned_observed_costs',
                trade_id=110, basis=C.BASIS, rule=C.RULE, execution_binding=calculation['execution_binding'],
                legacy_validated_performance_eligible=False, canonical_account_uuid_proved=False,
                global_accounting_complete=False, entry_ready=False, record_sha256=digest,
                net_pnl=N._text(net), allocated_fee_cost=N._text(fee), as_of=binding['cutoff'],
                binding=binding, bound_row=wire)
    context = dict(schema_version=1, status='CURRENT_BINDINGS_VERIFIED', source='postgres_current_observed_cost_read',
                   evidence_status='PINNED_OWNER_APPROVED', allocation_state='ACTIVE', checked_at=now.isoformat(), binding=binding)
    return context, data
