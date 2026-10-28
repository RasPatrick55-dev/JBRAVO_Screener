"""Offline, evidence-bound net accounting for the CRC110 native-fill allocation.

No clients, configuration discovery, persistence, activation or eligibility
mutation. Acceptance hashes must come from separately reviewed authoritative
captures, not be copied from the candidate documents being evaluated. This is
an explicit input boundary, not a verifier of arbitrary provider assertions.

Recorded paper-account net and hypothetical paper-model net have different
namespaces. Neither is silently admitted to legacy VALIDATED_PERFORMANCE.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import re
from uuid import UUID
from scripts import trade_allocation as A, trade_disposition as D

RECORDED = 'ALLOCATION_RECORDED_PAPER_NET_V1'
MODEL = 'ALLOCATION_PAPER_MODEL_NET_V1'
ENDPOINT = 'https://paper-api.alpaca.markets'
MAX_COSTS = 1000
ACCOUNT_BINDING = 'CANONICAL_PAPER_ACCOUNT_UUID_V1'
EXECUTION_SET_BINDING = 'EXECUTION_SET_MATCH_ONLY_V1'


class QualificationError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise QualificationError('ALLOCATION_NET_' + code)


@dataclass(frozen=True)
class EvidenceAcceptance:
    """Caller-supplied trust anchors; this module never discovers authority."""
    source_identity_sha256: str
    cost_contract_sha256: str
    cost_capture_sha256: str
    source_fingerprint: str


def _digest(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None, 'DIGEST_INVALID')
    return value


def _time(value):
    try:
        return D.aware_time(value)
    except (D.TradePolicyError, TypeError, ValueError):
        raise QualificationError('ALLOCATION_NET_TIME_INVALID') from None


def _decimal(value):
    # No float conversion, implicit zero, nonfinite or unbounded arithmetic.
    require(type(value) in (str, Decimal, int) and type(value) is not bool, 'AMOUNT_INVALID')
    if isinstance(value, str):
        require(re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', value) is not None, 'AMOUNT_INVALID')
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        raise QualificationError('ALLOCATION_NET_AMOUNT_INVALID') from None
    require(number.is_finite() and len(number.as_tuple().digits) <= 128 and abs(number.as_tuple().exponent) <= 128, 'AMOUNT_INVALID')
    return number


def _sum(values):
    nonzero = [v for v in values if v]
    precision = 1 if not nonzero else max(v.adjusted() for v in nonzero) - min(v.as_tuple().exponent for v in nonzero) + 2 + len(str(len(values)))
    with localcontext() as context:
        context.prec = max(precision, 1)
        return sum(values, Decimal(0))


def _text(number):
    result = format(number, 'f')
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    return '0' if not number else result


def _accepted(document, expected_hash):
    _digest(expected_hash)
    def strict(value, depth=0):
        # D.document deliberately normalizes ordinary JSON floats. Accounting
        # must reject Python binary floats before that normalization loses type.
        require(depth <= 32, 'DOCUMENT_INVALID')
        require(not isinstance(value, float), 'BINARY_FLOAT_FORBIDDEN')
        if isinstance(value, dict):
            for nested in value.values():
                strict(nested, depth + 1)
        elif isinstance(value, list):
            for nested in value:
                strict(nested, depth + 1)
    strict(document)
    try:
        normalized = D.document(document)
        require(len(D.canonical(normalized).encode()) <= 1024 * 1024, 'DOCUMENT_OVERSIZE')
        require(D.fingerprint(normalized) == expected_hash, 'EVIDENCE_NOT_ACCEPTED')
        return normalized
    except D.TradePolicyError:
        raise QualificationError('ALLOCATION_NET_DOCUMENT_INVALID') from None


def _execution(policy, manifest):
    require(type(policy) is D.Policy and policy.scope == 'VALIDATED_PERFORMANCE', 'POLICY_REQUIRED')
    try:
        D.identity(policy.deployment_id, policy.database_id)
        D.revision(policy.revision)
        overlay = D.document(policy.allocation_metadata_json)
        require(overlay.get('status') == 'VALIDATED' and overlay.get('policy_revision') == policy.revision
                and (overlay.get('deployment_id'), overlay.get('database_id')) == (policy.deployment_id, policy.database_id)
                and type(overlay.get('active_count')) is int and overlay['active_count'] == len(policy.allocations)
                and overlay.get('allocations') == [item.metadata_dict() for item in policy.allocations]
                and overlay.get('binding_fingerprint') == D.fingerprint({k:v for k,v in overlay.items() if k != 'binding_fingerprint'}),
                'POLICY_BINDING_INVALID')
    except (D.TradePolicyError, AttributeError):
        raise QualificationError('ALLOCATION_NET_POLICY_BINDING_INVALID') from None
    require(110 not in policy.excluded_ids, 'TRADE_EXCLUDED')
    allocations = [item for item in policy.allocations if item.trade_id == 110]
    require(len(allocations) == 1, 'ACTIVE_ALLOCATION_REQUIRED')
    allocation = allocations[0]
    require(type(allocation) is A.Allocation and allocation.state == 'ACTIVE' and allocation.financial_eligible is False
            and allocation.fees is None and allocation.net is None
            and type(allocation.revision) is int and allocation.revision > 0, 'ALLOCATION_INVALID')
    require(type(manifest) is dict and manifest.get('trade_id') == 110 and manifest.get('allocation_id') == allocation.allocation_id
            and (manifest.get('deployment_id'), manifest.get('database_id')) == (policy.deployment_id, policy.database_id)
            and manifest.get('row_fingerprint') == A.ROW_FINGERPRINT and manifest.get('row_version') == A.ROW_VERSION
            and manifest.get('classification') == 'NATIVE_FILL_COHORT'
            and manifest.get('resolution_scope') == 'RECONCILIATION_ONLY'
            and manifest.get('source_pins') == A.SOURCE_PINS
            and manifest.get('financial_eligible') is False and manifest.get('fees') is None and manifest.get('net') is None,
            'MANIFEST_BINDING')
    try:
        A.validate_trade(manifest.get('row_snapshot'))
        fills = A.validate_fills(manifest.get('fills'))
        D.require_order_owner(policy, 110, *A.ORDERS)
    except (A.AllocationError, D.TradePolicyError):
        raise QualificationError('ALLOCATION_NET_EXECUTION_INVALID') from None
    with localcontext() as context:
        context.prec = 80  # The exact, pinned native cohort is bounded by A.
        cost = sum((A.number(f['qty']) * A.number(f['price']) for f in fills.values() if f['side'] == 'buy'), Decimal(0))
        proceeds = sum((A.number(f['qty']) * A.number(f['price']) for f in fills.values() if f['side'] == 'sell'), Decimal(0))
        gross = proceeds - cost
    require(allocation.entry_quantity == allocation.exit_quantity == Decimal(30) and allocation.gross_result == gross, 'CONSERVATION_FAILED')
    binding = dict(deployment_id=policy.deployment_id, database_id=policy.database_id, policy_revision=policy.revision,
                   allocation_id=allocation.allocation_id, allocation_revision=allocation.revision, trade_id=110,
                   manifest_sha256=D.fingerprint(manifest), original_row_sha256=D.fingerprint(manifest['row_snapshot']),
                   fills_sha256=D.fingerprint({'fills': manifest['fills']}), policy_sha256=D.fingerprint(policy.metadata()))
    times = [_time(f['transaction_time']) for f in fills.values()]
    return binding, cost, proceeds, gross, min(times), max(times)


def execution_set_binding(observations):
    """Recognize five source-pinned fills without claiming account identity.

    The plugin may omit account UUID/endpoint. That does not invalidate exact
    execution observations, but this weaker evidence cannot qualify net.
    No pagination, account coverage or authenticated-host claim is made here.
    """
    require(type(observations) is list and len(observations) <= 400, 'EXECUTION_OBSERVATIONS_INVALID')
    expected = {fill[1]: fill for fill in A.FILLS}
    matched = []
    seen = set()
    for item in observations:
        require(type(item) is dict, 'EXECUTION_OBSERVATIONS_INVALID')
        activity_id = item.get('activity_id', item.get('id'))
        if activity_id not in expected:
            continue
        require(activity_id not in seen, 'EXECUTION_OBSERVATIONS_DUPLICATE')
        seen.add(activity_id)
        _, _, side, order, qty, price, instant = expected[activity_id]
        require(item.get('activity_type') == 'FILL' and item.get('symbol') == 'CRC'
                and item.get('side') == side and item.get('order_id') == order
                and _decimal(item.get('qty')) == Decimal(qty) and _decimal(item.get('price')) == Decimal(price)
                and _time(item.get('transaction_time')) == _time(instant), 'EXECUTION_OBSERVATIONS_MISMATCH')
        matched.append(dict(activity_id=activity_id, activity_type='FILL', symbol='CRC', side=side,
                            order_id=order, qty=qty, price=price, transaction_time=instant))
    require(len(matched) == 5, 'EXECUTION_SET_INCOMPLETE')
    matched.sort(key=lambda item: item['activity_id'])
    return dict(evidence_class=EXECUTION_SET_BINDING, matched_native_fills=5,
                source_document_sha256=A.SOURCE_PINS['CRC110-SOURCE-BACKED-RESOLUTION-003.json'],
                execution_set_sha256=D.fingerprint({'fills':matched}),
                canonical_source_identity_proved=False, net_qualified=False,
                account_history_complete=False)


def qualify(policy, manifest, source_identity, cost_contract, cost_capture, *, accepted, now):
    """Calculate one auditable derived record from independently accepted evidence.

    All source/contract/capture hashes and current policy/fill bindings are
    rechecked on consumption. A bounded activity scan alone cannot provide
    posting finality or prove zero cost. These must be in the accepted contract.
    """
    require(type(accepted) is EvidenceAcceptance, 'ACCEPTANCE_REQUIRED')
    binding, entry_cost, proceeds, gross, first, last = _execution(policy, manifest)
    identity = _accepted(source_identity, accepted.source_identity_sha256)
    contract = _accepted(cost_contract, accepted.cost_contract_sha256)
    capture = _accepted(cost_capture, accepted.cost_capture_sha256)
    source = _digest(accepted.source_fingerprint)
    checked_at = _time(now)
    for document in (identity, contract, capture):
        require(document.get('schema_version') == 1 and type(document.get('schema_version')) is int
                and document.get('source_fingerprint') == source and document.get('endpoint') == ENDPOINT
                and document.get('environment') == 'paper' and document.get('currency') == 'USD'
                and document.get('asset_class') == 'us_equity', 'SOURCE_SCOPE_MISMATCH')
    require(identity.get('binding_kind') == ACCOUNT_BINDING and identity.get('endpoint_attested') is True, 'CANONICAL_ACCOUNT_REQUIRED')
    try:
        account = str(UUID(identity.get('account_uuid')))
    except (ValueError, TypeError, AttributeError):
        raise QualificationError('ALLOCATION_NET_CANONICAL_ACCOUNT_REQUIRED') from None
    require(account == identity['account_uuid'] and hashlib.sha256((ENDPOINT + '\n' + account).encode()).hexdigest() == source,
            'CANONICAL_ACCOUNT_REQUIRED')
    require(identity.get('execution_binding') == binding and identity.get('matched_native_fills') == 5
            and type(identity.get('matched_native_fills')) is int and identity.get('historical_account_match') is True
            and identity.get('reset_or_recreation_detected') is False, 'HISTORICAL_ACCOUNT_UNPROVED')
    require(last <= _time(identity.get('observed_at')) <= checked_at, 'IDENTITY_TIME_INVALID')
    require(capture.get('execution_binding') == binding and contract.get('symbol') == 'CRC', 'EXECUTION_SCOPE_MISMATCH')
    require(_time(contract.get('effective_after')) <= first <= last <= _time(contract.get('effective_until')),
            'HISTORICAL_COST_CONTRACT_UNPROVED')
    basis = contract.get('basis')
    require(basis in (RECORDED, MODEL), 'BASIS_INVALID')
    require(contract.get('cost_scope') == 'COMPLETE_FOR_DECLARED_BASIS', 'COST_SCOPE_INCOMPLETE')
    _digest(contract.get('cost_scope_source_sha256'))
    # A source-backed finality assertion is an explicit reviewed dependency,
    # not inferred from elapsed time, an empty API response or a page limit.
    finality = contract.get('posting_finality')
    require(isinstance(finality, dict) and finality.get('status') == 'ESTABLISHED'
            and finality.get('execution_binding') == binding, 'POSTING_FINALITY_UNPROVED')
    _digest(finality.get('source_sha256'))
    rules = contract.get('cost_categories')
    require(type(rules) is dict and 1 <= len(rules) <= 32 and all(isinstance(k, str) and k and len(k) <= 64 for k in rules), 'COST_CATEGORIES_REQUIRED')
    require(capture.get('query_axis') == 'ALPACA_CREATED_AT' and capture.get('category') in ('non_trade_activity', 'all_activity')
            and capture.get('activity_types_filter') is None
            and capture.get('order_filter') is None and capture.get('symbol_filter') is None, 'INCOMPLETE_ACTIVITY_SCOPE')
    after, until = _time(capture.get('query_after')), _time(capture.get('query_until'))
    require(after < first and until > last and until >= _time(finality.get('created_until'))
            and until <= _time(capture.get('observed_at')) <= checked_at, 'COST_COVERAGE_INCOMPLETE')
    pages = capture.get('pages')
    require(type(pages) is list and 1 <= len(pages) <= 32, 'PAGINATION_INCOMPLETE')
    previous, activity_count, seen_tokens, seen_bodies = None, 0, set(), set()
    for index, page in enumerate(pages):
        require(type(page) is dict and type(page.get('http_status')) is int and page['http_status'] == 200 and page.get('request_token_sha256') == previous
                and type(page.get('row_count')) is int and 0 <= page['row_count'] <= 100, 'PAGINATION_INCOMPLETE')
        body_hash = _digest(page.get('body_sha256'))
        require(body_hash not in seen_bodies, 'PAGINATION_INCOMPLETE')
        seen_bodies.add(body_hash)
        count = page['row_count']
        token = page.get('last_activity_sha256')
        require((count == 0 and token is None) or (count > 0 and _digest(token) not in seen_tokens), 'PAGINATION_INCOMPLETE')
        require((index == len(pages) - 1 and count < 100) or (index < len(pages) - 1 and count == 100), 'PAGINATION_INCOMPLETE')
        previous = token
        seen_tokens.add(token)
        activity_count += count
    require(type(capture.get('unattributed_count')) is int and capture['unattributed_count'] == 0
            and capture.get('all_non_trade_records_classified') is True, 'UNATTRIBUTED_COSTS')
    costs = capture.get('costs')
    require(type(costs) is list and len(costs) <= MAX_COSTS, 'COST_ROWS_INVALID')
    classifications = capture.get('record_classifications')
    require(type(classifications) is list and len(classifications) == activity_count, 'CLASSIFICATION_INCOMPLETE')
    classified_ids, cost_ids = set(), set()
    for item in classifications:
        require(type(item) is dict, 'CLASSIFICATION_INCOMPLETE')
        record_id = _digest(item.get('source_record_sha256'))
        require(record_id not in classified_ids, 'CLASSIFICATION_INCOMPLETE')
        classified_ids.add(record_id)
        classification = item.get('classification')
        require(classification in ('COST', 'NON_COST_CONFIRMED', 'OUT_OF_SCOPE_CONFIRMED'), 'CLASSIFICATION_INCOMPLETE')
        if classification == 'COST':
            cost_ids.add(record_id)
        else:
            _digest(item.get('reason_source_sha256'))
    seen, charges, rebates = set(), [], []
    by_category = {key: [] for key in rules}
    for record in costs:
        require(type(record) is dict and record.get('category') in rules, 'COST_CATEGORY_UNREVIEWED')
        cost_id = _digest(record.get('source_record_sha256'))
        require(cost_id not in seen, 'DUPLICATE_COST')
        seen.add(cost_id)
        require(record.get('kind') in ('CHARGE', 'REBATE'), 'COST_KIND_INVALID')
        amount = _decimal(record.get('amount'))
        require(amount >= 0, 'AMOUNT_INVALID')
        require(record.get('execution_binding') == binding, 'COST_ATTRIBUTION_UNPROVED')
        if record.get('attribution') == 'ORDER':
            require(record.get('order_id') in A.ORDERS, 'COST_ATTRIBUTION_UNPROVED')
        else:
            require(record.get('attribution') == 'ACCOUNT_ALLOCATION', 'COST_ATTRIBUTION_UNPROVED')
            split = record.get('allocation')
            require(type(split) is dict and split.get('rule') == 'EXPLICIT_SOURCE_BACKED_CONSERVATION'
                    and split.get('this_trade_id') == 110, 'COST_ATTRIBUTION_UNPROVED')
            _digest(split.get('source_sha256'))
            population = split.get('shares')
            require(type(population) is list and 1 <= len(population) <= 1000, 'COST_ATTRIBUTION_UNPROVED')
            ids = [item.get('trade_id') for item in population]
            require(all(type(i) is int and i > 0 for i in ids) and len(set(ids)) == len(ids), 'COST_ATTRIBUTION_UNPROVED')
            amounts = [_decimal(item.get('amount')) for item in population]
            require(all(value >= 0 for value in amounts) and _sum(amounts) == _decimal(split.get('total_amount'))
                    and ids.count(110) == 1 and amounts[ids.index(110)] == amount, 'COST_ATTRIBUTION_UNPROVED')
        by_category[record['category']].append(record)
        (charges if record['kind'] == 'CHARGE' else rebates).append(amount)
    require(seen == cost_ids, 'CLASSIFICATION_INCOMPLETE')
    for category, rule in rules.items():
        require(type(rule) is dict, 'COST_RULE_INVALID')
        _digest(rule.get('source_sha256'))
        status = rule.get('status')
        require(status in ('RECORDED_COMPLETE', 'ZERO_CONFIRMED', 'NOT_APPLICABLE_CONFIRMED', 'MODEL_EXCLUDED'), 'COST_RULE_INVALID')
        require(status != 'RECORDED_COMPLETE' or bool(by_category[category]), 'COST_ROWS_MISSING')
        require(status == 'RECORDED_COMPLETE' or not by_category[category], 'COST_RULE_CONFLICT')
        require(status != 'MODEL_EXCLUDED' or basis == MODEL, 'MODEL_NOT_RECORDED_NET')
    charge_total, rebate_total = _sum(charges), _sum(rebates)
    net_cost = _sum([charge_total, rebate_total.copy_negate()])
    net = _sum([gross, net_cost.copy_negate()])
    result = dict(schema_version=1, namespace=basis, qualification_status='QUALIFIED_WITHIN_ACCEPTED_SCOPE',
                  cost_scope='COMPLETE_FOR_DECLARED_BASIS',
                  source_fingerprint=source, execution_binding=binding, entry_quantity='30', exit_quantity='30',
                  entry_cost=_text(entry_cost), proceeds=_text(proceeds), gross_pnl=_text(gross),
                  fees=_text(charge_total), rebates=_text(rebate_total), net_cost=_text(net_cost), net_pnl=_text(net),
                  cost_evidence_sha256=accepted.cost_capture_sha256, source_identity_sha256=accepted.source_identity_sha256,
                  cost_contract_sha256=accepted.cost_contract_sha256, financial_eligible=False,
                  legacy_validated_performance_eligible=False, recovery_complete=False, account_history_complete=False)
    result['consumer_fingerprint'] = D.fingerprint(result)
    return result


def consume(artifact, *args, **kwargs):
    """No cached qualification survives changed evidence, policy or fill bindings."""
    expected = qualify(*args, **kwargs)
    candidate = _accepted(artifact, D.fingerprint(expected))
    require(type(candidate.get('schema_version')) is int and candidate == D.document(expected), 'QUALIFICATION_STALE_OR_TAMPERED')
    return expected


def consumer_snapshot(qualifications, *args, namespace, **kwargs):
    """Revalidate current evidence before consuming the single CRC110 record.

    A self-hashed candidate or an earlier successful calculation is insufficient.
    No synthetic scalar exit price or legacy financial promotion is performed.
    """
    require(namespace in (RECORDED, MODEL) and type(qualifications) is list and len(qualifications) == 1, 'CONSUMER_SCOPE_INVALID')
    current = consume(qualifications[0], *args, **kwargs)
    require(current['namespace'] == namespace, 'CONSUMER_SCOPE_INVALID')
    return dict(namespace=namespace, rows=[current], input_count=1, net_pnl=current['net_pnl'],
                input_fingerprint=D.fingerprint({'rows': [current]}), legacy_validated_performance_eligible=False)
