"""Explicitly scoped, proposed net after allocated observed posted fees.

This is not canonical UUID accounting, broker-provided per-trade fees, or legacy
validated performance. No I/O, persistence, authority discovery or activation.
Input is a separately pinned parent-sanitized projection with original response
artifact hashes. Original response bytes remain with their acquiring operator.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from zoneinfo import ZoneInfo
import hashlib
from scripts import allocation_net_accounting as N, trade_allocation as A, trade_disposition as D

BASIS = 'CRC110_ALLOCATED_OBSERVED_POSTED_COST_NET_V1'
RULE = 'posted_daily_taf_sell_share_largest_remainder_v1'


def check(condition, reason):
    if not condition:
        raise N.QualificationError('COHORT_POSTED_COST_' + reason)


def _money(cents):
    whole, fraction = divmod(cents, 100)
    return str(whole) + '.' + str(fraction).zfill(2)


def _allocate(budget, executions):
    denominator = sum((Fraction(row['qty']) for row in executions), Fraction(0))
    check(denominator > 0, 'SELL_COHORT_MISSING')
    quotas = [Fraction(budget) * Fraction(row['qty']) / denominator for row in executions]
    cents = [value.numerator // value.denominator for value in quotas]
    remaining = budget - sum(cents)
    # Preserve the proposed atom and tie order exactly: execution first, then
    # aggregate. Grouping by symbol before rounding is a different policy.
    ranked = sorted(range(len(executions)), key=lambda i:(-(quotas[i]-cents[i]),
        executions[i]['instant'], executions[i]['symbol'], executions[i]['price'], executions[i]['ordinal']))
    for index in ranked[:remaining]:
        cents[index] += 1
    check(sum(cents) == budget, 'CONSERVATION_FAILED')
    return quotas, cents


def derive(projection_bytes, *, expected_file_sha256):
    """Return a deterministic proposal from pinned retained observations.

    File SHA256 binds the sanitized projection, not an original broker response.
    Original CallToolResult hashes and their exact encoding are carried separately.
    Missing canonical identity is reported, never silently satisfied.
    """
    check(type(projection_bytes) is bytes and len(projection_bytes) <= 1024*1024, 'PROJECTION_INVALID')
    N._digest(expected_file_sha256)
    check(hashlib.sha256(projection_bytes).hexdigest() == expected_file_sha256, 'PROJECTION_CHANGED')
    evidence = D.document(projection_bytes.decode('utf-8'))
    check(evidence.get('schema_version') == 1 and evidence.get('provenance_version') == 2
          and evidence.get('source_kind') == 'PARENT_PROVIDED_SANITIZED_PROJECTION_OF_OBSERVED_BROKER_RESPONSES'
          and evidence.get('paper_environment_basis') == 'CONNECTOR_CONTRACT_PAPER', 'PROVENANCE_INVALID')
    check(evidence.get('canonical_account_uuid_available') is False and evidence.get('endpoint_independently_attested') is False
          and evidence.get('provider_activity_ids_omitted') is True, 'EVIDENCE_CLASS_CHANGED')
    queries = evidence.get('query_provenance', {})
    fill_query, cost_query = queries.get('fill', {}), queries.get('nontrade', {})
    for query in (fill_query, cost_query):
        check(query.get('direction') == 'asc' and query.get('page_size') == 100
              and query.get('terminal_empty') is True and query.get('terminal_next_page_token_absent') is True,
              'QUERY_NOT_EXHAUSTED')
    receipts = queries.get('original_response_hashes')
    check(type(receipts) is list and len(receipts) == 6, 'RESPONSE_PROVENANCE_MISSING')
    check('full CallToolResult' in queries.get('original_response_hash_encoding', ''), 'RESPONSE_HASH_ENCODING_MISSING')
    check([row.get('row_count') for row in receipts] == [27,0,100,100,16,0], 'RESPONSE_COVERAGE_CHANGED')
    check(len({row.get('name') for row in receipts}) == 6, 'RESPONSE_PROVENANCE_MISSING')
    for row in receipts:
        N._digest(row.get('sha256'))
        check(type(row.get('bytes')) is int and row['bytes'] > 0, 'RESPONSE_PROVENANCE_MISSING')
    coverage = evidence.get('parent_coverage_attestations', {})
    required = ('all_27_fill_rows_in_projection','all_216_nontrade_rows_examined',
        'every_fee_row_economic_dates_2026_04_16_and_17_in_projection',
        'no_CAT_REG_refund_correction_or_other_status_for_these_dates_returned')
    check(all(coverage.get(key) is True for key in required), 'SCOPED_COVERAGE_INCOMPLETE')
    check(cost_query.get('category') == 'non_trade_activity' and cost_query.get('parent_total_records') == 216
          and fill_query.get('parent_page_counts') == [27,0] and cost_query.get('parent_page_counts') == [100,100,16,0],
          'SCOPED_COVERAGE_INCOMPLETE')
    first, last = N._time(fill_query.get('after')), N._time(fill_query.get('until'))
    cost_after, cutoff = N._time(cost_query.get('after')), N._time(cost_query.get('until'))
    times = queries.get('parent_actual_fetch_timestamps', {})
    check(type(times) is dict and cutoff <= N._time(times.get('started_utc')) <= N._time(times.get('completed_utc')),
          'RECEIPT_TIME_INVALID')
    fills = evidence.get('fills')
    check(type(fills) is list and len(fills) == 27, 'FILL_COVERAGE_CHANGED')
    executions, crc, seen = [], [], set()
    ny = ZoneInfo('America/New_York')
    for ordinal, row in enumerate(fills):
        instant = N._time(row.get('transaction_time'))
        quantity, price = N._decimal(row.get('qty')), N._decimal(row.get('price'))
        symbol, side = row.get('symbol'), row.get('side')
        check(isinstance(symbol,str) and symbol and side in ('buy','sell') and quantity > 0 and price > 0,
              'FILL_INVALID')
        check(first < instant < last and cost_after <= first and cutoff > last, 'QUERY_SCOPE_CHANGED')
        economic_day = instant.astimezone(ny).date().isoformat()
        # This proposal is restricted to the supplied daytime population. No
        # UTC date convention is generalized to evening/overnight executions.
        check(economic_day == instant.date().isoformat(), 'DATE_CONVENTION_REVIEW_REQUIRED')
        key = (symbol,side,quantity,price,instant,row.get('order_id_sha256'))
        check(key not in seen, 'AMBIGUOUS_DUPLICATE_EXECUTION')
        seen.add(key)
        execution = dict(ordinal=ordinal,symbol=symbol,side=side,qty=quantity,price=price,instant=instant,date=economic_day)
        executions.append(execution)
        if symbol == 'CRC':
            crc.append((execution,row.get('order_id_sha256')))
    expected = {(side,Decimal(qty),Decimal(price),N._time(stamp),hashlib.sha256(order.encode()).hexdigest())
                for _,_,side,order,qty,price,stamp in A.FILLS}
    actual = {(row['side'],row['qty'],row['price'],row['instant'],order_hash) for row,order_hash in crc}
    check(len(crc) == 5 and actual == expected, 'CRC_EXECUTION_FIELDS_MISMATCH')
    fees = evidence.get('fees')
    check(type(fees) is list and len(fees) <= 32, 'FEE_PROJECTION_INVALID')
    daily, seen_days, crc_cents = [], set(), 0
    for fee in fees:
        day = fee.get('date')
        check(day in ('2026-04-16','2026-04-17') and day not in seen_days, 'FEE_DAY_OR_CORRECTION_REVIEW_REQUIRED')
        seen_days.add(day)
        check(fee.get('activity_type') == 'FEE' and fee.get('activity_sub_type') == 'TAF'
              and fee.get('currency') == 'USD' and fee.get('status') == 'executed', 'FEE_RULE_REVIEW_REQUIRED')
        created = N._time(fee.get('created_at'))
        check(cost_after < created < cutoff, 'FEE_OUTSIDE_CAPTURE')
        amount = N._decimal(fee.get('net_amount'))
        exact_cents = -Fraction(amount)*100
        check(amount < 0 and exact_cents.denominator == 1, 'FEE_REVERSAL_OR_PRECISION_REVIEW_REQUIRED')
        budget = int(exact_cents)
        population = [row for row in executions if row['date'] == day and row['side'] == 'sell']
        quotas, costs = _allocate(budget,population)
        rows=[]
        for row,quota,cents in zip(population,quotas,costs):
            rows.append(dict(source_projection_ordinal_zero_based=row['ordinal'],symbol=row['symbol'],
                transaction_time=row['instant'].isoformat(),sell_quantity=N._text(row['qty']),
                unrounded_quota_cents_exact=str(quota),allocated_cost_usd=_money(cents)))
            if row['symbol'] == 'CRC':
                crc_cents += cents
        daily.append(dict(economic_date=day,posted_cost_usd=_money(budget),
                          sell_quantity=N._text(N._sum([row['qty'] for row in population])),execution_allocations=rows,
                          allocated_cents_conserved=sum(costs) == budget))
    daily.sort(key=lambda row:row['economic_date'])
    with localcontext() as context:
        context.prec=80
        entry_cost=sum((row['qty']*row['price'] for row,_ in crc if row['side']=='buy'),Decimal(0))
        proceeds=sum((row['qty']*row['price'] for row,_ in crc if row['side']=='sell'),Decimal(0))
        gross=proceeds-entry_cost
    allocated = Decimal(_money(crc_cents))
    net=N._sum([gross,allocated.copy_negate()])
    result=dict(schema_version=1,basis=BASIS,rule=RULE,status='PROPOSED_RULE_NOT_ACTIVATED',trade_id=110,
        evidence_file_sha256=expected_file_sha256,source_response_artifacts=receipts,query_creation_cutoff=cutoff.isoformat(),
        parent_fetch_time=times,execution_binding='EXACT_COMMON_FIELDS_AND_THREE_ORDER_HASHES',matched_crc_executions=5,
        entry_quantity='30',exit_quantity='30',gross_pnl=N._text(gross),allocated_observed_fee_cost=_money(crc_cents),
        derived_net_after_observed_posted_fees=N._text(net),daily_allocations=daily,
        scope='Observed posted fee buckets for economic dates April16-17, within the captured creation-time query, allocated per execution then aggregated.',
        canonical_account_uuid_proved=False,provider_activity_id_match_proved=False,independent_asset_exemption_attestation=False,
        global_accounting_complete=False,legacy_validated_performance_eligible=False,entry_ready=False,
        convention='Explicit proposed share-based allocation of a posted pool; not an official broker per-trade fee formula. No FIFO/scalar-exit or future-fee claim.')
    result['consumer_fingerprint']=D.fingerprint(result)
    return result


def consume(artifact, projection_bytes, *, expected_file_sha256):
    current=derive(projection_bytes,expected_file_sha256=expected_file_sha256)
    try:
        accepted=N._accepted(artifact,D.fingerprint(current))
    except N.QualificationError:
        raise N.QualificationError('COHORT_POSTED_COST_STALE_OR_TAMPERED_RECORD') from None
    check(accepted==D.document(current),'STALE_OR_TAMPERED_RECORD')
    return current
