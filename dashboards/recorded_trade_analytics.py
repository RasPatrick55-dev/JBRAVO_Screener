"""Bounded read-only recorded GROSS view, separate from qualified net analytics.

Uses existing policy exclusions and the existing gross-accounting basis. Unknown
rows are counted, never assigned zero. No persistence, clients or configuration.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import re
from scripts import trade_disposition as D, gross_paper_accounting as G

PERIODS = {'d': 1, 'w': 7, 'm': 30, 'y': 365, 'all': None}
LABELS = dict(d='DAILY', w='WEEKLY', m='MONTHLY', y='YEARLY', all='ALL')


def _time(value):
    try:
        return D.aware_time(value).astimezone(timezone.utc)
    except (D.TradePolicyError, TypeError, ValueError, OverflowError):
        return None


def _selected(stamp, period, now):
    return period == 'all' or stamp is not None and now-timedelta(days=PERIODS[period]) <= stamp <= now


def project(policy, records, *, identity, now, kind='stats', period='all', mode='winners', limit=12):
    """One full bounded CLOSED-row observation; G validates every included row."""
    D.identity(*identity)
    if type(records) is not list or len(records) > G.MAX_ROWS or period not in PERIODS or kind not in {'stats', 'leaderboard'}:
        raise ValueError('recorded_gross_scope_invalid')
    if now.utcoffset() is None or mode not in {'winners', 'losers'}:
        raise ValueError('recorded_gross_scope_invalid')
    now = now.astimezone(timezone.utc)
    allocations = {item.trade_id: item for item in policy.allocations}
    known, unresolved, excluded, seen = [], [], [], set()
    for row in records:
        identifier = D.positive_id(row.get('trade_id'))
        if identifier in seen or row.get('status') != 'CLOSED':
            raise ValueError('recorded_gross_population_invalid')
        seen.add(identifier)
        exited = _time(row.get('exit_time'))
        if identifier in policy.excluded_ids:
            excluded.append(exited)
            continue
        if identifier in allocations:
            # Recheck the queried full physical row, not merely quantity/price.
            # The canonical policy reader validates the current fill/claim case.
            G.A.validate_trade(row)
        reason = None
        symbol = row.get('symbol')
        if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9./^-]{0,31}', symbol):
            reason = 'SYMBOL_UNKNOWN'
        entered = _time(row.get('entry_time'))
        if entered is None or exited is None or exited < entered or exited > now:
            reason = 'TIME_UNKNOWN_OR_INVALID'
        for field, positive in [('qty', False), ('entry_price', True), ('exit_price', True)]:
            if field == 'exit_price' and identifier in allocations:
                continue  # Existing audited multi-fill gross; never invent a scalar exit.
            try:
                G._number(row.get(field), positive=positive, nonzero=field == 'qty')
            except D.TradePolicyError:
                reason = field.upper() + '_UNKNOWN_OR_INVALID'
                break
        if reason:
            unresolved.append((exited if exited is not None and exited <= now else None, reason))
        else:
            known.append(row)
    # This retains full overlay/ownership validation. Missing/invalid allocation
    # inputs cannot be silently removed to make a gross snapshot pass.
    snapshot = G.build_snapshot(policy, known, expected_identity=identity, expected_revision=policy.revision)
    gross_by_id = {row['trade_id']: Decimal(row['gross_pnl']) for row in snapshot['rows']}
    def scoped(key):
        included = [row for row in known if _selected(_time(row['exit_time']), key, now)]
        unknown = [reason for stamp, reason in unresolved if stamp is None or _selected(stamp, key, now)]
        coverage = dict(included=len(included), excluded=sum(_selected(stamp, key, now) for stamp in excluded),
                        unknown=len(unknown), unscoped_unknown=sum(stamp is None for stamp, _ in unresolved),
                        unscoped_excluded=sum(stamp is None for stamp in excluded),
                        reasons=dict(Counter(unknown)))
        return included, coverage
    def metric(key):
        rows, coverage = scoped(key)
        values = [gross_by_id[row['trade_id']] for row in rows]
        best = max(rows, key=lambda row:(gross_by_id[row['trade_id']], -row['trade_id'])) if rows else None
        worst = min(rows, key=lambda row:(gross_by_id[row['trade_id']], row['trade_id'])) if rows else None
        total = G._money_text(G._gross_sum(values)) if values or not coverage['unknown'] else None
        return dict(key=key, label=LABELS[key], winRatePct=(sum(value > 0 for value in values)/len(values)*100) if values else None,
            totalPL=total, topTrade={'symbol': best['symbol'] if best else '--', 'pl': G._money_text(gross_by_id[best['trade_id']]) if best else None},
            worstLoss={'symbol': worst['symbol'] if worst else '--', 'pl': G._money_text(gross_by_id[worst['trade_id']]) if worst else None},
            tradesCount=len(rows), coverage=coverage)
    selected, coverage = scoped(period)
    if kind == 'stats':
        rows = [metric(key) for key in PERIODS] if period == 'all' else [metric(period)]
    else:
        groups = {}
        for row in selected:
            groups.setdefault(row['symbol'], []).append(gross_by_id[row['trade_id']])
        totals = [(symbol, G._gross_sum(values)) for symbol, values in groups.items()]
        totals = [(symbol, value) for symbol, value in totals if value > 0] if mode == 'winners' else [(symbol, value) for symbol, value in totals if value < 0]
        totals.sort(key=lambda item:((item[1].copy_negate() if mode == 'winners' else item[1]), item[0]))
        rows = [dict(rank=index+1, symbol=symbol, pl=G._money_text(value)) for index,(symbol,value) in enumerate(totals[:max(1,min(50,int(limit)))])]
    return dict(ok=True, status='partial' if coverage['unknown'] else 'ready', source='postgres', view_scope='RECORDED_GROSS',
        pnl_basis=G.BASIS, performance_status='RECORDED_GROSS_ONLY', policy_validation_status='VALIDATED_POLICY',
        trade_policy={'policy': policy.metadata()}, range=period, mode=mode, rows=rows, coverage=coverage,
        closed_population_count=len(records), gross_input_complete=not bool(unresolved),
        input_fingerprint=D.fingerprint({'rows':records}), observation_basis='FULL_BOUNDED_CLOSED_ROWS_WITH_CURRENT_POLICY',
        date_basis='UTC exit instant; rolling1/7/30/365days; ALL has no date cutoff',
        observed_at=now.isoformat(), fees=None, net_pnl=None, fees_status='UNKNOWN', net_status='UNKNOWN',
        net_eligible=False, accounting_complete=False, recovery_complete=False, entry_ready=False,
        qualified_net_status='NOT_ESTABLISHED_BY_RECORDED_GROSS')


def read_view(connection_factory, identity, *, kind='stats', period='all', mode='winners', limit=12, now=None):
    connection = None
    try:
        if identity is None:
            raise ValueError('paper_read_context_required')
        connection = connection_factory()
        if connection is None or connection.autocommit or connection.get_transaction_status() != 0:
            raise ValueError('idle_connection_required')
        with connection:
            with connection.cursor() as cursor:
                cursor.execute('SET TRANSACTION READ ONLY')
                policy = D.read_policy(cursor, identity, scope='VALIDATED_PERFORMANCE')
                cursor.execute('SELECT pg_catalog.to_jsonb(t)::text FROM public.trades t '
                    "WHERE status='CLOSED' ORDER BY trade_id LIMIT %s", (G.MAX_ROWS+1,))
                records = [D.document(row[0]) for row in cursor.fetchall()]
                return project(policy, records, identity=identity, now=now or datetime.now(timezone.utc),
                               kind=kind, period=period, mode=mode, limit=limit)
    except Exception:
        return dict(ok=False, status='unavailable', source='postgres', view_scope='RECORDED_GROSS', pnl_basis=G.BASIS,
            reason='CURRENT_RECORDED_GROSS_INPUTS_UNAVAILABLE', rows=[], coverage=None,
            fees_status='UNKNOWN', net_status='UNKNOWN', net_eligible=False, accounting_complete=False, recovery_complete=False)
    finally:
        if connection is not None:
            try: connection.rollback()
            except Exception: pass
            try: connection.close()
            except Exception: pass
