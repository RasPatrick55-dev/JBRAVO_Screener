"""Supplied-data completion-clock contracts. No provider, database or config."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from dashboards import execution_gross_projection as E, recorded_trade_analytics as R
from scripts import trade_disposition as D
from test_execution_gross_view import inputs, Connection
from test_public_dashboard_integration import policy, IDENTITY, NOW

pytestmark = pytest.mark.alpaca_optional


def evaluate(row, fills, owners, *, now=NOW, **kwargs):
    p = policy()
    execution = E.reconstruct(p, [row], fills, owners)
    result = R.project(p, [row], identity=IDENTITY, now=now, execution=execution, **kwargs)
    return execution, result


@pytest.mark.parametrize('gap', [77536, 220247])
def test_exact_late_fill_is_completion_without_changing_ledger_or_net(gap):
    row, fills, owner = inputs()
    original_close = datetime.fromisoformat(row['exit_time'])
    final = original_close + timedelta(microseconds=gap)
    fills[-1]['transaction_time'] = fills[-1]['provider_transaction_time'] = final.isoformat()
    before = deepcopy((row, fills, owner))
    execution, result = evaluate(row, fills, [owner])
    clock = execution.completion[row['trade_id']]
    assert clock.recorded_exit == original_close and clock.completed_at == final
    assert clock.ledger_delta_microseconds == gap and clock.raw_clocks_match
    assert clock.holding_microseconds == 3600000000 + gap
    assert result['coverage']['included'] == 1 and result['coverage']['unknown'] == 0
    assert result['source_as_of'] == final.isoformat()
    assert result['completion_clock']['maximum_delta_microseconds'] == gap
    assert result['rows'][-1]['totalPL'] == '4'
    assert result['fees'] is result['net_pnl'] is None and result['net_eligible'] is False
    assert (row, fills, owner) == before
    with pytest.raises(FrozenInstanceError): clock.completed_at = NOW
    with pytest.raises(TypeError): execution.completion[row['trade_id']] = clock


@pytest.mark.parametrize('period,days', [('d', 1), ('w', 7), ('m', 30), ('y', 365)])
@pytest.mark.parametrize('offset', ['+00:00', '-05:00', '+05:30'])
def test_ranges_and_ranking_use_exact_completion_in_utc(period, days, offset):
    row, fills, owner = inputs()
    cutoff = NOW - timedelta(days=days)
    row['entry_time'] = (cutoff-timedelta(hours=2)).isoformat()
    row['exit_time'] = (cutoff-timedelta(microseconds=1)).isoformat()
    for index, fill in enumerate(fills):
        stamp = cutoff-timedelta(minutes=120 if index == 0 else 1 if index == 1 else 0)
        fill['transaction_time'] = stamp.isoformat()
        local = timezone(timedelta(hours=-5)) if offset == '-05:00' else timezone(timedelta(hours=5, minutes=30)) if offset == '+05:30' else timezone.utc
        fill['provider_transaction_time'] = stamp.astimezone(local).isoformat()
    execution, result = evaluate(row, fills, [owner], period=period)
    assert result['rows'][0]['tradesCount'] == 1
    leaders = R.project(policy(), [row], identity=IDENTITY, now=NOW, execution=execution, kind='leaderboard', period=period)
    assert leaders['rows'] == [dict(rank=1, symbol='FICT', pl='4')]
    assert result['source_as_of'] == cutoff.isoformat()
    assert 'execution' in result['date_basis'] and 'EXECUTION_COMPLETION' in result['source_as_of_basis']
    # One microsecond before the boundary is excluded by the same clock in both APIs.
    earlier = cutoff-timedelta(microseconds=1)
    fills[-1]['transaction_time'] = fills[-1]['provider_transaction_time'] = earlier.isoformat()
    _, result = evaluate(row, fills, [owner], period=period)
    _, leaders = evaluate(row, fills, [owner], period=period, kind='leaderboard')
    assert result['rows'][0]['tradesCount'] == 0 and leaders['rows'] == []


def test_no_ledger_time_fallback_when_completion_is_unknown():
    row, fills, owner = inputs()
    row['entry_time'] = (NOW-timedelta(days=20)).isoformat()
    row['exit_time'] = (NOW-timedelta(days=10)).isoformat()
    other = dict(owner, trade_id=9999, status='OPEN', entry_order_id=None)
    _, result = evaluate(row, fills, [owner, other], period='d')
    assert result['coverage']['unknown'] == result['coverage']['unscoped_unknown'] == 1
    assert result['source_as_of'] is None and result['rows'][0]['totalPL'] is None


@pytest.mark.parametrize('raw', [None, '', {}, 'bad', '2001-02-03T11:00:00',
    '2001-02-03T11:00:00.000000001Z', '2001-02-03T11:00:00+00:99', True])
def test_late_execution_requires_exact_primary_clock_no_fallback_or_rounding(raw):
    row, fills, owner = inputs()
    row['exit_time'] = (datetime.fromisoformat(row['exit_time'])-timedelta(microseconds=1)).isoformat()
    fills[-1]['provider_transaction_time'] = raw
    _, result = evaluate(row, fills, [owner])
    assert result['coverage']['reasons'] == {'LATE_EXECUTION_RAW_CLOCK_UNRESOLVED': 1}
    assert result['completion_clock']['included'] == 0


@pytest.mark.parametrize('suffix', ['Z', '+00:00', '-05:00'])
def test_raw_fraction_zero_padding_is_exact_and_bounded(suffix):
    stamp = datetime(2001, 2, 3, 11, 0, 0, 123456, tzinfo=timezone.utc)
    local = stamp if suffix != '-05:00' else stamp-timedelta(hours=5)
    text = local.strftime('%Y-%m-%dT%H:%M:%S')+'.123456000'+suffix
    assert E.exact_raw_clock_matches(text, stamp)
    assert not E.exact_raw_clock_matches(text.replace('123456000', '123456001'), stamp)


@pytest.mark.parametrize('mutation,reason', [
    ('reversed', 'TIME_UNKNOWN_OR_INVALID'),
    ('interleaved', 'EXECUTION_CHRONOLOGY_UNRESOLVED'),
    ('quantity', 'QUANTITY_MISMATCH'),
    ('owner', 'ORDER_OWNER_AMBIGUOUS'),
    ('future', 'TIME_UNKNOWN_OR_INVALID')])
def test_clock_change_does_not_relax_other_invariants(mutation, reason):
    row, fills, owner = inputs()
    owners = [owner]
    if mutation == 'reversed': row['entry_time'] = NOW.isoformat()
    if mutation == 'interleaved':
        fills[0]['transaction_time'] = fills[0]['provider_transaction_time'] = (NOW-timedelta(minutes=59)).isoformat()
    if mutation == 'quantity': fills[-1]['qty'] = fills[-1]['provider_qty'] = '0.5'
    if mutation == 'owner': owners.append(dict(owner, trade_id=9999, status='OPEN', entry_order_id=None))
    if mutation == 'future': fills[-1]['transaction_time'] = fills[-1]['provider_transaction_time'] = (NOW+timedelta(microseconds=1)).isoformat()
    _, result = evaluate(row, fills, owners)
    assert result['coverage']['reasons'] == {reason: 1}
    assert result['net_eligible'] is False and result['fees'] is result['net_pnl'] is None


def test_earlier_completed_fill_uses_its_clock_without_rewriting_delayed_close():
    row, fills, owner = inputs()
    row['exit_time'] = (NOW-timedelta(minutes=30)).isoformat()
    for fill in fills: fill.pop('provider_transaction_time')
    execution, result = evaluate(row, fills, [owner])
    assert result['source_as_of'] == (NOW-timedelta(hours=1)).isoformat()
    assert execution.completion[row['trade_id']].ledger_delta_microseconds == -1800000000
    assert result['completion_clock']['completion_before_recorded_count'] == 1


def test_raw_clock_or_current_row_drift_changes_bound_projection():
    row, fills, owner = inputs()
    p = policy(); captured = E.reconstruct(p, [row], fills, [owner])
    fills[-1]['provider_transaction_time'] = 'not-the-same-source'
    checked = E.reconstruct(p, [row], fills, [owner])
    assert captured.source_hash != checked.source_hash
    row['updated_at'] = NOW.isoformat()+'changed'
    with pytest.raises(ValueError, match='BINDING_CHANGED'):
        R.project(p, [row], identity=IDENTITY, now=NOW, execution=captured)


def test_real_read_entry_uses_new_fixed_projection_and_rolls_back(monkeypatch):
    conn = Connection()
    row = conn.records[0]
    row['exit_time'] = (datetime.fromisoformat(row['exit_time'])-timedelta(microseconds=77536)).isoformat()
    monkeypatch.setattr(D, 'read_policy', lambda *args, **kwargs: policy())
    result = R.read_view(lambda: conn, IDENTITY, now=NOW, basis='execution_gross')
    assert result['ok'] and result['coverage']['included'] == 1
    assert result['completion_clock']['maximum_delta_microseconds'] == 77536
    assert "'provider_transaction_time',raw::jsonb->>'transaction_time'" in E.FILLS_SQL
    assert result['read_consistency'] == 'READ_COMMITTED_VERSIONED_COHORT_AND_CANONICAL_RECHECK'
    conn.rollback.assert_called_once(); conn.close.assert_called_once()


def test_supplied_eight_case_profile_only_two_clock_cases_gain_gross_coverage():
    records, fills, owners = [], [], []
    for index, (qty, exit_qty) in enumerate([(12,12), (44,44), (2,4), (4,4), (11,3), (42,29), (5,6), (53,28)]):
        row, linked, owner = inputs(9001+index, qty=str(qty))
        if index < 2:
            row['exit_time'] = (datetime.fromisoformat(row['exit_time'])-timedelta(microseconds=[77536,220247][index])).isoformat()
        else:
            linked[-2]['qty'] = linked[-2]['provider_qty'] = str(exit_qty/2)
            linked[-1]['qty'] = linked[-1]['provider_qty'] = str(exit_qty/2)
        if index in (2,3):
            owners.append(dict(owner, trade_id=9900+index, status='OPEN', entry_order_id=None))
        records.append(row); fills.extend(linked); owners.append(owner)
    original = deepcopy(records)
    p = policy(); execution = E.reconstruct(p, records, fills, owners)
    result = R.project(p, records, identity=IDENTITY, now=NOW, execution=execution)
    assert result['coverage']['included'] == 2 and result['coverage']['unknown'] == 6
    assert result['coverage']['reasons'] == {'ORDER_OWNER_AMBIGUOUS':2, 'QUANTITY_MISMATCH':4}
    assert result['completion_clock']['completion_after_recorded_count'] == 2
    assert result['fees'] is result['net_pnl'] is None and result['net_eligible'] is False
    assert records == original


def test_midnight_boundary_preserves_both_instants_and_exact_holding_interval():
    now = datetime(2001, 2, 5, tzinfo=timezone.utc)
    cutoff = now-timedelta(days=1)
    row, fills, owner = inputs()
    row['entry_time'] = (cutoff-timedelta(hours=2)).isoformat()
    row['exit_time'] = (cutoff-timedelta(microseconds=77536)).isoformat()
    for index, fill in enumerate(fills):
        stamp = cutoff-timedelta(hours=2) if index == 0 else cutoff-timedelta(minutes=1) if index == 1 else cutoff
        fill['transaction_time'] = fill['provider_transaction_time'] = stamp.isoformat()
    execution, result = evaluate(row, fills, [owner], now=now, period='d')
    clock = execution.completion[row['trade_id']]
    assert result['rows'][0]['tradesCount'] == 1 and result['source_as_of'] == cutoff.isoformat()
    assert clock.completed_at - E.aware(row['entry_time']) == timedelta(hours=2)
    assert clock.holding_microseconds == 7200000000
    assert clock.recorded_exit.date() != clock.completed_at.date()


def test_real_read_refuses_raw_clock_drift_even_when_economics_return_unchanged(monkeypatch):
    from sql_transport_fixture import Cursor
    class ChangedClock(Cursor):
        def native(self, sql, params):
            rows = super().native(sql, params)
            if sql == E.FILLS_SQL and self.conn.source_reads > 1:
                document = D.document(rows[-1][0])
                document['row']['provider_transaction_time'] = 'different-primary-clock'
                rows[-1] = (D.canonical(document),)
            return rows
    conn = Connection()
    conn.cursor = lambda: ChangedClock(conn)
    monkeypatch.setattr(D, 'read_policy', lambda *args, **kwargs: policy())
    result = R.read_view(lambda: conn, IDENTITY, now=NOW, basis='execution_gross')
    assert result['status'] == 'unavailable' and result['coverage'] is None
    conn.rollback.assert_called_once(); conn.close.assert_called_once()
