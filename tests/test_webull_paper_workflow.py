from copy import deepcopy
from datetime import datetime, timedelta
import json
import re
from types import SimpleNamespace

import pytest

from scripts import setup_selection, strategy_pipeline, strategy_webull_market_data as market
from scripts import strategy_webull_selection as selection
from scripts.strategy_entry_only import run_entry_only

pytestmark = pytest.mark.alpaca_optional

CUTS = {'2026-11-27': '17:00'}
CALENDAR = [dict(date=day, open='09:30', close=close) for day, close in (
    ('2026-11-25', '16:00'), ('2026-11-27', '13:00'), ('2026-11-30', '16:00'))]


def at(iso):
    return datetime.fromisoformat(iso)


def test_calendar_carries_full_and_short_archives_and_keeps_two_cutoffs():
    context = strategy_pipeline.session_context(CALENDAR, at('2026-11-27T03:45:00+00:00'),
        webull_early_close_cutoffs=CUTS)
    assert context['signal_session'] == '2026-11-25'
    assert context['next_session'] == '2026-11-27'
    assert context['next_regular_close_at'] == '2026-11-27T13:00:00-05:00'

    weekend = at('2026-11-30T03:45:00+00:00')
    context = strategy_pipeline.session_context(CALENDAR, weekend, webull_early_close_cutoffs=CUTS)
    assert context['signal_session'] == '2026-11-27'
    assert context['next_session'] == '2026-11-30'
    assert context['signal_available_at'] == '2026-11-27T20:15:00-05:00'
    assert context['webull_hours'] == dict(regular_close_at='2026-11-27T13:00:00-05:00',
                                           after_hours_close_at='2026-11-27T17:00:00-05:00')
    with pytest.raises(ValueError, match='unverified_webull_early_close'):
        strategy_pipeline.session_context(CALENDAR, weekend, webull_early_close_cutoffs={})


def candle(stamp, session, close):
    return dict(time=stamp.isoformat(), open=close, high=close, low=close, close=close,
                volume='100', trading_session=session)


@pytest.mark.parametrize('value', [
    '2026-11-21T00:58:00.000+0000',
    '2026-11-20T19:58:00.000-0500',
    '2026-11-21T00:58:00Z',
    '2026-11-21T00:58:00+00:00',
])
def test_bar_timestamps_work_with_python_310_iso_parser(monkeypatch, value):
    def legacy_fromisoformat(text):
        assert not text.endswith('Z') and re.search(r'[+-][0-9]{4}$', text) is None
        return datetime.fromisoformat(text)

    monkeypatch.setattr(market, 'datetime', SimpleNamespace(fromisoformat=legacy_fromisoformat))
    start = at('2026-11-20T21:00:00+00:00')
    end = at('2026-11-21T01:00:00+00:00')
    bar = dict(candle(end, 'ATH', '100'), time=value)
    assert market.checked_bars([bar], start, end, 'ATH')[0]['time'] == '2026-11-21T00:58:00+00:00'
    with pytest.raises(ValueError, match='invalid_webull_bar_scope'):
        market.checked_bars([bar], start, end - timedelta(minutes=2), 'ATH')


def proposal(symbol, score):
    return dict(symbol=symbol, signal_id=symbol * 16, strategy_id='daily',
                signal_session='2026-11-27', next_session='2026-11-30',
                signal_close_raw=99, setup_ranking=dict(model=setup_selection.MODEL,
                    model_sha256=setup_selection.model_sha256(), feature_basis='COMPLETED_SPLIT_BARS',
                    signal_session='2026-11-27', score=score, relative_volume=1, signal_volume_raw=100))


def test_archived_webull_bars_reorder_finalists_and_build_replayable_tickets(monkeypatch):
    published = at('2026-11-30T03:45:00+00:00')
    session = strategy_pipeline.session_context(CALENDAR, published, webull_early_close_cutoffs=CUTS)
    session['webull_data_environment'] = 'sandbox'
    proposals = [proposal('AAAA', 4), proposal('BBBB', 3), proposal('CCCC', 2), proposal('DDDD', 1)]
    calls = []

    def bars(symbols, start, end, tag):
        calls.append((symbols, start.hour, end.hour, tag))
        stamp = end - timedelta(minutes=1)
        if tag == 'RTH':
            return [dict(symbol=s, result=[candle(stamp, tag, '100')])
                    for s in symbols if s != 'DDDD']
        return [dict(symbol='AAAA', result=[candle(stamp, tag, '99')]),
                dict(symbol='BBBB', result=[candle(stamp, tag, '105')]),
                dict(symbol='DDDD', result=[candle(stamp, tag, '106')])]

    capture = market.capture_after_hours({p['symbol'] for p in proposals}, session['webull_hours'],
        fetch=bars, now=lambda: at('2026-11-27T20:30:00-05:00'), environment='sandbox')
    assert calls == [(('AAAA', 'BBBB', 'CCCC', 'DDDD'), 13, 17, 'ATH'),
                     (('AAAA', 'BBBB', 'DDDD'), 12, 13, 'RTH')]
    archive = dict(run_id='archive', signal_session=session['signal_session'], webull_after_hours=capture)
    session[selection.ARCHIVE_KEY] = dict(id='archive', sha256=setup_selection.digest(archive),
                                          capture_sha256=setup_selection.digest(capture))
    parent = dict(primary_run_id='primary', primary_report_sha256='report', source_bindings={}, session=session)
    monkeypatch.setattr(setup_selection, 'inputs', lambda *args, **kwargs:
        ({}, {}, setup_selection.rank_setups(proposals)))
    rows = {'archive': ('spec', setup_selection.digest(archive), archive)}
    monkeypatch.setattr(selection, '_row', lambda conn, run_id: rows[run_id])
    record, _ = selection.expected_record(None, parent, now=published, published=published, base_dir=None)
    assert [s['proposal']['symbol'] for s in record['selection']['selected_setups']] == ['BBBB', 'AAAA']
    evidence = record['selection']['selected_setups'][0]['after_hours']
    assert evidence['regular_bar_close'] == '100' and evidence['dollar_volume_estimate'] == '10500'
    assert evidence['return_since_last_regular_bar'] == '0.05'
    assert record['execution_orders']['BBBB' * 16]['limit_price'] == '105.52'
    assert record['execution_orders']['BBBB' * 16]['price_provenance']['environment'] == 'sandbox'
    assert record['selection']['after_hours_coverage'] == dict(requested=4, available=2, missing_or_unusable=2)
    rows[record['run_id']] = (setup_selection.digest(record['specification']), setup_selection.digest(record), record)
    assert selection.read_selection(None, parent, now=published)['record'] == record
    wrong_environment = {**parent, 'session': {**session, 'webull_data_environment': 'production'}}
    with pytest.raises(ValueError, match='webull_capture_scope_changed'):
        selection.read_selection(None, wrong_environment, now=published)
    altered = deepcopy(record)
    altered['execution_orders']['BBBB' * 16]['limit_price'] = '105.53'
    rows[record['run_id']] = (setup_selection.digest(record['specification']), setup_selection.digest(altered), altered)
    with pytest.raises(ValueError, match='replay_changed'):
        selection.read_selection(None, parent, now=published)


def test_empty_finalists_need_no_webull_request():
    hours = market.session_hours(CALENDAR[1], CUTS)
    record = market.capture_after_hours(set(), hours,
        fetch=lambda *args: pytest.fail('empty batch queried Webull'),
        now=lambda: at('2026-11-27T20:20:00-05:00'))
    assert record['symbols'] == {}


def test_alpaca_extended_hours_fallback_only_requests_symbol_and_window_gaps():
    from functools import partial

    hours = market.session_hours(CALENDAR[1], CUTS)
    regular_close = at(hours['regular_close_at'])
    cutoff = at(hours['after_hours_close_at'])
    alpaca_calls, webull_calls = [], []

    def alpaca_bar(stamp, price):
        return dict(t=stamp.isoformat(), o=price, h=price, l=price, c=price, v=100)

    def get(url, params):
        assert url == 'https://data.alpaca.markets/v2/stocks/bars'
        assert (params['feed'], params['adjustment'], params['timeframe']) == ('iex', 'raw', '1Min')
        symbols = params['symbols'].split(',')
        alpaca_calls.append((symbols, params.get('page_token')))
        if at(params['start']) == regular_close:
            assert at(params['end']) == cutoff - timedelta(minutes=1)
            if params.get('page_token') is None:
                return dict(bars={'AAA': [alpaca_bar(cutoff - timedelta(minutes=1), 103)],
                                  'BBB': [alpaca_bar(cutoff - timedelta(minutes=45), 104)]},
                            next_page_token='next'), b''
            return dict(bars={'DDD': [alpaca_bar(cutoff - timedelta(minutes=1), 106)]},
                        next_page_token=None), b''
        assert at(params['end']) == regular_close - timedelta(minutes=1)
        return dict(bars={s: [alpaca_bar(regular_close - timedelta(minutes=1), 100)]
                          for s in ('AAA', 'BBB')}, next_page_token=None), b''

    def webull(symbols, start, end, session):
        webull_calls.append((symbols, session))
        symbol = 'BBB' if session == 'ATH' else 'DDD'
        return [dict(symbol=symbol, result=[candle(end - timedelta(minutes=1), session, '105')])]

    record = market.capture_after_hours({'AAA', 'BBB', 'CCC', 'DDD'}, hours,
        fetch=webull, primary_fetch=partial(market.alpaca_bars, SimpleNamespace(get=get)),
        now=lambda: cutoff + timedelta(hours=4), environment='sandbox')
    assert webull_calls == [(('BBB', 'CCC'), 'ATH'), (('DDD',), 'RTH')]
    assert alpaca_calls == [(['AAA', 'BBB', 'CCC', 'DDD'], None),
                            (['AAA', 'BBB', 'CCC', 'DDD'], 'next'), (['AAA', 'BBB', 'DDD'], None)]
    assert {symbol: (item['status'], item['after_hours_source'], item['regular_source'])
            for symbol, item in record['symbols'].items()} == {
                'AAA': ('AVAILABLE', 'alpaca_iex', 'alpaca_iex'),
                'BBB': ('AVAILABLE', 'webull', 'alpaca_iex'),
                'CCC': ('MISSING', None, None),
                'DDD': ('AVAILABLE', 'alpaca_iex', 'webull'),
            }


def test_unqualified_webull_does_not_block_complete_alpaca_data(monkeypatch):
    from contextlib import ExitStack

    hours = market.session_hours(CALENDAR[1], CUTS)
    cutoff = at(hours['after_hours_close_at'])
    monkeypatch.setattr(market, 'MarketData', lambda *a: pytest.fail('Webull initialized without a gap'))

    def alpaca(symbols, start, end, session):
        return [dict(symbol='AAA', result=[candle(end - timedelta(minutes=1), session, '100')])]

    with ExitStack() as resources:
        fallback = market.webull_fallback(resources, {}, 'sandbox', 'bars')
        record = market.capture_after_hours({'AAA'}, hours, fetch=fallback, primary_fetch=alpaca,
                                           now=lambda: cutoff + timedelta(hours=4))
    assert record['symbols']['AAA']['status'] == 'AVAILABLE'
    assert record['symbols']['AAA']['after_hours_source'] == 'alpaca_iex'


def depth(symbol, stamp, ask='12.34', bid='12.33', tag='PRE'):
    return dict(symbol=symbol, quote_time=int(stamp.timestamp() * 1000), trading_session=tag,
        bids=[dict(price=bid, size='100')], asks=[dict(price=ask, size='100')])


@pytest.mark.parametrize('age,ask,bid,tag,expected', [
    (1, '12.34', '12.33', 'PRE', 'PASS'),
    (2, '12.35', '12.33', 'PRE', 'QUOTE_ASK_ABOVE_PIPELINE_LIMIT'),
    (2, '12.34', '12.00', 'PRE', 'QUOTE_SPREAD_TOO_WIDE'),
    (16, '12.34', '12.33', 'PRE', 'QUOTE_STALE_OR_FUTURE'),
    (21, '12.34', '12.33', 'PRE', 'QUOTE_OUTSIDE_FIRST_PREMARKET_MINUTE'),
    (1, '12.34', '12.33', 'OVN', 'QUOTE_OUTSIDE_FIRST_PREMARKET_MINUTE'),
])
def test_quote_checks_use_provider_time_and_never_reprice(age, ask, bid, tag, expected):
    opening = at('2026-11-27T09:00:00+00:00')
    current = opening + timedelta(seconds=20)
    decision = market.premarket_decision(
        lambda s: depth(s, current - timedelta(seconds=age), ask, bid, tag), 'AAPL',
        ticket={'limit_price': '12.34'}, opening=opening, now=lambda: current)
    assert decision['status'] == expected


@pytest.mark.parametrize('primary,expected_status,expected_provider', [
    ('fresh', 'PASS', 'alpaca_iex'),
    ('expensive', 'QUOTE_ASK_ABOVE_PIPELINE_LIMIT', 'alpaca_iex'),
    ('wide', 'QUOTE_SPREAD_TOO_WIDE', 'alpaca_iex'),
    ('stale', 'PASS', 'webull'),
    ('missing', 'PASS', 'webull'),
    ('invalid', 'PASS', 'webull'),
])
def test_premarket_fills_data_gaps_without_bypassing_a_valid_alpaca_hold(
        primary, expected_status, expected_provider):
    opening = at('2026-11-27T09:00:00+00:00')
    observed = opening + timedelta(seconds=20)
    quote = dict(t='2026-11-27T09:00:19.123456789Z', bp='12.33', ap='12.34')
    if primary == 'expensive':
        quote['ap'] = '12.35'
    elif primary == 'wide':
        quote['bp'] = '12.00'
    elif primary == 'stale':
        quote['t'] = '2026-11-27T09:00:01Z'
    elif primary == 'missing':
        quote = None
    elif primary == 'invalid':
        quote['bp'] = 0
    calls = []

    def webull(symbol):
        calls.append(symbol)
        return depth(symbol, observed - timedelta(seconds=1))

    decision = market.premarket_decision(webull, 'AAA', ticket={'limit_price': '12.34'},
        opening=opening, now=lambda: observed, primary_fetch=lambda s: quote)
    assert (decision['status'], decision['provider']) == (expected_status, expected_provider)
    assert calls == (['AAA'] if expected_provider == 'webull' else [])


@pytest.mark.parametrize('dry,ask,primary,posted', [
    (False, '12.35', None, False), (False, '12.34', None, True), (True, '12.34', None, False),
    (False, '12.35', 'fresh', False), (False, '12.34', 'fresh', True),
    (False, '12.34', 'missing', True), (True, '12.34', 'missing', False),
])
def test_entry_uses_selected_candidate_quote_before_any_order_claim(dry, ask, primary, posted):
    from test_strategy_entry_only import NOW, fixture
    config, args, store, broker, _, _, handoff = fixture()
    config.dry_run = dry
    handoff['session']['next_regular_close_at'] = '2026-11-09T21:00:00+00:00'
    runtime = args['runtime']
    handoff['proposals'].append(dict(symbol='ZZZ', signal_id='z' * 64))
    all_setups = runtime.pipeline_execution_queue
    runtime.pipeline_execution_queue = lambda verified: all_setups(verified)[:1]
    original_reader = runtime.unified_handoff

    def read(*a, **kw):
        verified = original_reader(*a, **kw)
        verified['record']['selection'].update(policy=selection.POLICY)
        return verified

    runtime.unified_handoff = read
    original_marker = args['marker_for']

    def marker(*a, **kw):
        return dict(original_marker(*a, **kw), cancel_at=handoff['session']['next_regular_close_at'])

    calls = []
    primary_calls = []

    def quote(symbol):
        assert not store.rows and not store.outbox and not broker.posts
        calls.append(symbol)
        return depth(symbol, NOW, ask=ask)

    def alpaca(symbol):
        assert not store.rows and not store.outbox and not broker.posts
        primary_calls.append(symbol)
        if primary == 'fresh':
            return dict(t=NOW.isoformat(), bp='12.33', ap=ask)

    args.update(quote_fetch=quote, marker_for=marker)
    if primary is not None:
        args['primary_quote_fetch'] = alpaca
    result = run_entry_only(config, **args)
    assert calls == ([] if dry or primary == 'fresh' else ['AAA'])
    assert primary_calls == ([] if dry or primary is None else ['AAA'])
    assert bool(store.outbox) is posted
    assert bool(broker.posts) is posted
    if not dry:
        decision = result['webull_premarket' if primary is None else 'premarket_quotes']['AAA']
        assert decision['status'] == (
            'PASS' if posted else 'QUOTE_ASK_ABOVE_PIPELINE_LIMIT')
        assert decision['provider'] == ('alpaca_iex' if primary == 'fresh' else 'webull')
    if posted:
        assert broker.posts[0]['limit_price'] == 12.34


@pytest.mark.parametrize('environment', ['production', 'sandbox'])
def test_wire_is_restricted_to_webull_market_data(monkeypatch, environment):
    calls = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, _):
            return [b'[]']

    class Session:
        trust_env = True

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            assert self.trust_env is False
            return Response()

        def close(self):
            pass

    monkeypatch.setattr('requests.Session', Session)
    monkeypatch.setattr(market, 'sandbox_application', lambda: dict(app_key='sandbox-key', app_secret='sandbox-secret'))
    env = dict(JBR_WEBULL_DATA_ENV=environment, JBR_WEBULL_NON_DISPLAY_AUTHORIZED='true',
               JBR_WEBULL_HOSTED_QUALIFIED='true', JBR_WEBULL_DATA_APP_KEY='key',
               JBR_WEBULL_DATA_APP_SECRET='secret', JBR_WEBULL_DATA_ACCESS_TOKEN='token')
    market.require_hosted_qualification(env, environment)
    with pytest.raises(ValueError, match='webull_hosted_data_not_qualified'):
        market.require_hosted_qualification(env, 'sandbox' if environment == 'production' else 'production')
    with market.MarketData(env) as client:
        client.bars(('AAPL',), at('2026-11-27T18:00:00+00:00'), at('2026-11-27T22:00:00+00:00'), 'ATH')
        client.quote('AAPL')
    host = 'https://api.webull.com' if environment == 'production' else 'https://api.sandbox.webull.com'
    assert [(m, u) for m, u, _ in calls] == [('POST', host + market.BARS), ('GET', host + market.DEPTH)]
    body = json.loads(calls[0][2]['data'])
    assert body['trading_sessions'] == 'ATH'
    assert body['end_time'] == int(at('2026-11-27T22:00:00+00:00').timestamp() * 1000)
    assert body['real_time_required'] is (environment == 'production')
    assert calls[1][2]['params']['overnight_required'] == 'false'
    assert all(k['headers']['x-app-key'] == ('key' if environment == 'production' else 'sandbox-key')
               and k['headers'].get('x-access-token') == ('token' if environment == 'production' else None)
               for _, _, k in calls)
    assert all(k['headers'].get('x-signature') and not k['allow_redirects'] for _, _, k in calls)


@pytest.mark.parametrize('lead_minutes', [5, 65])
def test_scheduled_start_waits_without_contacting_webull_or_broker(lead_minutes):
    from test_strategy_entry_only import NOW, fixture
    from test_strategy_entry_recheck import Clock

    config, args, store, broker, _, _, handoff = fixture()
    opening = at(handoff['execution_policy']['entry_open_at'])
    clock = Clock(opening - timedelta(minutes=lead_minutes))
    handoff['session']['next_regular_close_at'] = '2026-11-09T21:00:00+00:00'
    runtime = args['runtime']
    original_reader = runtime.unified_handoff

    def read(*a, **kw):
        response = original_reader(*a, **kw)
        response['record']['selection'].update(policy=selection.POLICY)
        return response

    runtime.unified_handoff = read

    def authorized_broker(*a, **kw):
        assert clock.now() == opening and not store.rows
        return broker

    def quote(symbol):
        assert clock.now() == opening and clock.now() < NOW and not store.outbox
        return depth(symbol, clock.now())

    args.update(now=clock.now, monotonic=clock.monotonic, wait=clock.sleep,
                broker_factory=authorized_broker, quote_fetch=quote)
    result = run_entry_only(config, **args)
    assert sum(clock.sleeps) == lead_minutes * 60
    assert result['orders_submitted'] == 1
