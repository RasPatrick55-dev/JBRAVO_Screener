"""Local definitions for the Webull-assisted PythonAnywhere paper tasks."""
import argparse
from contextlib import ExitStack
from datetime import date, datetime, time, timedelta
from functools import partial
import json
import os
from pathlib import Path

from scripts.strategy_webull_market_data import (
    MarketData, NY, alpaca_quote, bar_groups, capture_after_hours, premarket_decision, require,
    session_hours, webull_fallback,
)

ROOT = Path(__file__).resolve().parents[1]


def private_environment():
    from dotenv import dotenv_values
    values = dotenv_values(Path.home() / '.config/jbravo/.env', interpolate=False)
    return {**{k: v for k, v in values.items() if v is not None}, **os.environ}


def read_unified(env, parent, *, base_dir):
    from datetime import timezone
    from scripts.research_store import connect
    from scripts.strategy_webull_selection import read_selection

    connection = connect(env=env)
    try:
        return read_selection(connection, parent, base_dir=base_dir, now=datetime.now(timezone.utc))
    finally:
        connection.rollback()
        connection.close()


def run_entry(env, *, place):
    from scripts import primary_paper_pipeline
    primary_paper_pipeline.configure_paper_context(os.environ)
    from scripts import strategy_only_monitor as monitor
    # For the scheduled invocation, skip non-sessions before reading a prior
    # Primary report. Input inspection remains available outside the slot.
    if place:
        from scripts.daily_strategy_scan import Provider, load_config
        environment = load_config().get('webull_data_environment', 'production')
        day = datetime.now(NY).date()
        calendar = monitor.calendar_session({**env, 'JBR_EXEC_PAPER': 'true'}, day=day)
        if calendar is None:
            print(json.dumps(dict(status='SKIPPED_NON_TRADING_DAY', broker_operations=0)))
            return 0

    from scripts import (
        execute_trades, strategy_entry_only, strategy_entry_preflight, strategy_execution_runtime,
        strategy_order_lifecycle, strategy_order_ownership_install,
    )
    from scripts.strategy_execution_engine import Engine

    def inspection_only_quote(_symbol):
        raise RuntimeError('webull_quote_forbidden_during_inspection')

    def read_entry_selection(runtime_env, parent, *, base_dir):
        if place:
            require(parent['session'].get('webull_data_environment') == environment,
                    'webull_selection_environment_changed')
        return read_unified(runtime_env, parent, base_dir=base_dir)

    with ExitStack() as resources:
        quote = inspection_only_quote
        primary_quote = None
        if place:
            quote = webull_fallback(resources, env, environment, 'quote')
            alpaca = Provider(env.get('APCA_API_KEY_ID'), env.get('APCA_API_SECRET_KEY'),
                              max_gets=3, max_seconds=67 * 60, max_bytes=192 * 1024)
            resources.callback(alpaca.close)
            primary_quote = partial(alpaca_quote, alpaca)
        strategy_order_ownership_install.install_entry(
            strategy_execution_runtime, Engine, monitor, strategy_entry_only,
            strategy_order_lifecycle, strategy_entry_preflight, source_variant='canonical',
            quote_fetch=quote, primary_quote_fetch=primary_quote, unified_reader=read_entry_selection)
        config = execute_trades.ExecutorConfig(paper_research_entries=True, strategy_premarket_entries=True,
            dry_run=not place, max_positions=4, max_new_positions=3, disable_open_position_cap=False,
            min_order_usd=0, allow_bump_to_one=False, limit_buffer_pct=0, cancel_after_min=35)
        result = strategy_execution_runtime.run_entry(config, base_dir=ROOT)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return int(result['status'] == 'blocked')


def run_probe(args, env):
    """Read-only provider qualification. No candidate, account, or order calls."""
    now = lambda: datetime.now(NY)
    opening = datetime.combine(args.session, time(4), NY)
    if args.kind == 'premarket':
        require(opening <= now() < opening + timedelta(minutes=1), 'webull_probe_outside_first_minute')
    with MarketData(env) as market:
        if args.kind == 'premarket':
            # Qualification instruments, independent of any Primary candidate list.
            # The high fixture limit isolates entitlement, timeliness and spread.
            outcomes = {symbol: premarket_decision(market.quote, symbol,
                ticket={'limit_price': '1000000.00'}, opening=opening, now=now)
                for symbol in ('AAPL', 'MSFT')}
            qualified = all(item['status'] == 'PASS' for item in outcomes.values())
        else:
            from scripts import daily_strategy_scan as producer
            day = args.session.isoformat()
            provider = producer.Provider(env.get('APCA_API_KEY_ID'), env.get('APCA_API_SECRET_KEY'),
                                         max_gets=1, max_seconds=15)
            try:
                rows, _ = provider.get(producer.CALENDAR, dict(start=day, end=day))
            finally:
                provider.close()
            require(isinstance(rows, list) and len(rows) == 1, 'webull_probe_calendar_unavailable')
            row = producer.normalize_calendar(rows)[0]
            require(row['date'] == day, 'webull_probe_calendar_unavailable')
            hours = session_hours(row, producer.load_config()['webull_early_close_cutoffs'])
            ordinary_close = datetime.combine(args.session, time(20), NY)
            require(now() >= ordinary_close + timedelta(minutes=15), 'webull_probe_after_hours_not_complete')
            symbols = ('AAPL', 'MSFT')
            capture = capture_after_hours(symbols, hours, fetch=market.bars, now=now,
                                          environment=market.environment)
            cutoff = datetime.fromisoformat(hours['after_hours_close_at'])
            late = bar_groups(market.bars, symbols, cutoff, ordinary_close, 'ATH') if cutoff < ordinary_close else {}
            outcomes = {symbol: dict(status=item['status'], bars=len(item['bars']),
                last_bar_at=item['bars'][-1]['time'] if item['bars'] else None,
                regular_bar_at=item['regular_bar']['time'] if item['regular_bar'] else None,
                bars_past_cutoff=None if late.get(symbol, []) is None else len(late.get(symbol, [])))
                for symbol, item in capture['symbols'].items()}
            qualified = all(item['status'] == 'AVAILABLE' and item['bars_past_cutoff'] == 0
                            for item in outcomes.values())
    print(json.dumps(dict(status='PASS' if qualified else 'HOLD', session=args.session.isoformat(),
        environment=env['JBR_WEBULL_DATA_ENV'], market_data_only=True, outcomes=outcomes), sort_keys=True))
    return 0 if qualified else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    sub.add_parser('producer', help='Capture and archive scanner finalists from the completed session')
    entry = sub.add_parser('entry', help='Inspect tickets or run the scheduled Alpaca paper entry')
    entry.add_argument('--place-paper-orders', action='store_true',
                       help='Enable paper placement in the first premarket minute')
    probe = sub.add_parser('probe', help='Run a market-data-only Webull qualification')
    probe.add_argument('kind', choices=('after-hours', 'premarket'))
    probe.add_argument('--session', required=True, type=date.fromisoformat)
    args = parser.parse_args(argv)
    from scripts.strategy_webull_quotes import QuoteError
    try:
        env = private_environment()
        if args.task == 'probe':
            return run_probe(args, env)
        from scripts import candidate007_worker_fence as fence

        def run_task():
            if args.task == 'entry':
                return run_entry(env, place=args.place_paper_orders)
            from scripts.daily_strategy_scan import main as producer
            config = json.loads((ROOT / 'config/strategy_pipeline.json').read_text(encoding='utf-8'))
            return producer(config['producer_root'])

        return fence.run_guarded(fence.DIRECTORY, run_task)
    except QuoteError as error:
        print(json.dumps(dict(status='HOLD', reason=str(error))))
    except Exception as error:
        safe = {'webull_hosted_data_not_qualified', 'webull_probe_outside_first_minute',
                'primary_paper_endpoint_required', 'primary_paper_context_conflict',
                'webull_data_environment_required', 'webull_non_display_authorization_required',
                'webull_probe_calendar_unavailable', 'webull_probe_after_hours_not_complete',
                'webull_data_credentials_unavailable', 'webull_sandbox_credentials_unavailable',
                'webull_capture_not_ready_or_too_large',
                'unverified_webull_early_close', 'webull_calendar_close_mismatch'}
        reason = str(error) if type(error) is ValueError and str(error) in safe else 'DETAIL_WITHHELD'
        print(json.dumps(dict(status='HOLD', reason=reason)))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
