"""Short, entry-only paper placement with independent broker preflight.

This module is inert on import. A reviewed launcher supplies the pinned native
runtime and Engine. Pipeline prices and historical handoffs remain unchanged.
Monitor Positions owns fill reconciliation, protective stops and close cleanup.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import time


def require(value, reason):
    if not value:
        raise ValueError(reason)


class EntryOnlyBroker:
    """Deny lifecycle mutations even if a native entry method changes later."""
    def __init__(self, broker):
        self.broker = broker

    def __getattr__(self, name):
        return getattr(self.broker, name)

    def submit_order(self, request):
        require(request.get('side') == 'buy' and request.get('type') == 'limit'
                and request.get('time_in_force') == 'day'
                and request.get('extended_hours') is True,
                'entry_only_buy_limit_required')
        return self.broker.submit_order(request)

    def cancel_order(self, *args, **kwargs):
        raise ValueError('entry_only_cancellation_forbidden')

    def replace_order(self, *args, **kwargs):
        raise ValueError('entry_only_replacement_forbidden')


class EntryOnlyStore:
    """Persist the separate close contract before the first order claim."""
    def __init__(self, store, calendar, marker_for):
        self.store, self.calendar, self.marker_for = store, calendar, marker_for

    def __getattr__(self, name):
        return getattr(self.store, name)

    def enroll(self, handoff, proposal, reference):
        marker = self.marker_for(handoff, self.calendar)
        signal = self.store.enroll(handoff, proposal, reference)
        existing = signal['payload'].get('entry_lifecycle')
        if existing is None:
            require(signal['state'] == 'PREPARED' and signal['revision'] == 0
                    and not self.store.orders(signal['signal_id']),
                    'entry_only_existing_signal_not_rebound')
            payload = deepcopy(signal['payload'])
            payload['entry_lifecycle'] = marker
            self.store.save(signal, payload=payload)
        else:
            require(existing == marker, 'entry_only_lifecycle_binding_changed')
        return signal

    def claim(self, **kwargs):
        require(kwargs.get('role') == 'entry' and kwargs.get('operation') == 'POST',
                'entry_only_lifecycle_claim_forbidden')
        return self.store.claim(**kwargs)


def entry_engine(NativeEngine):
    class EntryEngine(NativeEngine):
        def tick(self, **kwargs):
            # Native enter calls tick after each acceptance. Deliberately do not
            # reconcile fills or create stops while holding the entry lease.
            return dict(events=[], owned_symbols=[])

        def sync(self, *args, **kwargs):
            raise ValueError('entry_only_lifecycle_sync_forbidden')

        def manage(self, *args, **kwargs):
            raise ValueError('entry_only_lifecycle_management_forbidden')

    return EntryEngine


def run_entry_only(config, *, runtime, NativeEngine, calendar_reader,
                   marker_for, preflight, base_dir, wait=None, progress,
                   now=None, broker_factory=None):
    """Place the published batch at 04:00 NY, release the lease, then return.

    Fresh direct broker state gates existing owned exposure. Monitor health is
    not read. A successful receipt means accepted entries, never confirmed stops.
    """
    observe = now or (lambda: datetime.now(timezone.utc))
    from scripts.setup_selection import contract as selection_contract
    runtime.load_config(base_dir)
    require(config.paper_research_entries is True and config.source == 'db'
            and config.reconcile_auto is True and not config.reconcile_only
            and not config.diagnostic, 'strategy_entry_process_options_incompatible')
    require(config.allocation_pct == .05 and config.max_positions == 4
            and config.max_new_positions == 3 and config.disable_open_position_cap is False
            and config.min_order_usd == 0 and config.allow_bump_to_one is False
            and config.allow_fractional is False and config.submit_at_ny == '04:00'
            and config.extended_hours is True and config.limit_buffer_pct == 0
            and config.cancel_after_min == 35 and config.chase_enabled is False,
            'strategy_entry_risk_settings_mismatch')
    # The legacy35 flag remains a submission-ticket compatibility setting. It
    # no longer controls cancellation for signals enrolled by this release.
    env = runtime.environment()
    sleeper = wait or time.sleep
    with runtime.source_lease(base_dir):
        inspected = runtime.readonly_handoff(env, base_dir=base_dir, inspection=True)
        runtime.source_guard(inspected, base_dir=base_dir)
        progress['phase'] = 'UNIFIED_SELECTION_HANDOFF'
        verified = runtime.unified_handoff(env, inspected, base_dir=base_dir)
        queue = runtime.pipeline_execution_queue(verified)
        if not queue or config.dry_run:
            value = dict(status='skipped',
                exit_reason='STRATEGY_DRY_RUN_INPUTS_ONLY' if config.dry_run else 'NO_STRATEGY_BUY_SIGNALS',
                candidate_count=len(queue), orders_submitted=0, orders_filled=0,
                strategy_contract=selection_contract(),
                paper_only=True, broker_operations=0, finished_at=observe().isoformat(),
                source_bindings=inspected['source_bindings'])
            runtime.write_report(value, role='entry', base_dir=base_dir)
            return value
        handoff = runtime.readonly_handoff(env, base_dir=base_dir, pipeline_limits=True)
        verified = runtime.unified_handoff(env, handoff, base_dir=base_dir)
        queue = runtime.pipeline_execution_queue(verified)
        opening = runtime.rules.utc(handoff['execution_policy']['entry_open_at'])
        require(opening <= observe() < opening + timedelta(seconds=60),
                'entry_only_premarket_submission_window_closed')
        calendar = calendar_reader(env, day=date.fromisoformat(handoff['session']['next_session']))
        # Freeze one calendar-derived close for the whole batch. No fallback to
        # a guessed16:00 when the exchange is closed or calendar is unavailable.
        marker = marker_for(handoff, calendar)
        # Keep outer failure receipts truthful even if a later symbol's guard
        # fails after an earlier buy was accepted.
        config._entry_cancellation_policy = 'monitor_exchange_regular_close'
        config._entry_lifecycle_owner = 'MONITOR_POSITIONS'
        config._entry_cancel_at = marker['cancel_at']
        references = {s['proposal']['symbol']: s['handoff']['pipeline_limit_order'] for s in queue}
        runtime.expected_paper_account(env)
        if broker_factory is None:
            from scripts.strategy_execution_broker import PaperBroker
            broker_factory = PaperBroker
        progress.update(phase='BROKER_PREFLIGHT', broker_started=True)
        broker = broker_factory(env, max_requests=2048, market_data=False)
        conn = None
        try:
            runtime.entry_account(broker, env)
            conn, store = runtime.store_connection(env, broker)
            with runtime.entry_lease(store, opening, wait=sleeper):
                before = {s['signal_id'] for s in store.signals(active=False)}
                existing_orders = {r['client_order_id'] for s in store.signals(active=False)
                                   for r in store.orders(s['signal_id'])}
                proof = preflight(runtime, NativeEngine, store, broker, base_dir=base_dir,
                    handoff=handoff, proposed_symbols={s['proposal']['symbol'] for s in queue})
                scoped = EntryOnlyStore(store, calendar, marker_for)
                core_guard = runtime.guard_for(env, base_dir=base_dir)

                def guard(kind, signal):
                    require(kind == 'entry', 'entry_only_lifecycle_guard_forbidden')
                    require(opening <= observe() < opening + timedelta(seconds=60),
                            'entry_only_premarket_submission_window_closed')
                    require(signal['payload'].get('entry_lifecycle') ==
                            marker_for(signal['payload']['handoff'], calendar),
                            'entry_only_lifecycle_binding_changed')
                    return core_guard(kind, signal)

                engine = entry_engine(NativeEngine)(scoped, EntryOnlyBroker(broker),
                    guard=guard, pipeline_orders=True, now=observe)
                progress['phase'] = 'STRATEGY_PAPER_ENTRY'
                for setup in queue:
                    require(opening <= observe() < opening + timedelta(seconds=60),
                            'entry_only_premarket_submission_window_closed')
                    origin = setup['handoff']
                    runtime.source_guard(origin, base_dir=base_dir)
                    current = {s['signal_id'] for s in store.signals(active=False)}
                    preflight(runtime, NativeEngine, store, broker, base_dir=base_dir,
                        handoff=origin, proposed_symbols={setup['proposal']['symbol']},
                        exempt_signal_ids=current - before)
                    engine.enter(origin, references, now=observe(), clock=broker.clock(),
                        account=runtime.entry_account(broker, env), positions=broker.positions(),
                        open_orders=broker.open_orders(),
                        max_new_entries=runtime.available_entry_budget(store, handoff['session']['next_session']),
                        selected_signal_ids=[setup['proposal']['signal_id']])
                records = [r for s in store.signals(active=False) for r in store.orders(s['signal_id'])]
                new = [r for r in records if r['role'] == 'entry'
                       and r['client_order_id'] not in existing_orders]
                unresolved = any(r['state'] in {'CLAIMED', 'UNKNOWN'} for r in records)
                refused = any(r['state'] == 'REJECTED' or
                              (r.get('observation') or {}).get('status') == 'rejected' for r in new)
                observations = [r['observation'] for r in new if r.get('observation')]
                value = dict(status='blocked' if unresolved or refused else 'ok',
                    exit_reason='ORDER_OUTCOME_UNKNOWN' if unresolved else 'BROKER_REJECTED' if refused
                    else 'ENTRY_LIMITS_PLACED_MONITOR_OWNS_LIFECYCLE',
                    candidate_count=len(queue), orders_submitted=len(observations),
                    strategy_contract=selection_contract(),
                    orders_filled=sum(o['status'] == 'filled' for o in observations),
                    order_observations=observations, paper_only=True,
                    lifecycle_owner='MONITOR_POSITIONS', cancellation_deadline=marker['cancel_at'],
                    entry_lifecycle=marker, protection_confirmed=False,
                    extended_hours_protection_confirmed=False,
                    protection_status='MONITOR_RECONCILIATION_REQUIRED',
                    independent_monitor_status='NOT_A_SUBMISSION_DEPENDENCY',
                    preflight=proof, events=runtime.compact_events(engine.events),
                    handoff_id=handoff['handoff_id'], primary_report_sha256=handoff['primary_report_sha256'],
                    source_bindings=handoff['source_bindings'],
                    selection=verified['record']['selection'],
                    unified_selection_run_id=verified['record']['run_id'],
                    finished_at=observe().isoformat())
                value['preflight'] = dict(proof, candidate_binding=dict(
                    unified_selection_run_id=verified['record']['run_id'],
                    selected_setups=verified['record']['selection'].get('selected_setups', [])),
                    entry_authorization='paper_strategy_test', strategy_contract=selection_contract())
            # Lease has already been released, allowing the independent monitor
            # to reconcile accepted/partial fills even if receipt output fails.
            runtime.write_report(value, role='entry', base_dir=base_dir)
            return value
        finally:
            if conn is not None:
                conn.rollback()
                conn.close()
            broker.close()
