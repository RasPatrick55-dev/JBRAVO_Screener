"""Isolated native integration; explicit process-local installation only.

Never writes files, activates tasks, reloads web workers or submits orders.
Existing native source and ownership functions remain byte-identical.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from isolated_pricing_integration import need, verify_adapter_files
from shared_pricing_wiring import wire_shared

HELPERS = ('shared_pricing_contract.py', 'shared_pricing_wiring.py',
           'isolated_pricing_integration.py', 'pricing_adapter_prototype.py',
           'shared_native_integration.py')
ENDPOINTS = ('read_selection', 'execution_queue', 'dashboard_rows', 'dashboard_reports', 'publish')


def install_shared_native(native, *, root, adapter_pins, clock):
    root = Path(root).resolve(strict=True)
    need(set(HELPERS) <= set(adapter_pins), 'shared_native_helper_inventory_incomplete')
    verify_adapter_files(root, adapter_pins)
    selection = native.setup_selection
    pipeline = native.strategy_pipeline
    need(not getattr(selection, '_shared_pricing_installed', False), 'shared_native_duplicate_install')
    # Snapshot original functions before modifying supported module entry points.
    original = SimpleNamespace(**vars(selection))
    prior_sources = tuple(pipeline.EXTRA_SOURCES)
    extended_sources = tuple(dict.fromkeys((*prior_sources, *adapter_pins)))
    def check_sources(pins):
        verify_adapter_files(root, pins)
    def build(env, market_get=None, report_status=None):
        return wire_shared(root=root, pins=adapter_pins, rules=native.strategy_execution,
            selection=original, runtime=native.strategy_execution_runtime,
            connect=lambda **kw:native.research_store.connect(env=env, **kw),
            research_connect=native.research_store.connect, market_get=market_get,
            check_sources=check_sources, report_status=report_status, base_dir=root,
            challenger=native.research_strategies.CHALLENGER, clock=clock,
            broker_error_type=native.strategy_execution_broker.BrokerError)
    readers = build({})
    def publish(primary, *, env, base_dir=root):
        need(Path(base_dir).resolve() == root, 'shared_native_publication_root_changed')
        need(all(primary['source_bindings'].get(n) == pin for n,pin in adapter_pins.items()),
             'shared_native_publication_closure_missing')
        publication_env = dict(env)
        broker = None
        def get(path, params):
            nonlocal broker
            need(path in ('/v2/stocks/bars', '/v2/stocks/trades'), 'shared_native_data_path_forbidden')
            if broker is None:
                broker = native.strategy_execution_broker.PaperBroker(publication_env,max_requests=5,now=clock)
            return broker.request('GET', path, data=True, params=params)
        def inputs(primary):
            conn = native.research_store.connect(env=publication_env)
            try:
                return original.inputs(conn, original.parent_context(primary), now=clock(), base_dir=root)
            finally:
                conn.rollback(); conn.close()
        wired = build(publication_env, get,
            lambda status:native.bull_flag_companion._status(status,root,'unified_setups'))
        try:
            return wired.publish(primary, inputs=inputs)
        finally:
            if broker is not None:broker.close()
    replacements = dict(read_selection=readers.read_selection,
        execution_queue=readers.native_execution_queue,
        dashboard_rows=readers.dashboard_rows, dashboard_reports=readers.dashboard_reports,
        publish=publish)
    # All construction and source checks precede mutation. No provider/DB calls.
    for name, function in replacements.items():setattr(selection, name, function)
    pipeline.EXTRA_SOURCES = extended_sources
    selection._shared_pricing_installed = True
    return SimpleNamespace(readers=readers, original=original, entry_runtime=readers.entry_runtime,
        proof=dict(native_files_changed=False, order_actions=0, connections_opened=0,
                   source_closure=deepcopy(adapter_pins), installed_endpoints=list(ENDPOINTS),
                   prior_sources=list(prior_sources), extended_sources=list(extended_sources),
                   runtime_guard_changed=False, ownership_functions_changed=False))

