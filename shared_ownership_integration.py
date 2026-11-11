"""Explicit process-local ownership assembly; never run jobs or open accounts."""
from pathlib import Path
import importlib
import json
from types import SimpleNamespace

from isolated_pricing_integration import need, install_entry_runner, verify_adapter_files


def assemble_ownership(native, bound, *, root, pins):
    """Keep canonical guards and ownership modules; no historical overlay replay.

    The complete release loader must have verified fresh native modules first.
    This function only installs callables. It is not activation or host admission.
    """
    root = Path(root).resolve(strict=True)
    verify_adapter_files(root, pins)
    names = ('strategy_execution_engine', 'strategy_only_monitor',
             'strategy_entry_only', 'strategy_order_lifecycle', 'strategy_entry_preflight',
             'strategy_order_ownership_install', 'strategy_pending_stop_replacement', 'executor_health')
    modules = {name: importlib.import_module('scripts.' + name) for name in names}
    for name, module in modules.items():
        relative = 'scripts/' + name + '.py'
        need(relative in pins and Path(module.__file__).resolve() == root / relative,
             'shared_ownership_module_not_bound')
    runtime = native.strategy_execution_runtime
    need(Path(runtime.__file__).resolve() == root / 'scripts/strategy_execution_runtime.py',
         'shared_ownership_runtime_identity_changed')
    guard = runtime.source_guard
    lifecycle = modules['strategy_order_lifecycle']
    Engine = modules['strategy_execution_engine'].Engine
    # The lifecycle module checks exact sync/tick source and deadline expressions.
    lifecycle.monitor_engine(Engine)
    config_name='config/shared_pricing_execution.json'
    need(config_name in pins, 'shared_ownership_configuration_not_bound')
    def load_config(base_dir=root):
        need(Path(base_dir).resolve()==root, 'shared_ownership_config_root_changed')
        verify_adapter_files(root, {config_name:pins[config_name]})
        value=json.loads(native.pipeline_postflight._read(root/config_name,4096),
            object_pairs_hook=native.pipeline_postflight._strict_object,
            parse_constant=native.pipeline_postflight._invalid_constant)
        expected=dict(schema='jbravo.pipeline-limit-paper-execution-config.v1',
            enabled=True,paper_only=True,strategy_id='trend_pullback_reclaim_qualifying_close_v2',
            execution_model=native.strategy_execution.PIPELINE_LIMIT_MODEL,
            allocation_pct=.05,max_positions=4,extra_limit_buffer_pct=0,cancel_after_minutes=35,
            limit_source='PIPELINE_PUBLISHED_LIMIT',currency='USD',research_qualified=False)
        need(value==expected, 'shared_ownership_configuration_changed')
        return value
    load_config()
    class EntryRuntime:
        def __getattr__(self,name):return getattr(bound.entry_runtime,name)
    facade=EntryRuntime()
    facade.load_config=load_config
    # Validate the complete composed monitor class before any endpoint mutation.
    monitor=modules['strategy_only_monitor']
    ownership=modules['strategy_order_ownership_install']
    pending=modules['strategy_pending_stop_replacement']
    def canonical_pending(Engine, *, NativeEngine):
        return pending.monitor_engine(Engine,NativeEngine=NativeEngine,source_variant='canonical')
    pending_adapter=SimpleNamespace(monitor_engine=canonical_pending)
    canonical_pending(monitor.quote_safe_engine(lifecycle.monitor_engine(Engine)),
                      NativeEngine=Engine)
    health=modules['executor_health']
    receipt=SimpleNamespace(publish_receipt=health.publish_receipt)
    ownership.install_receipt_policy(receipt,expected_function_sha256=
        '819e10cb5851f12d6cb50adda92d9f7e226fa7f366460617988278560b7d2a8a')
    install_entry_runner(runtime, facade=facade, NativeEngine=Engine,
        entry=modules['strategy_entry_only'], monitor=modules['strategy_only_monitor'],
        lifecycle=lifecycle, preflight=modules['strategy_entry_preflight'])
    monitor=modules['strategy_only_monitor']
    def verify_native(base_dir):
        need(Path(base_dir).resolve()==root, 'shared_monitor_root_changed')
        verify_adapter_files(root,pins)
    # The new manifest's verified source closure supersedes historical hardcoded
    # interface pins only inside this explicitly assembled isolated process.
    monitor.verify_native=verify_native
    ownership=modules['strategy_order_ownership_install']
    monitor_proof=ownership.install_monitor(monitor,Engine,lifecycle,
        pending_replacement=pending_adapter)
    health.publish_receipt=receipt.publish_receipt
    need(runtime.source_guard is guard, 'shared_ownership_guard_changed')
    return {'entry_only': True, 'lifecycle_owner': 'MONITOR_POSITIONS',
            'source_guard_preserved': True, 'monitor_readiness_gate_preserved': True,
            'monitor_installation': monitor_proof, 'receipt_policy_composed': True,
            'entry_invocations': 0, 'broker_calls': 0, 'database_calls': 0,
            'activation_authorized': False}
