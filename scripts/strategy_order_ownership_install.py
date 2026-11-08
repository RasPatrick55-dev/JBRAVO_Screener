"""Explicit isolated-release wiring. No import-time installation or effects.

The bootstrap must verify the complete release and native source inventories
before calling these functions. Only process-local callable bindings change.
"""
import ast
import hashlib
import inspect
import textwrap


def require(value, reason):
    if not value:
        raise ValueError(reason)


def install_entry(runtime, NativeEngine, monitor, entry, lifecycle, preflight):
    require(not getattr(runtime, '_entry_monitor_ownership_installed', False),
            'entry_ownership_duplicate_install')
    # Fail before any CLI/account operation if the monitor cannot interpret the
    # native Engine's close contract. This is source verification, not health.
    lifecycle.monitor_engine(NativeEngine)
    def run(config, *, base_dir=runtime.ROOT, wait=None, progress):
        return entry.run_entry_only(config, runtime=runtime, NativeEngine=NativeEngine,
            calendar_reader=monitor.calendar_session, marker_for=lifecycle.marker_for,
            preflight=preflight.broker_entry_preflight, base_dir=base_dir,
            wait=wait, progress=progress)
    runtime._run_entry = run
    runtime._entry_monitor_ownership_installed = True
    return dict(lifecycle_owner='MONITOR_POSITIONS', entry_only=True,
                monitor_health_dependency=False, native_source_files_changed=False)


def install_monitor(monitor, NativeEngine, lifecycle):
    require(not getattr(monitor, '_entry_monitor_ownership_installed', False),
            'monitor_ownership_duplicate_install')
    Derived = lifecycle.monitor_engine(NativeEngine)
    original = monitor.quote_safe_engine
    def quote_safe(Engine):
        require(Engine is NativeEngine, 'monitor_ownership_engine_identity_changed')
        return original(Derived)
    monitor.quote_safe_engine = quote_safe
    monitor._entry_monitor_ownership_installed = True
    return dict(lifecycle_owner='MONITOR_POSITIONS',
                new_entry_deadline='EXCHANGE_REGULAR_CLOSE',
                existing_history_preserved=True, native_source_files_changed=False)


def install_receipt_policy(health, *, expected_function_sha256):
    """Keep native writer/integrity semantics; label the new cutoff honestly."""
    original = health.publish_receipt
    source = textwrap.dedent(inspect.getsource(original)).strip()
    require(hashlib.sha256(source.encode()).hexdigest() == expected_function_sha256,
            'entry_ownership_receipt_source_changed')
    tree = ast.parse(source)
    class Rewrite(ast.NodeTransformer):
        count = 0
        def visit_Constant(self, node):
            if node.value == 'frozen_premarket_session_0435':
                self.count += 1
                return ast.copy_location(ast.parse(
                    "getattr(config, '_entry_cancellation_policy', 'frozen_premarket_session_0435')",
                    mode='eval').body, node)
            return node
    rewrite = Rewrite()
    tree = rewrite.visit(tree)
    require(rewrite.count == 1, 'entry_ownership_receipt_policy_expression_changed')
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<entry-ownership-receipt>', 'exec'), namespace)
    health.publish_receipt = namespace['publish_receipt']
