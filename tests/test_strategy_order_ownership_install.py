from copy import deepcopy
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from scripts import strategy_order_lifecycle as lifecycle
from scripts import strategy_order_ownership_install as install
from tests.test_strategy_order_lifecycle import native, native_rules  # pinned deployed fixtures

pytestmark = pytest.mark.alpaca_optional


def publish_receipt(config):
    return {'cancellation_policy': ('frozen_premarket_session_0435'
            if getattr(config, 'strategy_premarket_entries', False) else 'minutes_from_submission'),
            'unchanged': ['retained']}


def test_receipt_policy_is_truthful_without_changing_original():
    health = SimpleNamespace(publish_receipt=publish_receipt)
    source = textwrap.dedent(inspect.getsource(publish_receipt)).strip()
    install.install_receipt_policy(health, expected_function_sha256=hashlib.sha256(source.encode()).hexdigest())
    legacy = SimpleNamespace(strategy_premarket_entries=True)
    updated = SimpleNamespace(strategy_premarket_entries=True,
                              _entry_cancellation_policy='monitor_exchange_regular_close')
    assert health.publish_receipt(updated)['cancellation_policy'] == 'monitor_exchange_regular_close'
    assert health.publish_receipt(legacy) == publish_receipt(legacy)
    assert publish_receipt(updated)['cancellation_policy'] == 'frozen_premarket_session_0435'
    assert health.publish_receipt(updated)['unchanged'] == ['retained']


def test_receipt_source_drift_refused():
    health = SimpleNamespace(publish_receipt=publish_receipt)
    with pytest.raises(ValueError, match='source_changed'):
        install.install_receipt_policy(health, expected_function_sha256='0' * 64)
    assert health.publish_receipt is publish_receipt


def test_monitor_installs_correct_derived_base_and_refuses_double_install(native):
    seen = []
    monitor = SimpleNamespace(quote_safe_engine=lambda e: seen.append(e) or e)
    old = native.Engine.sync
    install.install_monitor(monitor, native.Engine, lifecycle)
    derived = monitor.quote_safe_engine(native.Engine)
    assert issubclass(derived, native.Engine) and derived.sync is not old
    assert native.Engine.sync is old
    with pytest.raises(ValueError, match='duplicate_install'):
        install.install_monitor(monitor, native.Engine, lifecycle)
    with pytest.raises(ValueError, match='identity_changed'):
        monitor.quote_safe_engine(object)


def test_entry_install_preserves_outer_report_handler_and_only_replaces_inner(native):
    old = lambda *a, **k: 'outer'
    runtime = SimpleNamespace(ROOT='unused', run_entry=old, _run_entry=object())
    calendar = lambda *a, **k: None
    guard = lambda *a, **k: None
    monitor = SimpleNamespace(calendar_session=calendar)
    preflight = SimpleNamespace(broker_entry_preflight=guard)
    entry = SimpleNamespace(run_entry_only=lambda config, **kw: kw)
    install.install_entry(runtime, native.Engine, monitor, entry, lifecycle, preflight)
    result = runtime._run_entry('config', progress={'phase': 'test'})
    assert result['calendar_reader'] is calendar and result['preflight'] is guard
    assert runtime.run_entry is old
    with pytest.raises(ValueError, match='duplicate_install'):
        install.install_entry(runtime, native.Engine, monitor, entry, lifecycle, preflight)
