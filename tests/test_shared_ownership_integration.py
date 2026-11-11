"""Real canonical ownership composition in an isolated process, no runtime run."""
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.alpaca_optional


def test_real_canonical_ownership_assembly_without_runtime_effects():
    root = Path(__file__).resolve().parents[1]
    code = r"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
from datetime import datetime,timezone
import importlib,json,socket
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden'))
from scripts.shared_pricing_release import NATIVE
from shared_native_integration import install_shared_native
from shared_ownership_integration import assemble_ownership
root=Path.cwd()
paths=list(root.glob('*.py'))+list((root/'scripts').rglob('*.py'))+list((root/'config').glob('*.json'))
pins={p.relative_to(root).as_posix():{'bytes':len(p.read_bytes()),'sha256':sha256(p.read_bytes()).hexdigest()} for p in paths}
native=SimpleNamespace(**{n:importlib.import_module('scripts.'+n) for n in NATIVE})
guard=native.strategy_execution_runtime.source_guard
bound=install_shared_native(native,root=root,adapter_pins=pins,clock=lambda:datetime.now(timezone.utc))
proof=assemble_ownership(native,bound,root=root,pins=pins)
assert native.strategy_execution_runtime.source_guard is guard
assert proof['entry_invocations']==proof['broker_calls']==proof['database_calls']==0
assert proof['source_guard_preserved'] and proof['legacy_monitor_gate_preserved']
assert proof['entry_monitor_dependency'] is False
assert proof['direct_ownership_preflight_retained']
assert proof['monitor_installation']['existing_history_preserved']
assert proof['receipt_policy_composed']
from scripts import strategy_entry_only as entry
captured={}
def inspect_entry(*args,**kwargs):captured.update(kwargs);return 'BINDING_ONLY'
entry.run_entry_only=inspect_entry
assert native.strategy_execution_runtime._run_entry(None,progress={})=='BINDING_ONLY'
bound.entry_runtime.guard_for=lambda env,**kwargs:kwargs
assert captured['runtime'].guard_for({},base_dir=root)['pipeline_entry_only'] is True

print('OWNERSHIP_ASSEMBLED_NO_RUN')
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OWNERSHIP_ASSEMBLED_NO_RUN' in result.stdout


def test_canonical_pending_variant_requires_exact_source():
    from scripts.strategy_execution_engine import Engine
    from scripts import strategy_order_lifecycle as lifecycle
    from scripts import strategy_pending_stop_replacement as pending
    from scripts.strategy_only_monitor import quote_safe_engine
    wrapped=quote_safe_engine(lifecycle.monitor_engine(Engine,source_variant="canonical"))
    assert issubclass(pending.monitor_engine(wrapped,NativeEngine=Engine,
                                            source_variant='canonical'),Engine)
    with pytest.raises(ValueError,match='source_variant_invalid'):
        pending.monitor_engine(wrapped,NativeEngine=Engine,source_variant='unreviewed')
    class Changed(Engine):
        def manage(self,*args,**kwargs):return super().manage(*args,**kwargs)
    with pytest.raises(ValueError,match='manage_source_changed'):
        pending.monitor_engine(Changed,NativeEngine=Changed,source_variant='canonical')


def test_canonical_lifecycle_variant_rejects_unknown_and_drift():
    from scripts.strategy_execution_engine import Engine
    from scripts import strategy_order_lifecycle as lifecycle
    assert issubclass(lifecycle.monitor_engine(Engine,source_variant='canonical'),Engine)
    with pytest.raises(ValueError,match='source_variant_invalid'):
        lifecycle.monitor_engine(Engine,source_variant='unreviewed')
    class Changed(Engine):
        def sync(self,*args,**kwargs):return super().sync(*args,**kwargs)
    with pytest.raises(ValueError,match='sync_source_changed'):
        lifecycle.monitor_engine(Changed,source_variant='canonical')
