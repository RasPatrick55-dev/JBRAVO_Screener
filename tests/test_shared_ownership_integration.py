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
assert proof['source_guard_preserved'] and proof['monitor_readiness_gate_preserved']
assert proof['monitor_installation']['existing_history_preserved']
assert proof['receipt_policy_composed']
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
    wrapped=quote_safe_engine(lifecycle.monitor_engine(Engine))
    assert issubclass(pending.monitor_engine(wrapped,NativeEngine=Engine,
                                            source_variant='canonical'),Engine)
    with pytest.raises(ValueError,match='source_variant_invalid'):
        pending.monitor_engine(wrapped,NativeEngine=Engine,source_variant='unreviewed')
    class Changed(Engine):
        def manage(self,*args,**kwargs):return super().manage(*args,**kwargs)
    with pytest.raises(ValueError,match='manage_source_changed'):
        pending.monitor_engine(Changed,NativeEngine=Changed,source_variant='canonical')
