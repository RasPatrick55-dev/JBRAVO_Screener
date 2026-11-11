"""Actual deployed web-reader binding; independent of entry activation."""
import importlib.util
from hashlib import sha256
from pathlib import Path
from isolated_pricing_integration import need,verify_adapter_files,dashboard_adapter,IsolatedPricing,PairedResearchStore
from pricing_adapter_prototype import pricing_contract


def bind_deployed_web(*,preimage_path,preimage_pin,adapter_root,adapter_pins,rules,selection,clock):
    path=Path(preimage_path).resolve(strict=True)
    verify_adapter_files(path.parent,{path.name:preimage_pin})
    verify_adapter_files(adapter_root,adapter_pins)
    raw=path.read_bytes()
    need(dict(bytes=len(raw),sha256=sha256(raw).hexdigest())==preimage_pin,"web_preimage_changed_during_load")
    spec=importlib.util.spec_from_file_location('_isolated_verified_web_preimage',path)
    preimage=importlib.util.module_from_spec(spec)
    # Execute exact verified bytes; no pyc reuse or global module substitution.
    exec(compile(raw,str(path),'exec'),preimage.__dict__)
    adapter=IsolatedPricing(rules=rules,selection=selection,contract=pricing_contract(rules,selection),adapter_binding=adapter_pins,clock=clock)
    reader=PairedResearchStore(None,rules.digest)
    bound=dashboard_adapter(preimage,adapter,row_read=reader.read,parent_context=selection.parent_context,allow_legacy_display=True)
    class Export:
        snapshot=staticmethod(bound.snapshot)
        api_payload=staticmethod(bound.api_payload)
        def __getattr__(self,name):return getattr(preimage,name)
    return Export()
