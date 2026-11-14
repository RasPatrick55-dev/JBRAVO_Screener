"""Offline native provenance and dependency-closure contracts. No broker/DB IO."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.alpaca_optional
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'docs/verification/hourly-outer-dependencies-20261114.json').read_text())

@pytest.mark.parametrize('name', sorted(MANIFEST['sources']))
def test_exact_native_provenance(name):
    data = (ROOT / 'scripts' / name).read_bytes()
    record = MANIFEST['sources'][name]
    assert len(data) == record['bytes']
    assert hashlib.sha256(data).hexdigest() == record['sha256']

@pytest.mark.parametrize('name', sorted(MANIFEST['sources']))
def test_script_dependency_closure(name):
    tree = ast.parse((ROOT / 'scripts' / name).read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts':
            for alias in node.names:
                assert (ROOT / 'scripts' / (alias.name + '.py')).is_file(), alias.name

@pytest.mark.parametrize('name', sorted(MANIFEST['sources']))
def test_import_without_network_database_or_order_calls(name, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('Import must not connect to network or database')
    monkeypatch.setattr(socket, 'socket', forbidden)
    module = importlib.import_module('scripts.' + name[:-3])
    assert Path(module.__file__).resolve() == (ROOT / 'scripts' / name).resolve()
