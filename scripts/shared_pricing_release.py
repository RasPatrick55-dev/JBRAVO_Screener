"""Offline, exact-git-commit release builder and assembly-only qualification.

The manifest digest is an OUT-OF-BAND trust input, not a signature. Save the
builder's digest with the reviewed commit; never obtain it from an untrusted
release. No command here activates a scheduler, WSGI worker, or trading run.
Use a fresh subprocess for ``assemble``; its process-local bindings die on exit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import SimpleNamespace

SCHEMA = 'jbravo.shared-pricing-release.v1'
MANIFEST = 'release-manifest.json'
HELPERS = ('shared_ownership_integration.py', 'shared_native_integration.py', 'shared_pricing_contract.py',
           'shared_pricing_wiring.py', 'isolated_pricing_integration.py',
           'pricing_adapter_prototype.py')
NATIVE = ('strategy_execution', 'setup_selection', 'strategy_execution_runtime',
          'research_store', 'bull_flag_companion', 'pipeline_postflight',
          'strategy_execution_broker', 'research_strategies', 'primary_pipeline',
          'strategy_pipeline', 'strategy_dashboard')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and '\x00' not in name, 'release_path_invalid')
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
            'release_path_invalid')
    return name


def selected(name):
    # Conservative complete Python inventory, including lazy imports and tests.
    # Configuration directories are copied whole, not inferred from host imports.
    return (name.endswith('.py') or name.startswith(('config/', 'configs/'))
            or name in ('config.json', 'pyproject.toml', 'requirements.txt',
                        'requirements-dev.txt'))


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def build_release(repo, commit, destination):
    """Read raw object bytes, never checkout filters or working-tree contents."""
    require(re.fullmatch(r'[0-9a-fA-F]{7,40}', commit) is not None,
            'release_explicit_commit_required')
    oid = git(repo, 'rev-parse', '--verify', commit + '^{commit}').decode().strip()
    require(re.fullmatch(r'[0-9a-f]{40}', oid) is not None, 'release_commit_invalid')
    entries = git(repo, 'ls-tree', '-rz', '--full-tree', oid).split(b'\0')
    blobs = {}
    for entry in filter(None, entries):
        metadata, raw_name = entry.split(b'\t', 1)
        mode, kind, blob = metadata.decode('ascii').split()
        name = safe_name(raw_name.decode('utf-8'))
        if not selected(name):
            continue
        require(mode in ('100644', '100755') and kind == 'blob', 'release_nonregular_git_entry')
        blobs[name] = git(repo, 'cat-file', 'blob', blob)
    required = set(HELPERS) | {'scripts/__init__.py'} | {
        'scripts/' + name + '.py' for name in NATIVE}
    require(required <= blobs.keys(), 'release_required_sources_missing')
    # Reject case-folding collisions for identical releases on Windows and Linux.
    require(len({n.casefold() for n in blobs}) == len(blobs), 'release_path_collision')
    manifest = {'schema': SCHEMA, 'commit': oid,
                'files': {name: pin(raw) for name, raw in sorted(blobs.items())}}
    raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'release_destination_exists')
    require(not any(p.is_symlink() for p in destination.absolute().parents), 'release_path_redirected')
    destination.mkdir(parents=True)
    for name, raw in blobs.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (destination / MANIFEST).write_bytes(raw_manifest)
    return {'commit': oid, 'manifest_sha256': digest(raw_manifest), 'file_count': len(blobs)}


def regular_root(root):
    root = Path(root).absolute()
    require(not any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                    for p in (root, *root.parents)), 'release_path_redirected')
    require(root.is_dir(), 'release_root_missing')
    return root.resolve(strict=True)


def verify_release(root, manifest_sha256):
    root = regular_root(root)
    require(re.fullmatch(r'[0-9a-f]{64}', manifest_sha256 or '') is not None,
            'release_trusted_manifest_digest_required')
    manifest_path = root / MANIFEST
    require(not manifest_path.is_symlink(), 'release_path_redirected')
    raw = manifest_path.read_bytes()
    require(digest(raw) == manifest_sha256, 'release_manifest_changed')
    manifest = json.loads(raw)
    require(set(manifest) == {'schema', 'commit', 'files'} and manifest['schema'] == SCHEMA
            and re.fullmatch(r'[0-9a-f]{40}', manifest['commit']) is not None,
            'release_manifest_invalid')
    pins = manifest['files']
    require(isinstance(pins, dict) and pins, 'release_inventory_invalid')
    for name, expected in pins.items():
        safe_name(name)
        require(selected(name) and isinstance(expected, dict)
                and set(expected) == {'bytes', 'sha256'}
                and type(expected['bytes']) is int and expected['bytes'] >= 0
                and re.fullmatch(r'[0-9a-f]{64}', expected['sha256']) is not None,
                'release_pin_invalid')
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            item = Path(directory) / name
            require(not item.is_symlink() and not getattr(item, 'is_junction', lambda: False)(),
                    'release_path_redirected')
        actual.update((Path(directory) / n).relative_to(root).as_posix() for n in files)
    require(actual == set(pins) | {MANIFEST}, 'release_inventory_changed')
    for name, expected in pins.items():
        path = root / name
        require(path.is_file() and pin(path.read_bytes()) == expected, 'release_source_changed')
    required = set(HELPERS) | {'scripts/__init__.py'} | {
        'scripts/' + name + '.py' for name in NATIVE}
    require(required <= pins.keys(), 'release_required_sources_missing')
    return manifest


class PinnedImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile only verified source bytes, never timestamp or hash pyc caches."""
    def __init__(self, root, pins):
        self.root, self.pins = root, pins
        self.prefixes = {n.split('/')[0].removesuffix('.py') for n in pins if n.endswith('.py')}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in self.prefixes:
            return None
        relative = fullname.replace('.', '/')
        package = relative + '/__init__.py'
        source = package if package in self.pins else relative + '.py'
        if source not in self.pins:
            if any(n.startswith(relative + '/') for n in self.pins):
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = [str(self.root / relative)]
                return spec
            raise ImportError('release_unlisted_module: ' + fullname)
        return importlib.util.spec_from_file_location(fullname, self.root / source, loader=self,
            submodule_search_locations=[str(self.root / relative)] if source == package else None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = Path(module.__spec__.origin)
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'release_path_redirected')
        raw = path.read_bytes()
        require(pin(raw) == self.pins[path.relative_to(self.root).as_posix()], 'release_source_changed')
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)


@contextmanager
def assembly_guard(root):
    """Deny observable network, DB, process and filesystem mutation in assembly.

    This is a qualification guard for reviewed source, not a hostile-code sandbox.
    Audit hooks cannot be removed, so the retained hook is inert after the scope.
    """
    state = {'active': True}

    def audit(event, args):
        if not state['active']:
            return
        if (event.startswith(('socket.', 'subprocess.', 'sqlite3.', 'os.exec', 'os.spawn'))
                or event in {'os.system', 'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir'}):
            raise RuntimeError('release_assembly_effect_forbidden: ' + event)
        if event == 'open':
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')
                    or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
                raise RuntimeError('release_assembly_write_forbidden')

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name in {
                'main', 'run', 'connect', 'request', 'publish', 'submit_order'}:
            filename = Path(frame.f_code.co_filename)
            if root in filename.parents:
                raise RuntimeError('release_assembly_runtime_call_forbidden')

    old_profile = sys.getprofile()
    old_bytecode = sys.dont_write_bytecode
    sys.addaudithook(audit)
    sys.dont_write_bytecode = True
    sys.setprofile(profile)
    try:
        yield
    finally:
        state['active'] = False
        sys.setprofile(old_profile)
        sys.dont_write_bytecode = old_bytecode


def assemble_release(root, manifest_sha256):
    """Verify and install endpoints in this process only; never invoke them."""
    root = regular_root(root)
    manifest = verify_release(root, manifest_sha256)
    finder = PinnedImporter(root, manifest['files'])
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, '__file__', None)
        require(name.split('.')[0] not in finder.prefixes
                and not (location and root in Path(location).resolve().parents),
                'release_fresh_process_required')
    old_path = list(sys.path)
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, str(root))
    try:
        with assembly_guard(root):
            native = SimpleNamespace(**{name: importlib.import_module('scripts.' + name)
                                        for name in NATIVE})
            integration = importlib.import_module('shared_native_integration')
            installed = integration.install_shared_native(native, root=root,
                adapter_pins=manifest['files'], clock=lambda: datetime.now(timezone.utc))
            ownership = importlib.import_module('shared_ownership_integration')
            ownership_proof = ownership.assemble_ownership(native, installed, root=root,
                pins=manifest['files'])
            verify_release(root, manifest_sha256)
        return {'commit': manifest['commit'], 'manifest_sha256': manifest_sha256,
                'qualification': 'ASSEMBLY_ONLY', 'activation_authorized': False,
                'proof': installed.proof, 'ownership_proof': ownership_proof}
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = old_path
        # Process-local native changes remain intentionally. Reassembly is refused.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--repo', required=True)
    build.add_argument('--commit', required=True)
    build.add_argument('--output', required=True)
    for name in ('verify', 'assemble'):
        command = commands.add_parser(name)
        command.add_argument('--root', required=True)
        command.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        result = build_release(args.repo, args.commit, args.output)
    elif args.command == 'verify':
        manifest = verify_release(args.root, args.manifest_sha256)
        result = {'commit': manifest['commit'], 'verified': True, 'file_count': len(manifest['files'])}
    else:
        result = assemble_release(args.root, args.manifest_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
