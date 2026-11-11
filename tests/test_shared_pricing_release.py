"""Release provenance and fail-closed assembly tests, entirely offline."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / 'scripts/shared_pricing_release.py'
spec = importlib.util.spec_from_file_location('release_builder_tested', TOOL)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
pytestmark = pytest.mark.alpaca_optional


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.email', 'offline@example.invalid')
    git(repo, 'config', 'user.name', 'Offline tests')
    git(repo, 'config', 'core.autocrlf', 'false')
    files = {name: b'# fixture\n' for name in release.HELPERS}
    files.update({'scripts/' + name + '.py': b'# native fixture\n' for name in release.NATIVE})
    files['scripts/__init__.py'] = b''
    files['config/strategy_execution.json'] = b'{"enabled":true}\n'
    files['shared_native_integration.py'] = (
        b'from types import SimpleNamespace\n'
        b'def install_shared_native(native, **kwargs):\n'
        b'    return SimpleNamespace(proof={"connections_opened":0,"order_actions":0})\n')
    files['shared_ownership_integration.py'] = (
        b'def assemble_ownership(native, bound, **kwargs):\n'
        b'    return {"activation_authorized":False}\n')
    files['scripts/newline.py'] = b'VALUE = 1\r\n# retained CRLF\r\n'
    for name, raw in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'offline fixture')
    return repo


@pytest.fixture
def artifact(repository, tmp_path):
    commit = git(repository, 'rev-parse', 'HEAD').decode().strip()
    root = tmp_path / 'release'
    result = release.build_release(repository, commit, root)
    return root, result


def cli(root, result, command='assemble'):
    return subprocess.run([sys.executable, str(TOOL), command, '--root', str(root),
                           '--manifest-sha256', result['manifest_sha256']],
                          capture_output=True, text=True)


def test_build_uses_commit_blobs_not_dirty_checkout(repository, tmp_path):
    commit = git(repository, 'rev-parse', 'HEAD').decode().strip()
    expected = git(repository, 'show', commit + ':scripts/newline.py')
    (repository / 'scripts/newline.py').write_bytes(b'changed working tree\n')
    root = tmp_path / 'release'
    result = release.build_release(repository, commit, root)
    manifest = release.verify_release(root, result['manifest_sha256'])
    assert (root / 'scripts/newline.py').read_bytes() == expected
    assert manifest['files']['scripts/newline.py'] == {
        'bytes': len(expected), 'sha256': hashlib.sha256(expected).hexdigest()}
    assert b'\r\n' in expected
    assert result['commit'] == commit


def test_builder_never_imports_source(repository, tmp_path):
    (repository / 'shared_native_integration.py').write_text('raise AssertionError("imported")\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-qm', 'explosive import')
    commit = git(repository, 'rev-parse', 'HEAD').decode().strip()
    release.build_release(repository, commit, tmp_path / 'release')


def test_requires_exact_commit_and_new_destination(repository, tmp_path):
    with pytest.raises(ValueError, match='explicit_commit'):
        release.build_release(repository, 'HEAD', tmp_path / 'release')
    commit = git(repository, 'rev-parse', 'HEAD').decode().strip()
    with pytest.raises(ValueError, match='destination_exists'):
        release.build_release(repository, commit, tmp_path)


@pytest.mark.parametrize('name', ['../escape.py', '/escape.py', 'C:/escape.py',
                                  'scripts/../escape.py', 'scripts\\escape.py', 'a//b.py'])
def test_rejects_path_escapes(name):
    with pytest.raises(ValueError, match='path_invalid'):
        release.safe_name(name)


def test_source_hash_tampering(artifact):
    root, result = artifact
    (root / 'scripts/setup_selection.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        release.verify_release(root, result['manifest_sha256'])


def test_manifest_needs_external_digest(artifact):
    root, result = artifact
    manifest = root / release.MANIFEST
    manifest.write_bytes(manifest.read_bytes() + b' ')
    with pytest.raises(ValueError, match='manifest_changed'):
        release.verify_release(root, result['manifest_sha256'])
    with pytest.raises(ValueError, match='trusted_manifest'):
        release.verify_release(root, '')


@pytest.mark.parametrize('name', ['scripts/unlisted.py', 'unlisted.py',
                                  'scripts/__pycache__/setup_selection.pyc', 'config/extra.json'])
def test_rejects_extra_files(artifact, name):
    root, result = artifact
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'extra')
    with pytest.raises(ValueError, match='inventory_changed'):
        release.verify_release(root, result['manifest_sha256'])


def test_missing_source(artifact):
    root, result = artifact
    (root / 'scripts/setup_selection.py').unlink()
    with pytest.raises(ValueError, match='inventory_changed'):
        release.verify_release(root, result['manifest_sha256'])


def test_symlink_refused(artifact, tmp_path):
    root, result = artifact
    link = root / 'link'
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip('OS does not permit unprivileged symlinks')
    with pytest.raises(ValueError, match='redirected'):
        release.verify_release(root, result['manifest_sha256'])


def test_preloaded_native_package_refused(artifact, monkeypatch):
    root, result = artifact
    monkeypatch.setitem(sys.modules, 'scripts', object())
    with pytest.raises(ValueError, match='fresh_process'):
        release.assemble_release(root, result['manifest_sha256'])


def test_unlisted_import_refused(artifact):
    root, result = artifact
    manifest = release.verify_release(root, result['manifest_sha256'])
    finder = release.PinnedImporter(root, manifest['files'])
    with pytest.raises(ImportError, match='unlisted_module'):
        finder.find_spec('scripts.absent')


def test_assembly_does_not_activate(artifact):
    root, result = artifact
    proc = cli(root, result)
    assert proc.returncode == 0, proc.stderr
    proof = json.loads(proc.stdout)
    assert proof['qualification'] == 'ASSEMBLY_ONLY'
    assert proof['activation_authorized'] is False
    assert proof['proof'] == {'connections_opened': 0, 'order_actions': 0}
    assert not list(root.rglob('*.pyc'))


@pytest.mark.parametrize('source,reason', [
    ('import socket\nsocket.socket()\n', 'effect_forbidden'),
    ('import sqlite3\nsqlite3.connect(":memory:")\n', 'effect_forbidden'),
    ('def main(): pass\nmain()\n', 'runtime_call_forbidden'),
    ('open(__file__ + ".changed", "w")\n', 'write_forbidden'),
])
def test_assembly_blocks_import_side_effects(repository, tmp_path, source, reason):
    (repository / 'scripts/setup_selection.py').write_text(source)
    git(repository, 'add', '.')
    git(repository, 'commit', '-qm', 'side effect fixture')
    root = tmp_path / 'release'
    result = release.build_release(repository, git(repository, 'rev-parse', 'HEAD').decode().strip(), root)
    proc = cli(root, result)
    assert proc.returncode != 0
    assert reason in proc.stderr


def test_git_symlink_is_rejected_without_checkout_privileges(repository, tmp_path):
    blob = subprocess.check_output(['git', '-C', str(repository), 'hash-object', '-w', '--stdin'],
                                   input=b'../outside.py').decode().strip()
    git(repository, 'update-index', '--add', '--cacheinfo', '120000,' + blob + ',scripts/link.py')
    git(repository, 'commit', '-qm', 'symlink object')
    commit = git(repository, 'rev-parse', 'HEAD').decode().strip()
    with pytest.raises(ValueError, match='nonregular_git_entry'):
        release.build_release(repository, commit, tmp_path / 'release')


def test_preloaded_helper_refused(artifact, monkeypatch):
    root, result = artifact
    monkeypatch.setitem(sys.modules, 'shared_pricing_contract', object())
    with pytest.raises(ValueError, match='fresh_process'):
        release.assemble_release(root, result['manifest_sha256'])
