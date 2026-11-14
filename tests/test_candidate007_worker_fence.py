import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.candidate007_worker_fence import Lease, child_options, run_guarded

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.alpaca_optional


@unittest.skipUnless(os.name == "posix", "POSIX deployment lease tests")
class FenceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.lock = self.path / "generation.lock"
        self.lock.touch(mode=0o600)

    def test_worker_excludes_installer_and_installer_excludes_worker(self):
        with Lease(self.path) as worker:
            with Lease(self.path):
                with self.assertRaisesRegex(RuntimeError, "GENERATION_LEASE_BUSY"):
                    with Lease(self.path, exclusive=True):
                        self.fail("installer entered")
            with self.assertRaisesRegex(RuntimeError, "GENERATION_EXCLUSIVE_LEASE_REQUIRED"):
                worker.assert_held()
        with Lease(self.path, exclusive=True) as installer:
            installer.assert_held()
            with self.assertRaisesRegex(RuntimeError, "GENERATION_LEASE_BUSY"):
                run_guarded(self.path, lambda: self.fail("worker entered"))

    def test_refuses_replaced_inode(self):
        with Lease(self.path, exclusive=True) as lease:
            self.lock.rename(self.path / "retired.lock")
            self.lock.touch(mode=0o600)
            with self.assertRaisesRegex(RuntimeError, "GENERATION_IDENTITY_CHANGED"):
                lease.assert_held()

    def test_missing_symlink_and_hardlink_refused(self):
        old = self.path / "old"
        self.lock.rename(old)
        for kind in ("missing", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    self.lock.symlink_to(old)
                if kind == "hardlink":
                    os.link(old, self.lock)
                with self.assertRaises((OSError, RuntimeError)):
                    with Lease(self.path):
                        self.fail("invalid lock accepted")
                self.lock.unlink(missing_ok=True)

    def test_child_preserves_lease_after_worker_is_killed(self):
        # A pipe into the inert stage controls its lifetime; no fixed sleep or
        # external service. The stage still owns the lease after SIGKILL.
        script = (
            "import sys,subprocess\n"
            "from scripts.candidate007_worker_fence import run_guarded,child_options\n"
            "def stage():\n"
            "    child=subprocess.Popen([sys.executable,'-c',"
            "'import sys; print(1, flush=True); sys.stdin.buffer.read(1)'], **child_options())\n"
            "    return child.wait()\n"
            "run_guarded(sys.argv[1],stage)\n"
        )
        with subprocess.Popen([sys.executable, "-B", "-c", script, str(self.path)],
                              cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE) as parent:
            try:
                self.assertTrue(select.select([parent.stdout], [], [], 5)[0])
                self.assertEqual(parent.stdout.readline(), b"1\n")
                parent.kill()
                parent.wait(timeout=5)
                with self.assertRaisesRegex(RuntimeError, "GENERATION_LEASE_BUSY"):
                    with Lease(self.path, exclusive=True):
                        self.fail("surviving stage lost worker lease")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                parent.stdin.close()
                self.assertTrue(select.select([parent.stdout], [], [], 5)[0])
                self.assertEqual(parent.stdout.read(), b"")
        with Lease(self.path, exclusive=True) as lease:
            lease.assert_held()

    def test_child_options_merge_existing_role_lock(self):
        with tempfile.TemporaryFile() as retained:
            def worker():
                passed = child_options(retained.fileno())["pass_fds"]
                self.assertEqual(len(passed), 2)
                self.assertEqual(passed[0], retained.fileno())
                self.assertFalse(os.get_inheritable(passed[1]))
            run_guarded(self.path, worker)
            self.assertEqual(child_options(retained.fileno()), {"pass_fds": (retained.fileno(),)})
        self.assertEqual(child_options(), {})

    def test_interrupted_installer_keeps_workers_blocked_until_verified_recovery(self):
        with Lease(self.path, exclusive=True) as installer:
            installer.begin_maintenance()
            installer.assert_maintenance()
        with self.assertRaisesRegex(RuntimeError, "GENERATION_MAINTENANCE_BLOCKED"):
            run_guarded(self.path, lambda: self.fail("worker entered after interrupted installer"))
        with Lease(self.path, exclusive=True) as recovery:
            with self.assertRaises(FileExistsError):
                recovery.begin_maintenance()
            with self.assertRaisesRegex(RuntimeError, "GENERATION_TERMINAL_STATE_UNVERIFIED"):
                recovery.end_maintenance(lambda: False)
            recovery.assert_maintenance()
            recovery.end_maintenance(lambda: True)
        self.assertEqual(run_guarded(self.path, lambda: "resumed"), "resumed")

    def test_primary_refuses_before_work_when_maintenance_is_present(self):
        from scripts import candidate007_worker_fence as fence
        from scripts import primary_paper_pipeline as primary
        from scripts import primary_pipeline as implementation
        with Lease(self.path, exclusive=True) as installer:
            installer.begin_maintenance()
        with mock.patch.object(fence, 'DIRECTORY', str(self.path)), \
             mock.patch.dict(os.environ, {'APCA_API_BASE_URL': primary.PAPER_ENDPOINT,
                                          'JBR_EXEC_PAPER': 'true'}), \
             mock.patch.object(implementation, 'main') as work:
            with self.assertRaisesRegex(RuntimeError, 'GENERATION_MAINTENANCE_BLOCKED'):
                primary.main([])
            work.assert_not_called()

    def test_primary_holds_shared_lease_during_work(self):
        from scripts import candidate007_worker_fence as fence
        from scripts import primary_paper_pipeline as primary
        from scripts import primary_pipeline as implementation
        def work(argv):
            with self.assertRaisesRegex(RuntimeError, 'GENERATION_LEASE_BUSY'):
                with Lease(self.path, exclusive=True):
                    self.fail('primary did not hold generation lease')
            return 17
        with mock.patch.object(fence, 'DIRECTORY', str(self.path)), \
             mock.patch.dict(os.environ, {'APCA_API_BASE_URL': primary.PAPER_ENDPOINT,
                                          'JBR_EXEC_PAPER': 'true'}), \
             mock.patch.object(implementation, 'main', side_effect=work):
            self.assertEqual(primary.main([]), 17)

    def test_hourly_refuses_before_work_when_maintenance_is_present(self):
        from scripts import candidate007_worker_fence as fence
        from scripts import account_sync
        with Lease(self.path, exclusive=True) as installer:
            installer.begin_maintenance()
        with mock.patch.object(fence, 'DIRECTORY', str(self.path)), \
             mock.patch.object(account_sync, 'run_workflow') as work:
            self.assertEqual(account_sync.main([]), 1)
            work.assert_not_called()
