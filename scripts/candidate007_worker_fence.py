"""Linux shared worker lease and exclusive installer lease.

The operator preprovisions one file; there is no create, unlink or unlock
operation. Closing is essential: an inherited descriptor can outlive its parent.
"""
import errno
import os
from pathlib import PurePosixPath
import stat

LOCK_NAME = "generation.lock"
BLOCK_NAME = "maintenance.block"
DIRECTORY = "/home/RasPatrick/jbravo-research-tools/CANDIDATE007_GENERATION_FENCE"
_active = None


def _directory(path):
    path = PurePosixPath(path)
    if not path.is_absolute() or path.anchor != "/" or ".." in path.parts:
        raise ValueError("GENERATION_DIRECTORY_INVALID")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            parent = fd
            fd = os.open(component, flags, dir_fd=parent)
            os.close(parent)
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("GENERATION_DIRECTORY_NOT_PRIVATE")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _identity(st):
    return st.st_dev, st.st_ino


def _file_ok(st):
    return (stat.S_ISREG(st.st_mode) and st.st_uid == os.geteuid()
            and st.st_nlink == 1 and st.st_size == 0 and stat.S_IMODE(st.st_mode) == 0o600)


class Lease:
    def __init__(self, directory, *, exclusive=False):
        self.directory = str(directory)
        self.exclusive = exclusive
        self.fd = self.parent = None

    def __enter__(self):
        import fcntl

        if self.fd is not None:
            raise RuntimeError("GENERATION_LEASE_ALREADY_OPEN")
        self.parent = _directory(self.directory)
        try:
            self.fd = os.open(LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
                              | os.O_NONBLOCK, dir_fd=self.parent)
            self.assert_held(check_only=True)
            try:
                fcntl.flock(self.fd, (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
                            | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise RuntimeError("GENERATION_LEASE_BUSY") from None
                raise
            self.assert_held(check_only=True)
            if not self.exclusive and self._blocked():
                raise RuntimeError("GENERATION_MAINTENANCE_BLOCKED")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def assert_held(self, *, check_only=False):
        """The installer callback only accepts an exclusive local lease."""
        if self.fd is None or (not self.exclusive and not check_only):
            raise RuntimeError("GENERATION_EXCLUSIVE_LEASE_REQUIRED")
        directory = _directory(self.directory)
        try:
            actual = os.stat(LOCK_NAME, dir_fd=directory, follow_symlinks=False)
            opened = os.fstat(self.fd)
            if (not _file_ok(opened) or not _file_ok(actual)
                    or _identity(opened) != _identity(actual)
                    or _identity(os.fstat(directory)) != _identity(os.fstat(self.parent))):
                raise RuntimeError("GENERATION_IDENTITY_CHANGED")
        finally:
            os.close(directory)

    def _blocked(self):
        try:
            os.stat(BLOCK_NAME, dir_fd=self.parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True  # Any existing inode, even malformed, denies worker entry.

    def begin_maintenance(self):
        """Persist a stop before source changes. Never overwrite an old stop."""
        self.assert_held()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(BLOCK_NAME, flags, 0o600, dir_fd=self.parent)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self.parent)
        self.assert_maintenance()

    def assert_maintenance(self):
        """Use this callback for each installer/rollback source replacement."""
        self.assert_held()
        info = os.stat(BLOCK_NAME, dir_fd=self.parent, follow_symlinks=False)
        if not _file_ok(info):
            raise RuntimeError("GENERATION_MAINTENANCE_MARKER_INVALID")

    def end_maintenance(self, verify_terminal_state):
        """Remove the stop only after the reviewed external verifier succeeds."""
        self.assert_maintenance()
        if verify_terminal_state() is not True:
            raise RuntimeError("GENERATION_TERMINAL_STATE_UNVERIFIED")
        self.assert_maintenance()
        os.unlink(BLOCK_NAME, dir_fd=self.parent)
        os.fsync(self.parent)

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.parent is not None:
            os.close(self.parent)
            self.parent = None


def child_options(*existing):
    """Merge a guarded parent's lease with the role-specific descriptors.

    An unguarded legacy entry gets only its existing descriptors, if any.
    The descriptor is passed explicitly, never through the environment.
    """
    if _active is not None:
        _active.assert_held(check_only=True)
        existing += (_active.fd,)
    return {"pass_fds": tuple(dict.fromkeys(existing))} if existing else {}


def run_guarded(directory, entry):
    """Run the supplied fixed in-process entry under a shared lease."""
    global _active
    if _active is not None:
        raise RuntimeError("GENERATION_NESTED_LEASE")
    with Lease(directory) as lease:
        _active = lease
        try:
            return entry()
        finally:
            _active = None
