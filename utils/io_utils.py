"""I/O helpers for atomic file writes used across the project."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str]]


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomically write ``data`` to ``path``.

    The data is first written to a temporary file in the target directory and
    then moved into place via :func:`os.replace` to guarantee that readers never
    observe a partially written file.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # ``NamedTemporaryFile`` on Windows does not allow reopening the file while
    # it is still open, so we create the file manually using ``mkstemp``.
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        # ``os.replace`` removes the temp file on success; if an exception is
        # raised before that point we remove the temporary file to avoid
        # littering the filesystem.
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


__all__ = ["atomic_write_bytes"]
