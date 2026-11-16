"""Configuration-bound immutable daily archive identities; no I/O on import."""
from datetime import date
import hashlib
import json
import os
import stat
from pathlib import Path

PRIMARY = "trend_pullback_reclaim_qualifying_close_v2"
PIPELINE = "pullback-reclaim-daily-paper-v1"

def configuration_hash(cfg, strategy_id=None):
    if not isinstance(cfg, dict) or cfg.get("feed") not in ("iex", "sip"):
        raise ValueError("archive_configuration_invalid")
    value = dict(cfg)
    if strategy_id is not None:
        value["strategy_id"] = strategy_id
    if not isinstance(value.get("strategy_id"), str) or not value["strategy_id"]:
        raise ValueError("archive_strategy_invalid")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()

def session_value(session):
    if not isinstance(session, str) or date.fromisoformat(session).isoformat() != session:
        raise ValueError("archive_session_invalid")
    return session

def archive_id(cfg, session, strategy_id=None):
    strategy = strategy_id if strategy_id is not None else cfg.get("strategy_id")
    value = dict(schema="jbravo.daily-strategy-archive-id.v2", pipeline=PIPELINE,
                 strategy_id=strategy, signal_session=session_value(session),
                 configuration_sha256=configuration_hash(cfg, strategy))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()

def session_root(output, session, cfg):
    # Companions share the captured market generation, but own separate scan folders.
    return Path(output) / session_value(session) / "revisions" / configuration_hash(cfg, PRIMARY)


def create_session_root(output, session, cfg):
    """Create one immutable revision through directory descriptors, never symlinks.

    POSIX directory-fd operations are mandatory. No insecure platform fallback.
    Existing session/revisions directories are opened, not replaced or chmodded.
    """
    root = session_root(output, session, cfg)
    output = Path(output)
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError("archive_output_invalid")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in output.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        for part in (session_value(session), "revisions"):
            try: os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError: pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        name = root.name
        os.mkdir(name, mode=0o700, dir_fd=fd)  # Existing claim is never reused.
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("archive_revision_identity_changed")
            current = os.stat(root, follow_symlinks=False)
            if (info.st_dev,info.st_ino)!=(current.st_dev,current.st_ino):
                raise ValueError("archive_revision_path_changed")
        finally: os.close(child)
        return root
    finally: os.close(fd)
