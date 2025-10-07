"""Simple rate-limiting primitives."""
from __future__ import annotations

import collections
import threading
import time


class TokenBucket:
    """Token bucket limiting requests per minute."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max(1, int(max_per_minute or 1))
        self._timestamps = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = max(self._timestamps[0] + 60 - now, 0.01)
            time.sleep(min(sleep_for, 0.5))


__all__ = ["TokenBucket"]

