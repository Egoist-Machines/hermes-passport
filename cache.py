"""Bounded TTL cache that keeps expired entries as fallback material.

Two different callers want two different things from the same store: the
per-turn prefetch wants "fresh enough to inject", and the failure path wants
"the last thing Passport actually said" so an unreachable backend degrades to
slightly stale context instead of pretending the owner has no memory. So
``get()`` reports freshness rather than hiding a miss.

Locked because Hermes calls ``prefetch()`` from whatever thread is running the
turn, and a gateway serves several sessions of one install at once.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, NamedTuple


class CacheRead(NamedTuple):
    hit: bool
    fresh: bool
    value: Any


class TtlCache:
    def __init__(self, ttl_ms: int, max_entries: int = 32, clock=time.monotonic):
        self._ttl_s = max(0.0, ttl_ms / 1000.0)
        self._max_entries = max(1, max_entries)
        self._clock = clock
        self._entries: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> CacheRead:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return CacheRead(False, False, None)
            # Touch on read so eviction is least-recently-USED: the recent-rows
            # entry is read every turn but rewritten only per TTL window, and
            # write-order eviction would churn out exactly that hot entry under
            # a stream of novel prompt queries.
            self._entries.move_to_end(key)
            value, expires_at = entry
            return CacheRead(True, expires_at > self._clock(), value)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Re-insert so the eviction order is last-write, not first-write.
            self._entries.pop(key, None)
            self._entries[key] = (value, self._clock() + self._ttl_s)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)
