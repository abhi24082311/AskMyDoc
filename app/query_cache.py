"""Small in-memory TTL cache for completed ask responses."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from app.schemas import AskResponse


class AskResponseCache:
    """Cache complete question responses with bounded LRU and TTL eviction."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[tuple[str, int, str], tuple[float, AskResponse]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple[str, int, str]) -> AskResponse | None:
        """Return a copy of a cached response, or None when absent or expired."""
        if self._max_size == 0 or self._ttl_seconds == 0:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, response = entry
            if expires_at <= time.monotonic():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return response.model_copy(deep=True)

    def put(self, key: tuple[str, int, str], response: AskResponse) -> None:
        """Store a response and evict the least recently used entry if needed."""
        if self._max_size == 0 or self._ttl_seconds == 0:
            return
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, response.model_copy(deep=True))
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Clear responses after the indexed document set changes."""
        with self._lock:
            self._entries.clear()
