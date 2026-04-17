"""Thread-safe TTL cache used by API tool clients."""
from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Hashable, Optional, Tuple


class TTLCache:
    def __init__(self, ttl_seconds: float, maxsize: int = 2048):
        self.ttl_seconds = ttl_seconds
        self.maxsize = maxsize
        self._lock = Lock()
        self._data: Dict[Hashable, Tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._data) >= self.maxsize:
                drop_key = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(drop_key, None)
            self._data[key] = (now + self.ttl_seconds, value)
