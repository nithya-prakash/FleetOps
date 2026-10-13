"""Bounded, thread-safe token-bucket rate limiter (in-process; one control-plane replica)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TokenBucketLimiter:
    """``burst`` tokens, refilled at ``rate_per_second``. Keys are bounded: idle buckets are evicted oldest-first."""

    def __init__(
        self,
        rate_per_second: float,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if rate_per_second <= 0 or burst < 1:
            raise ValueError("rate_per_second must be > 0 and burst >= 1")
        self._rate, self._burst, self._clock, self._max_keys = rate_per_second, float(burst), clock, max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, last_refill)
        self._lock = threading.Lock()

    def _refilled(self, key: str, now: float) -> float:
        tokens, last = self._buckets.get(key, (self._burst, now))
        return min(self._burst, tokens + (now - last) * self._rate)

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens if available."""
        with self._lock:
            now = self._clock()
            tokens = self._refilled(key, now)
            ok = tokens >= cost
            self._buckets[key] = (tokens - cost if ok else tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            return ok

    def has_token(self, key: str, cost: float = 1.0) -> bool:
        """Would ``allow`` succeed? Does not consume."""
        with self._lock:
            return self._refilled(key, self._clock()) >= cost

    def consume(self, key: str, cost: float = 1.0) -> None:
        """Consume unconditionally (floors at zero); used to count failures."""
        with self._lock:
            now = self._clock()
            self._buckets[key] = (max(0.0, self._refilled(key, now) - cost), now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)

    def retry_after(self, key: str, cost: float = 1.0) -> float:
        with self._lock:
            missing = cost - self._refilled(key, self._clock())
            return max(0.0, missing / self._rate)

    def __len__(self) -> int:
        return len(self._buckets)
