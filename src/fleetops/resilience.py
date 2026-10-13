"""Bounded retries with exponential backoff."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def backoff_delay(attempt: int, base: float, maximum: float) -> float:
    """Delay before retry number ``attempt`` (1-based): base * 2^(attempt-1), capped."""
    return float(min(maximum, base * (2 ** max(0, attempt - 1))))


def jittered(delay: float, fraction: float, rng: Callable[[], float] = random.random) -> float:
    """Spread retries: ``delay * (1 +/- fraction)``. ``rng`` returns [0, 1) and is injectable for tests."""
    if fraction <= 0:
        return delay
    return float(delay * (1 + fraction * (2 * rng() - 1)))


def retry[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base: float,
    maximum: float,
    retry_on: tuple[type[Exception], ...],
    sleep: Callable[[float], None] = time.sleep,
    jitter: float = 0.0,
    rng: Callable[[], float] = random.random,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt == attempts:
                raise
            sleep(jittered(backoff_delay(attempt, base, maximum), jitter, rng))
    raise AssertionError("unreachable")  # pragma: no cover
