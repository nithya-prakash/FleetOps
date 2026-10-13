from __future__ import annotations

import threading

import pytest

from fleetops.security.ratelimit import TokenBucketLimiter


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_burst_then_refill_at_the_configured_rate() -> None:
    clock = Clock()
    lim = TokenBucketLimiter(rate_per_second=2, burst=3, clock=clock)
    assert [lim.allow("a") for _ in range(4)] == [True, True, True, False]  # burst of 3, 4th rejected
    assert lim.retry_after("a") == pytest.approx(0.5)  # one token at 2/s
    clock.t += 0.5
    assert lim.allow("a") and not lim.allow("a")
    clock.t += 100  # refill never exceeds the burst size
    assert [lim.allow("a") for _ in range(4)] == [True, True, True, False]


def test_keys_are_independent() -> None:
    lim = TokenBucketLimiter(1, 1, clock=Clock())
    assert lim.allow("a") and not lim.allow("a") and lim.allow("b")


def test_peek_and_consume_for_failure_counting() -> None:
    clock = Clock()
    lim = TokenBucketLimiter(1 / 60, 3, clock=clock)  # 3 failures per minute
    for _ in range(3):
        assert lim.has_token("ip")
        lim.consume("ip")
    assert not lim.has_token("ip")  # locked out
    assert lim.retry_after("ip") == pytest.approx(60)
    clock.t += 61
    assert lim.has_token("ip")  # and recovers on its own
    lim.consume("ip", 10)  # consuming never goes negative
    assert lim.retry_after("ip") <= 180


def test_memory_is_bounded_by_max_keys() -> None:
    lim = TokenBucketLimiter(1, 1, clock=Clock(), max_keys=100)
    for i in range(5000):
        lim.allow(f"client-{i}")
    assert len(lim) <= 100  # an attacker rotating source addresses cannot grow memory without bound


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(0, 1)
    with pytest.raises(ValueError):
        TokenBucketLimiter(1, 0)


def test_thread_safety_never_grants_more_than_the_burst() -> None:
    lim = TokenBucketLimiter(rate_per_second=0.001, burst=50, clock=Clock())
    granted: list[bool] = []

    def worker() -> None:
        granted.extend(lim.allow("shared") for _ in range(100))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) == 50
