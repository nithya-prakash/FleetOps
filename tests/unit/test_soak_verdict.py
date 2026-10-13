"""The soak verdict rules decide what the project may claim, so they are tested like any other logic."""

from __future__ import annotations

from typing import Any

import pytest
from scripts.load_test.soak import MIN_STEADY_SECONDS, evaluate, fit

OK: dict[str, Any] = {
    "restarts": 0,
    "http_5xx": 0.0,
    "mqtt_failures": 0.0,
    "rollouts": ["COMPLETED", "COMPLETED"],
    "recovery_seconds": 6.0,
    "offline_samples": 0,
}


def series(minutes: int, rss: Any, fds: Any = lambda _t: 120) -> list[dict[str, Any]]:
    return [{"t": t, "cp_rss_mb": rss(t), "cp_fds": fds(t), "cp_threads": 30} for t in range(0, minutes * 60 + 1, 30)]


def test_fit_recovers_a_known_slope() -> None:
    slope, r2 = fit([(float(t), 100 + 0.01 * t) for t in range(0, 3600, 30)])
    assert slope == pytest.approx(36.0, rel=1e-6) and r2 == pytest.approx(1.0)


def test_flat_memory_over_a_long_enough_steady_state_passes() -> None:
    wobble = lambda t: 330 + (t % 90) / 30  # noqa: E731 - noise, no trend
    v = evaluate(series(60, wobble), 600, **OK)
    assert v.verdict == "PASS", v.reasons


def test_warmup_growth_is_not_a_leak() -> None:
    warm = lambda t: 330 - 60 * max(0.0, 1 - t / 600)  # noqa: E731 - fills up during the first 10 minutes, then flat
    assert evaluate(series(60, warm), 600, **OK).verdict == "PASS"


def test_steady_linear_growth_fails_as_a_possible_leak() -> None:
    v = evaluate(series(60, lambda t: 300 + 0.01 * t), 600, **OK)  # +36 MB/h, perfectly linear
    assert v.verdict == "FAIL" and any("possible leak" in r for r in v.reasons)


def test_a_late_step_in_memory_is_flagged_conservatively() -> None:
    """A single step looks like growth to a linear fit; the rule errs towards flagging it for a human to look at."""
    step = lambda t: 300 if t < 2000 else 340  # noqa: E731
    v = evaluate(series(60, step), 600, **OK)
    assert v.verdict == "FAIL" and any("possible leak" in r for r in v.reasons)


def test_growth_below_the_threshold_is_not_a_leak() -> None:
    v = evaluate(series(60, lambda t: 330 + 0.001 * t), 600, **OK)  # +3.6 MB/h
    assert v.verdict == "PASS", v.reasons


def test_short_runs_can_never_pass() -> None:
    v = evaluate(series(15, lambda _t: 330), 300, **OK)
    assert v.verdict == "INCONCLUSIVE" and MIN_STEADY_SECONDS > 15 * 60


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"restarts": 1}, "restart"),
        ({"http_5xx": 3.0}, "5xx"),
        ({"mqtt_failures": 2.0}, "MQTT"),
        ({"rollouts": ["COMPLETED", "ROLLED_BACK"]}, "rollouts"),
        ({"recovery_seconds": None}, "did not recover"),
        ({"recovery_seconds": 200.0}, "recovery took"),
        ({"offline_samples": 4}, "offline"),
    ],
)
def test_any_hard_failure_fails_the_run(override: dict[str, Any], needle: str) -> None:
    v = evaluate(series(60, lambda _t: 330), 600, **{**OK, **override})
    assert v.verdict == "FAIL" and any(needle in r for r in v.reasons)


def test_file_descriptor_growth_fails() -> None:
    v = evaluate(series(60, lambda _t: 330, fds=lambda t: 100 + t // 20), 600, **OK)
    assert v.verdict == "FAIL" and any("file descriptors" in r for r in v.reasons)
