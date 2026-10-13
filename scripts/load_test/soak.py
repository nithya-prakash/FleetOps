"""Pass / fail / inconclusive rules for the soak test, kept free of I/O so they can be unit-tested.

Design notes
* Only the STEADY-STATE segment (after the warm-up) is judged for memory trends: caches and connection pools fill during warm-up.
* A leak verdict needs both a material growth rate AND a consistent (linear) trend; noise or a one-off step is not a leak.
* A run that is too short to separate warm-up from a slow trend is INCONCLUSIVE, never a quiet PASS.
* PASS only means "no problem detected over this duration on this machine"; it is not a long-term reliability claim.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

MIN_STEADY_SECONDS = 40 * 60  # below this a PASS cannot be awarded
LEAK_SLOPE_MB_PER_HOUR = 10.0  # steady-state RSS growth that is treated as material
MIN_R2 = 0.7  # how linear the growth must be to count as a trend instead of noise
FD_GROWTH_LIMIT = 0.10  # open file descriptors / threads may not grow more than 10% in steady state
RECOVERY_LIMIT_SECONDS = 60.0


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares (slope per hour, R^2) of (seconds, value)."""
    n = len(points)
    if n < 3:
        return float("nan"), 0.0
    mx, my = sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    if sxx == 0:
        return float("nan"), 0.0
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return slope * 3600, r2


def quartile_medians(values: list[float]) -> tuple[float, float]:
    q = max(1, len(values) // 4)
    return statistics.median(values[:q]), statistics.median(values[-q:])


@dataclass
class Verdict:
    verdict: str  # PASS | FAIL | INCONCLUSIVE
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def evaluate(
    series: list[dict[str, Any]],
    warmup_seconds: float,
    *,
    restarts: int,
    http_5xx: float,
    mqtt_failures: float,
    rollouts: list[str],
    recovery_seconds: float | None,
    offline_samples: int,
) -> Verdict:
    steady = [s for s in series if s["t"] >= warmup_seconds]
    hard: list[str] = []
    soft: list[str] = []
    details: dict[str, Any] = {"steady_samples": len(steady)}

    if restarts:
        hard.append(f"{restarts} container restart(s) during the run")
    if http_5xx:
        hard.append(f"{http_5xx:.0f} HTTP 5xx responses")
    if mqtt_failures:
        hard.append(f"{mqtt_failures:.0f} MQTT messages failed processing")
    bad = [r for r in rollouts if r != "COMPLETED"]
    if bad:
        hard.append(f"{len(bad)} of {len(rollouts)} scheduled rollouts did not complete ({sorted(set(bad))})")
    if offline_samples:
        hard.append(f"devices were offline in {offline_samples} sample(s)")
    if recovery_seconds is None:
        hard.append("the control plane did not recover after the controlled restart")
    elif recovery_seconds > RECOVERY_LIMIT_SECONDS:
        hard.append(f"recovery took {recovery_seconds:.0f}s (limit {RECOVERY_LIMIT_SECONDS:.0f}s)")

    steady_span = (steady[-1]["t"] - steady[0]["t"]) if len(steady) > 1 else 0.0
    details["steady_seconds"] = round(steady_span)
    if len(steady) >= 3:
        slope, r2 = fit([(s["t"], s["cp_rss_mb"]) for s in steady])
        first_q, last_q = quartile_medians([s["cp_rss_mb"] for s in steady])
        details |= {
            "cp_rss_slope_mb_per_hour": round(slope, 2),
            "cp_rss_r2": round(r2, 3),
            "cp_rss_first_last_quartile_median_mb": [round(first_q, 1), round(last_q, 1)],
        }
        if slope > LEAK_SLOPE_MB_PER_HOUR and r2 >= MIN_R2:
            hard.append(
                f"steady-state RSS grows {slope:.1f} MB/h with a consistent linear trend (R^2 {r2:.2f}): possible leak"
            )
        elif slope > LEAK_SLOPE_MB_PER_HOUR:
            soft.append(f"RSS slope {slope:.1f} MB/h but the trend is not linear (R^2 {r2:.2f}); needs a longer run")
        for key, label in (("cp_fds", "open file descriptors"), ("cp_threads", "threads")):
            vals = [s[key] for s in steady if s.get(key) is not None]
            if len(vals) >= 4:
                a, b = quartile_medians([float(v) for v in vals])
                details[f"{key}_first_last_quartile_median"] = [a, b]
                if a > 0 and (b - a) / a > FD_GROWTH_LIMIT:
                    hard.append(f"{label} grew {100 * (b - a) / a:.0f}% in steady state")
    else:
        soft.append("fewer than 3 steady-state samples")

    if steady_span < MIN_STEADY_SECONDS:
        soft.append(
            f"steady state lasted {steady_span / 60:.0f} min (< {MIN_STEADY_SECONDS // 60} min needed to tell warm-up "
            "from a slow trend)"
        )

    if hard:
        return Verdict("FAIL", hard + soft, details)
    if soft:
        return Verdict("INCONCLUSIVE", soft, details)
    return Verdict("PASS", ["no leak or failure detected in this window (not a long-term guarantee)"], details)
