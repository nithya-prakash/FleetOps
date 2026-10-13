"""Rollout batch planning (pure)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from fleetops.domain.models import RolloutStrategy


def plan_batches(device_ids: Sequence[str], strategy: RolloutStrategy) -> list[list[str]]:
    """Deterministically split devices into [canary, *followers].

    Order is by device id so reruns and tests are reproducible. The canary batch is capped at
    the number of devices; the rest are split in chunks of ``batch_percent`` of the total.
    """
    ordered = sorted(device_ids)
    if not ordered:
        return []
    canary = ordered[: strategy.canary_count]
    rest = ordered[strategy.canary_count :]
    batches = [canary]
    if rest:
        chunk = max(1, math.ceil(len(ordered) * strategy.batch_percent / 100))
        batches.extend(rest[i : i + chunk] for i in range(0, len(rest), chunk))
    return batches
