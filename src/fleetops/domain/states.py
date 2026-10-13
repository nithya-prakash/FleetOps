"""Deployment / target state machines. Pure; persistence layers call ``ensure_*_transition``."""

from __future__ import annotations

from enum import StrEnum


class InvalidTransitionError(ValueError):
    pass


class TargetState(StrEnum):
    PENDING = "PENDING"
    DISTRIBUTING = "DISTRIBUTING"
    STAGED = "STAGED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    CANCELLED = "CANCELLED"


class DeploymentState(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


T = TargetState
D = DeploymentState

TARGET_TRANSITIONS: dict[TargetState, frozenset[TargetState]] = {
    T.PENDING: frozenset({T.DISTRIBUTING, T.CANCELLED}),
    T.DISTRIBUTING: frozenset({T.STAGED, T.FAILED, T.CANCELLED, T.ROLLED_BACK}),
    T.STAGED: frozenset({T.ACTIVE, T.FAILED, T.CANCELLED, T.ROLLED_BACK}),
    T.ACTIVE: frozenset({T.FAILED, T.ROLLED_BACK}),
    T.FAILED: frozenset({T.ROLLED_BACK}),
    T.ROLLED_BACK: frozenset(),
    T.CANCELLED: frozenset(),
}

DEPLOYMENT_TRANSITIONS: dict[DeploymentState, frozenset[DeploymentState]] = {
    D.PENDING: frozenset({D.IN_PROGRESS, D.CANCELLED}),
    D.IN_PROGRESS: frozenset({D.COMPLETED, D.ROLLING_BACK, D.FAILED, D.CANCELLED}),
    D.COMPLETED: frozenset({D.ROLLING_BACK}),
    D.ROLLING_BACK: frozenset({D.ROLLED_BACK, D.FAILED}),
    D.ROLLED_BACK: frozenset(),
    D.FAILED: frozenset({D.ROLLING_BACK}),
    D.CANCELLED: frozenset(),
}

TERMINAL_DEPLOYMENT_STATES = frozenset({D.COMPLETED, D.ROLLED_BACK, D.FAILED, D.CANCELLED})
IN_FLIGHT_DEPLOYMENT_STATES = frozenset({D.PENDING, D.IN_PROGRESS, D.ROLLING_BACK})
TERMINAL_TARGET_STATES = frozenset({T.ROLLED_BACK, T.CANCELLED})


def ensure_target_transition(current: TargetState, new: TargetState) -> None:
    if new not in TARGET_TRANSITIONS[current]:
        raise InvalidTransitionError(f"target transition {current} -> {new} is not allowed")


def ensure_deployment_transition(current: DeploymentState, new: DeploymentState) -> None:
    if new not in DEPLOYMENT_TRANSITIONS[current]:
        raise InvalidTransitionError(f"deployment transition {current} -> {new} is not allowed")
