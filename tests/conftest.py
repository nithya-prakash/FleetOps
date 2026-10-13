from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from fleetops.config.settings import ControlPlaneSettings
from tests.harness import Stack

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture
def cp_settings(tmp_path: Path) -> ControlPlaneSettings:
    return ControlPlaneSettings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'fleetops.db'}",
        artifact_root=tmp_path / "artifacts",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        admin_api_key="admin-key-for-tests-0001",  # type: ignore[arg-type]
        operator_api_key="operator-key-for-tests-02",  # type: ignore[arg-type]
        viewer_api_key="viewer-key-for-tests-0003",  # type: ignore[arg-type]
        device_key_master="device-master-for-tests-04",  # type: ignore[arg-type]
        simulation_enabled=True,
        rate_limit_per_second=2000,  # unrelated tests must not be throttled; security tests lower these
        rate_limit_burst=4000,
        auth_failure_limit_per_minute=600,
        metrics_token="metrics-token-for-tests-0007",  # type: ignore[arg-type]
        mqtt_enabled=False,
        tick_interval_seconds=0.05,
        offline_after_seconds=3.0,
        retry_backoff_base_seconds=0.2,
        retry_backoff_max_seconds=1.0,
    )


@pytest.fixture
def stack(tmp_path: Path, cp_settings: ControlPlaneSettings) -> Iterator[Stack]:
    s = Stack(tmp_path, cp_settings).start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def make_stack(tmp_path: Path, cp_settings: ControlPlaneSettings) -> Iterator[Callable[..., Stack]]:
    """Factory for stacks with per-test control-plane settings: ``make_stack(command_ttl_seconds=2)``."""
    started: list[Stack] = []

    def build(**overrides: Any) -> Stack:
        stack = Stack(tmp_path, cp_settings.model_copy(update=overrides)).start()
        started.append(stack)
        return stack

    yield build
    for stack in started:
        stack.stop()
