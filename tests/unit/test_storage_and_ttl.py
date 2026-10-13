"""Edge storage safety (scenario 8) and command lifetime rules in the engine."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import httpx
import pytest

from fleetops.deployment.engine import DeploymentSpec
from fleetops.domain.messages import AckOutcome, CommandType, Heartbeat
from fleetops.domain.models import HealthCriteria, RolloutStrategy
from tests.helpers import faults, make_command
from tests.unit.conftest import Bundle
from tests.unit.test_edge_agent import HEALTHY, SECRET, deploy_and_activate, make_agent, run
from tests.unit.test_engine import Env


def counting_transport(bundle: Bundle) -> tuple[httpx.MockTransport, list[int]]:
    calls: list[int] = []

    def handler(_req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=bundle.path.read_bytes())

    return httpx.MockTransport(handler), calls


# ---- free-space pre-check -----------------------------------------------------------------------------------------


def test_download_is_not_started_when_free_space_is_insufficient(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    transport, calls = counting_transport(bundles[2])
    agent._http = httpx.Client(base_url="http://cp.test", transport=transport)
    run(agent, CommandType.SET_FAULTS, faults=faults(fake_free_disk_mb=50))  # below the 128 MB reserve
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    assert ack.outcome is AckOutcome.FAILED and ack.error_code == "INSUFFICIENT_STORAGE"
    assert "need" in (ack.detail or "") and calls == []  # not a single byte was requested
    snap = agent.store.snapshot()
    assert snap.active == bundles[1].ident and snap.staged is None
    assert agent.infer(HEALTHY).model.version == 1  # last known-good still serving
    assert not list(agent.store.staging_dir.iterdir())


@pytest.mark.parametrize(("delta", "ok"), [(-1, False), (0, True), (1, True)])
def test_storage_threshold_boundary(tmp_path: Path, bundles: dict[int, Bundle], delta: int, ok: bool) -> None:
    agent, _, _ = make_agent(
        tmp_path, bundles, settings_overrides={"min_free_disk_mb": 1, "storage_headroom_factor": 2.0}
    )
    needed = int(bundles[1].size * 2.0) + 1024 * 1024
    agent._free_disk_bytes = lambda: needed + delta  # type: ignore[method-assign]
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[1].manifest())
    assert (ack.outcome is AckOutcome.STAGED) is ok
    assert ok or ack.error_code == "INSUFFICIENT_STORAGE"


def test_storage_override_is_ignored_when_faults_are_not_allowed(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, allow_faults=False)
    agent.engine.faults = faults(fake_free_disk_mb=1)  # would starve the agent if it were honoured
    assert run(agent, CommandType.DEPLOY, manifest=bundles[1].manifest()).outcome is AckOutcome.STAGED


# ---- ENOSPC while writing -------------------------------------------------------------------------------------------


def test_disk_full_while_writing_cleans_up_and_keeps_the_running_model(
    tmp_path: Path, bundles: dict[int, Bundle], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    real_open = Path.open

    def full_disk(self: Path, mode: str = "r", *a: object, **k: object):  # type: ignore[no-untyped-def]
        if self.suffix == ".part" and "w" in mode:
            fh = real_open(self, mode, *a, **k)  # type: ignore[arg-type]
            fh.write(b"partial")  # some bytes land before the volume fills up
            fh.flush()
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(self, mode, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", full_disk)
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    monkeypatch.undo()
    assert ack.outcome is AckOutcome.FAILED and ack.error_code == "STORAGE_FULL"
    assert not list(agent.store.staging_dir.iterdir())  # partial file removed
    assert agent.store.snapshot().active == bundles[1].ident and agent.infer(HEALTHY).model.version == 1
    assert run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest()).outcome is AckOutcome.STAGED  # recovers


def test_state_persistence_failure_leaves_the_previous_state_file_valid(
    tmp_path: Path, bundles: dict[int, Bundle], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    state_file = tmp_path / "edge" / "state.json"
    before = state_file.read_text()

    def enospc(*_a: object, **_k: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("fleetops.edge.model_store.tempfile.mkstemp", enospc)
    ack = run(agent, CommandType.ACTIVATE, activate=bundles[2].manifest().model)
    monkeypatch.undo()
    assert (
        ack.outcome is AckOutcome.FAILED
        and ack.error_code == "ACTIVATION_FAILED"
        and "cannot persist" in (ack.detail or "")
    )
    assert (
        state_file.read_text() == before and json.loads(before)["active"]["version"] == 1
    )  # untouched and still valid JSON
    assert agent.infer(HEALTHY).model.version == 1
    assert not list((tmp_path / "edge").glob("*.tmp"))
    assert (
        run(agent, CommandType.ACTIVATE, activate=bundles[2].manifest().model).outcome is AckOutcome.ACTIVE
    )  # retry works


def test_failure_to_persist_the_dedup_record_does_not_crash_the_handler(
    tmp_path: Path, bundles: dict[int, Bundle], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    monkeypatch.setattr(
        "fleetops.edge.model_store.tempfile.mkstemp",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")),
    )
    ack = agent.handle_command(make_command(SECRET, CommandType.SET_FAULTS, faults=faults()))
    assert ack.outcome is AckOutcome.ACCEPTED  # answered despite the full disk


def test_stale_partial_downloads_are_removed_at_startup(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    leftover = agent.store.staging_dir / "dep-crashed-1234.part"
    leftover.write_bytes(b"half a model")  # what a kill -9 mid-download leaves behind
    agent.stop()
    reborn, _, _ = make_agent(tmp_path, bundles)
    assert not leftover.exists() and not list(reborn.store.staging_dir.iterdir())


def test_free_disk_is_reported_in_heartbeats_and_metrics(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    seen: list[Heartbeat] = []

    def capture(topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        if topic.endswith("/heartbeat"):
            seen.append(Heartbeat.model_validate_json(payload))

    agent.bus.publish = capture  # type: ignore[method-assign]
    assert agent.send_heartbeat()
    assert seen[0].disk_free_mb is not None and seen[0].disk_free_mb > 0
    assert agent.metrics.registry.get_sample_value("fleetops_edge_disk_free_mb", {"device_id": "edge-01"}) > 0  # type: ignore[operator]
    run(agent, CommandType.SET_FAULTS, faults=faults(fake_free_disk_mb=7))
    seen.clear()
    agent.send_heartbeat()
    assert seen[0].disk_free_mb == pytest.approx(7 * 1024 * 1024 / 1e6)


def test_redelivered_commands_are_counted_and_answered_from_the_cache(
    tmp_path: Path, bundles: dict[int, Bundle]
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    cmd = make_command(SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest())
    for _ in range(4):
        assert agent.handle_command(cmd).outcome is AckOutcome.STAGED
    assert (
        agent.metrics.registry.get_sample_value(
            "fleetops_edge_duplicate_commands_total", {"device_id": "edge-01", "type": "DEPLOY"}
        )
        == 3
    )
    assert (
        agent.metrics.registry.get_sample_value(
            "fleetops_edge_commands_total", {"device_id": "edge-01", "type": "DEPLOY", "outcome": "STAGED"}
        )
        == 1
    )


# ---- engine: command lifetime / best-effort abort --------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def spec(env: Env, ack_timeout: float = 10) -> DeploymentSpec:
    return DeploymentSpec(
        model_name="anomaly",
        model_version=2,
        group="line-a",
        strategy=RolloutStrategy(canary_count=1),
        health_criteria=HealthCriteria(ack_timeout_seconds=ack_timeout, max_command_attempts=3),
    )


def test_forward_commands_expire_shortly_after_the_ack_timeout_but_rollback_commands_do_not(env: Env) -> None:
    dep_id = env.engine.create_deployment(spec(env, ack_timeout=10), "op").deployment_id
    deploy = env.pub.of(CommandType.DEPLOY)[0]
    assert (deploy.expires_at - deploy.issued_at).total_seconds() == pytest.approx(
        15
    )  # ack timeout + 5s, below the 100s TTL
    env.ack(deploy, AckOutcome.STAGED)
    activate = env.pub.of(CommandType.ACTIVATE)[0]
    assert (activate.expires_at - activate.issued_at).total_seconds() == pytest.approx(15)
    env.ack(activate, AckOutcome.ACTIVE)
    env.engine.rollback(dep_id, "op", "test")
    rollback = env.pub.of(CommandType.ROLLBACK)[0]
    assert (rollback.expires_at - rollback.issued_at).total_seconds() == pytest.approx(
        100
    )  # full TTL: must survive an outage


def test_a_target_that_times_out_before_activation_gets_a_best_effort_abort(env: Env) -> None:
    dep_id = env.engine.create_deployment(spec(env, ack_timeout=10), "op").deployment_id
    first = env.pub.of(CommandType.DEPLOY)[0]
    env.clock.advance(11)  # device never answers (e.g. offline): the target fails
    env.engine.advance(dep_id)
    aborts = env.pub.of(CommandType.ABORT, "edge-01")
    assert (
        len(aborts) == 1 and aborts[0].generation > first.generation
    )  # queued behind the DEPLOY on the device's session
    assert aborts[0].activate is not None and aborts[0].activate.version == 2
    assert env.dep(dep_id).state == "ROLLED_BACK"  # nothing waits for the abort: no activation can follow
    assert env.targets(dep_id)["edge-01"].pending_command is None
    env.clock.advance(1000)
    env.engine.advance(dep_id)
    assert len(env.pub.of(CommandType.ABORT, "edge-01")) == 1  # fire-and-forget: never retried


def test_failed_publishes_do_not_burn_the_attempt_budget_but_the_ack_timeout_still_bounds_the_wait(env: Env) -> None:
    env.pub.up = False  # broker unreachable: every publish fails
    dep_id = env.engine.create_deployment(spec(env, ack_timeout=40), "op").deployment_id
    for _ in range(4):  # more retries than max_command_attempts (3), still inside the 40 s ack timeout
        env.clock.advance(6)
        env.engine.advance(dep_id)
    target = env.targets(dep_id)["edge-01"]
    assert target.attempts == 0 and target.state == "DISTRIBUTING"  # still waiting: nothing was ever delivered
    assert env.dep(dep_id).state == "IN_PROGRESS"
    assert env.engine._metrics.commands_published.labels("DEPLOY", "link_down")._value.get() >= 5
    ids = {c.command_id for c in env.pub.of(CommandType.DEPLOY)}
    assert (
        len(ids) <= 3
    )  # the same command (or a freshly signed one after expiry) - never an unbounded flood of new ones

    env.pub.up = True  # broker back: the next retry gets through and only now counts as an attempt
    env.clock.advance(6)
    env.engine.advance(dep_id)
    assert env.targets(dep_id)["edge-01"].attempts == 1

    env.pub.up = False  # but silence is still bounded by the acknowledgement timeout
    env.clock.advance(41)
    env.engine.advance(dep_id)
    assert env.dep(dep_id).state == "ROLLED_BACK" and "no STAGED acknowledgement" in (
        env.targets(dep_id)["edge-01"].last_error or ""
    )
