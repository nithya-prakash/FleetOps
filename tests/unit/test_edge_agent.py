from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from fleetops.config.settings import EdgeSettings
from fleetops.domain.messages import AckOutcome, CommandType
from fleetops.domain.models import ModelRef, utcnow
from fleetops.edge.agent import EdgeAgent
from fleetops.edge.inference import InjectedFaultError, NoModelError
from fleetops.messaging.bus import InMemoryBroker, InMemoryBus
from fleetops.messaging.topics import Channel, topic
from fleetops.registry.modelformat import InputRejectedError
from fleetops.resilience import backoff_delay, retry
from tests.helpers import faults, make_command, make_manifest, rollback_to
from tests.unit.conftest import Bundle

SECRET = "edge-secret-for-tests-0001"
HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]


def make_agent(
    tmp: Path,
    bundles: dict[int, Bundle],
    allow_faults: bool = True,
    broker: InMemoryBroker | None = None,
    *,
    settings_overrides: dict[str, object] | None = None,
    transport: httpx.MockTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> tuple[EdgeAgent, InMemoryBus, InMemoryBroker]:
    served = {f"/api/v1/device/artifacts/anomaly-detector/{v}": b.path for v, b in bundles.items()}

    def handler(req: httpx.Request) -> httpx.Response:
        path = served.get(req.url.path)
        if path is None:
            return httpx.Response(404)
        return httpx.Response(200, content=path.read_bytes())

    http = httpx.Client(base_url="http://cp.test", transport=transport or httpx.MockTransport(handler))
    broker = broker or InMemoryBroker()
    settings = EdgeSettings(
        device_id="edge-01",
        device_secret=SECRET,
        data_dir=tmp / "edge",
        allow_faults=allow_faults,  # type: ignore[arg-type]
        _env_file=None,
        **{"sensor_hz": 0, "download_max_attempts": 2, **(settings_overrides or {})},  # type: ignore[arg-type]
    )
    bus = InMemoryBus(broker, "edge-01", will=EdgeAgent.will_for("edge-01"))
    agent = EdgeAgent(settings, bus, http=http, sleep=sleep or (lambda _s: None))
    agent.start(loops=False)
    return agent, bus, broker


def run(agent: EdgeAgent, ctype: CommandType, **kw: object):  # type: ignore[no-untyped-def]
    return agent.handle_command(make_command(SECRET, ctype, **kw))


def deploy_and_activate(agent: EdgeAgent, b: Bundle) -> None:
    assert run(agent, CommandType.DEPLOY, manifest=b.manifest()).outcome is AckOutcome.STAGED
    assert (
        run(agent, CommandType.ACTIVATE, activate=ModelRef(name="anomaly-detector", version=b.version)).outcome
        is AckOutcome.ACTIVE
    )


def test_stage_activate_and_local_inference(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    with pytest.raises(NoModelError):
        agent.infer(HEALTHY)
    deploy_and_activate(agent, bundles[1])
    r = agent.infer(HEALTHY)
    assert r.model.version == 1 and 0 <= r.prediction.score <= 1
    assert agent.store.snapshot().active == bundles[1].ident
    assert agent.store.snapshot().staged is None


def test_checksum_mismatch_rejected_and_active_untouched(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    bad = make_manifest("anomaly-detector", 2, "0" * 64, bundles[2].size, bundles[2].meta)
    ack = run(agent, CommandType.DEPLOY, manifest=bad)
    assert ack.outcome is AckOutcome.FAILED and ack.error_code == "CHECKSUM_MISMATCH"
    assert agent.store.snapshot().staged is None
    assert agent.infer(HEALTHY).model.version == 1
    assert not list(agent.store.staging_dir.iterdir())  # partial download cleaned up


def test_injected_corrupt_download(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    assert (
        run(agent, CommandType.SET_FAULTS, faults=faults(corrupt_download_versions=[2])).outcome is AckOutcome.ACCEPTED
    )
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    assert ack.error_code == "CHECKSUM_MISMATCH"


def test_faults_ignored_when_disabled(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, allow_faults=False)
    ack = run(agent, CommandType.SET_FAULTS, faults=faults(fail_inference_versions=[1]))
    assert ack.outcome is AckOutcome.FAILED and ack.error_code == "FAULTS_DISABLED"


def test_activation_failure_keeps_last_known_good(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    run(agent, CommandType.SET_FAULTS, faults=faults(fail_activation_versions=[2]))
    run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    ack = run(agent, CommandType.ACTIVATE, activate=ModelRef(name="anomaly-detector", version=2))
    assert ack.outcome is AckOutcome.FAILED
    assert agent.infer(HEALTHY).model.version == 1
    assert agent.store.snapshot().active == bundles[1].ident


def test_activate_requires_staged(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    ack = run(agent, CommandType.ACTIVATE, activate=ModelRef(name="anomaly-detector", version=2))
    assert ack.error_code == "NOT_STAGED"


def test_rollback_is_a_real_model_switch(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    deploy_and_activate(agent, bundles[2])
    assert agent.infer(HEALTHY).model.version == 2
    assert agent.store.snapshot().previous == bundles[1].ident
    ack = run(agent, CommandType.ROLLBACK, rollback_to=rollback_to(bundles[1].ident))
    assert ack.outcome is AckOutcome.ROLLED_BACK
    assert agent.infer(HEALTHY).model.version == 1  # v1 really loaded and predicting
    assert agent.store.snapshot().active == bundles[1].ident
    assert not agent.store.path_of(bundles[2].ident).exists()  # failed model garbage-collected


def test_rollback_refetches_when_local_copy_missing(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[2])
    ack = run(agent, CommandType.ROLLBACK, rollback_to=rollback_to(bundles[1].ident, bundles[1].manifest()))
    assert ack.outcome is AckOutcome.ROLLED_BACK
    assert agent.infer(HEALTHY).model.version == 1


def test_rollback_without_known_good_deactivates(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    assert run(agent, CommandType.ROLLBACK, rollback_to=rollback_to(None)).outcome is AckOutcome.ROLLED_BACK
    with pytest.raises(NoModelError):
        agent.infer(HEALTHY)


def test_stale_generation_cannot_activate_older_model(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    run(agent, CommandType.DEPLOY, generation=10, manifest=bundles[2].manifest())
    old = make_command(SECRET, CommandType.DEPLOY, generation=5, manifest=bundles[1].manifest())
    ack = agent.handle_command(old)
    assert ack.outcome is AckOutcome.REJECTED and ack.error_code == "STALE"
    assert agent.store.snapshot().staged == bundles[2].ident


def test_duplicate_command_replays_ack_without_reexecuting(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    cmd = make_command(SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest())
    first = agent.handle_command(cmd)
    second = agent.handle_command(cmd)
    assert first.outcome is second.outcome is AckOutcome.STAGED
    assert agent.metrics.commands.labels("edge-01", "DEPLOY", "STAGED")._value.get() == 1  # type: ignore[attr-defined]


def test_bad_signature_and_expired_rejected(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    forged = make_command("wrong-secret-0000000000", CommandType.DEPLOY, manifest=bundles[1].manifest())
    assert agent.handle_command(forged).error_code == "BAD_SIGNATURE"
    expired = make_command(
        SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest(), expires_at=utcnow().replace(year=2020)
    )
    assert agent.handle_command(expired).error_code == "EXPIRED"
    assert agent.store.snapshot().staged is None


def test_malformed_mqtt_command_is_dropped_not_fatal(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, broker = make_agent(tmp_path, bundles)
    broker.route(topic("edge-01", Channel.COMMANDS), b"{not json", 1, False)
    assert broker.flush()
    assert agent.metrics.mqtt_failures.labels("edge-01")._value.get() == 1  # type: ignore[attr-defined]


def test_input_validation_and_injected_inference_failure(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    with pytest.raises(InputRejectedError):
        agent.infer([1.0, 2.0])
    run(agent, CommandType.SET_FAULTS, faults=faults(fail_inference_versions=[1]))
    for _ in range(3):
        with pytest.raises(InjectedFaultError):
            agent.infer(HEALTHY)
    stats = agent.engine.stats()
    assert stats is not None and stats.inference_errors == 3 and stats.inference_count == 3


def test_restart_recovers_active_model(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    agent.stop()
    reborn, _, _ = make_agent(tmp_path, bundles)
    assert reborn.infer(HEALTHY).model.version == 1
    assert reborn.store.snapshot().last_generation > 0


def test_corrupted_active_falls_back_to_previous(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    deploy_and_activate(agent, bundles[2])
    path = agent.store.path_of(bundles[2].ident)
    path.write_bytes(path.read_bytes()[:-20])  # simulate disk corruption
    agent.stop()
    reborn, _, _ = make_agent(tmp_path, bundles)
    assert reborn.infer(HEALTHY).model.version == 1
    assert reborn.store.snapshot().active == bundles[1].ident


def test_offline_inference_continues_and_updates_fail_safely(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, bus, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    agent.set_network(False)
    assert not bus.connected
    for _ in range(5):
        agent.sensor_step()
    assert agent.infer(HEALTHY).model.version == 1
    assert len(agent._pending) == 5
    assert agent.flush_predictions() is False and len(agent._pending) == 5
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    assert ack.error_code == "DOWNLOAD_FAILED"
    agent.set_network(True)
    assert agent.flush_predictions() is True and len(agent._pending) == 0


def test_retry_backoff() -> None:
    assert [backoff_delay(i, 0.5, 5.0) for i in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 5.0]
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("x")
        return "ok"

    assert retry(flaky, attempts=5, base=1, maximum=10, retry_on=(ConnectionError,), sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]

    def always() -> None:
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        retry(always, attempts=3, base=1, maximum=10, retry_on=(ConnectionError,), sleep=lambda _s: None)
    with pytest.raises(KeyError):  # non-retryable errors propagate immediately
        retry(lambda: {}["x"], attempts=3, base=1, maximum=1, retry_on=(ConnectionError,), sleep=lambda _s: None)
