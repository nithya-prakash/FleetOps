"""Reliability scenarios 2, 3, 4, 6, 10 (expectations are written in docs/reliability-test-results.md)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from fleetops.edge.agent import EdgeAgent
from tests.harness import Stack, wait_until
from tests.scenarios import DEVICES, FAST, HEALTHY, TERMINAL, fleet, manual_rollout, still_serving

MakeStack = Callable[..., Stack]


def cmd_count(stack: Stack, device: str, ctype: str, outcome: str) -> float:
    return stack.agent_metric(device, "fleetops_edge_commands_total", type=ctype, outcome=outcome)


# ---------------------------------------------------------------------------------------------------- scenario 2


def test_s2_control_plane_restart_inside_the_health_window(make_stack: MakeStack) -> None:
    stack = make_stack()
    _, v2 = fleet(stack)
    base = {
        (d, t, o): cmd_count(stack, d, t, o) for d in DEVICES for t, o in (("DEPLOY", "STAGED"), ("ACTIVATE", "ACTIVE"))
    }
    crit = {**FAST, "observation_seconds": 4}  # the canary sits in its health window for 4 s
    dep = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 100}, health_criteria=crit)
    dep_id = dep["deployment_id"]
    wait_until(lambda: stack.deployment(dep_id)["targets"][0]["state"] == "ACTIVE", 15, msg="canary active")
    assert stack.deployment(dep_id)["targets"][0]["verified_at"] is None and stack.active_version("edge-01") == v2

    stack.restart_control_plane()  # mid-rollout: only the database knows where we were

    again = stack.deployment(dep_id)
    assert again["state"] == "IN_PROGRESS" and again["targets"][0]["state"] == "ACTIVE"  # recovered, not restarted
    final = stack.wait_state(dep_id, {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] == "COMPLETED"
    still_serving(stack, v2)
    for d in DEVICES:  # every device executed DEPLOY and ACTIVATE exactly once: nothing was repeated by the recovery
        assert cmd_count(stack, d, "DEPLOY", "STAGED") - base[(d, "DEPLOY", "STAGED")] == 1
        assert cmd_count(stack, d, "ACTIVATE", "ACTIVE") - base[(d, "ACTIVATE", "ACTIVE")] == 1
    actions = [e["action"] for e in stack.audit(deployment_id=dep_id)]
    assert actions.count("deployment.created") == 1 and "deployment.completed" in actions  # one continuous history
    assert stack.metric("fleetops_deployments_total", state="COMPLETED") == 1  # counted by the new process


def test_s2_control_plane_restart_with_unacknowledged_commands(make_stack: MakeStack) -> None:
    stack = make_stack()
    _, v2 = fleet(stack, DEVICES[:2])
    r = stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"drop_acks": True})
    assert r.status_code == 202
    wait_until(lambda: stack.agents["edge-01"].engine.faults.drop_acks, msg="acks dropped from now on")
    staged_before = cmd_count(stack, "edge-01", "DEPLOY", "STAGED")

    dep_id = stack.deploy(v2, strategy={"canary_count": 2, "batch_percent": 100})["deployment_id"]
    wait_until(
        lambda: stack.active_version("edge-01") == v2, 15, msg="device ran the commands although its acks are lost"
    )
    stack.restart_control_plane()  # the engine wakes up with commands in flight whose acks never arrived
    final = stack.wait_state(dep_id, {"COMPLETED", "ROLLED_BACK", "FAILED"})

    assert final["state"] == "COMPLETED"  # healed from device-reported state, not from acks
    still_serving(stack, v2, DEVICES[:2])
    assert cmd_count(stack, "edge-01", "DEPLOY", "STAGED") - staged_before == 1  # executed once despite the re-sends
    deploys = [c for c in stack.wire_commands("edge-01", "DEPLOY") if c.deployment_id == dep_id]
    assert len({c.command_id for c in deploys}) <= 2 and len(deploys) <= 6  # bounded re-sends (same id)


# ---------------------------------------------------------------------------------------------------- scenario 3


class StallingTransport(httpx.BaseTransport):
    """Sends the first ``stall_after`` bytes of the next response, then hangs until released (a link that dies
    mid-download). Only armed responses stall, so baseline rollouts are unaffected."""

    def __init__(self, inner: httpx.BaseTransport, stall_after: int) -> None:
        self.inner, self.stall_after = inner, stall_after
        self.armed, self.started, self.gate = threading.Event(), threading.Event(), threading.Event()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.inner.handle_request(request)
        if not self.armed.is_set():
            return response
        self.armed.clear()
        original, outer = response.stream, self

        class Stream(httpx.SyncByteStream):
            def __iter__(self) -> Any:
                sent = 0
                for chunk in original:  # type: ignore[union-attr]
                    take = chunk[: max(0, outer.stall_after - sent)]
                    if take:
                        yield take
                        sent += len(take)
                    if sent >= outer.stall_after:
                        outer.started.set()
                        outer.gate.wait(120)
                        raise httpx.ReadError("connection lost")

            def close(self) -> None:
                original.close()  # type: ignore[union-attr]

        return httpx.Response(
            response.status_code, headers=response.headers, stream=Stream(), extensions=response.extensions
        )


def test_s3_agent_killed_in_the_middle_of_an_artifact_download(make_stack: MakeStack) -> None:
    stack = make_stack()
    transport_holder: list[StallingTransport] = []

    def hang_downloads(agent: EdgeAgent) -> None:
        old = agent._http
        transport = StallingTransport(httpx.HTTPTransport(), stall_after=300)
        transport_holder.append(transport)
        agent._http = httpx.Client(
            base_url=str(old.base_url), headers=dict(old.headers), transport=transport, timeout=60
        )
        original = agent._on_command_message
        # commands run on their own thread so a hung download cannot block the (shared, in-process) broker
        agent._on_command_message = lambda t, p: threading.Thread(target=original, args=(t, p), daemon=True).start()  # type: ignore[method-assign]

    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    stack.add_agent("edge-01", pre_start=hang_downloads)
    stack.add_agent("edge-02")
    stack.wait_online("edge-01", "edge-02")
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})
    victim = stack.agents["edge-01"]
    transport_holder[0].armed.set()  # the NEXT download will hang half way

    crit = {**FAST, "ack_timeout_seconds": 25}
    dep_id = stack.deploy(v2, strategy={"canary_count": 2, "batch_percent": 100}, health_criteria=crit)["deployment_id"]
    assert transport_holder[0].started.wait(20), "download never started"
    partial = list(victim.store.staging_dir.iterdir())
    assert len(partial) == 1 and partial[0].suffix == ".part"  # an unfinished download is on disk
    assert stack.active_version("edge-01") == v1 and victim.infer(HEALTHY).model.version == v1  # inference unaffected

    reborn = stack.restart_agent("edge-01", crash=True)  # kill -9: nothing was cleaned up by the dead process

    assert not list(reborn.store.staging_dir.iterdir())  # stale partial download removed at startup
    snap = reborn.store.snapshot()
    assert snap.active is not None and snap.active.version == v1 and snap.staged is None  # persisted state intact
    assert reborn.infer(HEALTHY).model.version == v1  # serving again immediately from the last known-good model
    final = stack.wait_state(dep_id, TERMINAL)  # the engine re-sends the same DEPLOY; the new process completes it
    assert final["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])
    assert cmd_count(stack, "edge-01", "DEPLOY", "STAGED") == 1  # the new process executed it exactly once
    sends = [c for c in stack.wire_commands("edge-01", "DEPLOY") if c.deployment_id == dep_id]
    assert (
        2 <= len(sends) <= 5 and len({c.command_id for c in sends}) == 1
    )  # one command, re-sent a bounded number of times
    assert stack.metric("fleetops_command_retries_total", type="DEPLOY") >= 1


# ---------------------------------------------------------------------------------------------------- scenario 4


def test_s4_link_lost_before_the_update_reaches_the_device(make_stack: MakeStack) -> None:
    stack = make_stack()
    v1, v2 = fleet(stack, DEVICES[:2])
    downloads = stack.metric("fleetops_artifact_downloads_total", result="ok")
    staged_before = cmd_count(stack, "edge-01", "DEPLOY", "STAGED")
    stack.buses["edge-01"].stop()  # silent disconnect: no last will, the control plane still believes it is online
    crit = {**FAST, "ack_timeout_seconds": 2}  # DEPLOY expires after ack timeout + 5 s = 7 s
    dep = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 100}, health_criteria=crit)
    final = stack.wait_state(dep["deployment_id"], TERMINAL)

    assert final["state"] == "ROLLED_BACK"
    targets = {t["device_id"]: t for t in final["targets"]}
    assert targets["edge-01"]["state"] == "FAILED" and "no STAGED acknowledgement" in targets["edge-01"]["last_error"]
    assert targets["edge-02"]["state"] == "CANCELLED"  # never contacted: the canary failed first
    aborts = [c for c in stack.wire_commands("edge-01", "ABORT") if c.deployment_id == dep["deployment_id"]]
    assert len(aborts) == 1  # best-effort, once

    time.sleep(7.5)  # the queued DEPLOY is now past its expiry
    stack.buses["edge-01"].start()  # link restored: the broker hands over everything queued for the device
    stack.agents["edge-01"].announce_online()
    wait_until(lambda: cmd_count(stack, "edge-01", "ABORT", "ABORTED") == 1, 10, msg="queued ABORT processed")

    assert cmd_count(stack, "edge-01", "DEPLOY", "REJECTED") == 1  # expired command refused...
    assert cmd_count(stack, "edge-01", "DEPLOY", "STAGED") == staged_before  # no new staging happened
    snap = stack.agents["edge-01"].store.snapshot()
    assert snap.staged is None and snap.active is not None and snap.active.version == v1
    assert stack.metric("fleetops_artifact_downloads_total", result="ok") == downloads  # ...so nothing was downloaded
    assert stack.deployment(dep["deployment_id"])["state"] == "ROLLED_BACK"  # and the stale command changed nothing
    still_serving(stack, v1, DEVICES[:2])


def test_s4_link_lost_right_after_activation_the_device_keeps_serving_until_it_can_be_told(
    make_stack: MakeStack,
) -> None:
    stack = make_stack()
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    agent = stack.add_agent("edge-01", heartbeat_interval_seconds=30)  # no reconciling heartbeat can sneak in
    stack.wait_online("edge-01")
    manual_rollout(stack, "edge-01", v1)
    r = stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"drop_ack_outcomes": ["ACTIVE"]})
    assert r.status_code == 202
    wait_until(lambda: agent.engine.faults.drop_ack_outcomes == ["ACTIVE"], msg="fault applied")

    crit = {**FAST, "ack_timeout_seconds": 3, "health_timeout_seconds": 3}
    dep_id = stack.deploy(v2, health_criteria=crit)["deployment_id"]
    wait_until(lambda: stack.active_version("edge-01") == v2, 15, msg="device switched; its ACTIVE ack is lost")
    agent.set_network(False)  # and now the link dies

    # Documented limitation: the control plane cannot reach the device, so v2 keeps serving until the link is back.
    assert agent.infer(HEALTHY).model.version == v2
    wait_until(
        lambda: stack.deployment(dep_id)["state"] == "ROLLING_BACK", 15, msg="activation never confirmed -> rollback"
    )
    time.sleep(1.5)
    during = stack.deployment(dep_id)
    assert during["state"] == "ROLLING_BACK" and during["targets"][0]["state"] == "FAILED"  # waiting, not pretending
    assert stack.api("GET", "/api/v1/devices/edge-01", role="viewer").json()["status"] == "OFFLINE"
    assert stack.active_version("edge-01") == v2
    rollbacks = [c for c in stack.wire_commands("edge-01", "ROLLBACK") if c.deployment_id == dep_id]
    assert rollbacks and rollbacks[0].rollback_to is not None and rollbacks[0].rollback_to.model is not None
    assert rollbacks[0].rollback_to.model.version == v1

    agent.set_network(True)
    agent.announce_online()
    final = stack.wait_state(dep_id, {"ROLLED_BACK", "FAILED", "COMPLETED"})
    assert final["state"] == "ROLLED_BACK" and final["targets"][0]["state"] == "ROLLED_BACK"
    still_serving(stack, v1, ("edge-01",))  # the queued ROLLBACK ran on reconnect: v1 is really back
    assert len({c.command_id for c in stack.wire_commands("edge-01", "ROLLBACK") if c.deployment_id == dep_id}) <= 2
    assert stack.metric("fleetops_rollbacks_total", trigger="automatic") == 1
    actions = [e["action"] for e in stack.audit(deployment_id=dep_id)]
    assert "deployment.rolling_back" in actions and "deployment.rolled_back" in actions


def test_s4_link_lost_after_a_completed_rollout_changes_nothing(make_stack: MakeStack) -> None:
    stack = make_stack()
    _, v2 = fleet(stack, DEVICES[:2])
    dep_id = stack.deploy(v2)["deployment_id"]
    stack.wait_state(dep_id, {"COMPLETED"})
    stack.agents["edge-02"].set_network(False)
    for _ in range(30):
        assert stack.agents["edge-02"].infer(HEALTHY).model.version == v2
    wait_until(lambda: stack.api("GET", "/api/v1/devices/edge-02", role="viewer").json()["status"] == "OFFLINE", 10)
    assert stack.deployment(dep_id)["state"] == "COMPLETED"  # an outage after success does not rewrite history
    stack.agents["edge-02"].set_network(True)
    stack.agents["edge-02"].announce_online()
    wait_until(lambda: stack.api("GET", "/api/v1/devices/edge-02", role="viewer").json()["status"] == "ONLINE", 10)
    assert stack.api("GET", "/api/v1/devices/edge-02/health", role="viewer").json()["active_model"]["version"] == v2


# ---------------------------------------------------------------------------------------------------- scenario 6


def test_s6_expired_command_is_rejected_and_reissued_fresh(make_stack: MakeStack) -> None:
    stack = make_stack(command_ttl_seconds=2, retry_backoff_base_seconds=1.0, retry_backoff_max_seconds=2.0)
    _, v2 = fleet(stack, ("edge-01",))
    deploy_base = cmd_count(stack, "edge-01", "DEPLOY", "STAGED")
    stack.buses["edge-01"].stop()  # offline while the control plane still thinks it is reachable
    crit = {**FAST, "ack_timeout_seconds": 30}  # patience is long, but each command only lives 2 s
    dep_id = stack.deploy(v2, health_criteria=crit)["deployment_id"]
    time.sleep(2.6)  # everything queued for the device is now expired
    stack.buses["edge-01"].start()
    stack.agents["edge-01"].announce_online()
    final = stack.wait_state(dep_id, TERMINAL)

    assert final["state"] == "COMPLETED"
    still_serving(stack, v2, ("edge-01",))
    assert cmd_count(stack, "edge-01", "DEPLOY", "REJECTED") >= 1  # the device refused the stale copies
    assert cmd_count(stack, "edge-01", "DEPLOY", "STAGED") - deploy_base == 1  # and executed only the fresh one
    sends = [c for c in stack.wire_commands("edge-01", "DEPLOY") if c.deployment_id == dep_id]
    assert len({c.command_id for c in sends}) >= 2  # a new command id was issued after expiry
    assert len(sends) <= 6  # bounded
    assert all(c.generation > 0 for c in sends) and sends[-1].generation > sends[0].generation


def test_s6_superseded_deployment_never_activates_on_the_reconnecting_device(make_stack: MakeStack) -> None:
    stack = make_stack()
    _, v2, v3 = fleet(stack, ("edge-01",), models=3)
    stack.buses["edge-01"].stop()  # still 'ONLINE' in the control plane for a few seconds
    dep_a = stack.deploy(v2)["deployment_id"]
    dep_b = stack.deploy(v3)["deployment_id"]  # supersedes A while A's commands sit in the device's queue
    assert stack.deployment(dep_a)["state"] == "CANCELLED" and stack.deployment(dep_a)["superseded_by"] == dep_b
    stack.buses["edge-01"].start()
    stack.agents["edge-01"].announce_online()
    final = stack.wait_state(dep_b, TERMINAL)

    assert final["state"] == "COMPLETED"
    still_serving(stack, v3, ("edge-01",))
    agent = stack.agents["edge-01"]
    ident_v2 = agent.store.path_of(type("I", (), {"name": "anomaly-detector", "version": v2})())  # type: ignore[arg-type]
    assert not ident_v2.exists()  # v2 was never stored, let alone activated
    assert (
        cmd_count(stack, "edge-01", "DEPLOY", "FAILED") == 1
    )  # A's DEPLOY: download refused for a superseded deployment
    assert cmd_count(stack, "edge-01", "ACTIVATE", "ACTIVE") == 2  # baseline v1 + v3 only
    assert stack.audit("artifact.denied")  # the control plane refused to hand out v2
    assert stack.deployment(dep_a)["state"] == "CANCELLED"


# ---------------------------------------------------------------------------------------------------- scenario 10


def test_s10_device_reconnects_after_several_versions_were_deployed(make_stack: MakeStack) -> None:
    stack = make_stack()
    v1, v2, v3 = fleet(stack, models=3)
    sleeper = stack.agents["edge-03"]
    sleeper.set_network(False)
    wait_until(lambda: stack.api("GET", "/api/v1/devices/edge-03", role="viewer").json()["status"] == "OFFLINE", 10)
    wire_before = len(stack.wire_commands("edge-03"))
    handled_before = sum(cmd_count(stack, "edge-03", t, o) for t, o in (("DEPLOY", "STAGED"), ("ACTIVATE", "ACTIVE")))

    d2 = stack.deploy(v2)
    assert d2["skipped"]["edge-03"].startswith("not online")
    stack.wait_state(d2["deployment_id"], {"COMPLETED"})
    d3 = stack.deploy(v3)
    stack.wait_state(d3["deployment_id"], {"COMPLETED"})
    assert {stack.active_version(d) for d in DEVICES[:2]} == {v3}
    assert len(stack.wire_commands("edge-03")) == wire_before  # nothing was queued for the sleeper

    sleeper.set_network(True)
    sleeper.announce_online()
    wait_until(lambda: stack.api("GET", "/api/v1/devices/edge-03", role="viewer").json()["status"] == "ONLINE", 10)
    sleeper.send_heartbeat()
    wait_until(
        lambda: stack.api("GET", "/api/v1/devices/edge-03", role="viewer").json()["active_model_version"] == v1, 10
    )
    assert stack.active_version("edge-03") == v1  # inventory shows the truth: it missed two versions
    assert (
        sum(cmd_count(stack, "edge-03", t, o) for t, o in (("DEPLOY", "STAGED"), ("ACTIVATE", "ACTIVE")))
        == handled_before
    )
    still_serving(stack, v1, ("edge-03",))

    catch_up = stack.deploy(v3)  # bring it current: only the sleeper needs work
    assert catch_up["batches"] == [["edge-03"]] and set(catch_up["skipped"]) == set(DEVICES[:2])
    assert stack.wait_state(catch_up["deployment_id"], TERMINAL)["state"] == "COMPLETED"
    still_serving(stack, v3)
    previous = stack.agents["edge-03"].store.snapshot().previous
    assert previous is not None and previous.version == v1  # its own last known-good was retained
