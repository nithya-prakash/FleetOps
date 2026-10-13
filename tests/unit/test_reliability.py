"""Edge reliability: download retry budget, jitter, last-known-good during outages, clock skew, expiry."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from fleetops.config.settings import EdgeSettings
from fleetops.domain.messages import AckOutcome, CommandType
from fleetops.domain.models import ModelRef, utcnow
from fleetops.resilience import backoff_delay, jittered, retry
from tests.helpers import make_command
from tests.unit.conftest import Bundle
from tests.unit.test_edge_agent import (
    HEALTHY,
    SECRET,
    deploy_and_activate,
    make_agent,
    run,
)

# ---- jitter / backoff ---------------------------------------------------------------------------------


def test_jitter_stays_within_the_configured_band_and_is_deterministic_with_injected_rng() -> None:
    assert jittered(10, 0.2, rng=lambda: 0.0) == pytest.approx(8.0)  # lower edge
    assert jittered(10, 0.2, rng=lambda: 0.5) == pytest.approx(10.0)
    assert jittered(10, 0.2, rng=lambda: 0.999999) == pytest.approx(12.0, abs=1e-3)  # upper edge
    assert jittered(10, 0.0, rng=lambda: 0.0) == 10  # disabled
    values = {jittered(5, 0.3) for _ in range(50)}
    assert len(values) > 1 and all(3.5 <= v <= 6.5 for v in values)  # real randomness stays bounded


def test_retry_applies_exponential_backoff_with_jitter_and_respects_the_cap() -> None:
    sleeps: list[float] = []

    def always() -> None:
        raise ConnectionError

    with pytest.raises(ConnectionError):
        retry(
            always,
            attempts=6,
            base=1,
            maximum=8,
            retry_on=(ConnectionError,),
            sleep=sleeps.append,
            jitter=0.25,
            rng=lambda: 1.0,
        )
    assert len(sleeps) == 5  # bounded: attempts - 1 sleeps
    assert sleeps == pytest.approx([b * 1.25 for b in (1, 2, 4, 8, 8)])  # exponential, capped at 8, jittered up by 25%
    assert backoff_delay(10, 1, 8) == 8


# ---- configurable download budget -----------------------------------------------------------------------


def test_download_budget_validation() -> None:
    base = {"device_id": "edge-01", "device_secret": "x" * 20, "_env_file": None}
    with pytest.raises(ValidationError):
        EdgeSettings(**base, download_max_attempts=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EdgeSettings(**base, download_jitter=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="download_backoff_max_seconds"):
        EdgeSettings(**base, download_backoff_base_seconds=10, download_backoff_max_seconds=1)  # type: ignore[arg-type]
    ok = EdgeSettings(**base)  # type: ignore[arg-type]
    assert (ok.download_max_attempts, ok.download_jitter) == (4, 0.2)


def flaky_transport(failures: int, bundle: Bundle) -> tuple[httpx.MockTransport, list[int]]:
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= failures:
            raise httpx.ConnectError("control plane unreachable", request=req)
        return httpx.Response(200, content=bundle.path.read_bytes())

    return httpx.MockTransport(handler), calls


def test_transient_outage_within_budget_is_ridden_out(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    transport, calls = flaky_transport(2, bundles[1])
    sleeps: list[float] = []
    agent, _, _ = make_agent(
        tmp_path,
        bundles,
        transport=transport,
        sleep=sleeps.append,
        settings_overrides={"download_max_attempts": 4, "download_backoff_base_seconds": 1.0, "download_jitter": 0.0},
    )
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[1].manifest())
    assert ack.outcome is AckOutcome.STAGED and len(calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff between the failed attempts


def test_outage_beyond_budget_fails_fast_and_keeps_last_known_good(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    transport, calls = flaky_transport(99, bundles[2])
    agent._http = httpx.Client(base_url="http://cp.test", transport=transport)
    agent._sleep = lambda _s: None
    agent.s.download_max_attempts = 3
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[2].manifest())
    assert (
        ack.outcome is AckOutcome.FAILED and ack.error_code == "DOWNLOAD_FAILED" and len(calls) == 3
    )  # bounded by the budget
    snap = agent.store.snapshot()
    assert snap.active is not None and snap.active.version == 1 and snap.staged is None  # last known-good untouched
    assert agent.infer(HEALTHY).model.version == 1  # and still serving
    assert not list(agent.store.staging_dir.iterdir())  # no partial download left behind


def test_larger_budget_changes_behaviour(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    transport, calls = flaky_transport(5, bundles[1])
    agent, _, _ = make_agent(tmp_path, bundles, transport=transport, settings_overrides={"download_max_attempts": 6})
    assert run(agent, CommandType.DEPLOY, manifest=bundles[1].manifest()).outcome is AckOutcome.STAGED
    assert len(calls) == 6


def test_auth_failures_are_not_retried(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    calls: list[int] = []

    def deny(_req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403)

    agent, _, _ = make_agent(
        tmp_path, bundles, transport=httpx.MockTransport(deny), settings_overrides={"download_max_attempts": 5}
    )
    ack = run(agent, CommandType.DEPLOY, manifest=bundles[1].manifest())
    assert ack.error_code == "DOWNLOAD_DENIED" and len(calls) == 1  # retrying a denial would just hammer the server


# ---- clock skew and expired commands ---------------------------------------------------------------------


def test_future_dated_command_is_rejected_with_actionable_detail_and_structured_log(
    tmp_path: Path,
    bundles: dict[int, Bundle],
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, settings_overrides={"max_clock_skew_seconds": 30})
    cmd = make_command(SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest())
    future = cmd.model_copy(
        update={"issued_at": utcnow() + timedelta(seconds=300), "expires_at": utcnow() + timedelta(seconds=900)}
    )
    from fleetops.security.device_identity import sign_command  # noqa: PLC0415

    with caplog.at_level(logging.WARNING, logger="fleetops.edge.agent"):
        ack = agent.handle_command(sign_command(SECRET, future))
    assert ack.outcome is AckOutcome.REJECTED and ack.error_code == "CLOCK_SKEW"
    assert (
        "device clock is" in (ack.detail or "")
        and "behind" in (ack.detail or "")
        and "tolerance 30s" in (ack.detail or "")
    )
    rec = next(r for r in caplog.records if getattr(r, "reason", "") == "clock_skew")
    assert 290 < rec.skew_seconds < 310 and rec.tolerance_seconds == 30  # type: ignore[attr-defined]
    assert agent.store.snapshot().staged is None  # nothing was executed


def test_small_clock_skew_within_tolerance_is_accepted(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, settings_overrides={"max_clock_skew_seconds": 30})
    cmd = make_command(SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest())
    from fleetops.security.device_identity import sign_command  # noqa: PLC0415

    slightly_ahead = cmd.model_copy(update={"issued_at": utcnow() + timedelta(seconds=10)})
    assert agent.handle_command(sign_command(SECRET, slightly_ahead)).outcome is AckOutcome.STAGED


def test_expired_command_reports_how_overdue_and_logs_structured_fields(
    tmp_path: Path,
    bundles: dict[int, Bundle],
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    expired = make_command(
        SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest(), expires_at=utcnow() - timedelta(seconds=120)
    )
    with caplog.at_level(logging.WARNING, logger="fleetops.edge.agent"):
        ack = agent.handle_command(expired)
    assert ack.outcome is AckOutcome.REJECTED and ack.error_code == "EXPIRED"
    assert "expired" in (ack.detail or "") and "NTP" in (ack.detail or "")
    rec = next(r for r in caplog.records if getattr(r, "reason", "") == "expired")
    assert 110 < rec.overdue_seconds < 130  # type: ignore[attr-defined]
    assert agent.store.snapshot().staged is None


def test_command_older_than_max_age_is_rejected_even_if_not_yet_expired(
    tmp_path: Path, bundles: dict[int, Bundle]
) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, settings_overrides={"command_max_age_seconds": 60})
    from fleetops.security.device_identity import sign_command  # noqa: PLC0415

    old = make_command(SECRET, CommandType.DEPLOY, manifest=bundles[1].manifest())
    old = old.model_copy(
        update={"issued_at": utcnow() - timedelta(seconds=120), "expires_at": utcnow() + timedelta(seconds=600)}
    )
    assert agent.handle_command(sign_command(SECRET, old)).error_code == "EXPIRED"


def test_rejected_commands_do_not_advance_the_generation_fence(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    expired = make_command(
        SECRET,
        CommandType.DEPLOY,
        generation=50,
        manifest=bundles[1].manifest(),
        expires_at=utcnow() - timedelta(seconds=5),
    )
    assert agent.handle_command(expired).outcome is AckOutcome.REJECTED
    assert agent.store.snapshot().last_generation == 0  # a bogus/expired command must not poison later valid ones
    ok = run(agent, CommandType.DEPLOY, generation=3, manifest=bundles[1].manifest())
    assert ok.outcome is AckOutcome.STAGED
    assert ModelRef(name="anomaly-detector", version=1) and agent.store.snapshot().last_generation == 3
