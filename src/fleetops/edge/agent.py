"""The edge agent: local inference + command handling + heartbeats over an unreliable link."""

from __future__ import annotations

import errno
import logging
import shutil
import ssl
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
from pydantic import ValidationError

from fleetops import __version__
from fleetops.config.logging import log_context
from fleetops.config.settings import EdgeSettings
from fleetops.domain.messages import (
    AckOutcome,
    Command,
    CommandType,
    Heartbeat,
    PredictionBatch,
    PredictionSample,
    StatusMessage,
    Telemetry,
)
from fleetops.domain.models import (
    MODEL_FORMAT,
    DeviceCapabilities,
    FaultConfig,
    Manifest,
    ModelIdentity,
    compatibility_problems,
    utcnow,
)
from fleetops.edge.inference import InferenceEngine, InferenceResult, InjectedFaultError, NoModelError
from fleetops.edge.metrics import EdgeMetrics
from fleetops.edge.model_store import ModelStore, StoreError
from fleetops.messaging.bus import BusDisconnectedError, MessageBus, Will
from fleetops.messaging.topics import QOS, RETAIN, Channel, topic
from fleetops.registry.modelformat import InputRejectedError, ModelFormatError, load_bundle
from fleetops.resilience import retry
from fleetops.security.artifact import ChecksumMismatchError, verify_sha256
from fleetops.security.device_identity import verify_command
from fleetops.training.sensors import FEATURE_NAMES, generate_arrays

logger = logging.getLogger(__name__)

MAX_PENDING_SAMPLES = 2000


class CommandFailedError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class EdgeAgent:
    def __init__(
        self,
        settings: EdgeSettings,
        bus: MessageBus,
        http: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.s = settings
        self.device_id = settings.device_id
        self._secret = settings.device_secret.get_secret_value()
        self.bus = bus
        self.store = ModelStore(settings.data_dir)
        self.engine = InferenceEngine()
        self.metrics = EdgeMetrics(self.device_id)
        self.boot_id = uuid.uuid4().hex
        self._started = time.monotonic()
        self._http = http or httpx.Client(
            # Chain and hostname are verified; a configured CA only widens trust (verification stays on).
            verify=ssl.create_default_context(cafile=str(settings.control_plane_ca))
            if settings.control_plane_ca
            else True,
            base_url=settings.control_plane_url.rstrip("/"),
            timeout=settings.http_timeout_seconds,
            headers={"Authorization": f"Bearer {self._secret}", "X-Device-ID": self.device_id},
        )
        self._sleep = sleep
        self.network_enabled = True
        self._cmd_lock = threading.Lock()
        self._pending: deque[PredictionSample] = deque(maxlen=MAX_PENDING_SAMPLES)
        self._sensor = _SensorFeed(settings.sensor_seed, settings.sensor_scenario, settings.label_fraction)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        self._last_sample_count = (time.monotonic(), 0)
        if settings.allow_faults and settings.initial_faults:
            self.engine.faults = FaultConfig.model_validate_json(settings.initial_faults)
        self.capabilities = DeviceCapabilities(agent_version=__version__, formats=[MODEL_FORMAT])
        self._recover()

    # ---------------------------------------------------------------- lifecycle

    def _recover(self) -> None:
        model, note = self.store.recover_active()
        if note:
            logger.warning("startup recovery: %s", note)
        self.engine.swap(model, self.store.state.active if model else None)
        self._update_model_gauge()

    def start(self, loops: bool = True) -> None:
        self.bus.subscribe(topic(self.device_id, Channel.COMMANDS), self._on_command_message, qos=1)
        self.bus.start()
        if loops:
            for name, fn, period in [
                ("sensor", self.sensor_step, 1.0 / self.s.sensor_hz if self.s.sensor_hz else 1.0),
                ("heartbeat", self.send_heartbeat, self.s.heartbeat_interval_seconds),
                ("telemetry", self.send_telemetry, self.s.telemetry_interval_seconds),
                ("flush", self.flush_predictions, self.s.prediction_flush_seconds),
            ]:
                if name == "sensor" and not self.s.sensor_hz:
                    continue
                t = threading.Thread(target=self._loop, args=(name, fn, period), daemon=True, name=name)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self.bus.stop()
        self._http.close()

    def _loop(self, name: str, fn: Any, period: float) -> None:
        next_run = time.monotonic()
        while not self._stop.is_set():
            try:
                fn()
            except Exception:
                logger.exception("loop %s iteration failed", name)
            next_run += period
            self._stop.wait(max(0.0, next_run - time.monotonic()))

    def set_network(self, enabled: bool) -> None:
        """Simulated outage: MQTT and control-plane HTTP unavailable; local inference unaffected."""
        self.network_enabled = enabled
        if enabled:
            self.bus.restore_network()
        else:
            self.bus.drop_network()
        logger.warning("network %s (simulated)", "restored" if enabled else "cut")

    # ---------------------------------------------------------------- inference

    def infer(self, features: list[float]) -> InferenceResult:
        """Local inference: never touches the network."""
        did = self.device_id
        try:
            result = self.engine.predict(features)
        except NoModelError:
            raise
        except InputRejectedError:
            self.metrics.inputs_rejected.labels(did).inc()
            raise
        except Exception:
            ident = self.engine.active
            v = str(ident.version) if ident else "none"
            self.metrics.inference_total.labels(did, v).inc()
            self.metrics.inference_failures.labels(did, v).inc()
            raise
        v = str(result.model.version)
        self.metrics.inference_total.labels(did, v).inc()
        self.metrics.inference_latency.labels(did, v).observe(result.latency_ms / 1000)
        return result

    def sensor_step(self) -> None:
        """Read one simulated sensor sample, run local inference, buffer the result."""
        features, label = self._sensor.next()
        try:
            r = self.infer(features)
        except (NoModelError, InputRejectedError, InjectedFaultError):
            return
        except Exception:
            logger.exception("inference failed")
            return
        self._pending.append(
            PredictionSample(
                ts=utcnow(),
                model_name=r.model.name,
                model_version=r.model.version,
                features=features,
                score=r.prediction.score,
                anomaly=r.prediction.anomaly,
                latency_ms=r.latency_ms,
                label=label,
            )
        )

    # ---------------------------------------------------------------- uplink

    def _publish(self, channel: Channel, payload: str) -> bool:
        try:
            self.bus.publish(topic(self.device_id, channel), payload.encode(), QOS[channel], RETAIN.get(channel, False))
            return True
        except BusDisconnectedError:
            return False

    def send_heartbeat(self) -> bool:
        cpu = min(100.0, self._proc.cpu_percent(None) / (psutil.cpu_count() or 1))  # % of total host CPU
        mem = self._proc.memory_info().rss / 1e6
        disk_free = self._free_disk_bytes()
        self.metrics.disk_free.labels(self.device_id).set(disk_free / 1e6)
        self.metrics.cpu.labels(self.device_id).set(cpu)
        self.metrics.memory.labels(self.device_id).set(mem)
        self.metrics.mqtt_connected.labels(self.device_id).set(1 if self.bus.connected else 0)
        hb = Heartbeat(
            device_id=self.device_id,
            sent_at=utcnow(),
            boot_id=self.boot_id,
            agent_version=__version__,
            uptime_seconds=time.monotonic() - self._started,
            state=self.store.snapshot(),
            capabilities=self.capabilities,
            model_stats=self.engine.stats(),
            cpu_percent=cpu,
            memory_mb=mem,
            mqtt_connected=self.bus.connected,
            disk_free_mb=disk_free / 1e6,
        )
        return self._publish(Channel.HEARTBEAT, hb.model_dump_json())

    def send_telemetry(self) -> bool:
        now, prev = time.monotonic(), self._last_sample_count
        stats = self.engine.stats()
        count = stats.inference_count if stats else 0
        rate = max(0.0, (count - prev[1]) / max(now - prev[0], 1e-6)) if count >= prev[1] else 0.0
        self._last_sample_count = (now, count)
        recent = list(self._pending)[-50:]
        means = (
            dict(zip(FEATURE_NAMES, np.mean([s.features for s in recent], axis=0).tolist(), strict=True))
            if recent
            else {}
        )
        t = Telemetry(
            device_id=self.device_id,
            sent_at=utcnow(),
            inference_rate_hz=rate,
            inputs_rejected=self.engine.inputs_rejected,
            sensor_means=means,
        )
        return self._publish(Channel.TELEMETRY, t.model_dump_json())

    def flush_predictions(self, max_batch: int = 100) -> bool:
        if not self._pending or not self.bus.connected:
            return False
        batch = [self._pending.popleft() for _ in range(min(max_batch, len(self._pending)))]
        msg = PredictionBatch(device_id=self.device_id, sent_at=utcnow(), samples=batch)
        if self._publish(Channel.PREDICTIONS, msg.model_dump_json()):
            return True
        self._pending.extendleft(reversed(batch))  # keep for the next flush
        return False

    def announce_online(self) -> bool:
        msg = StatusMessage(
            kind="online",
            device_id=self.device_id,
            sent_at=utcnow(),
            boot_id=self.boot_id,
            state=self.store.snapshot(),
            capabilities=self.capabilities,
        )
        return self._publish(Channel.STATUS, msg.model_dump_json())

    @staticmethod
    def will_for(device_id: str) -> Will:
        msg = StatusMessage(kind="offline", device_id=device_id, sent_at=utcnow())
        return Will(topic(device_id, Channel.STATUS), msg.model_dump_json().encode(), qos=1, retain=True)

    # ---------------------------------------------------------------- commands

    def _on_command_message(self, _topic: str, payload: bytes) -> None:
        try:
            if len(payload) > 256 * 1024:
                raise ValueError("command too large")
            cmd = Command.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            self.metrics.mqtt_failures.labels(self.device_id).inc()
            logger.warning("dropping malformed command: %s", exc)
            return
        self.handle_command(cmd)

    def handle_command(self, cmd: Command) -> StatusMessage:
        with (
            self._cmd_lock,
            log_context(
                device_id=self.device_id,
                command_id=cmd.command_id,
                deployment_id=cmd.deployment_id,
                correlation_id=cmd.correlation_id,
            ),
        ):
            started = time.monotonic()
            cached = self.store.state.acks.get(cmd.command_id)
            if cached is not None:  # duplicate delivery or control-plane retry: replay the same ack
                self.metrics.duplicate_commands.labels(self.device_id, cmd.type.value).inc()
                ack = StatusMessage.model_validate(cached)
                ack = ack.model_copy(update={"sent_at": utcnow(), "state": self.store.snapshot()})
                self._send_ack(ack)
                return ack
            ack = self._decide(cmd, started)
            try:
                self.store.record_command(
                    cmd.generation if ack.outcome != AckOutcome.REJECTED else 0,
                    cmd.command_id,
                    ack.model_dump(mode="json"),
                )
            except StoreError:  # cannot persist the dedup record (e.g. disk full): still answer, but say so
                logger.exception("could not persist command record; duplicate protection degraded for this command")
            self.metrics.commands.labels(self.device_id, cmd.type.value, ack.outcome.value if ack.outcome else "").inc()
            self._send_ack(ack)
            return ack

    def _decide(self, cmd: Command, started: float) -> StatusMessage:
        def ack(outcome: AckOutcome, code: str | None = None, detail: str | None = None) -> StatusMessage:
            return StatusMessage(
                kind="ack",
                device_id=self.device_id,
                sent_at=utcnow(),
                boot_id=self.boot_id,
                state=self.store.snapshot(),
                capabilities=self.capabilities,
                command_id=cmd.command_id,
                command_type=cmd.type,
                deployment_id=cmd.deployment_id,
                generation=cmd.generation,
                outcome=outcome,
                error_code=code,
                detail=detail,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        if cmd.device_id != self.device_id or not verify_command(self._secret, cmd):
            return ack(AckOutcome.REJECTED, "BAD_SIGNATURE", "command signature invalid")
        now = utcnow()
        # Clock skew is the usual reason a healthy device rejects fresh commands (or accepts stale ones), so
        # every time-based rejection is logged with the numbers an operator needs.
        ahead_of_issue = (now - cmd.issued_at).total_seconds()  # >0 normally; <0 if our clock is behind
        if ahead_of_issue < -self.s.max_clock_skew_seconds:
            logger.warning(
                "command rejected: issued in the future (device clock behind control plane?)",
                extra={
                    "reason": "clock_skew",
                    "skew_seconds": round(-ahead_of_issue, 1),
                    "tolerance_seconds": self.s.max_clock_skew_seconds,
                    "issued_at": cmd.issued_at.isoformat(),
                },
            )
            return ack(
                AckOutcome.REJECTED,
                "CLOCK_SKEW",
                f"device clock is {-ahead_of_issue:.0f}s behind the command issue time "
                f"(tolerance {self.s.max_clock_skew_seconds:.0f}s)",
            )
        if now > cmd.expires_at or ahead_of_issue > self.s.command_max_age_seconds:
            overdue = max((now - cmd.expires_at).total_seconds(), 0.0)
            logger.warning(
                "command rejected: expired",
                extra={
                    "reason": "expired",
                    "overdue_seconds": round(overdue, 1),
                    "age_seconds": round(ahead_of_issue, 1),
                    "expires_at": cmd.expires_at.isoformat(),
                },
            )
            return ack(
                AckOutcome.REJECTED,
                "EXPIRED",
                f"command expired {overdue:.0f}s ago (age {ahead_of_issue:.0f}s); if this repeats, "
                "check the device clock (NTP)",
            )
        if cmd.generation < self.store.state.last_generation:
            return ack(
                AckOutcome.REJECTED, "STALE", f"generation {cmd.generation} < {self.store.state.last_generation}"
            )
        try:
            outcome = self._execute(cmd)
        except CommandFailedError as exc:
            logger.warning("command failed: %s %s", exc.code, exc.detail)
            return ack(AckOutcome.FAILED, exc.code, exc.detail)
        except Exception as exc:
            logger.exception("unexpected command failure")
            return ack(AckOutcome.FAILED, "INTERNAL", str(exc)[:500])
        self.metrics.update_duration.labels(self.device_id, cmd.type.value).observe(time.monotonic() - started)
        return ack(outcome)

    def _send_ack(self, ack: StatusMessage) -> None:
        faults = self.engine.faults
        selective = ack.outcome is not None and ack.outcome.value in faults.drop_ack_outcomes
        if (faults.drop_acks or selective) and self.s.allow_faults and ack.command_type != CommandType.SET_FAULTS:
            logger.warning("ack dropped (fault injection)")
            return
        if not self._publish(Channel.STATUS, ack.model_dump_json()):
            logger.warning("ack not sent (link down); will replay on command retry")

    def _execute(self, cmd: Command) -> AckOutcome:
        if cmd.type is CommandType.DEPLOY:
            if cmd.manifest is None:
                raise CommandFailedError("BAD_COMMAND", "DEPLOY without manifest")
            self._stage(cmd.manifest)
            return AckOutcome.STAGED
        if cmd.type is CommandType.ACTIVATE:
            if cmd.activate is None:
                raise CommandFailedError("BAD_COMMAND", "ACTIVATE without model")
            self._activate(cmd.activate.name, cmd.activate.version)
            return AckOutcome.ACTIVE
        if cmd.type is CommandType.ROLLBACK:
            self._rollback(cmd)
            return AckOutcome.ROLLED_BACK
        if cmd.type is CommandType.ABORT:
            if cmd.activate is not None:
                self.store.discard_staged(cmd.activate.name, cmd.activate.version)
            return AckOutcome.ABORTED
        if cmd.type is CommandType.SET_FAULTS:
            if not self.s.allow_faults:
                raise CommandFailedError("FAULTS_DISABLED", "fault injection disabled on this device")
            self.engine.faults = cmd.faults or FaultConfig()
            return AckOutcome.ACCEPTED
        raise CommandFailedError("BAD_COMMAND", f"unsupported command {cmd.type}")

    # -- DEPLOY

    def _stage(self, m: Manifest) -> None:
        problems = compatibility_problems(m.compatibility, self.capabilities)
        if problems:
            raise CommandFailedError("INCOMPATIBLE", "; ".join(problems))
        ident = ModelIdentity(name=m.model.name, version=m.model.version, sha256=m.artifact.sha256)
        if self.store.state.staged == ident and self.store.has(ident):
            return  # idempotent re-delivery
        needed = int(m.artifact.size_bytes * self.s.storage_headroom_factor) + self.s.min_free_disk_mb * 1024 * 1024
        free = self._free_disk_bytes()
        if free < needed:  # do not even start: a half-written download could crowd out the running model's volume
            raise CommandFailedError("INSUFFICIENT_STORAGE", f"need {needed} bytes free, only {free} available")
        part = self.store.staging_dir / f"{m.deployment_id}-{uuid.uuid4().hex[:8]}.part"
        try:
            self._download(m.artifact.path, part, m.artifact.size_bytes)
            if m.model.version in self.engine.faults.corrupt_download_versions and self.s.allow_faults:
                with part.open("r+b") as fh:  # deterministic corruption: flip the last byte
                    fh.seek(-1, 2)
                    last = fh.read(1)
                    fh.seek(-1, 2)
                    fh.write(bytes([last[0] ^ 0xFF]))
            try:
                verify_sha256(part, m.artifact.sha256)
            except ChecksumMismatchError as exc:
                raise CommandFailedError("CHECKSUM_MISMATCH", str(exc)) from exc
            try:
                bundle = load_bundle(part)
                bundle.smoke_test()
            except (ModelFormatError, ValueError) as exc:
                raise CommandFailedError("INVALID_MODEL", str(exc)) from exc
            if bundle.meta.input_schema.names != m.input_schema.names:
                raise CommandFailedError("SCHEMA_MISMATCH", "bundle features differ from manifest input schema")
            try:
                self.store.commit_staged(ident, part)
            except StoreError as exc:
                raise CommandFailedError("INVALID_MODEL", str(exc)) from exc
        except OSError as exc:  # disk full / read-only volume / I/O error while writing or moving the download
            code = "STORAGE_FULL" if exc.errno == errno.ENOSPC else "STORAGE_ERROR"
            raise CommandFailedError(code, f"{type(exc).__name__}: {exc}") from exc
        finally:
            part.unlink(missing_ok=True)

    def _free_disk_bytes(self) -> int:
        fake = self.engine.faults.fake_free_disk_mb
        if fake is not None and self.s.allow_faults:
            return fake * 1024 * 1024
        return shutil.disk_usage(self.s.data_dir).free

    def _download(self, path: str, dest: Path, expected_size: int) -> None:
        """Fetch from the configured control plane only (path is validated in the manifest)."""
        if not self.network_enabled:
            raise CommandFailedError("DOWNLOAD_FAILED", "control plane unreachable (network down)")
        limit = min(expected_size, self.s.max_artifact_bytes)

        def attempt() -> None:
            written = 0
            with self._http.stream("GET", path) as resp:
                if resp.status_code in (401, 403, 404):
                    raise CommandFailedError("DOWNLOAD_DENIED", f"HTTP {resp.status_code}")
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        written += len(chunk)
                        if written > limit:
                            raise CommandFailedError("ARTIFACT_TOO_LARGE", "download exceeds declared size")
                        fh.write(chunk)

        try:
            retry(
                attempt,
                attempts=self.s.download_max_attempts,
                base=self.s.download_backoff_base_seconds,
                maximum=self.s.download_backoff_max_seconds,
                retry_on=(httpx.TransportError, httpx.HTTPStatusError),
                sleep=self._sleep,
                jitter=self.s.download_jitter,
            )
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise CommandFailedError("DOWNLOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc

    # -- ACTIVATE / ROLLBACK

    def _activate(self, name: str, version: int) -> None:
        staged = self.store.state.staged
        active = self.store.state.active
        if active and (active.name, active.version) == (name, version) and staged is None:
            return  # already active: idempotent
        if staged is None or (staged.name, staged.version) != (name, version):
            raise CommandFailedError("NOT_STAGED", f"{name}:{version} is not staged")
        if version in self.engine.faults.fail_activation_versions and self.s.allow_faults:
            raise CommandFailedError("ACTIVATION_FAILED", f"injected activation failure for version {version}")
        try:
            model = self.store.activate_staged(staged)
        except StoreError as exc:
            raise CommandFailedError("ACTIVATION_FAILED", str(exc)) from exc
        self.engine.swap(model, staged)  # in-memory swap only after the pointer is durable
        self._update_model_gauge()

    def _rollback(self, cmd: Command) -> None:
        if cmd.rollback_to is None:
            raise CommandFailedError("BAD_COMMAND", "ROLLBACK without target")
        if self.engine.faults.fail_rollback and self.s.allow_faults:
            raise CommandFailedError("ROLLBACK_FAILED", "injected rollback failure")
        target = cmd.rollback_to.model
        if target is None:  # no known-good model exists: deactivate
            self.store.switch_to(None)
            self.engine.swap(None, None)
            self._update_model_gauge()
            return
        if self.engine.active == target:
            return
        if not self.store.has(target):
            if cmd.rollback_to.manifest is None:
                raise CommandFailedError("ROLLBACK_UNAVAILABLE", "known-good model missing locally")
            self._stage(cmd.rollback_to.manifest)
        try:
            model = self.store.switch_to(target)
        except StoreError as exc:
            raise CommandFailedError("ROLLBACK_FAILED", str(exc)) from exc
        self.engine.swap(model, target)
        self._update_model_gauge()

    def _update_model_gauge(self) -> None:
        ident = self.engine.active
        self.metrics.model_version.clear()
        self.metrics.model_version.labels(self.device_id, ident.name if ident else "none").set(
            ident.version if ident else 0
        )


class _SensorFeed:
    """Deterministic simulated sensor stream (stand-in for a PLC/OPC-UA feed)."""

    def __init__(self, seed: int, scenario: str, label_fraction: float, block: int = 256) -> None:
        self._seed, self._scenario, self._label_fraction, self._block = seed, scenario, label_fraction, block
        self._rng = np.random.default_rng(seed + 10_000)
        self._i = 0
        self._x = np.zeros((0, len(FEATURE_NAMES)))
        self._y = np.zeros(0, dtype=np.int64)
        self._cursor = 0

    def next(self) -> tuple[list[float], bool | None]:
        if self._cursor >= len(self._x):
            self._x, self._y, _ = generate_arrays(
                self._block,
                0.06,
                self._seed * 1000 + self._i,
                self._scenario,  # type: ignore[arg-type]
            )
            self._i += 1
            self._cursor = 0
        row, label = self._x[self._cursor], bool(self._y[self._cursor])
        self._cursor += 1
        reveal = self._label_fraction > 0 and self._rng.random() < self._label_fraction
        return [float(v) for v in row], (label if reveal else None)
