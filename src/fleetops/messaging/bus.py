"""Message bus abstraction with an MQTT (paho) implementation and an in-process test double.

The in-memory broker reproduces the MQTT behaviours the platform relies on: wildcard
subscriptions, retained messages, last-will on unclean disconnect, persistent sessions (QoS 1
messages queued while a client is offline), and loss of QoS 0 traffic. It lets the real agent and
control-plane code run deterministically in tests; the paho implementation is verified against a
real Mosquitto (see tests/integration/test_mqtt_bus.py and the Compose demo).
"""

from __future__ import annotations

import logging
import queue
import ssl
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from fleetops.messaging.topics import topic_matches

logger = logging.getLogger(__name__)

Handler = Callable[[str, bytes], None]
ErrorHook = Callable[[str, Exception], None]


class BusDisconnectedError(Exception):
    """Publish attempted while the transport is down."""


@dataclass(frozen=True)
class Will:
    topic: str
    payload: bytes
    qos: int = 1
    retain: bool = True


class MessageBus(Protocol):
    @property
    def connected(self) -> bool: ...
    def subscribe(self, pattern: str, handler: Handler, qos: int = 1) -> None: ...
    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def drop_network(self) -> None: ...
    def restore_network(self) -> None: ...


# ----------------------------------------------------------------------------- in-memory


class InMemoryBroker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clients: dict[str, InMemoryBus] = {}
        self._retained: dict[str, bytes] = {}
        self._queue: queue.Queue[tuple[InMemoryBus, str, bytes] | None] = queue.Queue()
        self._offline: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
        self._pending = 0
        self._idle = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="inmem-broker", daemon=True)
        self._thread.start()
        self.published: list[tuple[str, bytes]] = []  # audit trail for tests

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            client, topic, payload = item
            try:
                client._deliver(topic, payload)
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _enqueue(self, client: InMemoryBus, topic: str, payload: bytes) -> None:
        with self._idle:
            self._pending += 1
        self._queue.put((client, topic, payload))

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until every queued message has been handled (handlers may enqueue more)."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout=timeout)

    def register(self, bus: InMemoryBus) -> None:
        with self._lock:
            self._clients[bus.client_id] = bus

    def route(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        with self._lock:
            self.published.append((topic, payload))
            if retain:
                if payload:
                    self._retained[topic] = payload
                else:
                    self._retained.pop(topic, None)
            for client in self._clients.values():
                if not client.wants(topic):
                    continue
                if client.connected:
                    self._enqueue(client, topic, payload)
                elif qos >= 1 and client.persistent:
                    self._offline[client.client_id].append((topic, payload))

    def on_connect(self, bus: InMemoryBus) -> None:
        with self._lock:
            for topic, payload in self._offline.pop(bus.client_id, []):
                self._enqueue(bus, topic, payload)
            for topic, payload in list(self._retained.items()):
                if bus.wants(topic):
                    self._enqueue(bus, topic, payload)

    def retained(self, topic: str) -> bytes | None:
        with self._lock:
            return self._retained.get(topic)


class InMemoryBus:
    def __init__(
        self,
        broker: InMemoryBroker,
        client_id: str,
        will: Will | None = None,
        persistent: bool = True,
        on_error: ErrorHook | None = None,
    ) -> None:
        self._broker = broker
        self.client_id = client_id
        self.persistent = persistent
        self._will = will
        self._subs: list[tuple[str, Handler]] = []
        self._connected = False
        self._on_error = on_error
        broker.register(self)

    @property
    def connected(self) -> bool:
        return self._connected

    def wants(self, topic: str) -> bool:
        return any(topic_matches(p, topic) for p, _ in self._subs)

    def subscribe(self, pattern: str, handler: Handler, qos: int = 1) -> None:
        self._subs.append((pattern, handler))

    def start(self) -> None:
        self._connected = True
        self._broker.on_connect(self)

    def stop(self) -> None:  # clean disconnect: no will
        self._connected = False

    def drop_network(self) -> None:
        """Unclean disconnect: the broker publishes the last will."""
        if not self._connected:
            return
        self._connected = False
        if self._will:
            self._broker.route(self._will.topic, self._will.payload, self._will.qos, self._will.retain)

    def restore_network(self) -> None:
        self.start()

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        if not self._connected:
            raise BusDisconnectedError(self.client_id)
        self._broker.route(topic, payload, qos, retain)

    def _deliver(self, topic: str, payload: bytes) -> None:
        for pattern, handler in list(self._subs):
            if topic_matches(pattern, topic):
                try:
                    handler(topic, payload)
                except Exception as exc:
                    logger.exception("handler failed", extra={"topic": topic})
                    if self._on_error:
                        self._on_error(topic, exc)


# ----------------------------------------------------------------------------- MQTT (paho)


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int
    client_id: str
    username: str | None = None
    password: str | None = None
    tls_ca: Path | None = None
    tls_cert: Path | None = None  # client certificate for mutual TLS
    tls_key: Path | None = None
    keepalive: int = 15
    clean_session: bool = False
    # Event fired after each (re)connection; the application re-announces its presence here.
    reconnect_delay_min: int = 1
    reconnect_delay_max: int = 30


class MqttBus:
    """paho-mqtt 2.x client: automatic reconnect with backoff, persistent session, LWT."""

    def __init__(
        self,
        config: MqttConfig,
        will: Will | None = None,
        on_connect: Callable[[], None] | None = None,
        on_error: ErrorHook | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        self._cfg = config
        self._on_disconnect_cb = on_disconnect
        self._subs: list[tuple[str, Handler, int]] = []
        self._on_connect_cb = on_connect
        self._on_error = on_error
        self._connected = threading.Event()
        self._client = mqtt.Client(
            CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
            clean_session=config.clean_session,
        )
        if config.username:
            self._client.username_pw_set(config.username, config.password)
        if config.tls_ca:
            # Server certificate AND hostname are always verified (paho's tls_insecure_set is never called).
            self._client.tls_set(
                ca_certs=str(config.tls_ca),
                certfile=str(config.tls_cert) if config.tls_cert else None,
                keyfile=str(config.tls_key) if config.tls_key else None,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
        if will:
            self._client.will_set(will.topic, will.payload, qos=will.qos, retain=will.retain)
        self._client.reconnect_delay_set(config.reconnect_delay_min, config.reconnect_delay_max)
        self._client.max_queued_messages_set(1000)
        self._client.on_connect = self._handle_connect
        self._client.on_disconnect = self._handle_disconnect
        self._client.on_message = self._handle_message

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def subscribe(self, pattern: str, handler: Handler, qos: int = 1) -> None:
        self._subs.append((pattern, handler, qos))
        if self.connected:
            self._client.subscribe(pattern, qos)

    def start(self) -> None:
        # connect_async + loop_start: never blocks or raises if the broker is down; the network
        # thread keeps retrying with backoff.
        self._client.connect_async(self._cfg.host, self._cfg.port, self._cfg.keepalive)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()
        self._connected.clear()

    def drop_network(self) -> None:
        """Simulated outage: disconnect and stop reconnecting."""
        self.stop()

    def restore_network(self) -> None:
        self.start()

    def wait_connected(self, timeout: float) -> bool:
        return self._connected.wait(timeout)

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        if not self.connected:
            raise BusDisconnectedError(self._cfg.client_id)
        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise BusDisconnectedError(f"publish rc={info.rc}")

    # callbacks run on paho's network thread
    def _handle_connect(self, client: mqtt.Client, _ud: object, _flags: object, rc: object, _p: object) -> None:
        if getattr(rc, "is_failure", False):
            logger.error("mqtt connect refused: %s", rc)
            return
        for pattern, _h, qos in self._subs:
            client.subscribe(pattern, qos)
        self._connected.set()
        logger.info("mqtt connected", extra={"host": self._cfg.host})
        if self._on_connect_cb:
            try:
                self._on_connect_cb()
            except Exception as exc:
                logger.exception("on_connect hook failed")
                if self._on_error:
                    self._on_error("on_connect", exc)

    def _handle_disconnect(self, _c: mqtt.Client, _ud: object, _f: object, rc: object, _p: object) -> None:
        self._connected.clear()
        logger.warning("mqtt disconnected: %s", rc)
        if self._on_disconnect_cb:
            try:
                self._on_disconnect_cb()
            except Exception:
                logger.exception("on_disconnect hook failed")

    def _handle_message(self, _c: mqtt.Client, _ud: object, msg: mqtt.MQTTMessage) -> None:
        for pattern, handler, _qos in list(self._subs):
            if topic_matches(pattern, msg.topic):
                try:
                    handler(msg.topic, bytes(msg.payload))
                except Exception as exc:
                    logger.exception("handler failed", extra={"topic": msg.topic})
                    if self._on_error:
                        self._on_error(msg.topic, exc)
