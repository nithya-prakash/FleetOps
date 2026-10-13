"""Edge agent entrypoint: ``python -m fleetops.edge``."""

from __future__ import annotations

import logging
import signal
import threading

import uvicorn

from fleetops.config.logging import configure_logging
from fleetops.config.settings import EdgeSettings
from fleetops.edge.agent import EdgeAgent
from fleetops.edge.api import create_edge_app
from fleetops.messaging.bus import MqttBus, MqttConfig

logger = logging.getLogger(__name__)


def main() -> None:
    settings = EdgeSettings.load()
    configure_logging(f"edge-{settings.device_id}", settings.log_level)
    holder: dict[str, EdgeAgent] = {}

    def on_connect() -> None:
        holder["agent"].announce_online()

    bus = MqttBus(
        MqttConfig(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            client_id=f"fleetops-{settings.device_id}",
            username=settings.mqtt_username or settings.device_id,
            password=(settings.mqtt_password or settings.device_secret).get_secret_value(),
            tls_ca=settings.mqtt_tls_ca,
            tls_cert=settings.mqtt_tls_cert,
            tls_key=settings.mqtt_tls_key,
            keepalive=settings.mqtt_keepalive_seconds,
            reconnect_delay_min=settings.mqtt_reconnect_min_seconds,
            reconnect_delay_max=settings.mqtt_reconnect_max_seconds,
        ),
        will=EdgeAgent.will_for(settings.device_id),
        on_connect=on_connect,
    )
    if settings.api_token is None or settings.metrics_token is None:
        logger.warning(
            "agent API is not fully authenticated: set FLEETOPS_EDGE_API_TOKEN and FLEETOPS_EDGE_METRICS_TOKEN"
        )
    agent = EdgeAgent(settings, bus)
    holder["agent"] = agent
    agent.start()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    server = uvicorn.Server(
        uvicorn.Config(
            create_edge_app(agent),
            host=settings.admin_host,
            port=settings.admin_port,
            log_level="warning",
            server_header=False,
            timeout_keep_alive=5,
            limit_concurrency=100,
            h11_max_incomplete_event_size=16 * 1024,
            ssl_certfile=str(settings.tls_certfile) if settings.tls_certfile else None,
            ssl_keyfile=str(settings.tls_keyfile) if settings.tls_keyfile else None,
        )
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    logger.info("edge agent started", extra={"device_id": settings.device_id})
    stop.wait()
    server.should_exit = True
    agent.stop()


if __name__ == "__main__":
    main()
