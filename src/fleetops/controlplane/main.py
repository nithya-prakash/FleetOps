"""Control plane entrypoint: ``python -m fleetops.controlplane``."""

from __future__ import annotations

import os

import uvicorn

from fleetops.config.logging import configure_logging
from fleetops.config.settings import ControlPlaneSettings
from fleetops.controlplane.app import create_app


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    settings = ControlPlaneSettings.load()
    configure_logging("control-plane", settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        server_header=False,  # do not advertise the server software
        timeout_keep_alive=settings.keep_alive_timeout_seconds,
        limit_concurrency=settings.max_concurrent_requests,
        h11_max_incomplete_event_size=16 * 1024,  # cap header size
        ssl_certfile=str(settings.tls_certfile) if settings.tls_certfile else None,
        ssl_keyfile=str(settings.tls_keyfile) if settings.tls_keyfile else None,
    )


if __name__ == "__main__":
    main()
