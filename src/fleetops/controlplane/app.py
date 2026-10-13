"""FastAPI application factory for the control plane."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fleetops.config.settings import ControlPlaneSettings
from fleetops.controlplane import devices
from fleetops.controlplane.context import AppContext
from fleetops.controlplane.ingest import BusCommandPublisher, NullPublisher, UplinkIngest
from fleetops.controlplane.routes import router
from fleetops.deployment.engine import (
    CommandPublisher,
    ConflictError,
    EngineError,
    ForbiddenError,
    InvalidRequestError,
    NotFoundError,
    QualityGateError,
    RolloutEngine,
)
from fleetops.messaging.bus import MessageBus, MqttBus, MqttConfig
from fleetops.monitoring.drift import DriftThresholds
from fleetops.monitoring.drift_service import DriftService
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.db import make_engine, make_session_factory, session_scope, upgrade_database
from fleetops.registry.artifact_store import ArtifactStore
from fleetops.registry.mlflow_registry import MlflowRegistry, write_mlflow_credentials
from fleetops.registry.service import ModelRegistrationService
from fleetops.security.asgi import BodyLimitMiddleware, RateLimitMiddleware
from fleetops.security.auth import ApiKeyAuthenticator, Role
from fleetops.security.ratelimit import TokenBucketLimiter

logger = logging.getLogger(__name__)


def build_context(
    settings: ControlPlaneSettings, bus: MessageBus | None, registry: MlflowRegistry | None
) -> AppContext:
    upgrade_database(settings.database_url)
    db_engine = make_engine(settings.database_url)
    sessions = make_session_factory(db_engine)
    metrics = ControlPlaneMetrics(sessions)
    if settings.mlflow_username and settings.mlflow_password:
        # A private credentials file (tmpfs $HOME in containers) rather than environment variables.
        write_mlflow_credentials(settings.mlflow_username, settings.mlflow_password.get_secret_value())
    if settings.mlflow_ca_cert:  # MLflow verifies the server certificate against this bundle (path, not a secret)
        os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] = str(settings.mlflow_ca_cert)
    store = ArtifactStore(settings.artifact_root)
    registry = registry or MlflowRegistry(
        settings.mlflow_tracking_uri, settings.mlflow_experiment, settings.mlflow_artifact_location
    )
    publisher: CommandPublisher = BusCommandPublisher(bus) if bus is not None else NullPublisher()
    engine = RolloutEngine(
        sessions,
        publisher,
        metrics,
        settings.device_key_master.get_secret_value(),
        command_ttl_seconds=settings.command_ttl_seconds,
        require_quality_gate=settings.require_quality_gate,
        backoff_base=settings.retry_backoff_base_seconds,
        backoff_max=settings.retry_backoff_max_seconds,
    )
    ingest = UplinkIngest(sessions, engine, metrics, settings.sample_retention, settings.clock_skew_warn_seconds)
    drift = DriftService(
        sessions,
        DriftThresholds(
            settings.drift_psi_warn, settings.drift_psi_alert, settings.drift_p_value, settings.drift_min_samples
        ),
        settings.drift_window_samples,
        metrics,
    )
    auth = ApiKeyAuthenticator(
        {
            Role.ADMIN: settings.admin_api_key.get_secret_value(),
            Role.OPERATOR: settings.operator_api_key.get_secret_value(),
            Role.VIEWER: settings.viewer_api_key.get_secret_value(),
        }
    )
    if bus is not None:
        ingest.attach(bus)
    return AppContext(
        settings,
        db_engine,
        sessions,
        store,
        registry,
        ModelRegistrationService(sessions, store, registry),
        engine,
        metrics,
        ingest,
        drift,
        auth,
        bus,
        limiter=TokenBucketLimiter(settings.rate_limit_per_second, settings.rate_limit_burst),
        auth_failures=TokenBucketLimiter(
            settings.auth_failure_limit_per_minute / 60.0, settings.auth_failure_limit_per_minute
        ),
    )


def make_mqtt_bus(
    settings: ControlPlaneSettings,
    on_connect: Callable[[], None] | None = None,
    on_disconnect: Callable[[], None] | None = None,
) -> MqttBus:
    return MqttBus(
        MqttConfig(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            client_id="fleetops-control-plane",
            username=settings.mqtt_username,
            password=settings.mqtt_password.get_secret_value() if settings.mqtt_password else None,
            tls_ca=settings.mqtt_tls_ca,
            tls_cert=settings.mqtt_tls_cert,
            tls_key=settings.mqtt_tls_key,
            keepalive=settings.mqtt_keepalive_seconds,
            reconnect_delay_min=settings.mqtt_reconnect_min_seconds,
            reconnect_delay_max=settings.mqtt_reconnect_max_seconds,
        ),
        on_connect=on_connect,
        on_disconnect=on_disconnect,
    )


async def _ticker(ctx: AppContext) -> None:
    """Drives rollouts (retries, timeouts, health) and marks silent devices offline."""
    interval = ctx.settings.tick_interval_seconds
    last_drift = 0.0
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.to_thread(ctx.engine.advance_all)

            def stale() -> None:
                with session_scope(ctx.sessions) as db:
                    devices.mark_stale(db, ctx.settings.offline_after_seconds)

            await asyncio.to_thread(stale)
            if loop.time() - last_drift >= ctx.settings.drift_interval_seconds:
                last_drift = loop.time()
                await asyncio.to_thread(ctx.drift.run_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ticker iteration failed")
        await asyncio.sleep(interval)


# Background work (ticker, drift, startup) runs one task at a time, so a small fixed pool is enough. asyncio's default
# executor would grow lazily up to cpu_count + 4 threads, which looked like a thread leak in the soak test.
BACKGROUND_WORKERS = 4


def create_app(
    settings: ControlPlaneSettings,
    bus: MessageBus | None = None,
    registry: MlflowRegistry | None = None,
    run_background: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        background = ThreadPoolExecutor(max_workers=BACKGROUND_WORKERS, thread_name_prefix="fleetops-bg")
        asyncio.get_running_loop().set_default_executor(background)
        chosen_bus = bus
        connects = {"n": 0}

        def mqtt_up() -> None:  # runs on the MQTT network thread
            ctx_now: AppContext | None = getattr(app.state, "ctx", None)
            connects["n"] += 1
            if ctx_now is not None:
                ctx_now.metrics.mqtt_connected.set(1)
                if connects["n"] > 1:
                    ctx_now.metrics.mqtt_reconnects.inc()

        def mqtt_down() -> None:
            ctx_now: AppContext | None = getattr(app.state, "ctx", None)
            if ctx_now is not None:
                ctx_now.metrics.mqtt_connected.set(0)

        if chosen_bus is None and settings.mqtt_enabled:
            chosen_bus = make_mqtt_bus(settings, mqtt_up, mqtt_down)
        ctx = await asyncio.to_thread(build_context, settings, chosen_bus, registry)
        app.state.ctx = ctx
        if chosen_bus is not None:
            chosen_bus.start()
        task = asyncio.create_task(_ticker(ctx)) if run_background else None
        logger.info("control plane started")
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if chosen_bus is not None:
                chosen_bus.stop()
            ctx.db_engine.dispose()
            background.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="FleetOps control plane", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    def count(metric: str, label: str) -> None:
        ctx: AppContext | None = getattr(app.state, "ctx", None)
        if ctx is not None:
            getattr(ctx.metrics, metric).labels(label).inc()

    # Order matters: the last middleware added is outermost, so rate limiting runs before body reading.
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
        read_timeout=settings.request_body_timeout_seconds,
        on_reject=lambda reason: count("requests_rejected", reason),
    )
    app.add_middleware(
        RateLimitMiddleware,
        get_limiter=lambda scope: getattr(getattr(scope["app"].state, "ctx", None), "limiter", None),
        on_limited=lambda scope_name: count("rate_limited", scope_name),
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        ctx: AppContext | None = getattr(request.app.state, "ctx", None)
        if ctx is not None and request.url.path.startswith("/api/"):
            ctx.metrics.http_requests.labels(request.method, str(response.status_code)).inc()
        return response

    @app.exception_handler(NotFoundError)
    async def _nf(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, "not_found", str(exc))

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return _error(409, "conflict", str(exc))

    @app.exception_handler(InvalidRequestError)
    async def _invalid(_: Request, exc: InvalidRequestError) -> JSONResponse:
        return _error(422, "invalid_request", str(exc))

    @app.exception_handler(ForbiddenError)
    async def _forbidden(_: Request, exc: ForbiddenError) -> JSONResponse:
        return _error(403, "forbidden", str(exc))

    @app.exception_handler(QualityGateError)
    async def _quality(_: Request, exc: QualityGateError) -> JSONResponse:
        # 422: the request is well-formed but the candidate model does not meet the requested policy.
        return _error(422, "quality_gate_failed", str(exc), exc.report.model_dump(mode="json"))

    @app.exception_handler(EngineError)
    async def _engine(_: Request, exc: EngineError) -> JSONResponse:
        return _error(400, "engine_error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", "request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(
            exc.status_code,
            {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(exc.status_code, "http_error"),
            str(exc.detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return _error(500, "internal_error", "internal server error")

    return app


def _error(
    status: int, code: str, message: str, details: object = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "details": details}}, status_code=status, headers=headers
    )


__all__ = ["HTTPException", "create_app"]
