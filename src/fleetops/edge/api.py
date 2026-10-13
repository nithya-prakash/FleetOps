"""Local HTTP surface of the edge agent: inference, health, metrics, simulation controls.

Authentication (bearer tokens, constant-time comparison):
* ``/predict`` and ``/healthz``: ``api_token`` (when configured);
* ``/metrics``: ``metrics_token`` (read-only scrape credential) or the ``api_token``;
* ``/admin/*``: the device credential, and only when simulation is allowed;
* ``/livez``: always open (container / orchestrator liveness; reveals nothing).
Inference never depends on any of this reaching the cloud.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from fleetops.edge.agent import EdgeAgent
from fleetops.edge.inference import NoModelError
from fleetops.registry.modelformat import InputRejectedError
from fleetops.security.asgi import BodyLimitMiddleware
from fleetops.security.ratelimit import TokenBucketLimiter


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=256)


class NetworkRequest(BaseModel):
    connected: bool


def _matches(presented: str | None, *expected: str | None) -> bool:
    if not presented or not presented.lower().startswith("bearer "):
        return False  # the Bearer scheme is mandatory
    token = presented[7:].strip()
    ok = False
    for candidate in expected:
        if candidate:
            ok |= hmac.compare_digest(token.encode(), candidate.encode())  # no short-circuit between candidates
    return ok and bool(token)


def create_edge_app(agent: EdgeAgent) -> FastAPI:
    app = FastAPI(title=f"FleetOps edge agent {agent.device_id}", docs_url=None, redoc_url=None, openapi_url=None)
    s = agent.s
    secret = s.device_secret.get_secret_value()
    api_token = s.api_token.get_secret_value() if s.api_token else None
    metrics_token = s.metrics_token.get_secret_value() if s.metrics_token else None
    limiter = TokenBucketLimiter(s.predict_rate_per_second, s.predict_burst)
    app.add_middleware(BodyLimitMiddleware, max_bytes=s.max_request_bytes, read_timeout=5.0)

    def require_api(authorization: Annotated[str | None, Header()] = None) -> None:
        if api_token is not None and not _matches(authorization, api_token):
            raise HTTPException(status_code=401, detail="api token required", headers={"WWW-Authenticate": "Bearer"})

    def require_metrics(authorization: Annotated[str | None, Header()] = None) -> None:
        if (api_token or metrics_token) and not _matches(authorization, metrics_token, api_token):
            raise HTTPException(
                status_code=401, detail="metrics token required", headers={"WWW-Authenticate": "Bearer"}
            )

    def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        if not _matches(authorization, secret):
            raise HTTPException(status_code=401, detail="invalid device credential")

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/healthz", dependencies=[Depends(require_api)])
    def healthz() -> dict[str, Any]:
        state = agent.store.snapshot()
        stats = agent.engine.stats()
        return {
            "device_id": agent.device_id,
            "status": "ok" if agent.engine.active else "no_model",
            "mqtt_connected": agent.bus.connected,
            "network_enabled": agent.network_enabled,
            "active": state.active.model_dump() if state.active else None,
            "previous": state.previous.model_dump() if state.previous else None,
            "staged": state.staged.model_dump() if state.staged else None,
            "last_generation": state.last_generation,
            "stats": stats.model_dump() if stats else None,
        }

    @app.post("/predict", dependencies=[Depends(require_api)])
    def predict(req: PredictRequest) -> dict[str, Any]:
        if not limiter.allow("predict"):
            raise HTTPException(status_code=429, detail="too many requests", headers={"Retry-After": "1"})
        try:
            r = agent.infer(req.features)
        except NoModelError as exc:
            raise HTTPException(status_code=503, detail="no active model") from exc
        except InputRejectedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="inference failed") from exc
        return {
            "score": r.prediction.score,
            "anomaly": r.prediction.anomaly,
            "latency_ms": r.latency_ms,
            "model": {"name": r.model.name, "version": r.model.version},
        }

    @app.get("/metrics", dependencies=[Depends(require_metrics)])
    def metrics() -> Response:
        return Response(generate_latest(agent.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/network", dependencies=[Depends(require_admin)])
    def network(req: NetworkRequest) -> dict[str, bool]:
        if not agent.s.allow_faults:
            raise HTTPException(status_code=403, detail="simulation controls disabled")
        agent.set_network(req.connected)
        return {"network_enabled": agent.network_enabled}

    return app
