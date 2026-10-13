"""Bootstrap the MLflow service account used by the control plane (least privilege).

MLflow runs with basic-auth. Humans use the ``admin`` account; the control plane authenticates as a separate
non-admin user, so a leaked control-plane credential cannot manage other users or permissions.

CLI (used by the ``mlflow-init`` Compose service): ``python -m fleetops.registry.mlflow_admin``.
"""

from __future__ import annotations

import logging
import os
import ssl
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SECRETS = Path(os.environ.get("FLEETOPS_SECRETS_DIR", "/run/secrets"))


class MlflowAdminError(Exception):
    pass


def ensure_service_user(
    base_url: str,
    admin_user: str,
    admin_password: str,
    username: str,
    password: str,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Create ``username`` (or reset its password so rotations take effect). Returns ``created``/``updated``."""
    ca = os.environ.get(
        "MLFLOW_TRACKING_SERVER_CERT_PATH"
    )  # trust anchor for https:// MLflow (verified, never disabled)
    http = client or httpx.Client(
        base_url=base_url.rstrip("/"), timeout=10, verify=ssl.create_default_context(cafile=ca) if ca else True
    )
    auth = (admin_user, admin_password)
    resp = http.post("/api/2.0/mlflow/users/create", json={"username": username, "password": password}, auth=auth)
    if resp.status_code == 200:
        return "created"
    if resp.status_code in (400, 409) and "RESOURCE_ALREADY_EXISTS" in resp.text:
        upd = http.patch(
            "/api/2.0/mlflow/users/update-password", json={"username": username, "password": password}, auth=auth
        )
        if upd.status_code == 200:
            return "updated"
        raise MlflowAdminError(f"password update failed: HTTP {upd.status_code}")
    if resp.status_code in (401, 403):
        raise MlflowAdminError("MLflow rejected the admin credentials")
    raise MlflowAdminError(f"unexpected response creating user: HTTP {resp.status_code}")


def main() -> int:
    logging.basicConfig(level="INFO", format="%(message)s")
    base = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
    admin_pw = (SECRETS / "mlflow_admin_password").read_text().strip()
    svc_pw = (SECRETS / "fleetops_mlflow_password").read_text().strip()
    svc_user = os.environ.get("FLEETOPS_MLFLOW_USERNAME", "fleetops-cp")
    for attempt in range(1, 31):
        try:
            outcome = ensure_service_user(base, "admin", admin_pw, svc_user, svc_pw)
        except (httpx.TransportError, MlflowAdminError) as exc:
            logger.warning("attempt %d: %s", attempt, exc)
            if isinstance(exc, MlflowAdminError) and "rejected" in str(exc):
                return 1
            time.sleep(2)
            continue
        logger.info("mlflow service user %r %s", svc_user, outcome)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
