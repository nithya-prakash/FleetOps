# syntax=docker/dockerfile:1
# Targets: edge | control-plane | mlflow-server | mosquitto
#
# Reproducibility: Python dependencies are installed from uv.lock (hash-verified, --require-hashes) and base
# images are pinned by digest. See docs/supply-chain.md for how digests and the lock are updated.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1
ARG MOSQUITTO_IMAGE=eclipse-mosquitto:2.0.22@sha256:199ea8ef2e35ec2b1b37e59cfd1dbae538ed4dfa4a2251a121a52215a6248a21

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 HOME=/tmp
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin fleetops \
 && mkdir -p /data && chown 10001:10001 /data
WORKDIR /app

# ---------- locked dependency export (shared by the python images) ----------
FROM base AS lock
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-header --emit-index-url -o /tmp/edge-requirements.txt \
 && uv export --frozen --no-dev --no-emit-project --no-header --emit-index-url --extra control --extra train -o /tmp/control-requirements.txt

# ---------- edge: no torch, no sqlalchemy/mlflow ----------
FROM base AS edge-build
COPY --from=lock /tmp/edge-requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes --no-deps --prefix=/install -r /tmp/requirements.txt

FROM base AS edge
COPY --from=edge-build /install /usr/local
COPY src/fleetops /usr/local/lib/python3.12/site-packages/fleetops
USER 10001
VOLUME ["/data"]
ENV FLEETOPS_EDGE_DATA_DIR=/data FLEETOPS_EDGE_ADMIN_PORT=9100
EXPOSE 9100
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:9100/livez', timeout=2)"]
CMD ["python", "-m", "fleetops.edge"]

# ---------- control plane: + torch (CPU) for the training endpoint ----------
FROM base AS control-build
COPY --from=lock /tmp/control-requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes --no-deps --prefix=/install -r /tmp/requirements.txt

FROM base AS control-plane
COPY --from=control-build /install /usr/local
COPY src/fleetops /usr/local/lib/python3.12/site-packages/fleetops
USER 10001
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["python", "-m", "fleetops.controlplane"]

# ---------- MLflow tracking server + registry (basic-auth enabled) ----------
FROM base AS mlflow-build
COPY deploy/docker/mlflow-requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes --no-deps --prefix=/install -r /tmp/requirements.txt

FROM base AS mlflow-server
COPY --from=mlflow-build /install /usr/local
COPY deploy/docker/mlflow_basic_auth.ini /app/basic_auth.ini
COPY --chmod=755 deploy/docker/mlflow-entrypoint.sh /usr/local/bin/mlflow-entrypoint.sh
USER 10001
VOLUME ["/data"]
EXPOSE 5000
# /health is exempt from authentication in MLflow's auth app.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health', timeout=2)"]
ENTRYPOINT ["/usr/local/bin/mlflow-entrypoint.sh"]

# ---------- Mosquitto with baked (non-secret) config; password file arrives as a secret ----------
FROM ${MOSQUITTO_IMAGE} AS mosquitto
COPY --chown=mosquitto:mosquitto --chmod=600 mqtt/mosquitto.conf /mosquitto/config/mosquitto.conf
COPY --chown=mosquitto:mosquitto --chmod=600 mqtt/mosquitto-tls.conf /mosquitto/config/mosquitto-tls.conf
COPY --chown=mosquitto:mosquitto --chmod=600 mqtt/mosquitto-mtls.conf /mosquitto/config/mosquitto-mtls.conf
COPY --chown=mosquitto:mosquitto --chmod=600 mqtt/acl.conf /mosquitto/config/acl
COPY --chmod=755 deploy/docker/mosquitto-entrypoint.sh /usr/local/bin/mosquitto-entrypoint.sh
USER mosquitto
ENTRYPOINT ["/usr/local/bin/mosquitto-entrypoint.sh"]
