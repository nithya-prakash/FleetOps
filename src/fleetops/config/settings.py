"""Validated environment-based settings for the control plane and the edge agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_KEY_LENGTH_PROD = 24
MIN_KEY_LENGTH_DEV = 12
DEFAULT_SECRETS_DIR = Path("/run/secrets")


def secrets_dir() -> Path | None:
    """Docker/Kubernetes secret files (one file per setting, named like the env var in lower case).

    Preferred over environment variables for credentials: values do not appear in ``docker inspect``,
    process environments of other tools, or compose output. ``FLEETOPS_SECRETS_DIR`` overrides the location.
    """
    path = Path(os.environ.get("FLEETOPS_SECRETS_DIR", str(DEFAULT_SECRETS_DIR)))
    return path if path.is_dir() else None


class MqttSettings(BaseSettings):
    """Connection parameters shared by both sides; each side sets its own env prefix."""

    mqtt_host: str = "localhost"
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    mqtt_username: str | None = None
    mqtt_password: SecretStr | None = None
    mqtt_tls_ca: Path | None = None  # set to enable TLS; the broker certificate is verified (chain AND hostname)
    mqtt_tls_cert: Path | None = None  # optional client certificate (mutual TLS)
    mqtt_tls_key: Path | None = None
    mqtt_keepalive_seconds: int = Field(default=15, ge=1, le=600)
    # Reconnect backoff after a lost broker connection: exponential from min to max seconds.
    mqtt_reconnect_min_seconds: int = Field(default=1, ge=1, le=60)
    mqtt_reconnect_max_seconds: int = Field(default=30, ge=1, le=600)


class ControlPlaneSettings(MqttSettings):
    model_config = SettingsConfigDict(env_prefix="FLEETOPS_", env_file=".env", extra="ignore", case_sensitive=False)

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"  # noqa: S104 - container bind; exposure is controlled by compose
    tls_certfile: Path | None = None  # serve the API over HTTPS
    tls_keyfile: Path | None = None
    port: int = Field(default=8000, ge=1, le=65535)

    database_url: str = "sqlite:///data/fleetops.db"
    artifact_root: Path = Path("data/artifacts")
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment: str = "fleetops-sensor-anomaly"
    mlflow_artifact_location: str | None = None  # None: let the tracking server decide
    mlflow_ca_cert: Path | None = None  # trust anchor when MLflow is served over HTTPS
    mlflow_username: str | None = None  # MLflow server uses basic-auth in the Compose/K8s profiles
    mlflow_password: SecretStr | None = None

    admin_api_key: SecretStr
    operator_api_key: SecretStr
    viewer_api_key: SecretStr
    device_key_master: SecretStr

    simulation_enabled: bool = False
    mqtt_enabled: bool = True
    # When true, POST /deployments without a quality_gate is rejected (policy enforcement).
    require_quality_gate: bool = False
    clock_skew_warn_seconds: float = Field(default=30.0, gt=0)

    # API protection (per client address; behind a proxy configure the proxy's limits as well)
    rate_limit_per_second: float = Field(default=50.0, gt=0)
    rate_limit_burst: int = Field(default=100, ge=1)
    auth_failure_limit_per_minute: int = Field(default=10, ge=1)  # 401s per client before it is answered with 429
    request_body_timeout_seconds: float = Field(default=10.0, gt=0)
    max_concurrent_requests: int = Field(default=200, ge=1)  # uvicorn answers 503 beyond this
    keep_alive_timeout_seconds: int = Field(default=5, ge=1)
    # Scrape credential that can read /metrics and nothing else (least privilege for Prometheus).
    metrics_token: SecretStr | None = None

    offline_after_seconds: float = Field(default=20.0, gt=0)
    tick_interval_seconds: float = Field(default=1.0, gt=0)
    command_ttl_seconds: float = Field(default=300.0, gt=0)
    retry_backoff_base_seconds: float = Field(default=5.0, gt=0)
    retry_backoff_max_seconds: float = Field(default=60.0, gt=0)
    max_request_bytes: int = Field(default=1_000_000, ge=1024)

    # drift monitoring
    drift_interval_seconds: float = Field(default=30.0, gt=0)
    drift_window_samples: int = Field(default=500, ge=10)
    drift_min_samples: int = Field(default=100, ge=10)
    drift_psi_warn: float = Field(default=0.10, gt=0)
    drift_psi_alert: float = Field(default=0.25, gt=0)
    drift_p_value: float = Field(default=0.01, gt=0, lt=1)
    sample_retention: int = Field(default=20_000, ge=100)

    @model_validator(mode="after")
    def _validate_secrets(self) -> Self:
        keys = {
            "admin_api_key": self.admin_api_key,
            "operator_api_key": self.operator_api_key,
            "viewer_api_key": self.viewer_api_key,
            "device_key_master": self.device_key_master,
        }
        minimum = MIN_KEY_LENGTH_PROD if self.env == "prod" else MIN_KEY_LENGTH_DEV
        for name, secret in keys.items():
            value = secret.get_secret_value()
            if len(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum} characters in env={self.env}")
            if self.env == "prod" and "change-me" in value:
                raise ValueError(f"{name} still contains a development placeholder")
        api = [self.admin_api_key, self.operator_api_key, self.viewer_api_key]
        if len({k.get_secret_value() for k in api}) != len(api):
            raise ValueError("admin, operator and viewer API keys must all differ")
        if self.env == "prod" and self.simulation_enabled:
            raise ValueError("simulation must be disabled when env=prod")
        if self.drift_psi_alert <= self.drift_psi_warn:
            raise ValueError("drift_psi_alert must exceed drift_psi_warn")
        if self.env == "prod":
            if self.mlflow_tracking_uri.startswith("http") and not (self.mlflow_username and self.mlflow_password):
                raise ValueError("MLflow credentials are required when env=prod and MLflow is remote")
            if self.mqtt_enabled and self.mqtt_password is None:
                raise ValueError("mqtt_password is required when env=prod")
            if self.mqtt_enabled and self.mqtt_tls_ca is None:
                raise ValueError("mqtt_tls_ca is required when env=prod (MQTT must use TLS)")
            if self.tls_certfile is None or self.tls_keyfile is None:
                raise ValueError("tls_certfile/tls_keyfile are required when env=prod (the API must use HTTPS)")
            if self.mlflow_tracking_uri.startswith("http://"):
                raise ValueError("mlflow_tracking_uri must be https:// when env=prod")
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ValueError("tls_certfile and tls_keyfile must be set together")
        if (self.mqtt_tls_cert is None) != (self.mqtt_tls_key is None):
            raise ValueError("mqtt_tls_cert and mqtt_tls_key must be set together")
        return self

    @classmethod
    def load(cls, **overrides: Any) -> Self:
        """Environment + ``.env`` + secret files (``/run/secrets``)."""
        return cls(_secrets_dir=secrets_dir(), **overrides)


class EdgeSettings(MqttSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLEETOPS_EDGE_", env_file=".env", extra="ignore", case_sensitive=False
    )

    device_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{1,62}$")
    device_secret: SecretStr
    data_dir: Path = Path("data/edge")
    control_plane_url: str = "http://localhost:8000"
    control_plane_ca: Path | None = None  # trust anchor for https://; the certificate chain AND hostname are verified
    require_tls: bool = False  # refuse http:// control plane and plaintext MQTT (set in production profiles)
    log_level: str = "INFO"

    admin_host: str = "0.0.0.0"  # noqa: S104 - container bind
    admin_port: int = Field(default=9100, ge=1, le=65535)

    heartbeat_interval_seconds: float = Field(default=5.0, gt=0)
    telemetry_interval_seconds: float = Field(default=10.0, gt=0)
    prediction_flush_seconds: float = Field(default=5.0, gt=0)

    # Simulated sensor stream (stand-in for a PLC / OPC-UA feed).
    sensor_hz: float = Field(default=5.0, ge=0, le=1000)
    sensor_scenario: Literal["normal", "drift"] = "normal"
    sensor_seed: int = 0
    label_fraction: float = Field(default=0.0, ge=0, le=1)

    command_max_age_seconds: float = Field(default=600.0, gt=0)
    http_timeout_seconds: float = Field(default=10.0, gt=0)
    # Artifact download retry budget: bounded exponential backoff with jitter. A budget that is too small
    # turns a short control-plane outage into a failed (rolled-back) rollout; too large delays the failure.
    download_max_attempts: int = Field(default=4, ge=1, le=20)
    download_backoff_base_seconds: float = Field(default=0.5, gt=0)
    download_backoff_max_seconds: float = Field(default=10.0, gt=0)
    download_jitter: float = Field(default=0.2, ge=0, le=1)  # +/- fraction of each delay
    max_clock_skew_seconds: float = Field(default=30.0, gt=0)  # tolerated future-dating of commands
    max_artifact_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    # A download only starts if free space >= artifact size * headroom + reserve (the previous model stays on disk).
    min_free_disk_mb: int = Field(default=128, ge=0)
    storage_headroom_factor: float = Field(default=2.0, ge=1.0)
    allow_faults: bool = False  # honour SET_FAULTS commands / env faults (simulation only)

    # Agent API protection. Tokens are optional so a bare development agent still starts, but every shipped
    # deployment (Compose, Kubernetes) sets them and the agent logs a warning when they are missing.
    api_token: SecretStr | None = None  # required by /predict and /healthz when set (PLC / gateway clients)
    metrics_token: SecretStr | None = None  # read-only /metrics scrape credential
    max_request_bytes: int = Field(default=64 * 1024, ge=1024)
    predict_rate_per_second: float = Field(default=500.0, gt=0)
    predict_burst: int = Field(default=1000, ge=1)
    tls_certfile: Path | None = None  # serve the agent API over HTTPS
    tls_keyfile: Path | None = None
    initial_faults: str = ""  # JSON FaultConfig for deterministic scripted scenarios

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.control_plane_url.startswith(("http://", "https://")):
            raise ValueError("control_plane_url must be http(s)")
        if self.download_backoff_max_seconds < self.download_backoff_base_seconds:
            raise ValueError("download_backoff_max_seconds must be >= download_backoff_base_seconds")
        if self.require_tls and (self.control_plane_url.startswith("http://") or self.mqtt_tls_ca is None):
            raise ValueError("require_tls: control_plane_url must be https:// and mqtt_tls_ca must be set")
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ValueError("tls_certfile and tls_keyfile must be set together")
        if (self.mqtt_tls_cert is None) != (self.mqtt_tls_key is None):
            raise ValueError("mqtt_tls_cert and mqtt_tls_key must be set together")
        return self

    @classmethod
    def load(cls, **overrides: Any) -> Self:
        return cls(_secrets_dir=secrets_dir(), **overrides)
