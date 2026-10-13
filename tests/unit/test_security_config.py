from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from fleetops.config.settings import ControlPlaneSettings
from fleetops.domain.messages import Command, CommandType
from fleetops.domain.models import utcnow
from fleetops.messaging.topics import Channel, parse_topic, topic, topic_matches
from fleetops.security.artifact import ChecksumMismatchError, sha256_bytes, verify_sha256
from fleetops.security.auth import ApiKeyAuthenticator, Role
from fleetops.security.device_identity import (
    derive_device_secret,
    sign_command,
    verify_command,
    verify_device_secret,
)


def test_api_key_roles() -> None:
    auth = ApiKeyAuthenticator({Role.ADMIN: "adm", Role.VIEWER: "vwr"})
    p = auth.authenticate("adm")
    assert p is not None and p.role is Role.ADMIN and p.allows(Role.OPERATOR)
    v = auth.authenticate("vwr")
    assert v is not None and not v.allows(Role.OPERATOR)
    assert auth.authenticate("nope") is None
    assert auth.authenticate(None) is None
    assert auth.authenticate("") is None


def test_device_secret_is_per_device_and_epoch() -> None:
    s1 = derive_device_secret("master", "edge-01", 1)
    assert verify_device_secret("master", "edge-01", 1, s1)
    assert not verify_device_secret("master", "edge-02", 1, s1)  # other device
    assert not verify_device_secret("master", "edge-01", 2, s1)  # rotated
    assert not verify_device_secret("other", "edge-01", 1, s1)
    assert not verify_device_secret("master", "edge-01", 1, "")


def _cmd() -> Command:
    now = utcnow()
    return Command(
        command_id="cmd-12345678",
        type=CommandType.ACTIVATE,
        device_id="edge-01",
        generation=3,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        correlation_id="c1",
    )


def test_command_signature_roundtrip_and_tamper() -> None:
    signed = sign_command("secret", _cmd())
    assert verify_command("secret", signed)
    wire = Command.model_validate_json(signed.model_dump_json())  # survives serialisation
    assert verify_command("secret", wire)
    assert not verify_command("wrong", wire)
    assert not verify_command("secret", wire.model_copy(update={"generation": 99}))
    assert not verify_command("secret", _cmd())  # unsigned


def test_topics() -> None:
    t = topic("edge-01", Channel.COMMANDS)
    assert t == "fleetops/devices/edge-01/commands"
    assert parse_topic(t) == ("edge-01", Channel.COMMANDS)
    assert parse_topic("other/devices/edge-01/commands") is None
    assert parse_topic("fleetops/devices/edge-01/unknown") is None
    assert parse_topic("fleetops/devices/../commands") is None
    with pytest.raises(ValueError, match="invalid device id"):
        topic("a/b", Channel.STATUS)
    assert topic_matches("fleetops/devices/+/heartbeat", "fleetops/devices/e1/heartbeat")
    assert not topic_matches("fleetops/devices/+/heartbeat", "fleetops/devices/e1/status")
    assert topic_matches("fleetops/#", "fleetops/devices/e1/status")


def test_checksum_verification(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "m.bin"
    f.write_bytes(b"hello")
    verify_sha256(f, sha256_bytes(b"hello"))
    with pytest.raises(ChecksumMismatchError):
        verify_sha256(f, sha256_bytes(b"other"))


def _settings(**kw: object) -> ControlPlaneSettings:
    base: dict[str, object] = {
        "admin_api_key": "a" * 30,
        "operator_api_key": "b" * 30,
        "viewer_api_key": "c" * 30,
        "device_key_master": "d" * 30,
    }
    base.update(kw)
    return ControlPlaneSettings(_env_file=None, **base)  # type: ignore[arg-type]


def test_settings_validation() -> None:
    assert _settings().env == "dev"
    with pytest.raises(ValidationError, match="differ"):
        _settings(operator_api_key="a" * 30)
    with pytest.raises(ValidationError, match="at least"):
        _settings(admin_api_key="short")
    with pytest.raises(ValidationError, match="placeholder"):
        _settings(env="prod", admin_api_key="change-me" + "x" * 20)
    with pytest.raises(ValidationError, match="simulation"):
        _settings(env="prod", simulation_enabled=True)
    with pytest.raises(ValidationError, match="drift_psi_alert"):
        _settings(drift_psi_alert=0.05)


def test_settings_require_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("ADMIN_API_KEY", "OPERATOR_API_KEY", "VIEWER_API_KEY", "DEVICE_KEY_MASTER"):
        monkeypatch.delenv(f"FLEETOPS_{k}", raising=False)
    with pytest.raises(ValidationError):
        ControlPlaneSettings(_env_file=None)  # type: ignore[call-arg]


# ---- TLS configuration rules ------------------------------------------------------------------------------------


def _prod(**kw: object) -> ControlPlaneSettings:
    base: dict[str, object] = {
        "env": "prod",
        "admin_api_key": "a" * 30,
        "operator_api_key": "b" * 30,
        "viewer_api_key": "c" * 30,
        "device_key_master": "d" * 30,
        "mqtt_password": "m" * 30,
        "mlflow_username": "svc",
        "mlflow_password": "p" * 30,
        "mlflow_tracking_uri": "https://mlflow.example",
        "mqtt_tls_ca": "/tls/ca.crt",
        "tls_certfile": "/tls/c.crt",
        "tls_keyfile": "/tls/c.key",
    }
    base.update(kw)
    return ControlPlaneSettings(_env_file=None, **base)  # type: ignore[arg-type]


def test_production_requires_tls_everywhere() -> None:
    assert _prod().env == "prod"
    with pytest.raises(ValidationError, match="mqtt_tls_ca"):
        _prod(mqtt_tls_ca=None)
    with pytest.raises(ValidationError, match="HTTPS"):
        _prod(tls_certfile=None, tls_keyfile=None)
    with pytest.raises(ValidationError, match="https"):
        _prod(mlflow_tracking_uri="http://mlflow:5000")


def test_certificate_and_key_must_come_in_pairs() -> None:
    dev = {
        "admin_api_key": "a" * 30,
        "operator_api_key": "b" * 30,
        "viewer_api_key": "c" * 30,
        "device_key_master": "d" * 30,
    }
    with pytest.raises(ValidationError, match="together"):
        ControlPlaneSettings(_env_file=None, tls_keyfile="/tls/c.key", **dev)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="together"):
        ControlPlaneSettings(_env_file=None, mqtt_tls_cert="/tls/client.crt", **dev)  # type: ignore[arg-type]


def test_edge_require_tls_refuses_plaintext_configuration() -> None:
    from fleetops.config.settings import EdgeSettings  # noqa: PLC0415

    base = {"device_id": "edge-01", "device_secret": "x" * 20, "_env_file": None, "require_tls": True}
    with pytest.raises(ValidationError, match="require_tls"):
        EdgeSettings(**base, control_plane_url="http://cp:8000", mqtt_tls_ca="/tls/ca.crt")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="require_tls"):
        EdgeSettings(**base, control_plane_url="https://cp:8000")  # type: ignore[arg-type]  # MQTT would be plaintext
    ok = EdgeSettings(**base, control_plane_url="https://cp:8000", mqtt_tls_ca="/tls/ca.crt")  # type: ignore[arg-type]
    assert ok.require_tls


def test_tls_verification_cannot_be_switched_off_in_code() -> None:
    """Guard against 'temporary' workarounds: no insecure-TLS switches may exist in shipped code."""
    from pathlib import Path  # noqa: PLC0415

    forbidden = (
        "tls_insecure_set(True",
        "CERT_NONE",
        "check_hostname = False",
        "verify=False",
        "_create_unverified_context",
    )
    for path in (Path(__file__).resolve().parents[2] / "src").rglob("*.py"):
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path.name} contains {needle!r}"
