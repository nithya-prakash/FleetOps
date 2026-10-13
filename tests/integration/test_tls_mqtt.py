"""MQTT over TLS against a real Mosquitto container: certificate chain, hostname, expiry, mutual TLS, no plaintext."""

from __future__ import annotations

import socket
import ssl
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts.devcerts import KeyPair, issue, make_ca
from scripts.provision import mosquitto_hash

from fleetops.messaging.bus import MqttBus, MqttConfig
from tests.harness import free_port

CP_PASSWORD = "tls-test-password-0123456789"
IMAGE = "fleetops-mosquitto:dev"


def docker_ready() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not docker_ready(), reason="docker is not available")


@pytest.fixture(scope="module", autouse=True)
def image() -> None:
    subprocess.run(
        ["docker", "build", "-q", "--target", "mosquitto", "-t", IMAGE, "."], check=True, capture_output=True
    )


@contextmanager
def broker(tmp: Path, ca: KeyPair, server: KeyPair, *, mtls: bool = False) -> Iterator[int]:
    """A TLS-only broker (no plaintext listener) built from the project image."""
    d = tmp / uuid.uuid4().hex[:8]
    d.mkdir()
    files = {"ca.crt": ca.cert_pem, "server.crt": server.cert_pem, "server.key": server.key_pem}
    mounts: list[str] = []
    for name, data in files.items():
        (d / name).write_bytes(data)
        (d / name).chmod(0o644)
        mounts += ["-v", f"{d / name}:/run/secrets/tls/{name}:ro"]
    (d / "passwd").write_text(f"control-plane:{mosquitto_hash(CP_PASSWORD)}\n")
    (d / "passwd").chmod(0o644)
    port, name = free_port(), f"fleetops-tls-{uuid.uuid4().hex[:8]}"
    conf = "mosquitto-mtls.conf" if mtls else "mosquitto-tls.conf"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:8883",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/mosquitto/run:uid=1883,gid=1883,mode=0700",
            "--tmpfs",
            "/tmp",
            "-v",
            f"{d / 'passwd'}:/run/secrets/mosquitto_passwd:ro",
            *mounts,
            "-e",
            f"MOSQUITTO_CONF=/mosquitto/config/{conf}",
            IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError(
                "broker did not start: "
                + subprocess.run(["docker", "logs", name], capture_output=True, text=True, check=False).stdout
            )
        time.sleep(0.5)
        yield port
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def pem(tmp: Path, name: str, data: bytes) -> Path:
    p = tmp / f"{uuid.uuid4().hex[:6]}-{name}"
    p.write_bytes(data)
    return p


def handshake(port: int, cafile: Path, hostname: str) -> str:
    """Raw TLS handshake with full verification; returns the negotiated version or raises SSLError."""
    ctx = ssl.create_default_context(cafile=str(cafile))
    with (
        socket.create_connection(("127.0.0.1", port), timeout=5) as raw,
        ctx.wrap_socket(raw, server_hostname=hostname) as tls,
    ):
        return str(tls.version())


def connects(
    port: int,
    ca: Path,
    host: str = "localhost",
    cert: Path | None = None,
    key: Path | None = None,
    username: str = "control-plane",
    password: str = CP_PASSWORD,
    timeout: float = 4.0,
) -> bool:
    bus = MqttBus(
        MqttConfig(
            host=host,
            port=port,
            client_id=f"t-{uuid.uuid4().hex[:6]}",
            username=username,
            password=password,
            tls_ca=ca,
            tls_cert=cert,
            tls_key=key,
            clean_session=True,
        )
    )
    bus.start()
    try:
        return bus.wait_connected(timeout)
    finally:
        bus.stop()


def test_valid_certificate_chain_and_hostname_connect_and_exchange_messages(tmp_path: Path) -> None:
    ca = make_ca()
    server = issue(ca, "mosquitto", ["mosquitto", "localhost", "127.0.0.1"])
    cafile = pem(tmp_path, "ca.crt", ca.cert_pem)
    with broker(tmp_path, ca, server) as port:
        assert handshake(port, cafile, "localhost") in ("TLSv1.2", "TLSv1.3")  # never an older protocol
        got: list[bytes] = []
        sub = MqttBus(
            MqttConfig(
                host="localhost",
                port=port,
                client_id="sub",
                username="control-plane",
                password=CP_PASSWORD,
                tls_ca=cafile,
                clean_session=True,
            )
        )
        sub.subscribe("fleetops/devices/+/heartbeat", lambda _t, p: got.append(p), qos=1)
        sub.start()
        assert sub.wait_connected(5)
        time.sleep(0.5)
        pub = MqttBus(
            MqttConfig(
                host="127.0.0.1",
                port=port,
                client_id="pub",
                username="control-plane",
                password=CP_PASSWORD,
                tls_ca=cafile,
                clean_session=True,
            )
        )
        pub.start()
        assert pub.wait_connected(5)  # IP address in the SAN list also verifies
        pub.publish("fleetops/devices/edge-01/commands", b"x", qos=1)
        time.sleep(0.3)
        sub.stop()
        pub.stop()
        assert connects(port, cafile)
        assert not connects(port, cafile, password="wrong-password")  # TLS does not replace authentication


def test_hostname_mismatch_is_rejected(tmp_path: Path) -> None:
    ca = make_ca()
    server = issue(ca, "mosquitto", ["mosquitto"])  # no localhost / 127.0.0.1 in the certificate
    cafile = pem(tmp_path, "ca.crt", ca.cert_pem)
    with broker(tmp_path, ca, server) as port:
        with pytest.raises(ssl.SSLCertVerificationError, match=r"Hostname mismatch|IP address mismatch"):
            handshake(port, cafile, "localhost")
        assert not connects(port, cafile, host="localhost")
        assert not connects(port, cafile, host="127.0.0.1")
        assert handshake(
            port, cafile, "mosquitto"
        )  # control: the very same broker accepts the name it is certified for


def test_untrusted_certificate_authority_is_rejected(tmp_path: Path) -> None:
    ca, other_ca = make_ca(), make_ca("Some Other CA")
    server = issue(ca, "mosquitto", ["localhost", "127.0.0.1"])
    trusted, untrusted = pem(tmp_path, "ca.crt", ca.cert_pem), pem(tmp_path, "other.crt", other_ca.cert_pem)
    with broker(tmp_path, ca, server) as port:
        with pytest.raises(
            ssl.SSLCertVerificationError, match=r"unable to get local issuer|self-signed|certificate verify failed"
        ):
            handshake(port, untrusted, "localhost")
        assert not connects(port, untrusted)
        assert connects(port, trusted)  # control


def test_expired_certificate_is_rejected(tmp_path: Path) -> None:
    ca = make_ca()
    past = datetime.now(UTC) - timedelta(days=2)
    expired = issue(ca, "mosquitto", ["localhost", "127.0.0.1"], not_before=past - timedelta(days=30), not_after=past)
    cafile = pem(tmp_path, "ca.crt", ca.cert_pem)
    with broker(tmp_path, ca, expired) as port:
        with pytest.raises(ssl.SSLCertVerificationError, match="expired"):
            handshake(port, cafile, "localhost")
        assert not connects(port, cafile)


def test_not_yet_valid_certificate_is_rejected(tmp_path: Path) -> None:
    ca = make_ca()
    future = datetime.now(UTC) + timedelta(days=2)
    pending = issue(ca, "mosquitto", ["localhost"], not_before=future, not_after=future + timedelta(days=30))
    with broker(tmp_path, ca, pending) as port, pytest.raises(ssl.SSLCertVerificationError, match="not yet valid"):
        handshake(port, pem(tmp_path, "ca.crt", ca.cert_pem), "localhost")


def test_plaintext_clients_cannot_use_the_tls_listener(tmp_path: Path) -> None:
    ca = make_ca()
    server = issue(ca, "mosquitto", ["localhost", "127.0.0.1"])
    with broker(tmp_path, ca, server) as port:
        plain = MqttBus(
            MqttConfig(
                host="127.0.0.1",
                port=port,
                client_id="plain",
                username="control-plane",
                password=CP_PASSWORD,
                clean_session=True,
            )
        )
        plain.start()
        try:
            assert not plain.wait_connected(3)  # a plaintext MQTT CONNECT is not accepted on the TLS port
        finally:
            plain.stop()


def test_mutual_tls_requires_a_client_certificate_from_the_trusted_ca(tmp_path: Path) -> None:
    ca, other_ca = make_ca(), make_ca("Rogue CA")
    server = issue(ca, "mosquitto", ["localhost", "127.0.0.1"])
    cafile = pem(tmp_path, "ca.crt", ca.cert_pem)
    good, rogue = issue(ca, "control-plane", [], client=True), issue(other_ca, "control-plane", [], client=True)
    good_c, good_k = pem(tmp_path, "c.crt", good.cert_pem), pem(tmp_path, "c.key", good.key_pem)
    rogue_c, rogue_k = pem(tmp_path, "r.crt", rogue.cert_pem), pem(tmp_path, "r.key", rogue.key_pem)
    with broker(tmp_path, ca, server, mtls=True) as port:
        assert not connects(port, cafile)  # no client certificate
        assert not connects(port, cafile, cert=rogue_c, key=rogue_k)  # certificate from an untrusted CA
        assert connects(port, cafile, cert=good_c, key=good_k)  # trusted client certificate + password
        assert not connects(port, cafile, cert=good_c, key=good_k, password="wrong")  # still needs the password
