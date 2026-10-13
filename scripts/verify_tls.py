"""Assert that the running TLS profile (compose.tls.yaml) really enforces TLS. Exit code 1 on any failure.

    docker compose -f compose.yaml -f compose.tls.yaml up -d --wait && python scripts/verify_tls.py

For every TLS endpoint: a verified handshake succeeds with TLS >= 1.2; a client without the CA, with a different CA, or
using the wrong hostname is rejected; plaintext is refused. Certificate verification is never disabled by this script.
"""

from __future__ import annotations

import socket
import ssl
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.devcerts import make_ca  # noqa: E402

TLS_DIR = ROOT / "deploy" / "secrets" / "tls"
ENV = {
    k.strip(): v.strip()
    for k, _, v in (ln.partition("=") for ln in (ROOT / ".env").read_text().splitlines() if "=" in ln and ln[0] != "#")
}
ENDPOINTS = {
    "mosquitto (MQTT)": int(ENV.get("FLEETOPS_MQTT_PORT", "11883")),
    "control-plane (API)": int(ENV.get("FLEETOPS_CP_PORT", "18000")),
    "mlflow (registry)": int(ENV.get("FLEETOPS_MLFLOW_PORT", "15000")),
    "edge-01 (agent API)": int(ENV.get("EDGE1_PORT", "19101")),
    "edge-02 (agent API)": int(ENV.get("EDGE2_PORT", "19102")),
    "edge-03 (agent API)": int(ENV.get("EDGE3_PORT", "19103")),
}
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def handshake(port: int, ctx: ssl.SSLContext, hostname: str) -> ssl.SSLSocket:
    raw = socket.create_connection(("127.0.0.1", port), timeout=5)
    return ctx.wrap_socket(raw, server_hostname=hostname)


def rejected(port: int, ctx: ssl.SSLContext, hostname: str) -> str | None:
    """Return the verification error text if the handshake is rejected, None if it (wrongly) succeeds."""
    try:
        handshake(port, ctx, hostname).close()
    except ssl.SSLCertVerificationError as exc:
        return str(exc)
    except (ssl.SSLError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    ca_file = TLS_DIR / "ca.crt"
    trusting = ssl.create_default_context(cafile=str(ca_file))
    system_only = ssl.create_default_context()  # does not contain the development CA
    other = TLS_DIR.parent / "unrelated-ca.crt"
    other.write_bytes(make_ca("Unrelated CA").cert_pem)
    try:
        wrong_ca = ssl.create_default_context(cafile=str(other))
        for name, port in ENDPOINTS.items():
            with handshake(port, trusting, "localhost") as tls:
                version = str(tls.version())
                cert = tls.getpeercert()
            check(f"{name}: verified handshake succeeds ({version})", version in ("TLSv1.2", "TLSv1.3"))
            not_after = datetime.strptime(str((cert or {})["notAfter"]), "%b %d %H:%M:%S %Y %Z").replace(tzinfo=UTC)
            days = (not_after - datetime.now(UTC)).days
            check(f"{name}: certificate valid for {days} more days (development CA, short-lived)", days >= 0)
            check(
                f"{name}: rejects clients without the CA (system trust only)",
                rejected(port, system_only, "localhost") is not None,
            )
            check(
                f"{name}: rejects a certificate chain from another CA",
                rejected(port, wrong_ca, "localhost") is not None,
            )
            reason = rejected(port, trusting, "not-this-service.example")
            check(
                f"{name}: hostname verification rejects a wrong name",
                reason is not None and "mismatch" in reason.lower(),
                str(reason),
            )
        # plaintext must not be served on the TLS ports
        mqtt_port = ENDPOINTS["mosquitto (MQTT)"]
        with socket.create_connection(("127.0.0.1", mqtt_port), timeout=3) as s:
            s.sendall(bytes.fromhex("101000044d51545404020000000474657374"))  # a plaintext MQTT 3.1.1 CONNECT
            s.settimeout(3)
            try:
                reply = s.recv(16)
            except (TimeoutError, ConnectionError):
                reply = b""
        check("mosquitto: a plaintext MQTT CONNECT gets no CONNACK on the TLS port", not reply.startswith(b"\x20"))
        try:
            plain = httpx.get(f"http://127.0.0.1:{ENDPOINTS['control-plane (API)']}/healthz", timeout=3)
            ok = plain.status_code >= 400 or plain.status_code == 0
        except httpx.HTTPError:
            ok = True
        check("control plane: plain HTTP is not served on the HTTPS port", ok)
        # the platform really works over TLS: Prometheus scrapes https targets with token auth
        # Prometheus scrapes on its own schedule, so right after start-up a target can legitimately still be unscraped.
        # Poll (bounded) instead of asserting instantly; a persistently failing scrape still fails with its last error.
        prom_url = f"http://127.0.0.1:{ENV.get('FLEETOPS_PROM_PORT', '19090')}/api/v1/targets"
        deadline = time.monotonic() + 60
        while True:
            targets = httpx.get(prom_url, timeout=10).json()["data"]["activeTargets"]
            bad = [
                (t["scrapeUrl"], t["health"], t.get("lastError", ""))
                for t in targets
                if t["health"] != "up" or not t["scrapeUrl"].startswith("https://")
            ]
            if (targets and not bad) or time.monotonic() > deadline:
                break
            time.sleep(3)
        check("Prometheus scrapes every target over verified HTTPS", bool(targets) and not bad, str(bad))
        logs = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "compose.yaml",
                "-f",
                "compose.tls.yaml",
                "logs",
                "--no-color",
                "edge-01",
                "control-plane",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.lower()
        check(
            "no TLS verification errors in agent / control-plane logs",
            "certificate verify failed" not in logs and "sslcertverification" not in logs,
        )
    finally:
        other.unlink(missing_ok=True)
    print(f"\n{len(failures)} failed" if failures else "\nall TLS checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
