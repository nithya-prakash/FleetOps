"""DEVELOPMENT-ONLY certificate tooling: a private CA and short-lived leaf certificates.

    python scripts/devcerts.py [--out deploy/secrets/tls]

These certificates exist so TLS can be exercised locally and in tests. They are NOT a production PKI:
the CA key sits on the developer's disk, there is no revocation, and lifetimes are short. See docs/security.md
("Development vs production certificates"). The helpers are also imported by the TLS tests to build
deliberately broken certificates (expired, wrong name, untrusted CA).
"""

from __future__ import annotations

import argparse
import ipaddress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

SERVICES = {
    "mosquitto": ["mosquitto", "localhost", "127.0.0.1"],
    "control-plane": ["control-plane", "localhost", "127.0.0.1"],
    "mlflow": ["mlflow", "localhost", "127.0.0.1"],
    "edge-01": ["edge-01", "localhost", "127.0.0.1"],
    "edge-02": ["edge-02", "localhost", "127.0.0.1"],
    "edge-03": ["edge-03", "localhost", "127.0.0.1"],
}


@dataclass(frozen=True)
class KeyPair:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey

    @property
    def cert_pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    @property
    def key_pem(self) -> bytes:
        return self.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FleetOps DEVELOPMENT ONLY"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def make_ca(common_name: str = "FleetOps Dev CA", days: int = 30) -> KeyPair:
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(common_name))
        .issuer_name(_name(common_name))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return KeyPair(cert, key)


def issue(
    ca: KeyPair,
    common_name: str,
    names: list[str],
    *,
    days: int = 30,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
    client: bool = False,
) -> KeyPair:
    """Leaf certificate for ``names`` (DNS names and IP addresses become subjectAltNames)."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    sans: list[x509.GeneralName] = []
    for n in names:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            sans.append(x509.DNSName(n))
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(common_name))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(minutes=5))
        .not_valid_after(not_after or now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        # Python >= 3.13 verifies with VERIFY_X509_STRICT, which rejects leaf certificates without these identifiers.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH if client else ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
    )
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return KeyPair(builder.sign(ca.key, hashes.SHA256()), key)


def write(path: Path, data: bytes, mode: int) -> None:
    path.write_bytes(data)
    path.chmod(mode)


def write_pair(directory: Path, stem: str, pair: KeyPair) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / f"{stem}.crt", pair.cert_pem, 0o644)
    write(directory / f"{stem}.key", pair.key_pem, 0o644)  # container users differ; the 0700 parent protects it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("deploy/secrets/tls"))
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    ca = make_ca(days=args.days)
    write(out / "ca.crt", ca.cert_pem, 0o644)
    write(out / "ca.key", ca.key_pem, 0o600)  # never mounted into a container
    for name, sans in SERVICES.items():
        write_pair(out, name, issue(ca, name, sans, days=args.days))
    write_pair(out, "mqtt-client", issue(ca, "control-plane", [], days=args.days, client=True))
    print(f"wrote CA + {len(SERVICES)} server certificates + 1 client certificate to {out} (valid {args.days} days)")
    print("DEVELOPMENT ONLY: do not use these certificates or this CA outside a developer machine.")


if __name__ == "__main__":
    main()
