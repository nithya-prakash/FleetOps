"""Generate local-development secrets from ``.env`` into ``deploy/secrets`` (gitignored).

Creates one file per secret (consumed as Docker secrets under /run/secrets, never as environment variables):
control-plane credentials, MLflow admin/service/flask secrets, Grafana admin password, the Mosquitto password
file (PBKDF2-SHA512 hashes) and one derived credential per edge device.

The directory is mode 0700 on the host. Files are 0644 because container users have different uids and read
them through bind mounts; the directory permission is what protects them from other host users.

Usage: ``python scripts/provision.py [--devices edge-01 edge-02 edge-03]``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import secrets
from pathlib import Path

from fleetops.security.device_identity import derive_device_secret

ROOT = Path(__file__).resolve().parent.parent
ITERATIONS = 101  # Mosquitto's PBKDF2-SHA512 iteration count

# secret file name -> .env variable it is generated from
CONTROL_PLANE_SECRETS = {
    "fleetops_admin_api_key": "FLEETOPS_ADMIN_API_KEY",
    "fleetops_operator_api_key": "FLEETOPS_OPERATOR_API_KEY",
    "fleetops_viewer_api_key": "FLEETOPS_VIEWER_API_KEY",
    "fleetops_device_key_master": "FLEETOPS_DEVICE_KEY_MASTER",
    "fleetops_mqtt_password": "FLEETOPS_MQTT_PASSWORD",
    "fleetops_mlflow_password": "FLEETOPS_MLFLOW_PASSWORD",
    "mlflow_admin_password": "FLEETOPS_MLFLOW_ADMIN_PASSWORD",
    "mlflow_flask_secret": "FLEETOPS_MLFLOW_FLASK_SECRET",
    "grafana_admin_password": "GRAFANA_ADMIN_PASSWORD",
    "fleetops_metrics_token": "FLEETOPS_METRICS_TOKEN",  # control plane: accepts it on /metrics
    "metrics_token": "FLEETOPS_METRICS_TOKEN",  # Prometheus: presents it
    "fleetops_edge_api_token": "FLEETOPS_EDGE_API_TOKEN",  # agents: required on /predict and /healthz
    "fleetops_edge_metrics_token": "FLEETOPS_METRICS_TOKEN",  # agents: accepted on /metrics
}


def mosquitto_hash(password: str) -> str:
    salt = secrets.token_bytes(12)
    digest = hashlib.pbkdf2_hmac("sha512", password.encode(), salt, ITERATIONS)
    return f"$7${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, v = stripped.split("=", 1)
                values[k.strip()] = v.strip()
    values.update({k: v for k, v in os.environ.items() if k.startswith(("FLEETOPS_", "GRAFANA_"))})
    return values


def write_secret(path: Path, value: str) -> None:
    path.write_text(value.strip() + "\n")
    path.chmod(0o644)


def provision(env: dict[str, str], out: Path, devices: list[str], load_devices: list[str] | None = None) -> list[Path]:
    missing = [var for var in CONTROL_PLANE_SECRETS.values() if not env.get(var)]
    if missing or not env.get("FLEETOPS_DEVICE_KEY_MASTER"):
        raise SystemExit(f"missing in .env: {', '.join(missing)} (copy .env.example to .env)")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    written: list[Path] = []
    for name, var in CONTROL_PLANE_SECRETS.items():
        write_secret(out / name, env[var])
        written.append(out / name)
    cp_user = env.get("FLEETOPS_MQTT_USERNAME", "control-plane")
    lines = [f"{cp_user}:{mosquitto_hash(env['FLEETOPS_MQTT_PASSWORD'])}"]
    for dev in devices:
        device_secret = derive_device_secret(env["FLEETOPS_DEVICE_KEY_MASTER"], dev, 1)  # epoch 1 = first registration
        lines.append(f"{dev}:{mosquitto_hash(device_secret)}")
        write_secret(out / f"{dev}.secret", device_secret)
        written.append(out / f"{dev}.secret")
    for dev in (
        load_devices or []
    ):  # simulated devices for load tests: broker credentials only (no container secret files)
        lines.append(f"{dev}:{mosquitto_hash(derive_device_secret(env['FLEETOPS_DEVICE_KEY_MASTER'], dev, 1))}")
    write_secret(out / "mosquitto_passwd", "\n".join(lines))
    written.append(out / "mosquitto_passwd")
    for stale in out.iterdir():  # the directory is fully generated: drop leftovers from older layouts
        if stale.is_file() and stale not in written:
            stale.unlink()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["edge-01", "edge-02", "edge-03"])
    ap.add_argument("--out", type=Path, default=ROOT / "deploy" / "secrets")
    ap.add_argument("--load-devices", type=int, default=0, help="also create N simulated load-test device credentials")
    ap.add_argument("--load-prefix", default="load-")
    args = ap.parse_args()
    load = [f"{args.load_prefix}{i:04d}" for i in range(1, args.load_devices + 1)]
    written = provision(load_env(ROOT / ".env"), args.out, args.devices, load)
    print(f"wrote {len(written)} secret files in {args.out} (directory mode 0700)")


if __name__ == "__main__":
    main()
