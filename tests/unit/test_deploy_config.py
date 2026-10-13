"""Static checks that lock in the security posture of the deployment configuration."""

from __future__ import annotations

import base64
import hashlib
import re
import stat
import subprocess
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import SecretStr
from scripts.provision import CONTROL_PLANE_SECRETS, ITERATIONS, mosquitto_hash, provision

from fleetops.config.settings import ControlPlaneSettings, EdgeSettings
from fleetops.registry.mlflow_admin import MlflowAdminError, ensure_service_user

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "compose.yaml").read_text())
SECRET_NAME = re.compile(r"(PASSWORD|SECRET|API_KEY|KEY_MASTER|TOKEN)$")


def services() -> dict[str, dict[str, object]]:
    return dict(COMPOSE["services"])


def test_every_published_port_binds_loopback_by_default() -> None:
    published = [(n, p) for n, s in services().items() for p in s.get("ports", [])]  # type: ignore[attr-defined]
    assert published, "expected published ports"
    for name, port in published:
        assert str(port).startswith("${FLEETOPS_BIND:-127.0.0.1}:"), f"{name} publishes {port!r} on all interfaces"
    assert "0.0.0.0" not in (ROOT / "compose.yaml").read_text().replace("FLEETOPS_BIND", "")  # noqa: S104


def test_no_credentials_in_service_environment() -> None:
    for name, svc in services().items():
        env = svc.get("environment", {}) or {}
        assert "env_file" not in svc, f"{name} must not load an env_file with secrets"
        for key, value in env.items():  # type: ignore[attr-defined]
            if SECRET_NAME.search(key):
                assert key.endswith("_FILE"), f"{name}: secret-looking variable {key} must use a secret file"
                assert str(value).startswith("/run/secrets/")


def test_all_services_are_hardened() -> None:
    for name, svc in services().items():
        assert svc.get("read_only") is True, f"{name} root filesystem must be read-only"
        assert svc.get("cap_drop") == ["ALL"], f"{name} must drop all capabilities"
        assert "no-new-privileges:true" in svc.get("security_opt", []), name  # type: ignore[operator]


def test_secrets_are_declared_files_under_deploy_secrets() -> None:
    declared = COMPOSE["secrets"]
    for name, spec in declared.items():
        assert spec["file"].startswith("./deploy/secrets/"), name
    for name, svc in services().items():
        for secret in svc.get("secrets", []):  # type: ignore[attr-defined]
            source = secret["source"] if isinstance(secret, dict) else secret
            assert source in declared, f"{name} uses undeclared secret {source}"


def test_third_party_images_and_base_images_are_pinned_by_digest() -> None:
    for name, svc in services().items():
        image = str(svc["image"])
        if "build" not in svc:
            assert "@sha256:" in image, f"{name}: {image} is not pinned by digest"
    dockerfile = (ROOT / "Dockerfile").read_text()
    for arg in ("PYTHON_IMAGE", "UV_IMAGE", "MOSQUITTO_IMAGE"):
        assert re.search(rf"^ARG {arg}=\S+@sha256:[0-9a-f]{{64}}$", dockerfile, re.M), arg


def test_dockerfile_final_stages_run_as_non_root() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    stages = re.split(r"^FROM ", dockerfile, flags=re.M)[1:]
    for target in ("edge", "control-plane", "mlflow-server", "mosquitto"):
        body = next(s for s in stages if s.splitlines()[0].endswith(f" AS {target}"))
        users = re.findall(r"^USER (\S+)", body, re.M)
        assert users and users[-1] not in ("root", "0"), f"{target} must end with a non-root USER"


def test_installs_use_hash_verified_lockfile() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "uv export --frozen" in dockerfile and dockerfile.count("--require-hashes") >= 3
    assert (ROOT / "uv.lock").exists()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout


# Names are assembled from fragments so that this guard does not itself contain the strings it forbids.
_TOOLING_NOTE = "CLAU" + "DE.md"
_TOOLING_DIR = ".clau" + "de/"


def test_ai_tooling_files_are_not_part_of_the_public_repo() -> None:
    visible = _git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    assert _TOOLING_NOTE not in visible and not any(f.startswith(_TOOLING_DIR) for f in visible)
    for path in visible:
        if path == "uv.lock" or not (ROOT / path).is_file() or (ROOT / path).stat().st_size > 2_000_000:
            continue
        assert _TOOLING_NOTE.lower() not in (ROOT / path).read_text(errors="ignore").lower(), path


def test_secret_directory_is_gitignored() -> None:
    assert _git("check-ignore", "deploy/secrets/fleetops_admin_api_key").strip()
    assert _git("check-ignore", ".env").strip()


# ---- provisioning -------------------------------------------------------------------------------


ENV = {
    "FLEETOPS_ADMIN_API_KEY": "a" * 20,
    "FLEETOPS_OPERATOR_API_KEY": "b" * 20,
    "FLEETOPS_VIEWER_API_KEY": "c" * 20,
    "FLEETOPS_DEVICE_KEY_MASTER": "master-" + "d" * 20,
    "FLEETOPS_MQTT_PASSWORD": "mqtt-" + "e" * 20,
    "FLEETOPS_MLFLOW_PASSWORD": "f" * 20,
    "FLEETOPS_MLFLOW_ADMIN_PASSWORD": "g" * 20,
    "FLEETOPS_MLFLOW_FLASK_SECRET": "h" * 20,
    "FLEETOPS_METRICS_TOKEN": "m" * 20,
    "FLEETOPS_EDGE_API_TOKEN": "t" * 20,
    "GRAFANA_ADMIN_PASSWORD": "i" * 20,
}


def test_provision_writes_private_directory_and_valid_hashes(tmp_path: Path) -> None:
    out = tmp_path / "secrets"
    out.mkdir()
    (out / "passwd").write_text("legacy")  # stale file from an older layout must be removed
    provision(ENV, out, ["edge-01", "edge-02"])
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert not (out / "passwd").exists()
    assert set(CONTROL_PLANE_SECRETS) <= {p.name for p in out.iterdir()}
    lines = (out / "mosquitto_passwd").read_text().splitlines()
    assert [ln.split(":")[0] for ln in lines] == ["control-plane", "edge-01", "edge-02"]
    user, _, rest = lines[0].partition(":")
    _, algo, iters, salt, digest = rest.split("$")
    assert (algo, int(iters)) == ("7", ITERATIONS)
    expected = hashlib.pbkdf2_hmac("sha512", ENV["FLEETOPS_MQTT_PASSWORD"].encode(), base64.b64decode(salt), ITERATIONS)
    assert base64.b64decode(digest) == expected and user == "control-plane"
    assert mosquitto_hash("x") != mosquitto_hash("x")  # random salt


def test_provision_requires_all_secrets(tmp_path: Path) -> None:
    broken = {k: v for k, v in ENV.items() if k != "FLEETOPS_MLFLOW_PASSWORD"}
    with pytest.raises(SystemExit, match="FLEETOPS_MLFLOW_PASSWORD"):
        provision(broken, tmp_path / "s", ["edge-01"])


def test_settings_load_credentials_from_secret_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "fleetops_admin_api_key": "admin-from-file-0001",
        "fleetops_operator_api_key": "operator-from-file-02",
        "fleetops_viewer_api_key": "viewer-from-file-0003",
        "fleetops_device_key_master": "master-from-file-0004",
        "fleetops_mlflow_password": "mlflow-from-file-0005",
        "fleetops_edge_device_secret": "edge-from-file-0006",
    }.items():
        (tmp_path / name).write_text(value + "\n")
    monkeypatch.setenv("FLEETOPS_SECRETS_DIR", str(tmp_path))
    for var in (
        "FLEETOPS_ADMIN_API_KEY",
        "FLEETOPS_OPERATOR_API_KEY",
        "FLEETOPS_VIEWER_API_KEY",
        "FLEETOPS_DEVICE_KEY_MASTER",
    ):
        monkeypatch.delenv(var, raising=False)
    cp = ControlPlaneSettings.load(_env_file=None)  # type: ignore[call-arg]
    assert cp.admin_api_key.get_secret_value() == "admin-from-file-0001"  # trailing newline stripped
    assert (
        isinstance(cp.mlflow_password, SecretStr) and cp.mlflow_password.get_secret_value() == "mlflow-from-file-0005"
    )
    edge = EdgeSettings.load(_env_file=None, device_id="edge-01")  # type: ignore[call-arg]
    assert edge.device_secret.get_secret_value() == "edge-from-file-0006"
    assert "from-file" not in repr(cp) and "from-file" not in repr(edge)  # never printed


# ---- MLflow service-account bootstrap --------------------------------------------------------------


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(base_url="http://mlflow.test", transport=handler)


def test_mlflow_service_user_created_updated_or_rejected() -> None:
    seen: list[tuple[str, str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.headers.get("authorization", "")[:10]))
        if req.url.path.endswith("/users/create"):
            return httpx.Response(200 if len(seen) == 1 else 400, text='{"error_code": "RESOURCE_ALREADY_EXISTS"}')
        return httpx.Response(200)

    c = _client(httpx.MockTransport(handler))
    assert ensure_service_user("http://x", "admin", "pw", "svc", "pw2", client=c) == "created"
    assert ensure_service_user("http://x", "admin", "pw", "svc", "pw2", client=c) == "updated"  # rotation
    assert seen[-1][:2] == ("PATCH", "/api/2.0/mlflow/users/update-password") and seen[-1][2].startswith("Basic")
    denied = _client(httpx.MockTransport(lambda _r: httpx.Response(401)))
    with pytest.raises(MlflowAdminError, match="rejected"):
        ensure_service_user("http://x", "admin", "bad", "svc", "pw2", client=denied)


# ---- supply chain / CI -------------------------------------------------------------------------------


def test_github_actions_are_pinned_to_commit_shas() -> None:
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        for line in wf.read_text().splitlines():
            if "uses:" in line and not line.strip().startswith("#"):
                assert re.search(r"uses: \S+@[0-9a-f]{40}\b", line), (
                    f"{wf.name}: action not pinned by SHA: {line.strip()}"
                )


def test_ci_runs_scan_security_verification_and_lock_check() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "scripts/trivy_scan.sh" in ci and "scripts/verify_security.py" in ci
    assert "uv lock --check" in ci and "uv sync --frozen" in ci
    assert "@" not in re.sub(r"uses: \S+@[0-9a-f]{40}", "", ci).replace("@sha256", "")  # no other floating refs


def test_trivy_policy_blocks_fixable_high_and_critical_with_pinned_scanner() -> None:
    script = (ROOT / "scripts" / "trivy_scan.sh").read_text()
    assert (
        'SEVERITY="${TRIVY_SEVERITY:-HIGH,CRITICAL}"' in script
        and "--ignore-unfixed" in script
        and "--exit-code 1" in script
    )
    assert re.search(r"aquasec/trivy:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", script)
    assert "uv.lock" in script


def test_trivy_exceptions_must_carry_an_expiry() -> None:
    entries = [
        ln for ln in (ROOT / ".trivyignore").read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    for entry in entries:
        assert re.fullmatch(r"(CVE-\d{4}-\d+|GHSA-[\w-]+)\s+exp:\d{4}-\d{2}-\d{2}", entry.strip()), (
            f"exception without expiry: {entry}"
        )


def test_lockfile_covers_every_runtime_dependency() -> None:
    import tomllib  # noqa: PLC0415

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {p["name"] for p in lock["package"]}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {re.split(r"[~=<>\[]", d)[0].lower() for d in project["dependencies"]}
    for extra in project["optional-dependencies"].values():
        declared |= {re.split(r"[~=<>\[]", d)[0].lower() for d in extra}
    assert declared <= locked, sorted(declared - locked)
    assert all("sdist" in p or "wheels" in p or p["name"] == "fleetops" for p in lock["package"])


# ---- Kubernetes manifest ----------------------------------------------------------------------------------


def k8s_docs() -> list[dict]:  # type: ignore[type-arg]
    return [d for d in yaml.safe_load_all((ROOT / "deploy" / "k3s" / "fleetops.yaml").read_text()) if d]


def pod_specs() -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    specs = []
    for d in k8s_docs():
        if d["kind"] in ("Deployment", "StatefulSet") or d["kind"] == "Job":
            specs.append((d["metadata"]["name"], d["spec"]["template"]["spec"]))
    return specs


def test_k8s_manifest_contains_no_secret_objects() -> None:
    assert not [d for d in k8s_docs() if d["kind"] == "Secret"], (
        "secrets are created by scripts/k8s_secrets.sh, not committed"
    )


def test_k8s_pods_are_non_root_read_only_and_do_not_automount_the_service_account() -> None:
    specs = pod_specs()
    assert {n for n, _ in specs} >= {"mosquitto", "mlflow", "control-plane", "edge-01", "mlflow-init"}
    for name, spec in specs:
        # automounting the token would collide with the /run/secrets volume (found on a real k3d cluster)
        assert spec.get("automountServiceAccountToken") is False, name
        assert spec["securityContext"]["runAsNonRoot"] is True, name
        for c in spec["containers"]:
            sc = c["securityContext"]
            assert sc["readOnlyRootFilesystem"] is True and sc["allowPrivilegeEscalation"] is False, name
            assert sc["capabilities"]["drop"] == ["ALL"], name


def test_k8s_control_plane_and_agents_authenticate_to_the_broker() -> None:
    """Regression: the control plane lacked FLEETOPS_MQTT_USERNAME and was refused by the broker on a real cluster."""
    specs = dict(pod_specs())
    cp_env = {e["name"]: e.get("value") for e in specs["control-plane"]["containers"][0]["env"]}
    assert cp_env["FLEETOPS_MQTT_USERNAME"] == "control-plane" and cp_env["FLEETOPS_MLFLOW_USERNAME"] == "fleetops-cp"
    edge_env = {e["name"]: e.get("value") for e in specs["edge-01"]["containers"][0]["env"]}
    assert edge_env["FLEETOPS_EDGE_DEVICE_ID"] == "edge-01" and edge_env["FLEETOPS_EDGE_ALLOW_FAULTS"] == "false"
    for name, spec in specs.items():
        for c in spec["containers"]:
            for e in c.get("env", []):
                assert not SECRET_NAME.search(e["name"]) or "valueFrom" in e, f"{name}: secret-looking env {e['name']}"


# ---- MLflow credentials are kept out of the process environment -----------------------------------------------


def test_mlflow_credentials_file_is_private_and_readable_by_the_mlflow_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.utils.credentials import read_mlflow_creds  # noqa: PLC0415

    from fleetops.registry.mlflow_registry import write_mlflow_credentials  # noqa: PLC0415

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    path = write_mlflow_credentials("fleetops-cp", "s3cret-value-xyz", home=tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    creds = read_mlflow_creds()
    assert (creds.username, creds.password) == ("fleetops-cp", "s3cret-value-xyz")  # the real client finds them
    import os  # noqa: PLC0415

    assert "MLFLOW_TRACKING_PASSWORD" not in os.environ  # and nothing leaked into the environment
    write_mlflow_credentials("fleetops-cp", "rotated", home=tmp_path)  # rewriting replaces, keeps the mode
    assert (
        "rotated" in path.read_text()
        and "s3cret" not in path.read_text()
        and stat.S_IMODE(path.stat().st_mode) == 0o600
    )


def test_mlflow_entrypoint_does_not_export_the_admin_password() -> None:
    script = (ROOT / "deploy" / "docker" / "mlflow-entrypoint.sh").read_text()
    assert "export MLFLOW_AUTH_ADMIN_PASSWORD" not in script and "/tmp/basic_auth.ini" in script
    assert "umask 077" in script


# ---- TLS profile ---------------------------------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """Compose-specific tags such as ``!override`` are structural hints, not data."""


_Loader.add_multi_constructor(
    "!",
    lambda loader, _suffix, node: (
        loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_scalar(node)
    ),
)


def _overlay() -> dict:  # type: ignore[type-arg]
    return yaml.load((ROOT / "compose.tls.yaml").read_text(), Loader=_Loader)  # noqa: S506 - restricted SafeLoader subclass


def test_tls_overlay_secrets_are_declared() -> None:
    overlay = _overlay()
    declared = set(COMPOSE["secrets"]) | set(overlay["secrets"])
    for name, svc in overlay["services"].items():
        for secret in svc.get("secrets", []):
            source = secret["source"] if isinstance(secret, dict) else secret
            assert source in declared, f"{name} uses undeclared secret {source}"
    for spec in overlay["secrets"].values():
        assert spec["file"].startswith("./deploy/secrets/tls/")


def test_tls_overlay_moves_every_hop_to_verified_tls() -> None:
    env = {n: s.get("environment", {}) for n, s in _overlay()["services"].items()}
    cp = env["control-plane"]
    assert cp["FLEETOPS_MLFLOW_TRACKING_URI"].startswith("https://") and cp["FLEETOPS_MQTT_PORT"] == "8883"
    assert (
        cp["FLEETOPS_MQTT_TLS_CA"]
        and cp["FLEETOPS_MLFLOW_CA_CERT"]
        and cp["FLEETOPS_TLS_CERTFILE"]
        and cp["FLEETOPS_TLS_KEYFILE"]
    )
    for edge in ("edge-01", "edge-02", "edge-03"):
        e = env[edge]
        assert e["FLEETOPS_EDGE_CONTROL_PLANE_URL"].startswith("https://") and e["FLEETOPS_EDGE_CONTROL_PLANE_CA"]
        assert (
            e["FLEETOPS_EDGE_MQTT_PORT"] == "8883"
            and e["FLEETOPS_EDGE_MQTT_TLS_CA"]
            and e["FLEETOPS_EDGE_REQUIRE_TLS"] == "true"
        )
    assert (
        env["mlflow-init"]["MLFLOW_URL"].startswith("https://")
        and env["mlflow-init"]["MLFLOW_TRACKING_SERVER_CERT_PATH"]
    )
    for svc in _overlay()["services"].values():
        for value in (svc.get("environment") or {}).values():
            assert not str(value).startswith("http://"), value  # no plaintext URL survives in the overlay


def test_tls_overlay_publishes_only_the_tls_broker_port() -> None:
    mosquitto = _overlay()["services"]["mosquitto"]
    assert mosquitto["ports"] == ["${FLEETOPS_BIND:-127.0.0.1}:${FLEETOPS_MQTT_PORT:-11883}:8883"]
    assert mosquitto["environment"]["MOSQUITTO_CONF"].endswith("mosquitto-tls.conf")


def test_broker_tls_profiles_have_no_plaintext_listener_and_no_anonymous_access() -> None:
    for name, mutual in (("mosquitto-tls.conf", False), ("mosquitto-mtls.conf", True)):
        conf = (ROOT / "mqtt" / name).read_text()
        active = [ln.strip() for ln in conf.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert "listener 8883" in active and "listener 1883" not in active, name
        assert "allow_anonymous false" in active and "cafile /mosquitto/run/tls/ca.crt" in active, name
        assert f"require_certificate {'true' if mutual else 'false'}" in active, name
        assert not any(ln.startswith("tls_version tlsv1.0") or ln.startswith("tls_version tlsv1.1") for ln in active)
        assert "acl_file /mosquitto/config/acl" in active


def test_prometheus_tls_profile_verifies_certificates() -> None:
    cfg = yaml.safe_load((ROOT / "monitoring" / "prometheus" / "prometheus-tls.yml").read_text())
    for job in cfg["scrape_configs"]:
        assert job["scheme"] == "https" and job["tls_config"]["ca_file"] == "/run/secrets/tls/ca.crt", job["job_name"]
        assert not job["tls_config"].get("insecure_skip_verify"), job["job_name"]
        assert job["authorization"]["credentials_file"] == "/run/secrets/metrics_token"


def test_development_certificates_are_short_lived_and_the_ca_key_is_never_mounted() -> None:
    text = (ROOT / "compose.tls.yaml").read_text() + (ROOT / "compose.yaml").read_text()
    assert "ca.key" not in text  # the CA private key never leaves the developer machine
    devcerts = (ROOT / "scripts" / "devcerts.py").read_text()
    assert "default=30" in devcerts and "DEVELOPMENT ONLY" in devcerts
