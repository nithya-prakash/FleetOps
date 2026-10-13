"""A real Mosquitto container (the project's hardened image) that tests can stop, start and restart."""

from __future__ import annotations

import socket
import subprocess
import time
import uuid
from pathlib import Path

from scripts.provision import mosquitto_hash

from tests.harness import free_port

IMAGE = "fleetops-mosquitto:dev"


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


class Mosquitto:
    """Plain-listener broker with persistence on a named volume (so a restart keeps sessions and queued messages)."""

    def __init__(self, tmp: Path, users: dict[str, str]) -> None:
        self.name = f"fleetops-rel-{uuid.uuid4().hex[:8]}"
        self.volume = f"{self.name}-data"
        self.port = free_port()
        self.passwd = tmp / f"{self.name}.passwd"
        self.passwd.write_text("".join(f"{user}:{mosquitto_hash(pw)}\n" for user, pw in users.items()))
        self.passwd.chmod(0o644)

    def start(self) -> None:
        _docker("build", "-q", "--target", "mosquitto", "-t", IMAGE, ".")
        result = _docker(
            "run",
            "-d",
            "--name",
            self.name,
            "-p",
            f"127.0.0.1:{self.port}:1883",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/mosquitto/run:uid=1883,gid=1883,mode=0700",
            "--tmpfs",
            "/tmp",
            "-v",
            f"{self.passwd}:/run/secrets/mosquitto_passwd:ro",
            "-v",
            f"{self.volume}:/mosquitto/data",
            IMAGE,
        )
        assert result.returncode == 0, result.stderr
        self._wait_listening()

    def _wait_listening(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                time.sleep(0.3)
                return
            except OSError:
                time.sleep(0.2)
        raise AssertionError("broker not reachable: " + _docker("logs", self.name).stdout)

    def stop(self) -> None:
        assert _docker("stop", "-t", "2", self.name).returncode == 0

    def start_again(self) -> None:
        assert _docker("start", self.name).returncode == 0
        self._wait_listening()

    def restart(self) -> None:
        assert _docker("restart", "-t", "2", self.name).returncode == 0
        self._wait_listening()

    def close(self) -> None:
        _docker("rm", "-f", self.name)
        _docker("volume", "rm", "-f", self.volume)
