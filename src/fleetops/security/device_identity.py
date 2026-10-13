"""Per-device credentials (HMAC key derivation) and command signing.

A device secret is ``base64url(HMAC-SHA256(master, "fleetops-device:<id>:<epoch>"))``. The
control plane never stores secrets: it recomputes and compares in constant time. Bumping a
device's ``key_epoch`` revokes the old secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from fleetops.domain.messages import Command


def derive_device_secret(master: str, device_id: str, epoch: int) -> str:
    msg = f"fleetops-device:{device_id}:{epoch}".encode()
    digest = hmac.new(master.encode(), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def verify_device_secret(master: str, device_id: str, epoch: int, presented: str) -> bool:
    expected = derive_device_secret(master, device_id, epoch)
    return hmac.compare_digest(expected.encode(), presented.encode())


def _canonical(command: Command) -> bytes:
    body = command.model_dump(mode="json", exclude={"signature"})
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_command(secret: str, command: Command) -> Command:
    sig = hmac.new(secret.encode(), _canonical(command), hashlib.sha256).hexdigest()
    return command.model_copy(update={"signature": sig})


def verify_command(secret: str, command: Command) -> bool:
    if not command.signature:
        return False
    expected = hmac.new(secret.encode(), _canonical(command), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, command.signature)
