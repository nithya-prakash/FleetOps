"""API-key authentication with role hierarchy for the management API."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import IntEnum


class Role(IntEnum):
    VIEWER = 1
    OPERATOR = 2
    ADMIN = 3


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role

    def allows(self, required: Role) -> bool:
        return self.role >= required


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


class ApiKeyAuthenticator:
    """Compares presented keys against configured ones in constant time."""

    def __init__(self, keys: dict[Role, str]) -> None:
        self._digests = [(role, _digest(key)) for role, key in keys.items()]

    def authenticate(self, presented: str | None) -> Principal | None:
        if not presented:
            return None
        probe = _digest(presented)
        found: Role | None = None
        for role, digest in self._digests:  # no early exit: constant-time over all keys
            if hmac.compare_digest(probe, digest) and found is None:
                found = role
        return Principal(name=found.name.lower(), role=found) if found else None
