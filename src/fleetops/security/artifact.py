"""Artifact integrity helpers."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

_CHUNK = 1024 * 1024


class ChecksumMismatchError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected.lower()):
        raise ChecksumMismatchError(f"sha256 mismatch for {path.name}: expected {expected}, got {actual}")
