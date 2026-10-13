"""Content-addressed, immutable-per-version artifact storage on a filesystem volume."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fleetops.domain.models import NAME_PATTERN
from fleetops.security.artifact import sha256_file

_NAME_RE = re.compile(NAME_PATTERN)
ARTIFACT_FILENAME = "model.safetensors"


class ArtifactStoreError(Exception):
    pass


class ArtifactIntegrityError(ArtifactStoreError):
    """The stored bytes no longer match the checksum recorded at registration."""


@dataclass(frozen=True)
class StoredArtifact:
    sha256: str
    size_bytes: int


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str, version: int) -> Path:
        """Resolve a model version to a path; inputs are validated so no traversal is possible."""
        if not _NAME_RE.match(name) or version < 1:
            raise ArtifactStoreError("invalid artifact reference")
        path = (self._root / name / str(version) / ARTIFACT_FILENAME).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ArtifactStoreError("artifact path escapes store root")
        return path

    def put(self, name: str, version: int, source: Path) -> StoredArtifact:
        dest = self.path_for(name, version)
        digest = sha256_file(source)
        if dest.exists():
            if sha256_file(dest) == digest:
                return StoredArtifact(digest, dest.stat().st_size)
            raise ArtifactStoreError(f"{name}:{version} already stored with different content")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, source.open("rb") as src:
                shutil.copyfileobj(src, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_name, dest)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        return StoredArtifact(digest, dest.stat().st_size)

    def verified(self, name: str, version: int, expected_sha256: str) -> Path:
        """Return the artifact path only if its content hash equals ``expected_sha256`` (registry value).

        Hashes are cached per (path, size, mtime) so repeated downloads of a large model are not re-hashed, while any
        change to the file invalidates the cache.
        """
        path = self.get(name, version)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        actual = self._hash_cache.get(key)
        if actual is None:
            actual = sha256_file(path)
            if len(self._hash_cache) >= 256:
                self._hash_cache.clear()
            self._hash_cache[key] = actual
        if actual != expected_sha256:
            raise ArtifactIntegrityError(
                f"artifact {name}:{version} hash {actual[:12]} != registered {expected_sha256[:12]}"
            )
        return path

    def get(self, name: str, version: int) -> Path:
        path = self.path_for(name, version)
        if not path.is_file():
            raise ArtifactStoreError(f"artifact {name}:{version} not found")
        return path

    def delete(self, name: str, version: int) -> None:
        path = self.path_for(name, version)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
