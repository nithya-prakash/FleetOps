"""On-device model store with atomic activation and last-known-good retention.

Layout::

    <data_dir>/models/<name>/<version>/model.safetensors
    <data_dir>/staging/<tmp>.part
    <data_dir>/state.json      (written write-temp -> fsync -> os.replace)

A failed stage/activation/rollback never modifies ``active`` before the new model has been
verified and loaded, so the last known-good model keeps serving.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fleetops.domain.messages import DeviceModelState
from fleetops.domain.models import ModelIdentity
from fleetops.registry.modelformat import MlpModel, ModelFormatError, load_bundle
from fleetops.security.artifact import ChecksumMismatchError, verify_sha256

logger = logging.getLogger(__name__)

MAX_REMEMBERED_ACKS = 64


class StoreError(Exception):
    pass


class PersistedState(BaseModel):
    active: ModelIdentity | None = None
    previous: ModelIdentity | None = None
    staged: ModelIdentity | None = None
    last_generation: int = 0
    acks: dict[str, dict[str, Any]] = Field(default_factory=dict)  # command_id -> ack payload


class ModelStore:
    def __init__(self, data_dir: Path) -> None:
        self._root = data_dir
        self._models = data_dir / "models"
        self.staging_dir = data_dir / "staging"
        self._models.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        for (
            leftover
        ) in self.staging_dir.iterdir():  # a crash mid-download leaves partial files; they are never reusable
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        self._state_path = data_dir / "state.json"
        self._lock = threading.RLock()
        self.state = self._read_state()

    # -- state persistence -------------------------------------------------------------------

    def _read_state(self) -> PersistedState:
        if not self._state_path.exists():
            return PersistedState()
        try:
            return PersistedState.model_validate_json(self._state_path.read_text())
        except Exception:
            logger.exception("state.json unreadable; starting with empty state")
            return PersistedState()

    def _write_state(self, state: PersistedState) -> None:
        """write-temp, fsync, atomic replace. On any I/O error the previous state.json and self.state are untouched."""
        tmp: str | None = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                fh.write(state.model_dump_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._state_path)
        except OSError as exc:
            raise StoreError(f"cannot persist device state: {exc}") from exc
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
        self.state = state

    def snapshot(self) -> DeviceModelState:
        s = self.state
        return DeviceModelState(
            active=s.active, staged=s.staged, previous=s.previous, last_generation=s.last_generation
        )

    def record_command(self, generation: int, command_id: str, ack: dict[str, Any]) -> None:
        with self._lock:
            acks = dict(self.state.acks)
            acks[command_id] = ack
            while len(acks) > MAX_REMEMBERED_ACKS:
                acks.pop(next(iter(acks)))
            self._write_state(
                self.state.model_copy(
                    update={"acks": acks, "last_generation": max(generation, self.state.last_generation)}
                )
            )

    # -- paths -------------------------------------------------------------------------------

    def path_of(self, ident: ModelIdentity) -> Path:
        return self._models / ident.name / str(ident.version) / "model.safetensors"

    def has(self, ident: ModelIdentity) -> bool:
        p = self.path_of(ident)
        return p.is_file()

    def _load_verified(self, ident: ModelIdentity) -> MlpModel:
        path = self.path_of(ident)
        try:
            verify_sha256(path, ident.sha256)
            model = load_bundle(path)
            model.smoke_test()
        except (ChecksumMismatchError, ModelFormatError, OSError, ValueError) as exc:
            raise StoreError(f"model {ident.name}:{ident.version} failed verification: {exc}") from exc
        return model

    # -- lifecycle ---------------------------------------------------------------------------

    def commit_staged(self, ident: ModelIdentity, downloaded: Path) -> None:
        """Move an already checksum-verified download into the store and mark it staged."""
        with self._lock:
            dest = self.path_of(ident)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, dest)
            try:
                self._load_verified(ident)
            except StoreError:
                if ident not in (self.state.active, self.state.previous):
                    shutil.rmtree(dest.parent, ignore_errors=True)
                raise
            self._write_state(self.state.model_copy(update={"staged": ident}))

    def activate_staged(self, ident: ModelIdentity) -> MlpModel:
        with self._lock:
            if self.state.staged != ident:
                raise StoreError(f"{ident.name}:{ident.version} is not staged")
            model = self._load_verified(ident)  # re-verify immediately before the switch
            previous = self.state.active if self.state.active != ident else self.state.previous
            self._write_state(self.state.model_copy(update={"active": ident, "previous": previous, "staged": None}))
            self._gc()
            return model

    def switch_to(self, target: ModelIdentity | None) -> MlpModel | None:
        """Rollback: make ``target`` active (None = deactivate). Previous becomes None: the model
        we left is not known-good."""
        with self._lock:
            model = self._load_verified(target) if target is not None else None
            self._write_state(self.state.model_copy(update={"active": target, "previous": None, "staged": None}))
            self._gc()
            return model

    def discard_staged(self, name: str, version: int) -> bool:
        with self._lock:
            s = self.state.staged
            if s is None or (s.name, s.version) != (name, version):
                return False
            self._write_state(self.state.model_copy(update={"staged": None}))
            self._gc()
            return True

    def recover_active(self) -> tuple[MlpModel | None, str | None]:
        """Startup: load the active model; fall back to previous if it is corrupted."""
        with self._lock:
            for label in ("active", "previous"):
                ident: ModelIdentity | None = getattr(self.state, label)
                if ident is None:
                    continue
                try:
                    model = self._load_verified(ident)
                except StoreError as exc:
                    logger.error("cannot load %s model: %s", label, exc)
                    continue
                if label == "previous":
                    self._write_state(self.state.model_copy(update={"active": ident, "previous": None}))
                    return model, "active model corrupted; recovered previous"
                return model, None
            if self.state.active is not None:
                self._write_state(self.state.model_copy(update={"active": None, "previous": None}))
            return None, "no loadable model" if self.state.active is not None else None

    def _gc(self) -> None:
        keep = {
            (i.name, str(i.version))
            for i in (self.state.active, self.state.previous, self.state.staged)
            if i is not None
        }
        for name_dir in self._models.iterdir():
            if not name_dir.is_dir():
                continue
            for ver_dir in name_dir.iterdir():
                if (name_dir.name, ver_dir.name) not in keep:
                    shutil.rmtree(ver_dir, ignore_errors=True)
