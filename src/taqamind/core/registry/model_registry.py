"""Core P1.8 local ModelRegistry implementation with single-active-model & audit log."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from taqamind.core.interfaces.dataset_registry import (
    RegistryConflictError,
    RegistryNotFoundError,
)
from taqamind.core.interfaces.evaluation import FinalEvaluationResult
from taqamind.core.interfaces.model_registry import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionRequest,
    ModelVersion,
    validate_model_transition,
)
from taqamind.core.interfaces.storage import ModelArtifactStorage
from taqamind.core.registry._concurrency import locked_path
from taqamind.core.registry._json_store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalModelRegistry(ModelRegistry):
    """JSON-backed factory-scoped model registry with atomic transitions."""

    def __init__(
        self,
        *,
        root_dir: Path,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._model_storage = model_storage

    def _scope_dir(self, scope: ModelScope) -> Path:
        return safe_resolve(
            self._root,
            f"{scope.factory_id}/{scope.domain}/{scope.process}/{scope.task_id}",
        )

    def _versions_dir(self, scope: ModelScope) -> Path:
        return safe_resolve(self._scope_dir(scope), "versions")

    def _version_path(self, scope: ModelScope, model_version_id: UUID) -> Path:
        return safe_resolve(self._versions_dir(scope), f"{model_version_id}.json")

    def _audit_log_path(self, scope: ModelScope) -> Path:
        return safe_resolve(self._scope_dir(scope), "transitions.jsonl")

    def _lock_path(self, scope: ModelScope) -> Path:
        return safe_resolve(self._scope_dir(scope), "registry.lock")

    def register_candidate(
        self,
        *,
        scope: ModelScope,
        version: ModelVersion,
    ) -> ModelRegistryEntry:
        scope.validate_identity(version.identity)
        # Verify model bundle exists and matches declared SHA-256 / size
        with self._model_storage.open_verified(
            factory_id=scope.factory_id,
            reference=version.artifact,
        ):
            pass

        path = self._version_path(scope, version.model_version_id)
        with locked_path(self._lock_path(scope)):
            if path.exists():
                existing = load_json_model(path, ModelRegistryEntry)
                if existing.version != version:
                    raise RegistryConflictError(
                        f"Conflicting model version '{version.model_version_id}' already exists."
                    )
                return existing
            entry = ModelRegistryEntry(
                version=version,
                status=ModelStatus.CANDIDATE,
                updated_at=max(version.created_at, datetime.now(UTC)),
            )
            atomic_write_text(path, dump_json_model(entry))
            return entry

    def apply_evaluation(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> ModelRegistryEntry:
        path = self._version_path(scope, model_version_id)
        with locked_path(self._lock_path(scope)):
            if not path.exists():
                raise RegistryNotFoundError(f"Model version '{model_version_id}' not found.")
            entry = load_json_model(path, ModelRegistryEntry)
            updated = entry.apply_evaluation(evaluation)
            atomic_write_text(path, dump_json_model(updated))
            return updated

    def get(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
    ) -> ModelRegistryEntry:
        path = self._version_path(scope, model_version_id)
        if not path.exists():
            raise RegistryNotFoundError(f"Model version '{model_version_id}' not found.")
        return load_json_model(path, ModelRegistryEntry)

    def get_active(
        self,
        *,
        scope: ModelScope,
    ) -> ModelRegistryEntry | None:
        active = [
            entry
            for entry in iter_json_models(self._versions_dir(scope), ModelRegistryEntry)
            if entry.status == ModelStatus.ACTIVE
        ]
        if len(active) > 1:
            raise RegistryConflictError("Multiple active models detected in scope.")
        return active[0] if active else None

    def list_versions(
        self,
        *,
        scope: ModelScope,
    ) -> tuple[ModelRegistryEntry, ...]:
        entries = list(iter_json_models(self._versions_dir(scope), ModelRegistryEntry))
        entries.sort(key=lambda e: (e.version.created_at, str(e.version.model_version_id)))
        return tuple(entries)

    def transition(
        self,
        *,
        request: ModelTransitionRequest,
    ) -> ModelRegistryEntry:
        scope = request.scope
        with locked_path(self._lock_path(scope)):
            target_path = self._version_path(scope, request.target_model_version_id)
            if not target_path.exists():
                raise RegistryNotFoundError(
                    f"Target model '{request.target_model_version_id}' not found."
                )
            target = load_json_model(target_path, ModelRegistryEntry)
            active = self.get_active(scope=scope)

            validate_model_transition(request=request, target=target, active=active)

            # Verify target bundle integrity before activating
            with self._model_storage.open_verified(
                factory_id=scope.factory_id,
                reference=target.version.artifact,
            ):
                pass

            now = datetime.now(UTC)
            active_path: Path | None = None
            active_backup: str | None = None
            target_backup = target_path.read_text(encoding="utf-8")
            if active is not None:
                active_path = self._version_path(scope, active.version.model_version_id)
                active_backup = active_path.read_text(encoding="utf-8")
                archived = ModelRegistryEntry(
                    version=active.version,
                    status=ModelStatus.ARCHIVED,
                    evaluation_id=active.evaluation_id,
                    updated_at=max(now, active.updated_at),
                )
                atomic_write_text(active_path, dump_json_model(archived))

            activated = ModelRegistryEntry(
                version=target.version,
                status=ModelStatus.ACTIVE,
                evaluation_id=target.evaluation_id,
                updated_at=max(now, target.updated_at),
            )
            try:
                atomic_write_text(target_path, dump_json_model(activated))

                audit_path = self._audit_log_path(scope)
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_record = {
                    "action": request.action.value,
                    "target_model_version_id": str(request.target_model_version_id),
                    "previous_active_model_version_id": (
                        str(active.version.model_version_id) if active is not None else None
                    ),
                    "requested_by": request.requested_by,
                    "reason": request.reason,
                    "timestamp": now.isoformat(),
                }
                with audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(audit_record) + "\n")
            except BaseException:
                atomic_write_text(target_path, target_backup)
                if active_path is not None and active_backup is not None:
                    atomic_write_text(active_path, active_backup)
                raise

            return activated


__all__ = ["LocalModelRegistry"]
