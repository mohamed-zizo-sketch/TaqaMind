"""Durable factory- and task-scoped ModelRegistry adapter implementing `ModelRegistry`.

Enforces:
    - Lifecycle states: CANDIDATE -> VALIDATED/REJECTED -> ACTIVE -> ARCHIVED
    - At most one ACTIVE model per `(factory_id, domain, process, task_id)`
    - Rejected models can never be activated
    - Atomic state transition and persistent audit trail
    - Artifact existence verification before registration/activation
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from taqamind.core.interfaces.dataset_registry import (
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
)
from taqamind.core.interfaces.evaluation import FinalEvaluationResult
from taqamind.core.interfaces.model_registry import (
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionRequest,
    ModelVersion,
    validate_model_transition,
)
from taqamind.core.interfaces.serialization import (
    deserialize_contract,
    serialize_contract,
)
from taqamind.core.interfaces.storage import ModelArtifactStorage
from taqamind.domains.iron_manufacturing.adapters._store import (
    atomic_write_json as _atomic_write_json,
)
from taqamind.domains.iron_manufacturing.adapters._store import (
    read_json as _read_json,
)


class LocalModelRegistry:
    """Durable local JSON-backed implementation of Shared Core `ModelRegistry`."""

    def __init__(
        self,
        root: Path | str,
        *,
        model_storage: ModelArtifactStorage | None = None,
    ) -> None:
        self._root = Path(root).resolve() / "registries" / "models"
        self._model_storage = model_storage
        self._root.mkdir(parents=True, exist_ok=True)

    def _scope_file(self, scope: ModelScope) -> Path:
        return (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / f"{scope.task_id}.json"
        )

    def _load(self, scope: ModelScope) -> dict[str, Any]:
        return _read_json(
            self._scope_file(scope),
            {"entries": {}, "active_model_version_id": None, "audit_log": []},
        )

    def _save(self, scope: ModelScope, data: dict[str, Any]) -> None:
        _atomic_write_json(self._scope_file(scope), data)

    def next_version_number(self, *, scope: ModelScope) -> int:
        data = self._load(scope)
        if not data["entries"]:
            return 1
        max_ver = max(item["version"]["version_number"] for item in data["entries"].values())
        return int(max_ver) + 1

    def register_candidate(
        self,
        *,
        scope: ModelScope,
        version: ModelVersion,
    ) -> ModelRegistryEntry:
        if version.scope != scope:
            raise RegistryScopeError("Model version scope does not match requested scope.")

        if self._model_storage is not None:
            # Verify integrity of persisted bundle before registration
            with self._model_storage.open_verified(
                factory_id=scope.factory_id,
                reference=version.artifact,
            ):
                pass

        data = self._load(scope)
        key = str(version.model_version_id)
        if key in data["entries"]:
            existing = deserialize_contract(ModelRegistryEntry, data["entries"][key])
            if existing.version != version:
                raise RegistryConflictError(f"Conflicting model version '{key}'.")
            return existing

        existing_numbers = {
            item["version"]["version_number"] for item in data["entries"].values()
        }
        if version.version_number in existing_numbers:
            raise RegistryConflictError(
                f"Model version number {version.version_number} already exists in scope."
            )

        entry = ModelRegistryEntry(version=version, status=ModelStatus.CANDIDATE)
        data["entries"][key] = serialize_contract(entry)
        data["audit_log"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "register_candidate",
                "model_version_id": key,
                "version_number": version.version_number,
            }
        )
        self._save(scope, data)
        return entry

    def apply_evaluation(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> ModelRegistryEntry:
        data = self._load(scope)
        key = str(model_version_id)
        if key not in data["entries"]:
            raise RegistryNotFoundError(f"Model version '{model_version_id}' not found.")
        entry = deserialize_contract(ModelRegistryEntry, data["entries"][key])
        updated = entry.apply_evaluation(evaluation)
        data["entries"][key] = serialize_contract(updated)
        data["audit_log"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "apply_evaluation",
                "model_version_id": key,
                "evaluation_id": str(evaluation.evaluation_id),
                "new_status": updated.status.value,
            }
        )
        self._save(scope, data)
        return updated

    def get(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
    ) -> ModelRegistryEntry:
        data = self._load(scope)
        key = str(model_version_id)
        if key not in data["entries"]:
            raise RegistryNotFoundError(f"Model version '{model_version_id}' not found.")
        return deserialize_contract(ModelRegistryEntry, data["entries"][key])

    def get_active(
        self,
        *,
        scope: ModelScope,
    ) -> ModelRegistryEntry | None:
        data = self._load(scope)
        active_id = data.get("active_model_version_id")
        if active_id is None or active_id not in data["entries"]:
            return None
        entry = deserialize_contract(ModelRegistryEntry, data["entries"][active_id])
        if entry.status != ModelStatus.ACTIVE:
            return None
        return entry

    def list_versions(
        self,
        *,
        scope: ModelScope,
    ) -> tuple[ModelRegistryEntry, ...]:
        data = self._load(scope)
        entries = [
            deserialize_contract(ModelRegistryEntry, payload)
            for payload in data["entries"].values()
        ]
        entries.sort(key=lambda item: item.version.version_number)
        return tuple(entries)

    def transition(
        self,
        *,
        request: ModelTransitionRequest,
    ) -> ModelRegistryEntry:
        scope = request.scope
        data = self._load(scope)

        target_key = str(request.target_model_version_id)
        if target_key not in data["entries"]:
            raise RegistryNotFoundError(
                f"Target model version '{request.target_model_version_id}' not found."
            )
        target = deserialize_contract(ModelRegistryEntry, data["entries"][target_key])

        active_id = data.get("active_model_version_id")
        active = (
            deserialize_contract(ModelRegistryEntry, data["entries"][active_id])
            if active_id is not None and active_id in data["entries"]
            else None
        )

        validate_model_transition(request=request, target=target, active=active)

        if self._model_storage is not None:
            with self._model_storage.open_verified(
                factory_id=scope.factory_id,
                reference=target.version.artifact,
            ):
                pass

        now = datetime.now(UTC)
        if active is not None:
            archived = ModelRegistryEntry(
                version=active.version,
                status=ModelStatus.ARCHIVED,
                evaluation_id=active.evaluation_id,
                updated_at=now,
            )
            data["entries"][str(active.version.model_version_id)] = serialize_contract(archived)

        activated = ModelRegistryEntry(
            version=target.version,
            status=ModelStatus.ACTIVE,
            evaluation_id=target.evaluation_id,
            updated_at=now,
        )
        data["entries"][target_key] = serialize_contract(activated)
        data["active_model_version_id"] = target_key
        data["audit_log"].append(
            {
                "timestamp": now.isoformat(),
                "action": request.action.value,
                "request_id": str(request.request_id),
                "target_model_version_id": target_key,
                "previous_active_model_version_id": active_id,
                "requested_by": request.requested_by,
                "reason": request.reason,
            }
        )
        self._save(scope, data)
        return activated
