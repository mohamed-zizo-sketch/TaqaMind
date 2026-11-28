"""Core P1.6 LocalSplitRegistry and LocalRecordUsageRegistry implementations."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from taqamind.core.interfaces.dataset_registry import (
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
)
from taqamind.core.interfaces.leakage import (
    LeakageValidationResult,
    TrainingAuthorization,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.training_registry import (
    RecordUsage,
    RecordUsageRegistry,
    SplitRegistry,
)
from taqamind.core.registry._concurrency import locked_path
from taqamind.core.registry._json_store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalSplitRegistry(SplitRegistry):
    """Persist immutable SplitManifest, LeakageValidationResult, and TrainingAuthorization."""

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve()

    def _split_path(self, factory_id: UUID, split_id: UUID) -> Path:
        return safe_resolve(self._root, "splits", str(factory_id), f"{split_id}.json")

    def _validation_path(self, factory_id: UUID, validation_id: UUID) -> Path:
        return safe_resolve(
            self._root, "leakage_validations", str(factory_id), f"{validation_id}.json"
        )

    def _factory_lock(self, factory_id: UUID) -> Path:
        return safe_resolve(self._root, "splits", str(factory_id), ".split_registry.lock")

    def register_split(self, *, factory_id: UUID, manifest: SplitManifest) -> SplitManifest:
        if manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Split manifest factory mismatch.")
        with locked_path(self._factory_lock(factory_id)):
            path = self._split_path(factory_id, manifest.split_id)
            if path.is_file():
                existing = load_json_model(path, SplitManifest)
                if existing == manifest:
                    return existing
                raise RegistryConflictError("Conflicting split manifest for existing split_id.")
            atomic_write_text(path, dump_json_model(manifest))
            return manifest

    def get_split(self, *, factory_id: UUID, split_id: UUID) -> SplitManifest:
        path = self._split_path(factory_id, split_id)
        if not path.is_file():
            raise RegistryNotFoundError(f"Split '{split_id}' not found.")
        return load_json_model(path, SplitManifest)

    def register_leakage_validation(
        self, *, factory_id: UUID, validation: LeakageValidationResult
    ) -> None:
        if validation.manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Leakage validation factory mismatch.")
        with locked_path(self._factory_lock(factory_id)):
            self.get_split(factory_id=factory_id, split_id=validation.manifest.split_id)
            path = self._validation_path(factory_id, validation.validation_id)
            if path.is_file():
                existing = load_json_model(path, LeakageValidationResult)
                if existing == validation:
                    return
                raise RegistryConflictError("Conflicting leakage validation record.")
            atomic_write_text(path, dump_json_model(validation))

    def issue_training_authorization(
        self, *, factory_id: UUID, validation_id: UUID
    ) -> TrainingAuthorization:
        path = self._validation_path(factory_id, validation_id)
        if not path.is_file():
            raise RegistryNotFoundError(f"Leakage validation '{validation_id}' not found.")
        validation = load_json_model(path, LeakageValidationResult)
        if validation.manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Validation factory mismatch.")
        return TrainingAuthorization.from_validation(validation)

    def verify_training_authorization(
        self, *, factory_id: UUID, authorization: TrainingAuthorization
    ) -> None:
        if authorization.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Authorization factory mismatch.")
        path = self._validation_path(factory_id, authorization.leakage_validation_id)
        if not path.is_file():
            raise RegistryNotFoundError("Authorization references unregistered leakage validation.")
        validation = load_json_model(path, LeakageValidationResult)
        validation.require_training_eligibility()
        if validation.manifest.split_id != authorization.development_split.split_id:
            raise RegistryConflictError("Authorization split mismatch.")


class LocalRecordUsageRegistry(RecordUsageRegistry):
    """Append-only local JSON persistence for historical ML record exposures."""

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve()

    def _scope_dir(self, scope: RecordScope) -> Path:
        return safe_resolve(
            self._root,
            "record_usage",
            str(scope.factory_id),
            scope.domain,
            scope.process,
            scope.dataset_id,
            scope.identity_namespace,
            scope.identity_version,
        )

    def _scope_lock(self, scope: RecordScope) -> Path:
        return safe_resolve(self._scope_dir(scope), ".usage.lock")

    def record_usage(self, *, scope: RecordScope, usages: tuple[RecordUsage, ...]) -> None:
        if not usages:
            return
        scope_dir = self._scope_dir(scope)
        with locked_path(self._scope_lock(scope)):
            pending: list[tuple[Path, RecordUsage]] = []
            for usage in usages:
                if usage.scope != scope:
                    raise RegistryScopeError("RecordUsage scope mismatch.")
                path = safe_resolve(scope_dir, f"{usage.usage_id}.json")
                if path.is_file():
                    existing = load_json_model(path, RecordUsage)
                    if existing != usage:
                        raise RegistryConflictError(
                            "Conflicting RecordUsage for existing usage_id."
                        )
                    continue
                pending.append((path, usage))

            created_paths: list[Path] = []
            try:
                for path, usage in pending:
                    atomic_write_text(path, dump_json_model(usage))
                    created_paths.append(path)
            except BaseException:
                for created in reversed(created_paths):
                    created.unlink(missing_ok=True)
                raise

    def get_history(
        self, *, scope: RecordScope, record_ids: tuple[UUID, ...]
    ) -> tuple[RecordUsage, ...]:
        wanted = set(record_ids)
        all_usages = iter_json_models(self._scope_dir(scope), RecordUsage)
        matched = [u for u in all_usages if u.record_id in wanted]
        matched.sort(key=lambda u: (u.exposed_at, str(u.usage_id)))
        return tuple(matched)
