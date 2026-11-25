"""Local durable adapters for SplitRegistry, RecordUsageRegistry, TrainingRegistry,
and EvaluationRegistry.

Migration note:
    Temporary local JSON-backed implementations of Shared Core ML governance
    registries. Designed for future migration into `taqamind.core.registry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from taqamind.core.interfaces.candidates import TrainingRunResult
from taqamind.core.interfaces.dataset_registry import (
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
)
from taqamind.core.interfaces.evaluation import (
    FinalEvaluationResult,
    FrozenModelSelection,
)
from taqamind.core.interfaces.leakage import (
    LeakageValidationResult,
    TrainingAuthorization,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.serialization import (
    deserialize_contract,
    serialize_contract,
)
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.training_registry import (
    EvaluationReservation,
    RecordUsage,
    RecordUsageRole,
    TrainingIntent,
    TrainingReuseKey,
    find_previously_exposed_test_records,
)
from taqamind.domains.iron_manufacturing.adapters._store import (
    atomic_write_json as _atomic_write_json,
)
from taqamind.domains.iron_manufacturing.adapters._store import (
    read_json as _read_json,
)


class LocalSplitRegistry:
    """Durable factory-scoped implementation of Shared Core `SplitRegistry`."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve() / "registries" / "splits"
        self._root.mkdir(parents=True, exist_ok=True)

    def _file(self, factory_id: UUID) -> Path:
        return self._root / f"{factory_id}.json"

    def _load(self, factory_id: UUID) -> dict[str, Any]:
        return _read_json(
            self._file(factory_id),
            {"splits": {}, "validations": {}, "authorizations": {}},
        )

    def _save(self, factory_id: UUID, data: dict[str, Any]) -> None:
        _atomic_write_json(self._file(factory_id), data)

    def register_split(
        self,
        *,
        factory_id: UUID,
        manifest: SplitManifest,
    ) -> SplitManifest:
        if manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Split manifest factory does not match requested factory.")
        data = self._load(factory_id)
        key = str(manifest.split_id)
        if key in data["splits"]:
            existing = deserialize_contract(SplitManifest, data["splits"][key])
            if existing != manifest:
                raise RegistryConflictError(f"Conflicting split registration for '{key}'.")
            return existing
        data["splits"][key] = serialize_contract(manifest)
        self._save(factory_id, data)
        return manifest

    def get_split(
        self,
        *,
        factory_id: UUID,
        split_id: UUID,
    ) -> SplitManifest:
        data = self._load(factory_id)
        key = str(split_id)
        if key not in data["splits"]:
            raise RegistryNotFoundError(f"Split manifest '{split_id}' not found.")
        return deserialize_contract(SplitManifest, data["splits"][key])

    def register_leakage_validation(
        self,
        *,
        factory_id: UUID,
        validation: LeakageValidationResult,
    ) -> None:
        if validation.manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Leakage validation factory mismatch.")
        data = self._load(factory_id)
        split_key = str(validation.manifest.split_id)
        if split_key not in data["splits"]:
            raise RegistryNotFoundError(
                f"Split '{split_key}' must be registered before leakage validation."
            )
        registered_split = deserialize_contract(SplitManifest, data["splits"][split_key])
        if registered_split != validation.manifest:
            raise RegistryConflictError("Leakage validation split does not match registered split.")

        val_key = str(validation.validation_id)
        if val_key in data["validations"]:
            existing = deserialize_contract(LeakageValidationResult, data["validations"][val_key])
            if existing != validation:
                raise RegistryConflictError(f"Conflicting leakage validation '{val_key}'.")
            return
        data["validations"][val_key] = serialize_contract(validation)
        self._save(factory_id, data)

    def issue_training_authorization(
        self,
        *,
        factory_id: UUID,
        validation_id: UUID,
    ) -> TrainingAuthorization:
        data = self._load(factory_id)
        val_key = str(validation_id)
        if val_key not in data["validations"]:
            raise RegistryNotFoundError(f"Leakage validation '{validation_id}' not found.")
        validation = deserialize_contract(LeakageValidationResult, data["validations"][val_key])
        auth = TrainingAuthorization.from_validation(validation)
        data["authorizations"][str(auth.authorization_id)] = serialize_contract(auth)
        self._save(factory_id, data)
        return auth

    def verify_training_authorization(
        self,
        *,
        factory_id: UUID,
        authorization: TrainingAuthorization,
    ) -> None:
        if authorization.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Training authorization factory mismatch.")
        data = self._load(factory_id)
        auth_key = str(authorization.authorization_id)
        if auth_key not in data["authorizations"]:
            raise RegistryNotFoundError("Training authorization is not registered.")
        registered_auth = deserialize_contract(
            TrainingAuthorization, data["authorizations"][auth_key]
        )
        if registered_auth != authorization:
            raise RegistryConflictError("Training authorization does not match registered copy.")
        val_key = str(authorization.leakage_validation_id)
        if val_key not in data["validations"]:
            raise RegistryNotFoundError("Underlying leakage validation is missing.")
        validation = deserialize_contract(LeakageValidationResult, data["validations"][val_key])
        validation.require_training_eligibility()


class LocalRecordUsageRegistry:
    """Durable scope-isolated implementation of Shared Core `RecordUsageRegistry`."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve() / "registries" / "usage"
        self._root.mkdir(parents=True, exist_ok=True)

    def _scope_file(self, scope: RecordScope) -> Path:
        return (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / f"{scope.dataset_id}__{scope.identity_namespace}__{scope.identity_version}.json"
        )

    def record_usage(
        self,
        *,
        scope: RecordScope,
        usages: tuple[RecordUsage, ...],
    ) -> None:
        if not usages:
            return
        for usage in usages:
            if usage.scope != scope:
                raise RegistryScopeError("Record usage scope does not match requested scope.")
        path = self._scope_file(scope)
        existing: list[dict[str, Any]] = _read_json(path, [])
        existing_ids = {item["usage_id"] for item in existing}
        for usage in usages:
            if str(usage.usage_id) not in existing_ids:
                existing.append(serialize_contract(usage))
                existing_ids.add(str(usage.usage_id))
        _atomic_write_json(path, existing)

    def get_history(
        self,
        *,
        scope: RecordScope,
        record_ids: tuple[UUID, ...],
    ) -> tuple[RecordUsage, ...]:
        path = self._scope_file(scope)
        raw_items: list[dict[str, Any]] = _read_json(path, [])
        target_set = set(record_ids)
        matches = [
            deserialize_contract(RecordUsage, item)
            for item in raw_items
            if UUID(item["record_id"]) in target_set
        ]
        return tuple(matches)

    def get_all_history(self, *, scope: RecordScope) -> tuple[RecordUsage, ...]:
        path = self._scope_file(scope)
        raw_items: list[dict[str, Any]] = _read_json(path, [])
        return tuple(deserialize_contract(RecordUsage, item) for item in raw_items)


class LocalTrainingRegistry:
    """Durable factory-scoped implementation of Shared Core `TrainingRegistry`."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve() / "registries" / "training"
        self._root.mkdir(parents=True, exist_ok=True)

    def _file(self, factory_id: UUID) -> Path:
        return self._root / f"{factory_id}.json"

    def _load(self, factory_id: UUID) -> dict[str, Any]:
        return _read_json(
            self._file(factory_id),
            {"intents": {}, "results": {}, "reuse_index": {}},
        )

    def _save(self, factory_id: UUID, data: dict[str, Any]) -> None:
        _atomic_write_json(self._file(factory_id), data)

    def register_intent(
        self,
        *,
        factory_id: UUID,
        intent: TrainingIntent,
    ) -> None:
        if intent.context.factory_id != factory_id:
            raise RegistryScopeError("Training intent factory mismatch.")
        data = self._load(factory_id)
        exec_key = str(intent.context.execution_id)
        if exec_key in data["intents"]:
            existing = deserialize_contract(TrainingIntent, data["intents"][exec_key])
            if existing != intent:
                raise RegistryConflictError(f"Conflicting training intent for '{exec_key}'.")
            return
        data["intents"][exec_key] = serialize_contract(intent)
        self._save(factory_id, data)

    def find_reusable_result(
        self,
        *,
        key: TrainingReuseKey,
    ) -> TrainingRunResult | None:
        data = self._load(key.factory_id)
        run_id = data["reuse_index"].get(key.fingerprint)
        if run_id is None or run_id not in data["results"]:
            return None
        return deserialize_contract(TrainingRunResult, data["results"][run_id])

    def register_result(
        self,
        *,
        factory_id: UUID,
        key: TrainingReuseKey,
        result: TrainingRunResult,
    ) -> None:
        if result.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Training result factory mismatch.")
        if key.factory_id != factory_id:
            raise RegistryScopeError("Training reuse key factory mismatch.")
        data = self._load(factory_id)
        run_key = str(result.training_run_id)
        if run_key in data["results"]:
            existing = deserialize_contract(TrainingRunResult, data["results"][run_key])
            if existing != result:
                raise RegistryConflictError(f"Conflicting training run result '{run_key}'.")
            return
        data["results"][run_key] = serialize_contract(result)
        data["reuse_index"][key.fingerprint] = run_key
        self._save(factory_id, data)

    def get_result(
        self,
        *,
        factory_id: UUID,
        training_run_id: UUID,
    ) -> TrainingRunResult:
        data = self._load(factory_id)
        key = str(training_run_id)
        if key not in data["results"]:
            raise RegistryNotFoundError(f"Training run '{training_run_id}' not found.")
        return deserialize_contract(TrainingRunResult, data["results"][key])

    def count_runs(self, *, factory_id: UUID) -> int:
        data = self._load(factory_id)
        return len(data["results"])


class LocalEvaluationRegistry:
    """Durable factory-scoped implementation of Shared Core `EvaluationRegistry`."""

    def __init__(
        self,
        root: Path | str,
        *,
        split_registry: LocalSplitRegistry,
        usage_registry: LocalRecordUsageRegistry,
    ) -> None:
        self._root = Path(root).resolve() / "registries" / "evaluations"
        self._split_registry = split_registry
        self._usage_registry = usage_registry
        self._root.mkdir(parents=True, exist_ok=True)

    def _file(self, factory_id: UUID) -> Path:
        return self._root / f"{factory_id}.json"

    def _load(self, factory_id: UUID) -> dict[str, Any]:
        return _read_json(
            self._file(factory_id),
            {"selections": {}, "reservations": {}, "exposed": {}, "evaluations": {}},
        )

    def _save(self, factory_id: UUID, data: dict[str, Any]) -> None:
        _atomic_write_json(self._file(factory_id), data)

    def register_selection(
        self,
        *,
        factory_id: UUID,
        selection: FrozenModelSelection,
    ) -> None:
        context = selection.training_run.development_split.context
        if context.factory_id != factory_id:
            raise RegistryScopeError("Frozen model selection factory mismatch.")
        data = self._load(factory_id)
        key = str(selection.selection_id)
        if key in data["selections"]:
            existing = deserialize_contract(FrozenModelSelection, data["selections"][key])
            if existing != selection:
                raise RegistryConflictError(f"Conflicting model selection '{key}'.")
            return
        data["selections"][key] = serialize_contract(selection)
        self._save(factory_id, data)

    def reserve_test_access(
        self,
        *,
        factory_id: UUID,
        reservation: EvaluationReservation,
    ) -> EvaluationReservation:
        if reservation.scope.factory_id != factory_id:
            raise RegistryScopeError("Evaluation reservation factory mismatch.")
        data = self._load(factory_id)

        sel_key = str(reservation.selection_id)
        if sel_key not in data["selections"]:
            raise RegistryNotFoundError(
                f"Frozen selection '{sel_key}' must be registered before reserving test access."
            )

        manifest = self._split_registry.get_split(
            factory_id=factory_id,
            split_id=reservation.split_id,
        )
        if set(reservation.test_record_ids) != set(manifest.test.record_ids):
            raise RegistryConflictError(
                "Reservation test records do not match registered immutable split manifest."
            )

        history = self._usage_registry.get_history(
            scope=reservation.scope,
            record_ids=reservation.test_record_ids,
        )
        previously_exposed = find_previously_exposed_test_records(
            manifest=manifest,
            scope=reservation.scope,
            history=history,
        )
        if previously_exposed:
            raise RegistryConflictError(
                f"Protected test partition contains {len(previously_exposed)} previously "
                "exposed record(s)."
            )

        requested_ids = set(reservation.test_record_ids)
        for existing_raw in data["reservations"].values():
            existing = deserialize_contract(EvaluationReservation, existing_raw)
            if existing.reservation_id == reservation.reservation_id:
                if existing != reservation:
                    raise RegistryConflictError("Conflicting reservation identifier.")
                return existing
            if existing.scope == reservation.scope and (
                set(existing.test_record_ids) & requested_ids
            ):
                raise RegistryConflictError(
                    "Competing reservation overlaps protected test records."
                )

        data["reservations"][str(reservation.reservation_id)] = serialize_contract(reservation)
        self._save(factory_id, data)
        return reservation

    def mark_test_exposed(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
    ) -> None:
        data = self._load(factory_id)
        res_key = str(reservation_id)
        if res_key not in data["reservations"]:
            raise RegistryNotFoundError(f"Evaluation reservation '{reservation_id}' not found.")
        reservation = deserialize_contract(EvaluationReservation, data["reservations"][res_key])
        if res_key in data["exposed"]:
            return

        usages = tuple(
            RecordUsage(
                scope=reservation.scope,
                record_id=rec_id,
                record_version_id=rec_id,
                execution_id=reservation.execution_id,
                split_id=reservation.split_id,
                role=RecordUsageRole.PROTECTED_TEST,
            )
            for rec_id in reservation.test_record_ids
        )
        self._usage_registry.record_usage(scope=reservation.scope, usages=usages)
        data["exposed"][res_key] = True
        self._save(factory_id, data)

    def get_reservation(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
    ) -> EvaluationReservation:
        data = self._load(factory_id)
        res_key = str(reservation_id)
        if res_key not in data["reservations"]:
            raise RegistryNotFoundError(f"Evaluation reservation '{reservation_id}' not found.")
        return deserialize_contract(EvaluationReservation, data["reservations"][res_key])

    def register_evaluation(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> None:
        if evaluation.split_manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Final evaluation factory mismatch.")
        data = self._load(factory_id)
        res_key = str(reservation_id)
        if res_key not in data["reservations"]:
            raise RegistryNotFoundError(f"Evaluation reservation '{reservation_id}' not found.")
        if res_key not in data["exposed"]:
            raise RegistryConflictError(
                "Protected test exposure must be persisted before registering evaluation."
            )
        reservation = deserialize_contract(EvaluationReservation, data["reservations"][res_key])
        if reservation.selection_id != evaluation.selection.selection_id:
            raise RegistryConflictError("Evaluation selection does not match reservation.")
        if reservation.split_id != evaluation.split_manifest.split_id:
            raise RegistryConflictError("Evaluation split does not match reservation.")

        eval_key = str(evaluation.evaluation_id)
        if eval_key in data["evaluations"]:
            existing = deserialize_contract(FinalEvaluationResult, data["evaluations"][eval_key])
            if existing != evaluation:
                raise RegistryConflictError(f"Conflicting final evaluation '{eval_key}'.")
            return
        data["evaluations"][eval_key] = serialize_contract(evaluation)
        self._save(factory_id, data)

    def get_evaluation(
        self,
        *,
        factory_id: UUID,
        evaluation_id: UUID,
    ) -> FinalEvaluationResult:
        data = self._load(factory_id)
        eval_key = str(evaluation_id)
        if eval_key not in data["evaluations"]:
            raise RegistryNotFoundError(f"Final evaluation '{evaluation_id}' not found.")
        return deserialize_contract(FinalEvaluationResult, data["evaluations"][eval_key])
