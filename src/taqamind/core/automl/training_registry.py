"""Core P1.7 local TrainingRegistry and EvaluationRegistry implementations."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from taqamind.core.interfaces.candidates import TrainingRunResult
from taqamind.core.interfaces.data_access import DataAccessDeniedError
from taqamind.core.interfaces.dataset_registry import (
    RecordLineageRegistry,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
)
from taqamind.core.interfaces.evaluation import FinalEvaluationResult, FrozenModelSelection
from taqamind.core.interfaces.training_registry import (
    EvaluationRegistry,
    EvaluationReservation,
    RecordUsage,
    RecordUsageRegistry,
    RecordUsageRole,
    SplitRegistry,
    TrainingIntent,
    TrainingRegistry,
    TrainingReuseKey,
    find_previously_exposed_test_records,
)
from taqamind.core.registry._concurrency import locked_path
from taqamind.core.registry._json_store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalTrainingRegistry(TrainingRegistry):
    """JSON-backed local training intent and reusable result registry."""

    def __init__(self, root_dir: Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _intent_path(self, *, factory_id: UUID, execution_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/intents/{execution_id}.json")

    def _result_path(self, *, factory_id: UUID, training_run_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/results/{training_run_id}.json")

    def _reuse_path(self, *, factory_id: UUID, fingerprint: str) -> Path:
        return safe_resolve(self._root, f"{factory_id}/reuse/{fingerprint}.json")

    def register_intent(
        self,
        *,
        factory_id: UUID,
        intent: TrainingIntent,
    ) -> None:
        if intent.context.factory_id != factory_id:
            raise RegistryScopeError("Training intent factory scope mismatch.")
        path = self._intent_path(factory_id=factory_id, execution_id=intent.context.execution_id)
        with locked_path(path.with_suffix(".lock")):
            if path.exists():
                existing = load_json_model(path, TrainingIntent)
                if existing != intent:
                    raise RegistryConflictError("Conflicting training intent already registered.")
                return
            atomic_write_text(path, dump_json_model(intent))

    def find_reusable_result(
        self,
        *,
        key: TrainingReuseKey,
    ) -> TrainingRunResult | None:
        reuse_path = self._reuse_path(factory_id=key.factory_id, fingerprint=key.fingerprint)
        if not reuse_path.exists():
            return None
        payload = json.loads(reuse_path.read_text(encoding="utf-8"))
        training_run_id = UUID(payload["training_run_id"])
        result_path = self._result_path(
            factory_id=key.factory_id,
            training_run_id=training_run_id,
        )
        if not result_path.exists():
            return None
        return load_json_model(result_path, TrainingRunResult)

    def register_result(
        self,
        *,
        factory_id: UUID,
        key: TrainingReuseKey,
        result: TrainingRunResult,
    ) -> None:
        if (
            key.factory_id != factory_id
            or result.development_split.context.factory_id != factory_id
        ):
            raise RegistryScopeError("Training result factory scope mismatch.")
        result_path = self._result_path(
            factory_id=factory_id,
            training_run_id=result.training_run_id,
        )
        reuse_path = self._reuse_path(factory_id=factory_id, fingerprint=key.fingerprint)
        factory_lock = safe_resolve(self._root, f"{factory_id}/.training_registry.lock")
        with locked_path(factory_lock):
            created_result = False
            if result_path.exists():
                existing = load_json_model(result_path, TrainingRunResult)
                if existing != result:
                    raise RegistryConflictError(
                        "Conflicting training result for existing training_run_id."
                    )
            if reuse_path.exists():
                existing_reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
                if UUID(existing_reuse["training_run_id"]) != result.training_run_id:
                    raise RegistryConflictError(
                        "TrainingReuseKey already bound to a different training_run_id."
                    )
            if not result_path.exists():
                atomic_write_text(result_path, dump_json_model(result))
                created_result = True
            if not reuse_path.exists():
                reuse_payload = json.dumps(
                    {
                        "training_run_id": str(result.training_run_id),
                        "fingerprint": key.fingerprint,
                    },
                    indent=2,
                )
                try:
                    atomic_write_text(reuse_path, reuse_payload)
                except BaseException:
                    if created_result:
                        result_path.unlink(missing_ok=True)
                    raise

    def get_result(
        self,
        *,
        factory_id: UUID,
        training_run_id: UUID,
    ) -> TrainingRunResult:
        path = self._result_path(factory_id=factory_id, training_run_id=training_run_id)
        if not path.exists():
            raise RegistryNotFoundError(f"Training run '{training_run_id}' not found.")
        return load_json_model(path, TrainingRunResult)


class LocalEvaluationRegistry(EvaluationRegistry):
    """JSON-backed protected evaluation reservation and result registry."""

    def __init__(
        self,
        *,
        root_dir: Path,
        split_registry: SplitRegistry,
        usage_registry: RecordUsageRegistry,
        lineage_registry: RecordLineageRegistry,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._split_registry = split_registry
        self._usage_registry = usage_registry
        self._lineage_registry = lineage_registry

    def _selection_path(self, *, factory_id: UUID, selection_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/selections/{selection_id}.json")

    def _reservation_path(self, *, factory_id: UUID, reservation_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/reservations/{reservation_id}.json")

    def _exposed_marker_path(self, *, factory_id: UUID, reservation_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/exposed/{reservation_id}.json")

    def _evaluation_path(self, *, factory_id: UUID, evaluation_id: UUID) -> Path:
        return safe_resolve(self._root, f"{factory_id}/evaluations/{evaluation_id}.json")

    def register_selection(
        self,
        *,
        factory_id: UUID,
        selection: FrozenModelSelection,
    ) -> None:
        if selection.training_run.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Selection factory scope mismatch.")
        path = self._selection_path(factory_id=factory_id, selection_id=selection.selection_id)
        with locked_path(path.with_suffix(".lock")):
            if path.exists():
                existing = load_json_model(path, FrozenModelSelection)
                if existing != selection:
                    raise RegistryConflictError(
                        "Conflicting FrozenModelSelection for existing selection_id."
                    )
                return
            atomic_write_text(path, dump_json_model(selection))

    def reserve_test_access(
        self,
        *,
        factory_id: UUID,
        reservation: EvaluationReservation,
    ) -> EvaluationReservation:
        if reservation.scope.factory_id != factory_id:
            raise RegistryScopeError("Reservation factory scope mismatch.")

        res_dir = safe_resolve(self._root, f"{factory_id}/reservations")
        lock_file = safe_resolve(self._root, f"{factory_id}/reserve.lock")
        with locked_path(lock_file):
            manifest = self._split_registry.get_split(
                factory_id=factory_id,
                split_id=reservation.split_id,
            )
            history = self._usage_registry.get_history(
                scope=reservation.scope,
                record_ids=reservation.test_record_ids,
            )
            exposed = find_previously_exposed_test_records(
                manifest=manifest,
                scope=reservation.scope,
                history=history,
            )
            if exposed:
                raise DataAccessDeniedError(
                    f"Protected test records already exposed in prior ML history: {exposed}"
                )

            requested = set(reservation.test_record_ids)
            for existing in iter_json_models(res_dir, EvaluationReservation):
                if existing.scope == reservation.scope and (
                    set(existing.test_record_ids) & requested
                ):
                    raise RegistryConflictError(
                        "Protected test records already reserved by another evaluation."
                    )

            path = self._reservation_path(
                factory_id=factory_id,
                reservation_id=reservation.reservation_id,
            )
            atomic_write_text(path, dump_json_model(reservation))
            return reservation

    def mark_test_exposed(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
    ) -> None:
        res_path = self._reservation_path(factory_id=factory_id, reservation_id=reservation_id)
        if not res_path.exists():
            raise RegistryNotFoundError(f"Reservation '{reservation_id}' not found.")
        reservation = load_json_model(res_path, EvaluationReservation)

        marker = self._exposed_marker_path(factory_id=factory_id, reservation_id=reservation_id)
        with locked_path(marker.with_suffix(".lock")):
            if marker.exists():
                raise DataAccessDeniedError(
                    "Protected test reservation was already exposed (single-use only)."
                )
            manifest = self._split_registry.get_split(
                factory_id=factory_id,
                split_id=reservation.split_id,
            )
            membership = self._lineage_registry.get_membership(
                scope=reservation.scope,
                dataset_revision_id=manifest.dataset_revision_id,
            )
            versions = {
                item.record_id: item.record_version_id for item in membership.memberships
            }
            usages = tuple(
                RecordUsage(
                    scope=reservation.scope,
                    record_id=rec_id,
                    record_version_id=versions[rec_id],
                    execution_id=reservation.execution_id,
                    split_id=reservation.split_id,
                    role=RecordUsageRole.PROTECTED_TEST,
                )
                for rec_id in reservation.test_record_ids
            )
            self._usage_registry.record_usage(scope=reservation.scope, usages=usages)
            atomic_write_text(
                marker,
                json.dumps({"reservation_id": str(reservation_id), "exposed": True}),
            )

    def register_evaluation(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> None:
        if evaluation.split_manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Evaluation factory scope mismatch.")
        res_path = self._reservation_path(factory_id=factory_id, reservation_id=reservation_id)
        if not res_path.exists():
            raise RegistryNotFoundError(f"Reservation '{reservation_id}' not found.")
        eval_path = self._evaluation_path(
            factory_id=factory_id,
            evaluation_id=evaluation.evaluation_id,
        )
        with locked_path(eval_path.with_suffix(".lock")):
            if eval_path.exists():
                existing = load_json_model(eval_path, FinalEvaluationResult)
                if existing != evaluation:
                    raise RegistryConflictError(
                        "Conflicting FinalEvaluationResult for existing evaluation_id."
                    )
                return
            atomic_write_text(eval_path, dump_json_model(evaluation))

    def get_evaluation(
        self,
        *,
        factory_id: UUID,
        evaluation_id: UUID,
    ) -> FinalEvaluationResult:
        eval_path = self._evaluation_path(factory_id=factory_id, evaluation_id=evaluation_id)
        if not eval_path.exists():
            raise RegistryNotFoundError(f"Evaluation '{evaluation_id}' not found.")
        return load_json_model(eval_path, FinalEvaluationResult)
