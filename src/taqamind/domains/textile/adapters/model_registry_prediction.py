"""Model registry, active model provider, and no-fit PredictionEngine adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

import joblib
import numpy as np
import pandas as pd

from taqamind.core.contracts import DatasetContract, DataType
from taqamind.core.interfaces import (
    ActiveModelProvider,
    DataAccessDeniedError,
    DatasetArtifact,
    DerivedRatePrediction,
    FinalEvaluationResult,
    ModelArtifactStorage,
    ModelRegistry,
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionBatchResult,
    PredictionEngine,
    PredictionModelIdentity,
    PredictionStatus,
    RecordLineageRegistry,
    RecordPrediction,
    RecordScope,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    StageTiming,
    validate_active_model_access,
    validate_model_transition,
    validate_prediction_request,
)
from taqamind.core.preprocessing import TabularPreprocessor
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.textile.adapters._store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalModelRegistry(ModelRegistry):
    """Factory/task-scoped lifecycle registry enforcing single active model invariant.

    Migration path: Replace with Shared Core P1.7 `LocalModelRegistry`.
    """

    def __init__(
        self,
        registry_dir: str | Path,
        *,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._root = Path(registry_dir).resolve()
        self._model_storage = model_storage

    def _scope_dir(self, scope: ModelScope) -> Path:
        return safe_resolve(
            self._root,
            "models",
            str(scope.factory_id),
            scope.domain,
            scope.process,
            scope.task_id,
        )

    def _entry_path(self, scope: ModelScope, model_version_id: UUID) -> Path:
        return safe_resolve(self._scope_dir(scope), "versions", f"{model_version_id}.json")

    def register_candidate(
        self, *, scope: ModelScope, version: ModelVersion
    ) -> ModelRegistryEntry:
        scope.validate_identity(version.identity)
        with self._model_storage.open_verified(
            factory_id=scope.factory_id, reference=version.artifact
        ):
            pass

        path = self._entry_path(scope, version.model_version_id)
        if path.is_file():
            existing = load_json_model(path, ModelRegistryEntry)
            if existing.version == version:
                return existing
            raise RegistryConflictError("Conflicting model version metadata.")

        entry = ModelRegistryEntry(version=version, status=ModelStatus.CANDIDATE)
        atomic_write_text(path, dump_json_model(entry))
        return entry

    def apply_evaluation(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> ModelRegistryEntry:
        entry = self.get(scope=scope, model_version_id=model_version_id)
        updated = entry.apply_evaluation(evaluation)
        path = self._entry_path(scope, model_version_id)
        atomic_write_text(path, dump_json_model(updated))
        return updated

    def get(self, *, scope: ModelScope, model_version_id: UUID) -> ModelRegistryEntry:
        path = self._entry_path(scope, model_version_id)
        if not path.is_file():
            raise RegistryNotFoundError(f"Model version '{model_version_id}' not found.")
        entry = load_json_model(path, ModelRegistryEntry)
        scope.validate_identity(entry.version.identity)
        return entry

    def get_active(self, *, scope: ModelScope) -> ModelRegistryEntry | None:
        actives = [
            entry
            for entry in self.list_versions(scope=scope)
            if entry.status == ModelStatus.ACTIVE
        ]
        if len(actives) > 1:
            raise RegistryConflictError("Multiple active models found for single scope.")
        return actives[0] if actives else None

    def list_versions(self, *, scope: ModelScope) -> tuple[ModelRegistryEntry, ...]:
        versions_dir = safe_resolve(self._scope_dir(scope), "versions")
        entries = iter_json_models(versions_dir, ModelRegistryEntry)
        entries.sort(key=lambda e: (e.version.version_number, str(e.version.model_version_id)))
        return tuple(entries)

    def transition(self, *, request: ModelTransitionRequest) -> ModelRegistryEntry:
        scope = request.scope
        target = self.get(scope=scope, model_version_id=request.target_model_version_id)
        active = self.get_active(scope=scope)

        validate_model_transition(request=request, target=target, active=active)
        with self._model_storage.open_verified(
            factory_id=scope.factory_id, reference=target.version.artifact
        ):
            pass

        if active is not None:
            archived = ModelRegistryEntry(
                version=active.version,
                status=ModelStatus.ARCHIVED,
                evaluation_id=active.evaluation_id,
            )
            atomic_write_text(
                self._entry_path(scope, active.version.model_version_id),
                dump_json_model(archived),
            )

        activated = ModelRegistryEntry(
            version=target.version,
            status=ModelStatus.ACTIVE,
            evaluation_id=target.evaluation_id,
        )
        atomic_write_text(
            self._entry_path(scope, target.version.model_version_id),
            dump_json_model(activated),
        )

        audit_path = safe_resolve(
            self._scope_dir(scope), "audit", f"{request.request_id}.json"
        )
        atomic_write_text(
            audit_path,
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "request": request.model_dump(mode="json"),
                    "previous_active_id": (
                        str(active.version.model_version_id) if active is not None else None
                    ),
                    "new_active_id": str(activated.version.model_version_id),
                },
                indent=2,
            ),
        )
        return activated


class LocalActiveModelProvider(ActiveModelProvider):
    """Read-only verified bundle provider for active models.

    Migration path: Replace with Shared Core P1.8 `ActiveModelProvider`.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._registry = model_registry
        self._storage = model_storage

    def get_active(self, *, scope: ModelScope) -> ModelRegistryEntry | None:
        return self._registry.get_active(scope=scope)

    @contextmanager
    def open_bundle(
        self,
        *,
        context: PipelineContext,
        entry: ModelRegistryEntry,
        contract: DatasetContract,
    ) -> Iterator[BinaryIO]:
        validate_active_model_access(context=context, entry=entry, contract=contract)
        with self._storage.open_verified(
            factory_id=context.factory_id, reference=entry.version.artifact
        ) as stream:
            yield stream


class TextilePredictionEngine(PredictionEngine):
    """Train-once / predict-many inference engine without any fit/retraining calls.

    Migration path: Replace with Shared Core P1.8 `PredictionEngine`.
    """

    def __init__(
        self,
        *,
        dataset_storage: LocalDatasetArtifactStorage,
        model_registry: ModelRegistry,
        active_model_provider: ActiveModelProvider,
        lineage_registry: RecordLineageRegistry | None = None,
        identity_namespace: str = "production_record",
    ) -> None:
        self._dataset_storage = dataset_storage
        self._model_registry = model_registry
        self._active_provider = active_model_provider
        self._lineage = lineage_registry
        self._identity_namespace = identity_namespace

    def predict(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        model: PredictionModelIdentity,
    ) -> PredictionBatchResult:
        started_at = datetime.now(UTC)
        validate_prediction_request(
            context=context, artifact=artifact, contract=contract, model=model
        )

        scope = ModelScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
        )
        active = self._active_provider.get_active(scope=scope)
        if active is None:
            raise DataAccessDeniedError("No active model registered for this factory and task.")
        if active.version.model_version_id != model.model_version_id:
            raise RegistryScopeError(
                "Requested model_version_id does not match the current active model."
            )

        with self._active_provider.open_bundle(
            context=context, entry=active, contract=contract
        ) as bundle_stream:
            bundle = joblib.load(bundle_stream)

        expected_features = tuple(active.version.feature_specification.feature_names)
        if tuple(bundle.get("feature_names", ())) != expected_features:
            raise ValueError("Model bundle feature order mismatch against ModelVersion.")

        dtype_hints = {
            f.name: "string"
            for f in contract.all_columns
            if f.data_type in {DataType.STRING, DataType.CATEGORY, DataType.DATE, DataType.DATETIME}
        }
        with self._dataset_storage.open_verified(
            factory_id=context.factory_id, reference=artifact.reference
        ) as data_stream:
            df = pd.read_csv(data_stream, keep_default_na=True, dtype=dtype_hints)

        # Resolve registered record IDs from membership if available.
        record_ids: list[UUID] | None = None
        if self._lineage is not None:
            rec_scope = RecordScope(
                factory_id=context.factory_id,
                domain=context.domain,
                process=context.process,
                dataset_id=context.dataset_id,
                identity_namespace=self._identity_namespace,
            )
            try:
                manifest = self._lineage.get_membership(
                    scope=rec_scope, dataset_revision_id=artifact.metadata.revision_id
                )
                manifest.validate_artifact(artifact)
                record_ids = [item.record_id for item in manifest.memberships]
            except RegistryNotFoundError:
                record_ids = None

        if record_ids is None:
            from uuid import uuid4

            record_ids = [uuid4() for _ in range(len(df))]

        # Strictly transform-only inference (NO fit / fit_transform / partial_fit).
        if bundle.get("bundle_kind") == "historical_rate":
            denom_col = bundle["denominator_column"]
            X_pred = np.asarray(
                pd.to_numeric(df[denom_col], errors="coerce"), dtype=float
            ).reshape(-1, 1)
            raw_preds = np.asarray(bundle["estimator"].predict(X_pred), dtype=float)
        else:
            preprocessor: TabularPreprocessor = bundle["preprocessor"]
            if not preprocessor.is_fitted:
                raise RuntimeError("Loaded preprocessor is not fitted.")
            ordered_df = df.loc[:, list(expected_features)]
            X_pred = preprocessor.transform(ordered_df)
            raw_preds = np.asarray(bundle["estimator"].predict(X_pred), dtype=float)

        target_constraints = contract.target.feature.constraints
        rate_def = contract.target.derived_rate

        row_predictions: list[RecordPrediction] = []
        for idx, pred_val in enumerate(raw_preds):
            rid = record_ids[idx]
            val = float(pred_val)
            if not np.isfinite(val):
                row_predictions.append(
                    RecordPrediction(
                        record_id=rid,
                        status=PredictionStatus.REJECTED,
                        rejection_reason="Model produced non-finite prediction.",
                    )
                )
                continue

            if target_constraints.minimum is not None:
                min_v = target_constraints.minimum
                if val < min_v and abs(val - min_v) < 1e-9:
                    val = float(min_v)
                if val < min_v or (val == min_v and not target_constraints.minimum_inclusive):
                    row_predictions.append(
                        RecordPrediction(
                            record_id=rid,
                            status=PredictionStatus.REJECTED,
                            rejection_reason=(
                                "Predicted quantity violates target minimum constraint."
                            ),
                        )
                    )
                    continue

            if target_constraints.maximum is not None and (
                val > target_constraints.maximum
                or (val == target_constraints.maximum and not target_constraints.maximum_inclusive)
            ):
                row_predictions.append(
                    RecordPrediction(
                        record_id=rid,
                        status=PredictionStatus.REJECTED,
                        rejection_reason="Predicted quantity violates target maximum constraint.",
                    )
                )
                continue

            derived_rate: DerivedRatePrediction | None = None
            if rate_def is not None:
                denom_name = rate_def.denominator
                denom_val = (
                    float(pd.to_numeric(df.iloc[idx][denom_name], errors="coerce"))
                    if denom_name in df.columns
                    else float("nan")
                )
                if not np.isfinite(denom_val) or denom_val <= 0:
                    derived_rate = DerivedRatePrediction(
                        value=None,
                        denominator_value=None if not np.isfinite(denom_val) else denom_val,
                        unit=rate_def.output_unit,
                        undefined_reason=(
                            f"Denominator '{denom_name}' is missing or non-positive."
                        ),
                    )
                else:
                    rate_value = (val / denom_val) * rate_def.multiplier
                    derived_rate = DerivedRatePrediction(
                        value=rate_value,
                        denominator_value=denom_val,
                        unit=rate_def.output_unit,
                    )

            row_predictions.append(
                RecordPrediction(
                    record_id=rid,
                    status=PredictionStatus.SUCCEEDED,
                    predicted_value=val,
                    derived_rate=derived_rate,
                )
            )

        return PredictionBatchResult(
            context=context,
            model=active.version.identity,
            contract=contract,
            input_artifact=artifact,
            predictions=tuple(row_predictions),
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
