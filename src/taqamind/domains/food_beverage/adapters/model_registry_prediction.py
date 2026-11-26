"""P1.8: Factory-scoped ModelRegistry, ActiveModelProvider, and no-fit PredictionEngine."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import joblib
import numpy as np
import pandas as pd

from taqamind.core.contracts import DatasetContract, DatasetPurpose
from taqamind.core.interfaces import (
    DatasetArtifact,
    DatasetState,
    DerivedRatePrediction,
    EvaluationDecision,
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionBatchResult,
    PredictionStatus,
    RecordPrediction,
    StageTiming,
    validate_active_model_access,
    validate_model_transition,
    validate_prediction_request,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage, ModelArtifactStorage
from taqamind.domains.food_beverage.adapters._store import (
    atomic_write_model,
    load_model_json,
)
from taqamind.domains.food_beverage.adapters.membership import RowLineageSummary


class LocalModelRegistry:
    """Persistent JSON-backed factory-scoped ModelRegistry.

    Enforces single ACTIVE model per (factory_id, domain, process, task_id) and
    strips @computed_field attributes cleanly on serialization.
    Candidate for migration to ``taqamind.core.registry``.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve() / "models"
        self._root.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, scope: ModelScope, model_version_id: UUID) -> Path:
        return (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / scope.task_id
            / f"{model_version_id}.json"
        )

    def _scope_dir(self, scope: ModelScope) -> Path:
        return (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / scope.task_id
        )

    def list_entries(self, scope: ModelScope) -> list[ModelRegistryEntry]:
        sdir = self._scope_dir(scope)
        if not sdir.exists():
            return []
        entries = [load_model_json(p, ModelRegistryEntry) for p in sdir.glob("*.json")]
        return sorted(entries, key=lambda e: e.version.version_number)

    def next_version_number(self, scope: ModelScope) -> int:
        entries = self.list_entries(scope)
        return max((e.version.version_number for e in entries), default=0) + 1

    def get_active_model(self, scope: ModelScope) -> ModelRegistryEntry | None:
        for entry in self.list_entries(scope):
            if entry.status == ModelStatus.ACTIVE:
                return entry
        return None

    def get_entry(self, scope: ModelScope, model_version_id: UUID) -> ModelRegistryEntry:
        path = self._entry_path(scope, model_version_id)
        if not path.exists():
            raise KeyError(f"Model version {model_version_id} not found.")
        return load_model_json(path, ModelRegistryEntry)

    def register_evaluated(
        self,
        *,
        version: ModelVersion,
        evaluation_id: UUID,
        decision: EvaluationDecision,
    ) -> ModelRegistryEntry:
        path = self._entry_path(version.scope, version.model_version_id)
        if path.exists():
            raise ValueError(f"Model version {version.model_version_id} already exists.")
        status = (
            ModelStatus.VALIDATED
            if decision == EvaluationDecision.ACCEPTED
            else ModelStatus.REJECTED
        )
        entry = ModelRegistryEntry(
            version=version,
            status=status,
            evaluation_id=evaluation_id,
        )
        atomic_write_model(path, entry)
        return entry

    def transition(self, request: ModelTransitionRequest) -> ModelRegistryEntry:
        target = self.get_entry(request.scope, request.target_model_version_id)
        active = self.get_active_model(request.scope)
        validate_model_transition(request=request, target=target, active=active)

        if (
            active is not None
            and active.version.model_version_id != target.version.model_version_id
        ):
            archived = ModelRegistryEntry(
                version=active.version,
                status=ModelStatus.ARCHIVED,
                evaluation_id=active.evaluation_id,
                updated_at=datetime.now(UTC),
            )
            atomic_write_model(
                self._entry_path(request.scope, active.version.model_version_id),
                archived,
            )

        activated = ModelRegistryEntry(
            version=target.version,
            status=ModelStatus.ACTIVE,
            evaluation_id=target.evaluation_id,
            updated_at=datetime.now(UTC),
        )
        atomic_write_model(
            self._entry_path(request.scope, target.version.model_version_id),
            activated,
        )
        return activated


class LocalActiveModelProvider:
    """Resolves and validates the active model entry for a factory/process/task scope.

    Candidate for migration to ``taqamind.core.data_access``.
    """

    def __init__(self, registry: LocalModelRegistry) -> None:
        self._registry = registry

    def get_verified_active(
        self,
        *,
        context: PipelineContext,
        contract: DatasetContract,
    ) -> ModelRegistryEntry:
        scope = ModelScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
        )
        entry = self._registry.get_active_model(scope)
        if entry is None:
            raise ValueError(
                f"No ACTIVE model registered for factory={context.factory_id}, "
                f"process={context.process}, task={context.task_id}."
            )
        validate_active_model_access(context=context, entry=entry, contract=contract)
        return entry


class FoodBeveragePredictionEngine:
    """Strictly transform-only PredictionEngine (zero .fit() or .fit_transform() calls).

    Candidate for migration to ``taqamind.core.prediction``.
    """

    def __init__(
        self,
        *,
        dataset_storage: DatasetArtifactStorage,
        model_storage: ModelArtifactStorage,
        active_provider: LocalActiveModelProvider,
    ) -> None:
        self._dataset_storage = dataset_storage
        self._model_storage = model_storage
        self._active_provider = active_provider

    def predict(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        lineage: RowLineageSummary,
    ) -> PredictionBatchResult:
        started_at = datetime.now(UTC)

        if context.purpose != DatasetPurpose.PREDICTION:
            raise ValueError("PredictionEngine requires DatasetPurpose.PREDICTION.")
        if artifact.metadata.state != DatasetState.ENGINEERED:
            raise ValueError("PredictionEngine requires an ENGINEERED dataset artifact.")

        entry = self._active_provider.get_verified_active(context=context, contract=contract)
        version = entry.version
        validate_prediction_request(
            context=context,
            artifact=artifact,
            contract=contract,
            model=version.identity,
        )

        with self._model_storage.open_verified(
            factory_id=version.identity.factory_id,
            reference=version.artifact,
        ) as stream:
            bundle = joblib.load(stream)

        preprocessor = bundle["preprocessor"]
        estimator = bundle["estimator"]

        with self._dataset_storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        # STRICTLY TRANSFORM ONLY — never fit or fit_transform
        X_t = preprocessor.transform(df)
        raw_preds = np.asarray(estimator.predict(X_t), dtype=float)

        target_constraints = contract.target.feature.constraints
        rate_def = contract.target.derived_rate

        record_predictions: list[RecordPrediction] = []
        for idx, row in df.iterrows():
            i = int(idx)
            rec_id = lineage.row_to_record_id[i]
            pred_val = float(raw_preds[i]) if i < len(raw_preds) else float("nan")

            rejection_reason: str | None = None
            if not math.isfinite(pred_val):
                rejection_reason = "Non-finite model prediction."
            elif target_constraints.minimum is not None and pred_val < target_constraints.minimum:
                rejection_reason = (
                    f"Predicted value {pred_val:.4f} violates target minimum "
                    f"{target_constraints.minimum}."
                )

            if rejection_reason is not None:
                record_predictions.append(
                    RecordPrediction(
                        record_id=rec_id,
                        status=PredictionStatus.REJECTED,
                        rejection_reason=rejection_reason,
                    )
                )
                continue

            derived_rate: DerivedRatePrediction | None = None
            if rate_def is not None:
                denom_col = rate_def.denominator
                denom_val = (
                    float(row[denom_col])
                    if denom_col in df.columns and pd.notna(row[denom_col])
                    else None
                )
                if denom_val is not None and math.isfinite(denom_val) and denom_val > 0.0:
                    rate_val = (pred_val / denom_val) * rate_def.multiplier
                    derived_rate = DerivedRatePrediction(
                        value=rate_val,
                        denominator_value=denom_val,
                        unit=rate_def.output_unit,
                    )
                else:
                    derived_rate = DerivedRatePrediction(
                        value=None,
                        denominator_value=denom_val,
                        unit=rate_def.output_unit,
                        undefined_reason=f"Denominator '{denom_col}' is missing or non-positive.",
                    )

            record_predictions.append(
                RecordPrediction(
                    record_id=rec_id,
                    status=PredictionStatus.SUCCEEDED,
                    predicted_value=pred_val,
                    derived_rate=derived_rate,
                )
            )

        finished_at = datetime.now(UTC)
        return PredictionBatchResult(
            context=context,
            model=version.identity,
            contract=contract,
            input_artifact=artifact,
            predictions=tuple(record_predictions),
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )
