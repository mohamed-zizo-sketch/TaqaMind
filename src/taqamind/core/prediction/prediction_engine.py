"""Core P1.8 LocalActiveModelProvider and CorePredictionEngine (zero-retraining)."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import BinaryIO

import numpy as np
import pandas as pd

from taqamind.core.automl.data_access import convert_dataframe_by_contract
from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.ingestion.artifact_csv import read_verified_csv
from taqamind.core.ingestion.resource_policy import CsvResourcePolicy, resolve_internal_csv_policy
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import (
    ActiveModelProvider,
    DataAccessDeniedError,
    validate_active_model_access,
)
from taqamind.core.interfaces.dataset import DatasetArtifact
from taqamind.core.interfaces.dataset_registry import (
    RecordLineageRegistry,
    RegistryNotFoundError,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.ml import PredictionEngine
from taqamind.core.interfaces.model_registry import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelScope,
)
from taqamind.core.interfaces.prediction import (
    DerivedRatePrediction,
    PredictionBatchResult,
    PredictionModelIdentity,
    PredictionStatus,
    RecordPrediction,
)
from taqamind.core.interfaces.results import StageTiming
from taqamind.core.interfaces.storage import DatasetArtifactStorage, ModelArtifactStorage
from taqamind.core.lineage.membership_builder import RecordMembershipBuilder
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


class LocalActiveModelProvider(ActiveModelProvider):
    """Read-only active model provider verifying scope, status, and SHA-256 integrity."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._model_registry = model_registry
        self._model_storage = model_storage

    def get_active(self, *, scope: ModelScope) -> ModelRegistryEntry | None:
        return self._model_registry.get_active(scope=scope)

    @contextmanager
    def open_bundle(
        self,
        *,
        context: PipelineContext,
        entry: ModelRegistryEntry,
        contract: DatasetContract,
    ) -> Iterator[BinaryIO]:
        validate_active_model_access(context=context, entry=entry, contract=contract)
        with self._model_storage.open_verified(
            factory_id=context.factory_id,
            reference=entry.version.artifact,
        ) as stream:
            yield stream


class CorePredictionEngine(PredictionEngine):
    """Generates batch predictions strictly from an existing active model without fitting."""

    def __init__(
        self,
        *,
        active_model_provider: ActiveModelProvider,
        dataset_storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        csv_reader: SafeCsvReader | None = None,
        resource_policy: CsvResourcePolicy | None = None,
        type_converter: ContractTypeConverter | None = None,
        identity_namespace: str = "production_record",
        membership_builder: RecordMembershipBuilder | None = None,
    ) -> None:
        self._provider = active_model_provider
        self._dataset_storage = dataset_storage
        self._lineage_registry = lineage_registry
        self._resource_policy = resolve_internal_csv_policy(resource_policy)
        self._csv_reader = (csv_reader or SafeCsvReader()).bounded_by(self._resource_policy)
        self._converter = type_converter or ContractTypeConverter()
        self._identity_namespace = identity_namespace
        self._membership_builder = membership_builder or RecordMembershipBuilder(
            self._dataset_storage,
            self._lineage_registry,
            identity_namespace=self._identity_namespace,
            resource_policy=self._resource_policy,
        )

    def predict(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        model: PredictionModelIdentity,
    ) -> PredictionBatchResult:
        started_at = datetime.now(UTC)
        model.validate_context(context)
        context.validate_dataset_contract(contract)
        artifact.metadata.validate_scope(context)

        scope = ModelScope.from_identity(model)
        active = self._provider.get_active(scope=scope)
        if active is None:
            raise DataAccessDeniedError("No active model is registered for this scope.")
        if active.version.model_version_id != model.model_version_id:
            raise DataAccessDeniedError(
                "Requested model version is not the currently active model."
            )

        with self._provider.open_bundle(
            context=context,
            entry=active,
            contract=contract,
        ) as bundle_stream:
            bundle = pickle.loads(bundle_stream.read())

        expected_predictors = active.version.feature_specification.feature_names
        if tuple(bundle.get("predictor_names", ())) != expected_predictors:
            raise DataAccessDeniedError("Model bundle predictor specification mismatch.")

        raw_df = read_verified_csv(
            storage=self._dataset_storage,
            factory_id=context.factory_id,
            reference=artifact.reference,
            reader=self._csv_reader,
            policy=self._resource_policy,
        )

        df = convert_dataframe_by_contract(
            raw_df,
            contract=contract,
            converter=self._converter,
        )

        rec_scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=self._identity_namespace,
        )
        if hasattr(self._lineage_registry, "get_prediction_observations"):
            try:
                obs_batch = self._lineage_registry.get_prediction_observations(
                    scope=rec_scope,
                    dataset_revision_id=artifact.metadata.revision_id,
                )
            except RegistryNotFoundError:
                obs_batch = self._membership_builder.resolve_prediction_observations(
                    scope=rec_scope,
                    artifact=artifact,
                    contract=contract,
                    context=context,
                )
        else:
            obs_batch = self._membership_builder.resolve_prediction_observations(
                scope=rec_scope,
                artifact=artifact,
                contract=contract,
                context=context,
            )
        if len(obs_batch.observations) != len(df):
            raise DataAccessDeniedError(
                "Prediction dataset row count does not match resolved observations."
            )

        preprocessor = bundle["preprocessor"]
        estimator = bundle["estimator"]
        X = preprocessor.transform(df)
        y_pred = np.asarray(estimator.predict(X), dtype=float)
        if bundle.get("must_be_non_negative", False):
            y_pred = np.maximum(0.0, y_pred)

        rate_def = contract.target.derived_rate
        denom_series = (
            pd.to_numeric(df[rate_def.denominator], errors="coerce").to_numpy(dtype=float)
            if rate_def is not None and rate_def.denominator in df.columns
            else None
        )

        predictions: list[RecordPrediction] = []
        for idx, obs in enumerate(obs_batch.observations):
            val = float(y_pred[idx])
            if not np.isfinite(val):
                predictions.append(
                    RecordPrediction(
                        record_id=obs.known_record_id,
                        observation_id=obs.observation_id,
                        row_position=obs.row_position,
                        status=PredictionStatus.REJECTED,
                        rejection_reason="NON_FINITE_PREDICTION",
                    )
                )
                continue

            derived_rate: DerivedRatePrediction | None = None
            if rate_def is not None:
                denom_val = float(denom_series[idx]) if denom_series is not None else float("nan")
                if not np.isfinite(denom_val) or denom_val <= 0.0:
                    derived_rate = DerivedRatePrediction(
                        value=None,
                        denominator_value=None if not np.isfinite(denom_val) else denom_val,
                        unit=rate_def.output_unit,
                        undefined_reason="NON_POSITIVE_DENOMINATOR",
                    )
                else:
                    rate_val = (val / denom_val) * rate_def.multiplier
                    if rate_val < 0.0:
                        rate_val = 0.0
                    derived_rate = DerivedRatePrediction(
                        value=float(rate_val),
                        denominator_value=denom_val,
                        unit=rate_def.output_unit,
                    )

            predictions.append(
                RecordPrediction(
                    record_id=obs.known_record_id,
                    observation_id=obs.observation_id,
                    row_position=obs.row_position,
                    status=PredictionStatus.SUCCEEDED,
                    predicted_value=val,
                    derived_rate=derived_rate,
                )
            )

        finished_at = datetime.now(UTC)
        return PredictionBatchResult(
            context=context,
            model=model,
            contract=contract,
            input_artifact=artifact,
            predictions=tuple(predictions),
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )


__all__ = ["CorePredictionEngine", "LocalActiveModelProvider"]
