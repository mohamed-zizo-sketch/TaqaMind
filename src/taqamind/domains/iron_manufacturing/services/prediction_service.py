"""Strict train-once / predict-many prediction service for iron_manufacturing.

Architectural invariant:
    This module NEVER imports or calls any training service, AutoML trainer,
    or sklearn `fit` / `fit_transform` / `partial_fit` routine.
    If no compatible ACTIVE model exists for `(factory_id, domain, process, task_id)`,
    prediction returns a clear BLOCKED outcome and NEVER starts training.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    IndustrialTaskContract,
    ValidationReport,
)
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import ActiveModelProvider
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetMetadata
from taqamind.core.interfaces.dataset_registry import (
    DatasetRegistry,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.ml import PredictionEngine
from taqamind.core.interfaces.model_registry import ModelScope
from taqamind.core.interfaces.prediction import PredictionBatchResult, PredictionStatus
from taqamind.core.interfaces.processing import (
    ColumnMapper,
    DataCleaner,
    DataIngestion,
    FeatureEngineer,
    SemanticValidator,
)
from taqamind.core.interfaces.processing_models import ApprovedMapping
from taqamind.core.interfaces.results import ExecutionStatus
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.domains.iron_manufacturing.adapters.processing import _read_artifact_df
from taqamind.domains.iron_manufacturing.common.hashing import calculate_contract_sha256
from taqamind.domains.iron_manufacturing.common.normalization import safe_export_csv
from taqamind.domains.iron_manufacturing.services.lineage_service import (
    register_dataset_lineage,
)


@dataclass(frozen=True)
class PredictionOutcome:
    """Structured prediction outcome including blocked reasons and record provenance."""

    status: str  # "succeeded" | "partial" | "blocked" | "rejected"
    execution_id: UUID | None
    model_version_id: UUID | None
    model_version_number: int | None
    model_artifact_sha256: str | None
    batch_result: PredictionBatchResult | None
    validation_report: ValidationReport | None
    previously_known_record_count: int
    new_record_count: int
    blocked_reason: str | None
    output_csv_path: Path | None
    warnings: tuple[str, ...] = ()

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_id": str(self.execution_id) if self.execution_id else None,
            "model_version_id": str(self.model_version_id) if self.model_version_id else None,
            "model_version_number": self.model_version_number,
            "model_artifact_sha256": self.model_artifact_sha256,
            "previously_known_record_count": self.previously_known_record_count,
            "new_record_count": self.new_record_count,
            "successful_predictions": (
                self.batch_result.successful_count if self.batch_result else 0
            ),
            "rejected_predictions": (
                self.batch_result.rejected_count if self.batch_result else 0
            ),
            "target_name": self.batch_result.target_name if self.batch_result else None,
            "target_unit": self.batch_result.target_unit if self.batch_result else None,
            "blocked_reason": self.blocked_reason,
            "output_csv_path": str(self.output_csv_path) if self.output_csv_path else None,
            "warnings": list(self.warnings),
        }


class IronPredictionService:
    """Executes inference against a persisted ACTIVE model without ever retraining."""

    def __init__(
        self,
        *,
        dataset_storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        active_model_provider: ActiveModelProvider,
        ingestion: DataIngestion,
        column_mapper: ColumnMapper,
        semantic_validator: SemanticValidator,
        data_cleaner: DataCleaner,
        feature_engineer: FeatureEngineer,
        prediction_engine: PredictionEngine,
    ) -> None:
        self._dataset_storage = dataset_storage
        self._dataset_registry = dataset_registry
        self._lineage_registry = lineage_registry
        self._active_model_provider = active_model_provider
        self._ingestion = ingestion
        self._column_mapper = column_mapper
        self._semantic_validator = semantic_validator
        self._data_cleaner = data_cleaner
        self._feature_engineer = feature_engineer
        self._prediction_engine = prediction_engine

    def predict_csv(
        self,
        *,
        factory_id: UUID,
        task: IndustrialTaskContract,
        csv_path: Path | str,
        approved_by: str = "factory_operator",
        confirm_all_proposed_mappings: bool = False,
        confirmed_source_columns: tuple[str, ...] = (),
        output_csv_path: Path | str | None = None,
    ) -> PredictionOutcome:
        """Predict waste/scrap quantities using the factory's existing ACTIVE model."""
        contract: DatasetContract = task.dataset_contract
        scope = ModelScope(
            factory_id=factory_id,
            domain=contract.domain,
            process=contract.process,
            task_id=task.task_id,
        )

        # 1. Resolve active model — NEVER start training if missing!
        active_entry = self._active_model_provider.get_active(scope=scope)
        if active_entry is None:
            return PredictionOutcome(
                status="blocked",
                execution_id=None,
                model_version_id=None,
                model_version_number=None,
                model_artifact_sha256=None,
                batch_result=None,
                validation_report=None,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason=(
                    f"No active model exists for factory '{factory_id}', "
                    f"process '{contract.process}', task '{task.task_id}'. "
                    "Prediction never triggers automatic training; run initial training first."
                ),
                output_csv_path=None,
            )

        # 2. Verify contract compatibility against persisted model version
        current_contract_sha = calculate_contract_sha256(contract)
        if active_entry.version.contract_sha256 != current_contract_sha:
            return PredictionOutcome(
                status="blocked",
                execution_id=None,
                model_version_id=active_entry.version.model_version_id,
                model_version_number=active_entry.version.version_number,
                model_artifact_sha256=active_entry.version.artifact.content_sha256,
                batch_result=None,
                validation_report=None,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason=(
                    "Active model contract hash does not match current DatasetContract; "
                    "incompatible contract rejected."
                ),
                output_csv_path=None,
            )

        # 3. Create prediction PipelineContext bound to active model version
        raw_bytes = Path(csv_path).read_bytes()
        source_sha = calculate_bytes_sha256(raw_bytes)
        raw_revision_id = uuid4()
        context = PipelineContext.from_task(
            factory_id=factory_id,
            dataset_revision_id=raw_revision_id,
            task=task,
            purpose=DatasetPurpose.PREDICTION,
            model_version_id=active_entry.version.model_version_id,
        )

        raw_ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=source_sha,
            size_bytes=len(raw_bytes),
        )
        self._dataset_storage.put(
            factory_id=factory_id,
            reference=raw_ref,
            stream=io.BytesIO(raw_bytes),
        )
        raw_artifact = DatasetArtifact(
            metadata=DatasetMetadata.create_raw(context=context, source_file_sha256=source_sha),
            reference=raw_ref,
        )
        self._dataset_registry.register(factory_id=factory_id, artifact=raw_artifact)

        # 4. Run processing pipeline (INGEST -> MAP -> VALIDATE -> CLEAN -> ENGINEER)
        ingest_res = self._ingestion.ingest(
            context=context,
            artifact=raw_artifact,
            contract=contract,
        )
        if ingest_res.status != ExecutionStatus.SUCCEEDED or ingest_res.output_artifact is None:
            return PredictionOutcome(
                status="blocked",
                execution_id=context.execution_id,
                model_version_id=active_entry.version.model_version_id,
                model_version_number=active_entry.version.version_number,
                model_artifact_sha256=active_entry.version.artifact.content_sha256,
                batch_result=None,
                validation_report=ingest_res.report,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason="Prediction CSV failed ingestion validation.",
                output_csv_path=None,
            )

        proposal = self._column_mapper.propose_mapping(
            context=context,
            artifact=ingest_res.output_artifact,
            contract=contract,
        )
        confirmed = (
            tuple(e.source_column for e in proposal.entries if e.requires_confirmation)
            if confirm_all_proposed_mappings
            else confirmed_source_columns
        )
        try:
            approved = ApprovedMapping(
                proposal=proposal,
                approved_by=approved_by,
                confirmed_source_columns=confirmed,
            )
            map_res = self._column_mapper.apply_mapping(
                context=context,
                artifact=ingest_res.output_artifact,
                contract=contract,
                mapping=approved,
            )
        except ValueError as exc:
            return PredictionOutcome(
                status="blocked",
                execution_id=context.execution_id,
                model_version_id=active_entry.version.model_version_id,
                model_version_number=active_entry.version.version_number,
                model_artifact_sha256=active_entry.version.artifact.content_sha256,
                batch_result=None,
                validation_report=None,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason=f"Prediction column mapping rejected: {exc}",
                output_csv_path=None,
            )

        if map_res.status != ExecutionStatus.SUCCEEDED or map_res.output_artifact is None:
            return PredictionOutcome(
                status="blocked",
                execution_id=context.execution_id,
                model_version_id=active_entry.version.model_version_id,
                model_version_number=active_entry.version.version_number,
                model_artifact_sha256=active_entry.version.artifact.content_sha256,
                batch_result=None,
                validation_report=map_res.report,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason="Prediction CSV failed column mapping / unit conversion.",
                output_csv_path=None,
            )

        sem_res = self._semantic_validator.validate(
            context=context,
            artifact=map_res.output_artifact,
            contract=contract,
        )
        if sem_res.status != ExecutionStatus.SUCCEEDED:
            return PredictionOutcome(
                status="blocked",
                execution_id=context.execution_id,
                model_version_id=active_entry.version.model_version_id,
                model_version_number=active_entry.version.version_number,
                model_artifact_sha256=active_entry.version.artifact.content_sha256,
                batch_result=None,
                validation_report=sem_res.report,
                previously_known_record_count=0,
                new_record_count=0,
                blocked_reason="Prediction CSV failed semantic validation.",
                output_csv_path=None,
            )

        clean_res = self._data_cleaner.clean(
            context=context,
            artifact=map_res.output_artifact,
            contract=contract,
        )
        assert clean_res.output_artifact is not None

        eng_res = self._feature_engineer.engineer(
            context=context,
            artifact=clean_res.output_artifact,
            contract=contract,
        )
        assert eng_res.output_artifact is not None

        # 5. Match record identities via trusted business keys (distinguishing old vs new rows)
        eng_df = _read_artifact_df(self._dataset_storage, factory_id, eng_res.output_artifact)
        lineage_summary = register_dataset_lineage(
            context=context,
            artifact=eng_res.output_artifact,
            df=eng_df,
            contract=contract,
            lineage_registry=self._lineage_registry,
            register_new_records=True,
        )

        # 6. Execute inference using saved ACTIVE model only (zero fit/training)
        batch_result = self._prediction_engine.predict(
            context=context,
            artifact=eng_res.output_artifact,
            contract=contract,
            model=active_entry.version.identity,
        )

        warnings: list[str] = []
        if lineage_summary.exact_matches > 0:
            warnings.append(
                f"{lineage_summary.exact_matches} record(s) matched previously registered "
                "identities; inference generated without retraining and must NOT be treated "
                "as unseen evaluation evidence."
            )

        exported_path: Path | None = None
        if output_csv_path is not None:
            out_rows: list[dict[str, Any]] = []
            bkey_col = contract.business_keys[0] if contract.business_keys else None
            for idx, rec_pred in enumerate(batch_result.predictions):
                row_dict: dict[str, Any] = {
                    "record_id": str(rec_pred.record_id),
                    "business_key": str(eng_df.iloc[idx][bkey_col]) if bkey_col else None,
                    "record_history_status": lineage_summary.row_match_statuses[idx].value,
                    "prediction_status": rec_pred.status.value,
                    f"predicted_{batch_result.target_name}": (
                        round(rec_pred.predicted_value, 6)
                        if rec_pred.status == PredictionStatus.SUCCEEDED
                        and rec_pred.predicted_value is not None
                        else None
                    ),
                    "target_unit": batch_result.target_unit,
                    "derived_rate_value": (
                        round(rec_pred.derived_rate.value, 6)
                        if rec_pred.derived_rate and rec_pred.derived_rate.value is not None
                        else None
                    ),
                    "derived_rate_unit": (
                        rec_pred.derived_rate.unit if rec_pred.derived_rate else None
                    ),
                    "derived_rate_undefined_reason": (
                        rec_pred.derived_rate.undefined_reason if rec_pred.derived_rate else None
                    ),
                    "model_version_id": str(active_entry.version.model_version_id),
                    "model_version_number": active_entry.version.version_number,
                    "execution_id": str(context.execution_id),
                    "warnings": "; ".join(rec_pred.warnings),
                    "rejection_reason": rec_pred.rejection_reason or "",
                }
                out_rows.append(row_dict)
            exported_path = safe_export_csv(pd.DataFrame(out_rows), Path(output_csv_path))

        return PredictionOutcome(
            status=batch_result.status.value,
            execution_id=context.execution_id,
            model_version_id=active_entry.version.model_version_id,
            model_version_number=active_entry.version.version_number,
            model_artifact_sha256=active_entry.version.artifact.content_sha256,
            batch_result=batch_result,
            validation_report=sem_res.report,
            previously_known_record_count=lineage_summary.exact_matches,
            new_record_count=lineage_summary.new_records,
            blocked_reason=None,
            output_csv_path=exported_path,
            warnings=tuple(warnings),
        )
