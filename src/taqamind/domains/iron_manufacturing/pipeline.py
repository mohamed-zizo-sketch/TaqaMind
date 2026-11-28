"""High-level injectable pipeline facade for the iron_manufacturing domain."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from taqamind.core.contracts import (
    DatasetPurpose,
    DomainContract,
    TargetConfirmation,
)
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetMetadata
from taqamind.core.interfaces.evaluation import AcceptancePolicy
from taqamind.core.interfaces.model_registry import (
    ModelRegistryEntry,
    ModelScope,
    PromotionComparison,
)
from taqamind.core.interfaces.processing_models import ApprovedMapping
from taqamind.core.interfaces.results import ExecutionStatus
from taqamind.core.interfaces.splitting import SplitConfiguration
from taqamind.core.interfaces.training import TrainingConfiguration
from taqamind.core.interfaces.training_registry import TrainingMode
from taqamind.core.pipeline import CorePlatformServices
from taqamind.core.registry import DomainRegistry
from taqamind.domains.iron_manufacturing.adapters import (
    IronSemanticValidator,
    IronUnitAwareColumnMapper,
    ProcessFeatureEngineerRouter,
)
from taqamind.domains.iron_manufacturing.config import DOMAIN_ID
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    PROCESS_DERIVED_FEATURE_SPECS,
    PROCESS_FORBIDDEN_PREDICTORS,
    PROCESS_PHYSICAL_VALIDATORS,
    get_iron_task,
    register_iron_manufacturing,
)
from taqamind.domains.iron_manufacturing.services import (
    IronPredictionService,
    IronTrainingService,
    PredictionOutcome,
    TrainingWorkflowOutcome,
    generate_process_templates,
)


class IronManufacturingPipeline:
    """Facade composing Shared Core protocols and replaceable local adapters."""

    def __init__(
        self,
        storage_root: Path | str = Path("artifacts/iron_manufacturing/store"),
        *,
        domain_registry: DomainRegistry | None = None,
        column_overrides: dict[str, str] | None = None,
        unit_overrides: dict[str, str] | None = None,
    ) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self.domain_registry = domain_registry or DomainRegistry()
        self.domain_contract: DomainContract = register_iron_manufacturing(self.domain_registry)

        # Shared Core P1.1-P1.8 platform stack wired directly via CorePlatformServices
        self.core = CorePlatformServices.create_local(self.storage_root)
        self.dataset_storage = self.core.dataset_storage
        self.model_storage = self.core.model_storage
        self.dataset_registry = self.core.dataset_registry
        self.lineage_registry = self.core.lineage_registry
        self.split_registry = self.core.split_registry
        self.usage_registry = self.core.usage_registry
        self.training_registry = self.core.training_registry
        self.evaluation_registry = self.core.evaluation_registry
        self.model_registry = self.core.model_registry

        # Restricted data access providers from Shared Core
        self.dev_data_provider = self.core.development_data_provider
        self.protected_test_provider = self.core.protected_test_data_provider
        self.active_model_provider = self.core.active_model_provider

        # Processing & ML engines from Shared Core (+ thin domain/AUD-001/AUD-004 wrappers)
        self.ingestion = self.core.ingestion
        self.profiler = self.core.profiler
        self.column_mapper = IronUnitAwareColumnMapper(
            self.dataset_storage,
            inner_mapper=self.core.mapper,
            explicit_column_overrides=column_overrides,
            explicit_unit_overrides=unit_overrides,
            forbidden_predictors_by_process=PROCESS_FORBIDDEN_PREDICTORS,
            implementation_version="0.1.0",
        )
        self.semantic_validator = IronSemanticValidator(
            self.dataset_storage,
            inner_validator=self.core.validator,
            process_validators=PROCESS_PHYSICAL_VALIDATORS,
        )
        self.data_cleaner = self.core.cleaner
        self.feature_engineer = ProcessFeatureEngineerRouter(
            self.dataset_storage,
            specs_by_process=PROCESS_DERIVED_FEATURE_SPECS,
            implementation_version="0.1.0",
        )
        self.readiness_checker = self.core.readiness_checker
        self.splitter = self.core.splitter
        self.leakage_validator = self.core.leakage_validator
        self.model_trainer = self.core.trainer
        self.prediction_engine = self.core.prediction_engine

        # Application services (training and prediction strictly separated)
        self.training_service = IronTrainingService(
            dataset_storage=self.dataset_storage,
            model_storage=self.model_storage,
            dataset_registry=self.dataset_registry,
            lineage_registry=self.lineage_registry,
            split_registry=self.split_registry,
            training_registry=self.training_registry,
            evaluation_registry=self.evaluation_registry,
            model_registry=self.model_registry,
            test_data_provider=self.protected_test_provider,
            evaluator_factory=self.core.create_evaluator,
            ingestion=self.ingestion,
            profiler=self.profiler,
            column_mapper=self.column_mapper,
            semantic_validator=self.semantic_validator,
            data_cleaner=self.data_cleaner,
            feature_engineer=self.feature_engineer,
            readiness_checker=self.readiness_checker,
            splitter=self.splitter,
            leakage_validator=self.leakage_validator,
            model_trainer=self.model_trainer,
        )
        self.prediction_service = IronPredictionService(
            dataset_storage=self.dataset_storage,
            dataset_registry=self.dataset_registry,
            lineage_registry=self.lineage_registry,
            active_model_provider=self.active_model_provider,
            ingestion=self.ingestion,
            column_mapper=self.column_mapper,
            semantic_validator=self.semantic_validator,
            data_cleaner=self.data_cleaner,
            feature_engineer=self.feature_engineer,
            prediction_engine=self.prediction_engine,
        )

    def inspect_dataset(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path | str,
        purpose: DatasetPurpose = DatasetPurpose.RESEARCH,
    ) -> dict[str, Any]:
        """Inspect a CSV file: SHA-256, profiling, and proposed canonical column mappings."""
        task = get_iron_task(process_id)
        contract = task.dataset_contract
        raw_bytes = Path(csv_path).read_bytes()
        source_sha = calculate_bytes_sha256(raw_bytes)
        context = PipelineContext.from_task(
            factory_id=factory_id,
            dataset_revision_id=uuid4(),
            task=task,
            purpose=purpose,
        )
        raw_ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=source_sha,
            size_bytes=len(raw_bytes),
        )
        self.dataset_storage.put(
            factory_id=factory_id, reference=raw_ref, stream=io.BytesIO(raw_bytes)
        )
        raw_artifact = DatasetArtifact(
            metadata=DatasetMetadata.create_raw(context=context, source_file_sha256=source_sha),
            reference=raw_ref,
        )
        self.dataset_registry.register(factory_id=factory_id, artifact=raw_artifact)

        ingest_res = self.ingestion.ingest(
            context=context, artifact=raw_artifact, contract=contract
        )
        if ingest_res.status != ExecutionStatus.SUCCEEDED or ingest_res.output_artifact is None:
            return {
                "status": "blocked",
                "source_file_sha256": source_sha,
                "issues": [i.model_dump(mode="json") for i in ingest_res.report.issues]
                if ingest_res.report
                else [],
            }
        self.dataset_registry.register(factory_id=factory_id, artifact=ingest_res.output_artifact)

        prof = self.profiler.profile(
            context=context, artifact=ingest_res.output_artifact, contract=contract
        )
        proposal = self.column_mapper.propose_mapping(
            context=context, artifact=ingest_res.output_artifact, contract=contract
        )
        return {
            "status": "succeeded",
            "domain_id": DOMAIN_ID,
            "process_id": process_id,
            "source_file_sha256": source_sha,
            "row_count": prof.profile.row_count if prof.profile else 0,
            "column_count": prof.profile.column_count if prof.profile else 0,
            "exact_duplicate_row_count": (
                prof.profile.exact_duplicate_row_count if prof.profile else 0
            ),
            "columns": (
                [c.model_dump(mode="json") for c in prof.profile.columns] if prof.profile else []
            ),
            "mapping_proposal": {
                "requires_review": proposal.requires_review,
                "mapped_entries": [e.model_dump(mode="json") for e in proposal.entries],
                "unmapped_columns": list(proposal.unmapped_columns),
            },
        }

    def validate_dataset(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path | str,
        purpose: DatasetPurpose = DatasetPurpose.TRAINING,
        confirmation: TargetConfirmation | None = None,
        approved_by: str = "factory_engineer",
        confirm_all_proposed_mappings: bool = True,
        confirmed_source_columns: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run ingestion, mapping, and contract-driven semantic validation without training."""
        task = get_iron_task(process_id, confirmation=confirmation)
        contract = task.dataset_contract
        raw_bytes = Path(csv_path).read_bytes()
        source_sha = calculate_bytes_sha256(raw_bytes)
        model_id = uuid4() if purpose == DatasetPurpose.PREDICTION else None
        context = PipelineContext.from_task(
            factory_id=factory_id,
            dataset_revision_id=uuid4(),
            task=task,
            purpose=purpose,
            model_version_id=model_id,
        )
        raw_ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=source_sha,
            size_bytes=len(raw_bytes),
        )
        self.dataset_storage.put(
            factory_id=factory_id, reference=raw_ref, stream=io.BytesIO(raw_bytes)
        )
        raw_artifact = DatasetArtifact(
            metadata=DatasetMetadata.create_raw(context=context, source_file_sha256=source_sha),
            reference=raw_ref,
        )
        self.dataset_registry.register(factory_id=factory_id, artifact=raw_artifact)

        ingest_res = self.ingestion.ingest(
            context=context, artifact=raw_artifact, contract=contract
        )
        if ingest_res.status != ExecutionStatus.SUCCEEDED or ingest_res.output_artifact is None:
            return {
                "status": "blocked",
                "stage": "ingestion",
                "issues": [i.model_dump(mode="json") for i in ingest_res.report.issues]
                if ingest_res.report
                else [],
            }
        self.dataset_registry.register(factory_id=factory_id, artifact=ingest_res.output_artifact)

        proposal = self.column_mapper.propose_mapping(
            context=context, artifact=ingest_res.output_artifact, contract=contract
        )
        confirmed_proposal = proposal
        if confirm_all_proposed_mappings:
            confirmed_entries = tuple(
                e.model_copy(update={"source_unit": e.canonical_unit})
                if e.canonical_unit is not None and e.source_unit is None
                else e
                for e in proposal.entries
            )
            confirmed_proposal = proposal.model_copy(update={"entries": confirmed_entries})
        confirmed = (
            tuple(e.source_column for e in confirmed_proposal.entries if e.requires_confirmation)
            if confirm_all_proposed_mappings
            else confirmed_source_columns
        )
        try:
            approved = ApprovedMapping(
                proposal=confirmed_proposal,
                approved_by=approved_by,
                confirmed_source_columns=confirmed,
            )
            map_res = self.column_mapper.apply_mapping(
                context=context,
                artifact=ingest_res.output_artifact,
                contract=contract,
                mapping=approved,
            )
        except ValueError as exc:
            return {
                "status": "blocked",
                "stage": "column_mapping",
                "error": str(exc),
                "unmapped_columns": list(proposal.unmapped_columns),
            }

        if map_res.status != ExecutionStatus.SUCCEEDED or map_res.output_artifact is None:
            return {
                "status": "blocked",
                "stage": "column_mapping",
                "issues": [i.model_dump(mode="json") for i in map_res.report.issues]
                if map_res.report
                else [],
            }
        self.dataset_registry.register(factory_id=factory_id, artifact=map_res.output_artifact)

        sem_res = self.semantic_validator.validate(
            context=context, artifact=map_res.output_artifact, contract=contract
        )
        return {
            "status": sem_res.report.status.value if sem_res.report else sem_res.status.value,
            "stage": "semantic_validation",
            "target_confirmed": contract.target.is_confirmed,
            "issues": [i.model_dump(mode="json") for i in sem_res.report.issues]
            if sem_res.report
            else [],
        }

    def generate_templates(
        self,
        *,
        process_id: str,
        output_dir: Path | str,
        sample_rows: int = 3,
    ) -> dict[str, Path]:
        """Generate historical training and prediction-only CSV templates for a process."""
        return generate_process_templates(process_id, output_dir, sample_rows=sample_rows)

    def train_initial(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path | str,
        confirmation: TargetConfirmation | None = None,
        requested_by: str = "factory_data_scientist",
        reason: str = "Initial factory model training",
        approved_by: str = "factory_engineer",
        confirm_all_proposed_mappings: bool = True,
        confirmed_source_columns: tuple[str, ...] = (),
        split_configuration: SplitConfiguration | None = None,
        training_configuration: TrainingConfiguration | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
        auto_activate_first: bool = True,
    ) -> TrainingWorkflowOutcome:
        """Execute initial controlled training for a factory and process."""
        task = get_iron_task(process_id, confirmation=confirmation)
        return self.training_service.run_training(
            factory_id=factory_id,
            task=task,
            csv_path=csv_path,
            mode=TrainingMode.INITIAL,
            requested_by=requested_by,
            reason=reason,
            approved_by=approved_by,
            confirm_all_proposed_mappings=confirm_all_proposed_mappings,
            confirmed_source_columns=confirmed_source_columns,
            split_configuration=split_configuration,
            training_configuration=training_configuration,
            acceptance_policy=acceptance_policy,
            auto_activate_first=auto_activate_first,
        )

    def retrain(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path | str,
        confirmation: TargetConfirmation | None = None,
        requested_by: str = "factory_data_scientist",
        reason: str = "Explicit retraining with new labeled factory records",
        approved_by: str = "factory_engineer",
        confirm_all_proposed_mappings: bool = True,
        confirmed_source_columns: tuple[str, ...] = (),
        split_configuration: SplitConfiguration | None = None,
        training_configuration: TrainingConfiguration | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
        allow_confirmed_corrections: bool = False,
        correction_reason: str | None = None,
    ) -> TrainingWorkflowOutcome:
        """Execute explicit retraining while keeping the current active model unchanged."""
        task = get_iron_task(process_id, confirmation=confirmation)
        return self.training_service.run_training(
            factory_id=factory_id,
            task=task,
            csv_path=csv_path,
            mode=TrainingMode.RETRAINING,
            requested_by=requested_by,
            reason=reason,
            approved_by=approved_by,
            confirm_all_proposed_mappings=confirm_all_proposed_mappings,
            confirmed_source_columns=confirmed_source_columns,
            split_configuration=split_configuration,
            training_configuration=training_configuration,
            acceptance_policy=acceptance_policy,
            allow_confirmed_corrections=allow_confirmed_corrections,
            correction_reason=correction_reason,
            auto_activate_first=False,
        )

    def predict(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path | str,
        confirmation: TargetConfirmation | None = None,
        approved_by: str = "factory_operator",
        confirm_all_proposed_mappings: bool = True,
        confirmed_source_columns: tuple[str, ...] = (),
        output_csv_path: Path | str | None = None,
    ) -> PredictionOutcome:
        """Execute strict zero-retraining inference using the factory's active model."""
        task = get_iron_task(process_id, confirmation=confirmation)
        return self.prediction_service.predict_csv(
            factory_id=factory_id,
            task=task,
            csv_path=csv_path,
            approved_by=approved_by,
            confirm_all_proposed_mappings=confirm_all_proposed_mappings,
            confirmed_source_columns=confirmed_source_columns,
            output_csv_path=output_csv_path,
        )

    def get_model_info(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Return active model summary and full version history for a factory and process."""
        resolved_task = task_id or PROCESS_DEFAULT_TASKS[process_id]
        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=resolved_task,
        )
        active = self.model_registry.get_active(scope=scope)
        versions = self.model_registry.list_versions(scope=scope)
        return {
            "factory_id": str(factory_id),
            "domain_id": DOMAIN_ID,
            "process_id": process_id,
            "task_id": resolved_task,
            "active_model": (
                {
                    "model_version_id": str(active.version.model_version_id),
                    "version_number": active.version.version_number,
                    "status": active.status.value,
                    "family": active.version.candidate.family.value,
                    "artifact_sha256": active.version.artifact.content_sha256,
                    "feature_names": list(active.version.feature_specification.feature_names),
                    "target_name": active.version.feature_specification.target_name,
                    "target_unit": active.version.feature_specification.target_unit,
                }
                if active
                else None
            ),
            "versions": [
                {
                    "model_version_id": str(entry.version.model_version_id),
                    "version_number": entry.version.version_number,
                    "status": entry.status.value,
                    "family": entry.version.candidate.family.value,
                    "artifact_sha256": entry.version.artifact.content_sha256,
                    "created_at": entry.version.created_at.isoformat(),
                }
                for entry in versions
            ],
        }

    def promote_model(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        target_model_version_id: UUID,
        comparison: PromotionComparison,
        requested_by: str,
        reason: str,
    ) -> ModelRegistryEntry:
        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=PROCESS_DEFAULT_TASKS[process_id],
        )
        return self.training_service.promote_model(
            scope=scope,
            target_model_version_id=target_model_version_id,
            comparison=comparison,
            requested_by=requested_by,
            reason=reason,
        )

    def rollback_model(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        target_archived_version_id: UUID,
        requested_by: str,
        reason: str,
    ) -> ModelRegistryEntry:
        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=PROCESS_DEFAULT_TASKS[process_id],
        )
        return self.training_service.rollback_model(
            scope=scope,
            target_archived_version_id=target_archived_version_id,
            requested_by=requested_by,
            reason=reason,
        )
