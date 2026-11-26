"""Core P1.6 TrainingReadinessChecker implementation."""

from __future__ import annotations

from datetime import UTC, datetime

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    RecordLineageRegistry,
    RegistryNotFoundError,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.ml import TrainingReadinessChecker
from taqamind.core.interfaces.processing import validate_processing_input
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


class CoreTrainingReadinessChecker(TrainingReadinessChecker):
    """Assess whether an engineered dataset artifact is eligible for splitting and training."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        min_rows: int = 12,
        min_positive_targets: int = 2,
        delimiter: str = ",",
        identity_namespace: str = "production_record",
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._min_rows = min_rows
        self._min_positive_targets = min_positive_targets
        self._reader = SafeCsvReader(delimiter=delimiter)
        self._converter = ContractTypeConverter()
        self._identity_namespace = identity_namespace

    def assess(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> TrainingReadinessResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.ENGINEERED,
        )
        if context.purpose != DatasetPurpose.TRAINING:
            raise ValueError("Training readiness requires a training context.")

        issues: list[ValidationIssue] = []

        if not contract.target.is_confirmed:
            issues.append(
                ValidationIssue(
                    code="TARGET_SEMANTICS_UNCONFIRMED",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Target semantics are unconfirmed; production training is blocked.",
                    columns=(contract.target.feature.name,),
                )
            )

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            raw_bytes = stream.read()

        try:
            df = self._reader.read(raw_bytes)
        except SafeCsvError as exc:
            stage_err = StageResult(
                context=context,
                stage=PipelineStage.READINESS,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc)),
            )
            return TrainingReadinessResult(result=stage_err, contract=contract)

        target_name = contract.target.feature.name
        if len(df) < self._min_rows:
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_ROW_COUNT",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=f"Dataset has {len(df)} rows; minimum required is {self._min_rows}.",
                )
            )

        if target_name not in df.columns:
            issues.append(
                ValidationIssue(
                    code="MISSING_TRAINING_TARGET",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Target column '{target_name}' is missing.",
                    columns=(target_name,),
                )
            )
        else:
            target_type = contract.target.feature.data_type
            conv = self._converter.convert(
                df[target_name],
                data_type=(
                    target_type
                    if target_type in {DataType.INTEGER, DataType.FLOAT}
                    else DataType.FLOAT
                ),
            )
            if not conv.is_valid or bool(conv.values.isna().any()):
                issues.append(
                    ValidationIssue(
                        code="MISSING_OR_INVALID_TARGET_VALUES",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Target contains missing or non-numeric values.",
                        columns=(target_name,),
                        affected_rows=int(
                            len(conv.invalid_positions) + int(conv.values.isna().sum())
                        ),
                    )
                )
            elif int((conv.values.astype(float) > 0).sum()) < self._min_positive_targets:
                issues.append(
                    ValidationIssue(
                        code="INSUFFICIENT_POSITIVE_TARGETS",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Insufficient positive target events for regression.",
                        columns=(target_name,),
                    )
                )

        scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=self._identity_namespace,
        )
        try:
            manifest = self._lineage.get_membership(
                scope=scope,
                dataset_revision_id=artifact.metadata.revision_id,
            )
            manifest.validate_artifact(artifact)
        except (RegistryNotFoundError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    code="MISSING_OR_INVALID_MEMBERSHIP",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Registered record membership unavailable or invalid: {exc}",
                )
            )

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.READINESS,),
            issues=tuple(issues),
        )
        status = (
            ExecutionStatus.BLOCKED
            if report.status == ReadinessStatus.BLOCKED
            else ExecutionStatus.SUCCEEDED
        )
        stage_result = StageResult(
            context=context,
            stage=PipelineStage.READINESS,
            status=status,
            input_artifact=artifact,
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
        return TrainingReadinessResult(result=stage_result, contract=contract)
