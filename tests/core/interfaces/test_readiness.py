"""Tests for training readiness contracts."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureContract,
    FeatureRole,
    IssueImpact,
    PipelineStage,
    PredictionContext,
    ReadinessStatus,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    PipelineContext,
    StageResult,
    StageTiming,
    TrainingReadinessResult,
)


def make_contract(confirmed=True):
    confirmation = TargetConfirmation()

    if confirmed:
        confirmation = TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="synthetic_reviewer",
            confirmed_at=datetime.now(UTC),
        )

    return DatasetContract(
        dataset_id="knitting_waste",
        domain="textile",
        process="knitting",
        description="Synthetic dataset contract.",
        granularity="production_run",
        features=(
            FeatureContract(
                name="planned_quantity_kg",
                display_name="Planned Quantity",
                description="Known planned production quantity.",
                data_type=DataType.FLOAT,
                role=FeatureRole.PREDICTOR,
                availability=FeatureAvailability.PRE_PRODUCTION,
                unit="kg",
            ),
        ),
        target=TargetContract(
            target_id="waste_prediction",
            feature=FeatureContract(
                name="actual_waste_kg",
                display_name="Actual Waste",
                description="Measured waste after production.",
                data_type=DataType.FLOAT,
                role=FeatureRole.TARGET,
                availability=FeatureAvailability.POST_PRODUCTION,
                unit="kg",
            ),
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(
                definition="Synthetic waste definition.",
                granularity="production_run",
            ),
            confirmation=confirmation,
        ),
        prediction_context=PredictionContext(
            description="Predict waste before production.",
        ),
    )


def make_stage(
    contract,
    impact=None,
    row_count=100,
    stage=PipelineStage.READINESS,
):
    context = PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        task_id="waste_prediction",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.TRAINING,
    )

    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256="a" * 64,
    )

    columns = (
        DatasetColumnSchema(
            name="planned_quantity_kg",
            physical_dtype="float64",
        ),
        DatasetColumnSchema(
            name="actual_waste_kg",
            physical_dtype="float64",
        ),
    )

    for state in (
        DatasetState.INGESTED,
        DatasetState.MAPPED,
        DatasetState.CLEANED,
        DatasetState.ENGINEERED,
    ):
        metadata = metadata.derive(
            context=context,
            state=state,
            row_count=row_count,
            columns=columns,
            operation="synthetic_transformation",
            implementation_version="1.0.0",
            description="Synthetic processing.",
        )

    artifact = DatasetArtifact(
        metadata=metadata,
        reference=ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256="b" * 64,
            size_bytes=1024,
        ),
    )

    issues = ()

    if impact is not None:
        issues = (
            ValidationIssue(
                code="SYNTHETIC_READINESS_ISSUE",
                stage=stage,
                severity=ValidationSeverity.WARNING,
                impact=impact,
                message="Synthetic readiness issue.",
            ),
        )

    report = ValidationReport(
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        dataset_revision_id=metadata.revision_id,
        domain=context.domain,
        process=context.process,
        purpose=context.purpose,
        completed_stages=(stage,),
        issues=issues,
    )

    status = (
        ExecutionStatus.BLOCKED
        if impact == IssueImpact.BLOCKING
        else ExecutionStatus.SUCCEEDED
    )

    return StageResult(
        context=context,
        stage=stage,
        status=status,
        input_artifact=artifact,
        report=report,
        timing=StageTiming(
            started_at=context.created_at,
            finished_at=datetime.now(UTC),
        ),
    )


def test_ready_dataset_can_proceed_to_split():
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(contract),
    )

    assert readiness.status == ReadinessStatus.READY
    assert readiness.eligible_for_split

    readiness.require_split_eligibility()


def test_unconfirmed_target_cannot_be_ready():
    contract = make_contract(confirmed=False)

    with pytest.raises(
        ValidationError,
        match="unconfirmed target",
    ):
        TrainingReadinessResult(
            contract=contract,
            result=make_stage(contract),
        )


def test_limited_dataset_cannot_proceed_automatically():
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(
            contract,
            impact=IssueImpact.LIMITED,
        ),
    )

    assert readiness.status == ReadinessStatus.LIMITED
    assert not readiness.eligible_for_split

    with pytest.raises(ValueError):
        readiness.require_split_eligibility()


def test_blocked_dataset_cannot_proceed():
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(
            contract,
            impact=IssueImpact.BLOCKING,
        ),
    )

    assert readiness.status == ReadinessStatus.BLOCKED
    assert not readiness.eligible_for_split


def test_empty_dataset_cannot_be_ready():
    contract = make_contract()

    with pytest.raises(
        ValidationError,
        match="empty dataset",
    ):
        TrainingReadinessResult(
            contract=contract,
            result=make_stage(
                contract,
                row_count=0,
            ),
        )


def test_wrong_stage_rejected():
    contract = make_contract()

    with pytest.raises(
        ValidationError,
        match="readiness stage",
    ):
        TrainingReadinessResult(
            contract=contract,
            result=make_stage(
                contract,
                stage=PipelineStage.PROFILING,
            ),
        )