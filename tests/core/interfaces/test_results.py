"""Tests for shared pipeline stage results."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
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
    ExecutionError,
    ExecutionStatus,
    PipelineContext,
    StageResult,
    StageTiming,
)

SOURCE_HASH = "a" * 64


@pytest.fixture
def context():
    return PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain="textile",
        process="knitting",
        task_id="waste_prediction",
        dataset_id="textile_knitting_waste",
        contract_version="1.0.0",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
        created_at=datetime.now(UTC) - timedelta(seconds=10),
    )


@pytest.fixture
def raw_artifact(context):
    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256=SOURCE_HASH,
    )

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=SOURCE_HASH,
        size_bytes=1024,
    )

    return DatasetArtifact(
        metadata=metadata,
        reference=reference,
    )


@pytest.fixture
def ingested_artifact(context, raw_artifact):
    metadata = raw_artifact.metadata.derive(
        context=context,
        state=DatasetState.INGESTED,
        row_count=1000,
        columns=(
            DatasetColumnSchema(
                name="planned_quantity_kg",
                physical_dtype="float64",
            ),
        ),
        operation="csv_parsing",
        implementation_version="1.0.0",
        description="Synthetic ingestion.",
    )

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256="b" * 64,
        size_bytes=2048,
    )

    return DatasetArtifact(
        metadata=metadata,
        reference=reference,
    )


def make_timing():
    now = datetime.now(UTC)

    return StageTiming(
        started_at=now - timedelta(seconds=2),
        finished_at=now,
    )


def make_report(
    context,
    artifact,
    stage=PipelineStage.PROFILING,
    issues=(),
):
    return ValidationReport(
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        dataset_revision_id=artifact.metadata.revision_id,
        domain=context.domain,
        process=context.process,
        purpose=context.purpose,
        completed_stages=(stage,),
        issues=issues,
    )


def make_issue(
    impact=IssueImpact.BLOCKING,
    stage=PipelineStage.PROFILING,
):
    return ValidationIssue(
        code="SYNTHETIC_DATA_ISSUE",
        stage=stage,
        severity=ValidationSeverity.WARNING,
        impact=impact,
        message="Synthetic validation issue.",
    )


def make_result(context, artifact, **overrides):
    data = {
        "context": context,
        "stage": PipelineStage.PROFILING,
        "status": ExecutionStatus.SUCCEEDED,
        "input_artifact": artifact,
        "report": make_report(context, artifact),
        "timing": make_timing(),
    }

    data.update(overrides)

    return StageResult(**data)


def test_stage_duration():
    timing = make_timing()

    assert timing.duration_seconds == 2.0


def test_negative_duration_rejected():
    now = datetime.now(UTC)

    with pytest.raises(ValidationError):
        StageTiming(
            started_at=now,
            finished_at=now - timedelta(seconds=1),
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        StageTiming(
            started_at=datetime(2026, 1, 1),
            finished_at=datetime(2026, 1, 2),
        )


def test_successful_inspection(context, ingested_artifact):
    result = make_result(context, ingested_artifact)

    assert result.is_successful
    assert result.can_continue
    assert not result.requires_review


def test_inspection_reuses_input_artifact(
    context,
    ingested_artifact,
):
    result = make_result(context, ingested_artifact)

    assert result.output_artifact is None
    assert result.next_artifact == ingested_artifact


def test_successful_transformation(
    context,
    raw_artifact,
    ingested_artifact,
):
    result = make_result(
        context,
        raw_artifact,
        stage=PipelineStage.INGESTION,
        output_artifact=ingested_artifact,
        report=make_report(
            context,
            raw_artifact,
            stage=PipelineStage.INGESTION,
        ),
    )

    assert result.can_continue
    assert result.next_artifact == ingested_artifact


def test_transformation_requires_output(
    context,
    raw_artifact,
):
    with pytest.raises(
        ValidationError,
        match="requires an output artifact",
    ):
        make_result(
            context,
            raw_artifact,
            stage=PipelineStage.INGESTION,
            report=make_report(
                context,
                raw_artifact,
                stage=PipelineStage.INGESTION,
            ),
        )


def test_success_requires_report(context, ingested_artifact):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            report=None,
        )


def test_nonblocking_warning_allows_continuation(
    context,
    ingested_artifact,
):
    report = make_report(
        context,
        ingested_artifact,
        issues=(
            make_issue(impact=IssueImpact.NONE),
        ),
    )

    result = make_result(
        context,
        ingested_artifact,
        report=report,
    )

    assert result.can_continue


def test_limited_result_requires_review(
    context,
    ingested_artifact,
):
    report = make_report(
        context,
        ingested_artifact,
        issues=(
            make_issue(impact=IssueImpact.LIMITED),
        ),
    )

    result = make_result(
        context,
        ingested_artifact,
        report=report,
    )

    assert result.is_successful
    assert not result.can_continue
    assert result.requires_review
    assert result.next_artifact is None


def test_valid_blocked_result(context, ingested_artifact):
    report = make_report(
        context,
        ingested_artifact,
        issues=(make_issue(),),
    )

    result = make_result(
        context,
        ingested_artifact,
        status=ExecutionStatus.BLOCKED,
        report=report,
    )

    assert not result.can_continue
    assert result.requires_review
    assert result.next_artifact is None


def test_success_cannot_have_blocked_report(
    context,
    ingested_artifact,
):
    report = make_report(
        context,
        ingested_artifact,
        issues=(make_issue(),),
    )

    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            report=report,
        )


def test_blocked_requires_blocking_issue(
    context,
    ingested_artifact,
):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            status=ExecutionStatus.BLOCKED,
        )


def test_valid_failed_result(context, ingested_artifact):
    result = make_result(
        context,
        ingested_artifact,
        status=ExecutionStatus.FAILED,
        report=None,
        error=ExecutionError(
            code="ARTIFACT_READ_FAILED",
            message="Unable to read dataset artifact.",
            retryable=True,
        ),
    )

    assert not result.can_continue
    assert result.error.retryable
    assert result.next_artifact is None


def test_failure_requires_error(context, ingested_artifact):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            status=ExecutionStatus.FAILED,
            report=None,
        )


def test_success_rejects_error(context, ingested_artifact):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            error=ExecutionError(
                code="UNEXPECTED_ERROR",
                message="Unexpected execution error.",
            ),
        )


def test_valid_skipped_result(context, ingested_artifact):
    result = make_result(
        context,
        ingested_artifact,
        status=ExecutionStatus.SKIPPED,
        report=None,
        skip_reason="Stage disabled by approved configuration.",
    )

    assert not result.can_continue
    assert result.next_artifact is None


def test_skipped_requires_reason(context, ingested_artifact):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            status=ExecutionStatus.SKIPPED,
            report=None,
        )


def test_skipped_cannot_declare_completed_report(
    context,
    ingested_artifact,
):
    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            status=ExecutionStatus.SKIPPED,
            skip_reason="Synthetic skip.",
        )


def test_report_revision_mismatch(
    context,
    raw_artifact,
    ingested_artifact,
):
    report = make_report(context, raw_artifact)

    with pytest.raises(
        ValidationError,
        match="dataset_revision_id",
    ):
        make_result(
            context,
            ingested_artifact,
            report=report,
        )


def test_report_stage_mismatch(
    context,
    ingested_artifact,
):
    report = make_report(
        context,
        ingested_artifact,
        stage=PipelineStage.CLEANING,
    )

    with pytest.raises(ValidationError):
        make_result(
            context,
            ingested_artifact,
            report=report,
        )


def test_wrong_factory_rejected(
    context,
    ingested_artifact,
):
    payload = context.model_dump()
    payload["factory_id"] = uuid4()

    other_context = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(
        ValidationError,
        match="factory_id",
    ):
        make_result(
            other_context,
            ingested_artifact,
        )


def test_output_cannot_reuse_input_revision(
    context,
    ingested_artifact,
):
    with pytest.raises(
        ValidationError,
        match="new revision",
    ):
        make_result(
            context,
            ingested_artifact,
            output_artifact=ingested_artifact,
        )


def test_output_from_different_execution_rejected(
    context,
    raw_artifact,
    ingested_artifact,
):
    payload = context.model_dump()
    payload["execution_id"] = uuid4()

    other_execution = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(
        ValidationError,
        match="execution mismatch",
    ):
        make_result(
            other_execution,
            raw_artifact,
            stage=PipelineStage.INGESTION,
            output_artifact=ingested_artifact,
            report=make_report(
                other_execution,
                raw_artifact,
                stage=PipelineStage.INGESTION,
            ),
        )


def test_stage_cannot_start_before_execution(
    context,
    ingested_artifact,
):
    timing = StageTiming(
        started_at=context.created_at - timedelta(seconds=1),
        finished_at=datetime.now(UTC),
    )

    with pytest.raises(
        ValidationError,
        match="before execution creation",
    ):
        make_result(
            context,
            ingested_artifact,
            timing=timing,
        )


def test_result_is_frozen(context, ingested_artifact):
    result = make_result(context, ingested_artifact)

    with pytest.raises(ValidationError):
        result.status = ExecutionStatus.FAILED


def test_result_serialization(
    context,
    ingested_artifact,
):
    result = make_result(context, ingested_artifact)

    payload = result.model_dump(mode="json")

    # ValidationReport contains computed fields that
    # must not be passed back as validation inputs.
    payload["report"] = result.report.to_payload()

    restored = StageResult.model_validate(payload)

    assert restored == result