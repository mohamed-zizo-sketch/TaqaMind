"""Tests for unified pipeline validation reports."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

TEST_REVISION_ID = UUID(
    "00000000-0000-0000-0000-000000000001"
)


def make_issue(**overrides):
    data = {
        "code": "MISSING_OPTIONAL_COLUMN",
        "stage": PipelineStage.PROFILING,
        "severity": ValidationSeverity.WARNING,
        "impact": IssueImpact.NONE,
        "message": "Optional column is missing.",
    }

    data.update(overrides)

    return ValidationIssue(**data)


def make_report(**overrides):
    data = {
        "dataset_id": "knitting_waste_dataset",
        "contract_version": "1.0.0",
        "dataset_revision_id": TEST_REVISION_ID,
        "domain": "textile",
        "process": "knitting",
        "purpose": DatasetPurpose.TRAINING,
        "completed_stages": (PipelineStage.PROFILING,),
        "issues": (),
    }

    data.update(overrides)

    return ValidationReport(**data)


def test_empty_report_is_ready():
    report = make_report()

    assert report.status == ReadinessStatus.READY
    assert report.summary.total_issues == 0
    assert not report.has_blocking_issues


def test_non_blocking_warning_is_ready():
    report = make_report(issues=(make_issue(),))

    assert report.status == ReadinessStatus.READY
    assert report.summary.warning_count == 1


def test_limited_issue():
    report = make_report(
        issues=(
            make_issue(impact=IssueImpact.LIMITED),
        )
    )

    assert report.status == ReadinessStatus.LIMITED
    assert report.summary.limited_count == 1


def test_blocking_issue():
    report = make_report(
        issues=(
            make_issue(
                severity=ValidationSeverity.ERROR,
                impact=IssueImpact.BLOCKING,
            ),
        )
    )

    assert report.status == ReadinessStatus.BLOCKED
    assert report.has_blocking_issues


def test_blocking_overrides_limited():
    report = make_report(
        issues=(
            make_issue(impact=IssueImpact.LIMITED),
            make_issue(
                severity=ValidationSeverity.CRITICAL,
                impact=IssueImpact.BLOCKING,
            ),
        )
    )

    assert report.status == ReadinessStatus.BLOCKED


def test_critical_requires_blocking_impact():
    with pytest.raises(ValidationError):
        make_issue(
            severity=ValidationSeverity.CRITICAL,
            impact=IssueImpact.NONE,
        )


def test_error_requires_operational_impact():
    with pytest.raises(ValidationError):
        make_issue(
            severity=ValidationSeverity.ERROR,
            impact=IssueImpact.NONE,
        )


def test_info_cannot_block_execution():
    with pytest.raises(ValidationError):
        make_issue(
            severity=ValidationSeverity.INFO,
            impact=IssueImpact.BLOCKING,
        )


@pytest.mark.parametrize(
    "code",
    ["invalid_code", "INVALID CODE", "", "123_CODE"],
)
def test_invalid_error_codes(code):
    with pytest.raises(ValidationError):
        make_issue(code=code)


def test_negative_affected_rows():
    with pytest.raises(ValidationError):
        make_issue(affected_rows=-1)


def test_negative_row_position():
    with pytest.raises(ValidationError):
        make_issue(row_samples=(-1,))


def test_duplicate_row_samples():
    with pytest.raises(ValidationError):
        make_issue(row_samples=(1, 1))


def test_row_sample_count():
    with pytest.raises(ValidationError):
        make_issue(
            affected_rows=1,
            row_samples=(1, 2),
        )


def test_row_sample_limit():
    with pytest.raises(ValidationError):
        make_issue(row_samples=tuple(range(21)))


def test_duplicate_columns():
    with pytest.raises(ValidationError):
        make_issue(columns=("waste", "waste"))


def test_summary_counts():
    report = make_report(
        issues=(
            make_issue(),
            make_issue(
                severity=ValidationSeverity.INFO,
                impact=IssueImpact.NONE,
            ),
            make_issue(
                severity=ValidationSeverity.ERROR,
                impact=IssueImpact.BLOCKING,
            ),
            make_issue(
                severity=ValidationSeverity.CRITICAL,
                impact=IssueImpact.BLOCKING,
            ),
        )
    )

    assert report.summary.total_issues == 4
    assert report.summary.info_count == 1
    assert report.summary.warning_count == 1
    assert report.summary.error_count == 1
    assert report.summary.critical_count == 1
    assert report.summary.blocking_count == 2


def test_filter_by_severity():
    report = make_report(
        issues=(
            make_issue(),
            make_issue(
                severity=ValidationSeverity.ERROR,
                impact=IssueImpact.BLOCKING,
            ),
        )
    )

    errors = report.issues_by_severity(
        ValidationSeverity.ERROR
    )

    assert len(errors) == 1


def test_filter_by_stage():
    report = make_report(issues=(make_issue(),))

    issues = report.issues_by_stage(
        PipelineStage.PROFILING
    )

    assert len(issues) == 1


def test_issue_stage_must_be_registered():
    with pytest.raises(ValidationError):
        make_report(
            issues=(
                make_issue(
                    stage=PipelineStage.INGESTION,
                ),
            )
        )


def test_duplicate_stages_are_rejected():
    with pytest.raises(ValidationError):
        make_report(
            completed_stages=(
                PipelineStage.PROFILING,
                PipelineStage.PROFILING,
            )
        )


def test_timezone_required():
    with pytest.raises(ValidationError):
        make_report(
            created_at=datetime(2026, 1, 1),
        )


def test_report_is_frozen():
    report = make_report()

    with pytest.raises(ValidationError):
        report.domain = "food_beverage"


def test_serialization_round_trip():
    report = make_report(issues=(make_issue(),))

    restored = ValidationReport.model_validate(
        report.to_payload()
    )

    assert restored == report


def test_status_cannot_be_overridden():
    with pytest.raises(ValidationError):
        make_report(status="ready")


def test_computed_fields_in_json():
    report = make_report(issues=(make_issue(),))

    serialized = report.model_dump(mode="json")

    assert serialized["status"] == "ready"
    assert serialized["summary"]["total_issues"] == 1


def test_merge_compatible_reports():
    first = make_report(issues=(make_issue(),))

    second = make_report(
        completed_stages=(PipelineStage.READINESS,),
        issues=(
            make_issue(
                stage=PipelineStage.READINESS,
                impact=IssueImpact.LIMITED,
            ),
        ),
    )

    merged = ValidationReport.merge(
        (first, second)
    )

    assert merged.summary.total_issues == 2
    assert merged.status == ReadinessStatus.LIMITED
    assert len(merged.completed_stages) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset_id": "different_dataset"},
        {"contract_version": "2.0.0"},
        {"dataset_revision_id": uuid4()},
        {"domain": "food_beverage"},
        {"process": "weaving"},
        {"purpose": DatasetPurpose.PREDICTION},
    ],
)
def test_incompatible_reports_cannot_merge(overrides):
    first = make_report()
    second = make_report(**overrides)

    with pytest.raises(ValueError, match="incompatible"):
        ValidationReport.merge((first, second))


def test_duplicate_reports_cannot_merge():
    report = make_report()

    with pytest.raises(ValueError, match="Duplicate reports"):
        ValidationReport.merge((report, report))


def test_overlapping_stages_cannot_merge():
    first = make_report()
    second = make_report()

    with pytest.raises(ValueError, match="Overlapping"):
        ValidationReport.merge((first, second))


def test_empty_merge():
    with pytest.raises(ValueError):
        ValidationReport.merge(())