"""Tests for leakage validation and training authorization."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    IssueImpact,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.interfaces import (
    LeakageCheck,
    LeakageCheckResult,
    LeakageCheckStatus,
    LeakageValidationResult,
    SplitStrategy,
    TrainingAuthorization,
    TrainingReadinessResult,
)
from tests.core.interfaces.test_readiness import (
    make_contract,
    make_stage,
)
from tests.core.interfaces.test_splitting import (
    make_configuration,
    make_manifest,
)


def make_validation(
    failed_check=None,
    unverified_check=None,
    omitted_check=None,
):
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(contract),
    )

    manifest = make_manifest(
        context=readiness.result.context,
        dataset_revision_id=(
            readiness.result.input_artifact.metadata.revision_id
        ),
        configuration=make_configuration(
            strategy=SplitStrategy.RANDOM,
            justification="Synthetic contract test only.",
        ),
    )

    required_checks = (
        LeakageCheck.RECORD_OVERLAP,
        LeakageCheck.RECORD_COVERAGE,
        LeakageCheck.RECORD_IDENTITY,
        LeakageCheck.DUPLICATE_CONTENT,
        LeakageCheck.HISTORICAL_USAGE,
        LeakageCheck.FEATURE_AVAILABILITY,
    )

    checks = []
    issues = []

    for check in required_checks:
        if check == omitted_check:
            continue

        status = LeakageCheckStatus.PASSED
        reason = None

        if check == failed_check:
            status = LeakageCheckStatus.FAILED
            reason = "Synthetic leakage failure."

            issues.append(
                ValidationIssue(
                    code="LEAKAGE_DETECTED",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=reason,
                )
            )

        elif check == unverified_check:
            status = LeakageCheckStatus.NOT_VERIFIED
            reason = "Synthetic unavailable verification."

            issues.append(
                ValidationIssue(
                    code="LEAKAGE_NOT_VERIFIED",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.WARNING,
                    impact=IssueImpact.LIMITED,
                    message=reason,
                )
            )

        checks.append(
            LeakageCheckResult(
                check=check,
                status=status,
                reason=reason,
            )
        )

    context = manifest.context

    report = ValidationReport(
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        dataset_revision_id=manifest.dataset_revision_id,
        domain=context.domain,
        process=context.process,
        purpose=context.purpose,
        completed_stages=(PipelineStage.SPLITTING,),
        issues=tuple(issues),
    )

    return LeakageValidationResult(
        readiness=readiness,
        manifest=manifest,
        checks=tuple(checks),
        report=report,
    )


def test_all_required_checks_passed():
    result = make_validation()

    assert result.eligible_for_training


@pytest.mark.parametrize(
    "check",
    [
        LeakageCheck.RECORD_OVERLAP,
        LeakageCheck.RECORD_COVERAGE,
        LeakageCheck.RECORD_IDENTITY,
        LeakageCheck.DUPLICATE_CONTENT,
        LeakageCheck.HISTORICAL_USAGE,
        LeakageCheck.FEATURE_AVAILABILITY,
    ],
)
def test_failed_check_blocks_training(check):
    result = make_validation(
        failed_check=check,
    )

    assert not result.eligible_for_training

    with pytest.raises(ValueError):
        result.require_training_eligibility()


def test_unverified_history_blocks_training():
    result = make_validation(
        unverified_check=LeakageCheck.HISTORICAL_USAGE,
    )

    assert not result.eligible_for_training


def test_required_check_cannot_be_omitted():
    with pytest.raises(
        ValidationError,
        match="Required leakage checks are missing",
    ):
        make_validation(
            omitted_check=LeakageCheck.RECORD_IDENTITY,
        )


def test_failed_check_requires_reason():
    with pytest.raises(ValidationError):
        LeakageCheckResult(
            check=LeakageCheck.RECORD_OVERLAP,
            status=LeakageCheckStatus.FAILED,
        )


def test_required_check_cannot_be_not_applicable():
    result = make_validation()

    payload = result.model_dump(
        exclude={
            "readiness": {"result": {"report": True}},
            "report": True,
        }
    )

    payload["readiness"]["result"]["report"] = (
        result.readiness.result.report.to_payload()
    )

    payload["report"] = result.report.to_payload()

    payload["checks"][0]["status"] = (
        LeakageCheckStatus.NOT_APPLICABLE
    )

    payload["checks"][0]["reason"] = "Synthetic reason."

    with pytest.raises(
        ValidationError,
        match="cannot be not applicable",
    ):
        LeakageValidationResult.model_validate(payload)


def test_duplicate_leakage_checks_rejected():
    result = make_validation()

    with pytest.raises(
        ValidationError,
        match="Leakage checks must be unique",
    ):
        LeakageValidationResult(
            readiness=result.readiness,
            manifest=result.manifest,
            report=result.report,
            checks=(
                *result.checks,
                result.checks[0],
            ),
        )


def test_report_revision_mismatch():
    result = make_validation()

    payload = result.report.to_payload()
    payload["dataset_revision_id"] = str(uuid4())

    report = ValidationReport.model_validate(payload)

    with pytest.raises(
        ValidationError,
        match="dataset_revision_id",
    ):
        LeakageValidationResult(
            readiness=result.readiness,
            manifest=result.manifest,
            checks=result.checks,
            report=report,
        )


def test_failed_checks_require_blocked_report():
    result = make_validation()

    failed = LeakageCheckResult(
        check=result.checks[0].check,
        status=LeakageCheckStatus.FAILED,
        reason="Synthetic failure.",
    )

    with pytest.raises(
        ValidationError,
        match="require a blocked report",
    ):
        LeakageValidationResult(
            readiness=result.readiness,
            manifest=result.manifest,
            report=result.report,
            checks=(
                failed,
                *result.checks[1:],
            ),
        )


def test_valid_training_authorization():
    result = make_validation()

    authorization = TrainingAuthorization.from_validation(
        result
    )

    assert authorization.leakage_validation_id == (
        result.validation_id
    )

    assert authorization.development_split.train == (
        result.manifest.train
    )


def test_blocked_validation_cannot_authorize_training():
    result = make_validation(
        failed_check=LeakageCheck.RECORD_OVERLAP,
    )

    with pytest.raises(
        ValueError,
        match="does not permit production training",
    ):
        TrainingAuthorization.from_validation(result)


def test_authorization_excludes_test_data():
    result = make_validation()

    authorization = TrainingAuthorization.from_validation(
        result
    )

    development = authorization.development_split

    assert not hasattr(development, "test")

    development_ids = (
        set(development.train.record_ids)
        | set(development.validation.record_ids)
    )

    assert not (
        development_ids
        & set(result.manifest.test.record_ids)
    )


def test_authorization_serialization():
    authorization = TrainingAuthorization.from_validation(
        make_validation()
    )

    restored = TrainingAuthorization.model_validate_json(
        authorization.model_dump_json()
    )

    assert restored == authorization