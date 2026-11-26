"""Shared validation report and issue builders for Food & Beverage adapters."""

from __future__ import annotations

from uuid import UUID

from taqamind.core.contracts import (
    IssueImpact,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.interfaces import PipelineContext


def make_issue(
    *,
    code: str,
    stage: PipelineStage,
    severity: ValidationSeverity,
    impact: IssueImpact,
    message: str,
    column: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        stage=stage,
        severity=severity,
        impact=impact,
        message=message,
        columns=(column,) if column is not None else (),
    )


def make_report(
    *,
    context: PipelineContext,
    stage: PipelineStage,
    issues: tuple[ValidationIssue, ...] | list[ValidationIssue] = (),
    revision_id: UUID | None = None,
) -> ValidationReport:
    return ValidationReport(
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        dataset_revision_id=revision_id or context.dataset_revision_id,
        domain=context.domain,
        process=context.process,
        purpose=context.purpose,
        completed_stages=(stage,),
        issues=tuple(issues),
    )
