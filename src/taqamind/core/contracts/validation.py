"""Unified reporting contracts for industrial pipeline validation."""

from datetime import UTC, datetime
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

from taqamind.core.contracts.enums import (
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import ContractVersion

ErrorCode = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    ),
]


class ValidationIssue(BaseModel):
    """A single issue detected during pipeline execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    code: ErrorCode

    stage: PipelineStage

    severity: ValidationSeverity

    impact: IssueImpact

    message: NonEmptyString

    columns: tuple[NonEmptyString, ...] = ()

    affected_rows: int | None = Field(
        default=None,
        ge=0,
    )

    row_samples: tuple[int, ...] = Field(
        default=(),
        max_length=20,
    )

    recommendation: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_issue(self) -> Self:
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "Affected columns must be unique."
            )

        if any(index < 0 for index in self.row_samples):
            raise ValueError(
                "Row positions cannot be negative."
            )

        if len(self.row_samples) != len(set(self.row_samples)):
            raise ValueError(
                "Row samples must be unique."
            )

        if self.affected_rows is not None:
            if len(self.row_samples) > self.affected_rows:
                raise ValueError(
                    "Row samples cannot exceed affected row count."
                )

        if self.severity == ValidationSeverity.CRITICAL:
            if self.impact != IssueImpact.BLOCKING:
                raise ValueError(
                    "Critical issues must have blocking impact."
                )

        if self.severity == ValidationSeverity.ERROR:
            if self.impact == IssueImpact.NONE:
                raise ValueError(
                    "Errors must have limited or blocking impact."
                )

        if self.severity == ValidationSeverity.INFO:
            if self.impact != IssueImpact.NONE:
                raise ValueError(
                    "Informational issues cannot restrict execution."
                )

        return self


class ValidationSummary(BaseModel):
    """Computed issue counts for a validation report."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    info_count: int = Field(ge=0)

    warning_count: int = Field(ge=0)

    error_count: int = Field(ge=0)

    critical_count: int = Field(ge=0)

    blocking_count: int = Field(ge=0)

    limited_count: int = Field(ge=0)

    @computed_field
    @property
    def total_issues(self) -> int:
        return (
            self.info_count
            + self.warning_count
            + self.error_count
            + self.critical_count
        )

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.blocking_count + self.limited_count
            > self.total_issues
        ):
            raise ValueError(
                "Impact counts cannot exceed total issues."
            )

        return self


class ValidationReport(BaseModel):
    """Unified validation output for one or more pipeline stages."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    report_id: UUID = Field(default_factory=uuid4)

    dataset_id: FeatureName

    contract_version: ContractVersion

    dataset_revision_id: UUID

    domain: FeatureName

    process: FeatureName

    purpose: DatasetPurpose

    completed_stages: tuple[PipelineStage, ...] = Field(
        min_length=1
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    issues: tuple[ValidationIssue, ...] = ()

    @computed_field
    @property
    def status(self) -> ReadinessStatus:
        """Determine operational status from issue impacts."""

        if any(
            issue.impact == IssueImpact.BLOCKING
            for issue in self.issues
        ):
            return ReadinessStatus.BLOCKED

        if any(
            issue.impact == IssueImpact.LIMITED
            for issue in self.issues
        ):
            return ReadinessStatus.LIMITED

        return ReadinessStatus.READY

    @computed_field
    @property
    def summary(self) -> ValidationSummary:
        """Calculate issue counts dynamically."""

        return ValidationSummary(
            info_count=sum(
                issue.severity == ValidationSeverity.INFO
                for issue in self.issues
            ),
            warning_count=sum(
                issue.severity == ValidationSeverity.WARNING
                for issue in self.issues
            ),
            error_count=sum(
                issue.severity == ValidationSeverity.ERROR
                for issue in self.issues
            ),
            critical_count=sum(
                issue.severity == ValidationSeverity.CRITICAL
                for issue in self.issues
            ),
            blocking_count=sum(
                issue.impact == IssueImpact.BLOCKING
                for issue in self.issues
            ),
            limited_count=sum(
                issue.impact == IssueImpact.LIMITED
                for issue in self.issues
            ),
        )

    @computed_field
    @property
    def has_blocking_issues(self) -> bool:
        return self.status == ReadinessStatus.BLOCKED

    def issues_by_severity(
        self,
        severity: ValidationSeverity,
    ) -> tuple[ValidationIssue, ...]:
        """Filter issues by severity."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity == severity
        )

    def issues_by_stage(
        self,
        stage: PipelineStage,
    ) -> tuple[ValidationIssue, ...]:
        """Filter issues by their originating stage."""

        return tuple(
            issue
            for issue in self.issues
            if issue.stage == stage
        )

    def to_payload(self) -> dict:
        """Return a JSON-compatible report without computed fields."""

        return self.model_dump(
            mode="json",
            exclude={
                "status",
                "summary",
                "has_blocking_issues",
            },
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Report timestamp must be timezone-aware."
            )

        if len(self.completed_stages) != len(
            set(self.completed_stages)
        ):
            raise ValueError(
                "Completed stages must be unique."
            )

        for issue in self.issues:
            if issue.stage not in self.completed_stages:
                raise ValueError(
                    "Issue stage must be registered in the report."
                )

        return self

    @classmethod
    def merge(
        cls,
        reports: tuple[Self, ...],
    ) -> Self:
        """Combine compatible reports for the same dataset revision."""

        if not reports:
            raise ValueError(
                "At least one validation report is required."
            )

        first = reports[0]

        identity_fields = (
            "dataset_id",
            "contract_version",
            "dataset_revision_id",
            "domain",
            "process",
            "purpose",
        )

        report_ids = set()
        completed_stages = []
        issues = []

        for report in reports:
            for field in identity_fields:
                if getattr(report, field) != getattr(first, field):
                    raise ValueError(
                        f"Reports have incompatible {field}."
                    )

            if report.report_id in report_ids:
                raise ValueError(
                    "Duplicate reports cannot be merged."
                )

            report_ids.add(report.report_id)

            for stage in report.completed_stages:
                if stage in completed_stages:
                    raise ValueError(
                        f"Overlapping report stage: {stage}."
                    )

                completed_stages.append(stage)

            issues.extend(report.issues)

        return cls(
            dataset_id=first.dataset_id,
            contract_version=first.contract_version,
            dataset_revision_id=first.dataset_revision_id,
            domain=first.domain,
            process=first.process,
            purpose=first.purpose,
            completed_stages=tuple(completed_stages),
            issues=tuple(issues),
        )