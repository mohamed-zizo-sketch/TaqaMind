"""Leakage validation contracts and training authorization metadata."""

from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.enums import (
    PipelineStage,
    ReadinessStatus,
)
from taqamind.core.contracts.feature import NonEmptyString
from taqamind.core.contracts.validation import ValidationReport
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.splitting import (
    SplitManifest,
    SplitStrategy,
)
from taqamind.core.interfaces.training import DevelopmentSplit


class LeakageCheck(StrEnum):
    """Required categories of leakage prevention checks."""

    RECORD_OVERLAP = "record_overlap"

    RECORD_COVERAGE = "record_coverage"

    RECORD_IDENTITY = "record_identity"

    DUPLICATE_CONTENT = "duplicate_content"

    GROUP_OVERLAP = "group_overlap"

    TEMPORAL_ORDER = "temporal_order"

    HISTORICAL_USAGE = "historical_usage"

    FEATURE_AVAILABILITY = "feature_availability"


class LeakageCheckStatus(StrEnum):
    """Outcome of an individual leakage check."""

    PASSED = "passed"

    FAILED = "failed"

    NOT_APPLICABLE = "not_applicable"

    NOT_VERIFIED = "not_verified"


class LeakageCheckResult(BaseModel):
    """Result of one leakage prevention check."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    check: LeakageCheck

    status: LeakageCheckStatus

    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_check(self) -> Self:
        if (
            self.status != LeakageCheckStatus.PASSED
            and self.reason is None
        ):
            raise ValueError(
                "Non-passing leakage checks require a reason."
            )

        return self


class LeakageValidationResult(BaseModel):
    """Leakage assessment bound to a specific registered split."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    validation_id: UUID = Field(default_factory=uuid4)

    readiness: TrainingReadinessResult

    manifest: SplitManifest

    checks: tuple[LeakageCheckResult, ...] = Field(
        min_length=1,
    )

    report: ValidationReport

    @property
    def required_checks(self) -> frozenset[LeakageCheck]:
        contract = self.readiness.contract

        required = {
            LeakageCheck.RECORD_OVERLAP,
            LeakageCheck.RECORD_COVERAGE,
            LeakageCheck.RECORD_IDENTITY,
            LeakageCheck.DUPLICATE_CONTENT,
            LeakageCheck.HISTORICAL_USAGE,
            LeakageCheck.FEATURE_AVAILABILITY,
        }

        if contract.grouping_columns:
            required.add(LeakageCheck.GROUP_OVERLAP)

        if self.manifest.configuration.strategy in {
            SplitStrategy.TEMPORAL,
            SplitStrategy.GROUP_TEMPORAL,
        }:
            required.add(LeakageCheck.TEMPORAL_ORDER)

        return frozenset(required)

    @property
    def eligible_for_training(self) -> bool:
        if not self.readiness.eligible_for_split:
            return False

        if self.report.status != ReadinessStatus.READY:
            return False

        check_map = {
            item.check: item.status
            for item in self.checks
        }

        if any(
            check_map.get(check) != LeakageCheckStatus.PASSED
            for check in self.required_checks
        ):
            return False

        return all(
            item.status in {
                LeakageCheckStatus.PASSED,
                LeakageCheckStatus.NOT_APPLICABLE,
            }
            for item in self.checks
        )

    def require_training_eligibility(self) -> None:
        if not self.eligible_for_training:
            raise ValueError(
                "Leakage validation does not permit production training."
            )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        self.manifest.validate_readiness(self.readiness)

        names = [
            item.check
            for item in self.checks
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Leakage checks must be unique."
            )

        if not self.required_checks <= set(names):
            raise ValueError(
                "Required leakage checks are missing."
            )

        for item in self.checks:
            if (
                item.check in self.required_checks
                and item.status == LeakageCheckStatus.NOT_APPLICABLE
            ):
                raise ValueError(
                    "Required leakage checks cannot be not applicable."
                )

        context = self.manifest.context

        expected = {
            "dataset_id": context.dataset_id,
            "contract_version": context.contract_version,
            "dataset_revision_id": self.manifest.dataset_revision_id,
            "domain": context.domain,
            "process": context.process,
            "purpose": context.purpose,
        }

        for field, value in expected.items():
            if getattr(self.report, field) != value:
                raise ValueError(
                    f"Leakage report mismatch for '{field}'."
                )

        if self.report.completed_stages != (
            PipelineStage.SPLITTING,
        ):
            raise ValueError(
                "Leakage report requires the splitting stage."
            )

        if self.report.created_at < self.manifest.created_at:
            raise ValueError(
                "Leakage report cannot precede split creation."
            )

        has_failed_checks = any(
            item.status == LeakageCheckStatus.FAILED
            for item in self.checks
        )

        if (
            has_failed_checks
            and self.report.status != ReadinessStatus.BLOCKED
        ):
            raise ValueError(
                "Failed leakage checks require a blocked report."
            )

        has_unverified_checks = any(
            item.status == LeakageCheckStatus.NOT_VERIFIED
            for item in self.checks
        )

        if (
            has_unverified_checks
            and self.report.status == ReadinessStatus.READY
        ):
            raise ValueError(
                "Unverified leakage checks cannot have a ready report."
            )

        return self


class TrainingAuthorization(BaseModel):
    """Development-only metadata issued after leakage validation.

    Protected test records are intentionally excluded.

    This object is not a security credential.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    authorization_id: UUID = Field(default_factory=uuid4)

    leakage_validation_id: UUID

    development_split: DevelopmentSplit

    @classmethod
    def from_validation(
        cls,
        validation: LeakageValidationResult,
    ) -> Self:
        validation.require_training_eligibility()

        return cls(
            leakage_validation_id=validation.validation_id,
            development_split=DevelopmentSplit.from_manifest(
                validation.manifest
            ),
        )