"""Shared pipeline stage results and execution status contracts."""

from datetime import UTC, datetime
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
from taqamind.core.contracts.validation import (
    ErrorCode,
    ValidationReport,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    STATE_ORDER,
    STATE_STAGE,
    DatasetArtifact,
)


class ExecutionStatus(StrEnum):
    """Final execution state of a pipeline stage."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ExecutionError(BaseModel):
    """Safe operational error information."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    code: ErrorCode

    message: NonEmptyString

    retryable: bool = False

    diagnostic_id: UUID = Field(
        default_factory=uuid4,
    )


class StageTiming(BaseModel):
    """Execution timing for a completed stage."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    started_at: datetime

    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        """Return elapsed execution time in seconds."""

        return (
            self.finished_at - self.started_at
        ).total_seconds()

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.started_at.utcoffset() is None:
            raise ValueError(
                "Start timestamp must be timezone-aware."
            )

        if self.finished_at.utcoffset() is None:
            raise ValueError(
                "Finish timestamp must be timezone-aware."
            )

        if self.finished_at < self.started_at:
            raise ValueError(
                "Stage cannot finish before it starts."
            )

        return self


class StageResult(BaseModel):
    """Unified result of a dataset-oriented pipeline stage."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    result_id: UUID = Field(
        default_factory=uuid4,
    )

    context: PipelineContext

    stage: PipelineStage

    status: ExecutionStatus

    input_artifact: DatasetArtifact

    output_artifact: DatasetArtifact | None = None

    report: ValidationReport | None = None

    timing: StageTiming

    error: ExecutionError | None = None

    skip_reason: NonEmptyString | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def is_successful(self) -> bool:
        """Return whether the stage completed successfully."""

        return self.status == ExecutionStatus.SUCCEEDED

    @property
    def can_continue(self) -> bool:
        """Return whether this result permits default continuation."""

        if self.status != ExecutionStatus.SUCCEEDED:
            return False

        if self.report is None:
            return False

        return self.report.status == ReadinessStatus.READY

    @property
    def requires_review(self) -> bool:
        """Return whether the stage requires a policy decision."""

        if self.status == ExecutionStatus.BLOCKED:
            return True

        if self.report is not None:
            return self.report.status == ReadinessStatus.LIMITED

        return False

    @property
    def next_artifact(self) -> DatasetArtifact | None:
        """Return the artifact available for default continuation."""

        if not self.can_continue:
            return None

        if self.output_artifact is not None:
            return self.output_artifact

        return self.input_artifact

    def validate_artifacts(self) -> None:
        """Validate artifact scope and direct revision lineage."""

        input_metadata = self.input_artifact.metadata

        input_metadata.validate_scope(self.context)

        if self.output_artifact is None:
            return

        output_metadata = self.output_artifact.metadata

        output_metadata.validate_scope(self.context)

        if output_metadata.revision_id == input_metadata.revision_id:
            raise ValueError(
                "Output artifact must have a new revision."
            )

        if output_metadata.parent_revision_id != input_metadata.revision_id:
            raise ValueError(
                "Output revision must reference the input revision."
            )

        if output_metadata.root_revision_id != input_metadata.root_revision_id:
            raise ValueError(
                "Output revision must preserve the root revision."
            )

        if output_metadata.source_file_sha256 != input_metadata.source_file_sha256:
            raise ValueError(
                "Output revision must preserve the source hash."
            )

        if (
            STATE_ORDER[output_metadata.state]
            != STATE_ORDER[input_metadata.state] + 1
        ):
            raise ValueError(
                "Output must advance exactly one dataset state."
            )

        transformation = output_metadata.transformation

        if transformation is None:
            raise ValueError(
                "Output requires transformation metadata."
            )

        if transformation.execution_id != self.context.execution_id:
            raise ValueError(
                "Output transformation execution mismatch."
            )

        if transformation.stage != self.stage:
            raise ValueError(
                "Output transformation stage mismatch."
            )

        if (
            self.output_artifact.reference.artifact_id
            == self.input_artifact.reference.artifact_id
        ):
            raise ValueError(
                "Output must use a separate artifact identifier."
            )

    def validate_report(self) -> None:
        """Validate the identity and scope of the stage report."""

        if self.report is None:
            return

        report = self.report

        expected = {
            "dataset_id": self.context.dataset_id,
            "contract_version": self.context.contract_version,
            "dataset_revision_id": (
                self.input_artifact.metadata.revision_id
            ),
            "domain": self.context.domain,
            "process": self.context.process,
            "purpose": self.context.purpose,
        }

        for field, expected_value in expected.items():
            if getattr(report, field) != expected_value:
                raise ValueError(
                    f"Validation report mismatch for '{field}'."
                )

        if report.completed_stages != (self.stage,):
            raise ValueError(
                "Stage result requires a report for its own stage only."
            )

    def validate_status(self) -> None:
        """Reject contradictory execution results."""

        if self.status == ExecutionStatus.SUCCEEDED:
            if self.error is not None:
                raise ValueError(
                    "Successful stages cannot contain execution errors."
                )

            if self.report is None:
                raise ValueError(
                    "Successful stages require a validation report."
                )

            if self.report.status == ReadinessStatus.BLOCKED:
                raise ValueError(
                    "Successful stage cannot have a blocked report."
                )

            is_transformation = self.stage in STATE_STAGE.values()

            if is_transformation and self.output_artifact is None:
                raise ValueError(
                    "Transformation stage requires an output artifact."
                )

            if not is_transformation and self.output_artifact is not None:
                raise ValueError(
                    "Inspection stage cannot create an output revision."
                )

        elif self.status == ExecutionStatus.FAILED:
            if self.error is None:
                raise ValueError(
                    "Failed stages require execution error information."
                )

            if self.output_artifact is not None:
                raise ValueError(
                    "Failed stages cannot expose an output artifact."
                )

            if self.report is not None:
                raise ValueError(
                    "Failed stages cannot declare a completed validation report."
                )

        elif self.status == ExecutionStatus.BLOCKED:
            if self.report is None:
                raise ValueError(
                    "Blocked stages require a validation report."
                )

            if self.report.status != ReadinessStatus.BLOCKED:
                raise ValueError(
                    "Blocked stages require blocking validation issues."
                )

            if self.output_artifact is not None:
                raise ValueError(
                    "Blocked stages cannot expose an output artifact."
                )

            if self.error is not None:
                raise ValueError(
                    "Blocked stages must use validation issues."
                )

        elif self.status == ExecutionStatus.SKIPPED:
            if self.skip_reason is None:
                raise ValueError(
                    "Skipped stages require a reason."
                )

            if (
                self.output_artifact is not None
                or self.error is not None
                or self.report is not None
            ):
                raise ValueError(
                    "Skipped stages cannot contain execution outputs."
                )

        if (
            self.status != ExecutionStatus.SKIPPED
            and self.skip_reason is not None
        ):
            raise ValueError(
                "Only skipped stages may contain a skip reason."
            )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate the complete stage result."""

        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Result timestamp must be timezone-aware."
            )

        if self.timing.started_at < self.context.created_at:
            raise ValueError(
                "Stage cannot start before execution creation."
            )

        if self.created_at < self.timing.finished_at:
            raise ValueError(
                "Result cannot be created before stage completion."
            )

        self.validate_artifacts()
        self.validate_report()
        self.validate_status()

        return self