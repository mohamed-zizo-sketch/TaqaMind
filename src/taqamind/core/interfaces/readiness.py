"""Training readiness contracts for industrial ML tasks."""

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    PipelineStage,
    ReadinessStatus,
)
from taqamind.core.interfaces.dataset import DatasetState
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
)


class TrainingReadinessResult(BaseModel):
    """Result of assessing a prepared dataset for training."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    result: StageResult

    contract: DatasetContract

    @property
    def status(self) -> ReadinessStatus:
        """Return the effective readiness status."""

        if self.result.status != ExecutionStatus.SUCCEEDED:
            return ReadinessStatus.BLOCKED

        if self.result.report is None:
            return ReadinessStatus.BLOCKED

        return self.result.report.status

    @property
    def eligible_for_split(self) -> bool:
        """Check eligibility to proceed to split validation."""

        return (
            self.status == ReadinessStatus.READY
            and self.contract.target.is_confirmed
        )

    def require_split_eligibility(self) -> None:
        """Reject automatic continuation when readiness fails."""

        if not self.eligible_for_split:
            raise ValueError(
                "Dataset is not eligible for production splitting."
            )

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if self.result.stage != PipelineStage.READINESS:
            raise ValueError(
                "Readiness result requires the readiness stage."
            )

        context = self.result.context

        if context.purpose != DatasetPurpose.TRAINING:
            raise ValueError(
                "Training readiness requires a training context."
            )

        context.validate_dataset_contract(self.contract)

        metadata = self.result.input_artifact.metadata

        if metadata.state != DatasetState.ENGINEERED:
            raise ValueError(
                "Training readiness requires an engineered dataset."
            )

        if (
            self.status == ReadinessStatus.READY
            and not self.contract.target.is_confirmed
        ):
            raise ValueError(
                "An unconfirmed target cannot be marked ready."
            )

        if (
            self.status == ReadinessStatus.READY
            and metadata.row_count == 0
        ):
            raise ValueError(
                "An empty dataset cannot be marked ready."
            )

        return self