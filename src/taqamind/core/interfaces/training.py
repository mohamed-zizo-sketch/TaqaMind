"""Configuration contracts for controlled industrial model training."""

from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DatasetPurpose, TaskType
from taqamind.core.contracts.feature import FeatureName, NonEmptyString
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.splitting import (
    SplitManifest,
    SplitPartition,
    SplitRole,
)


class ModelFamily(StrEnum):
    """Approved model families for industrial regression."""

    DUMMY_REGRESSOR = "dummy_regressor"
    HISTORICAL_RATE = "historical_rate"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    XGBOOST = "xgboost"
    HURDLE_REGRESSION = "hurdle_regression"


class CandidateRole(StrEnum):
    """Role of a model within the training process."""

    BASELINE = "baseline"
    CHALLENGER = "challenger"


BASELINE_FAMILIES = frozenset({
    ModelFamily.DUMMY_REGRESSOR,
    ModelFamily.HISTORICAL_RATE,
})


class RegressionMetric(StrEnum):
    """Supported regression evaluation metrics."""

    MAE = "mae"
    RMSE = "rmse"
    WAPE = "wape"
    R2 = "r2"


ParameterValue = str | int | float | bool | None


class ModelParameter(BaseModel):
    """One immutable model hyperparameter."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        strict=True,
    )

    name: FeatureName

    value: ParameterValue


class CandidateConfiguration(BaseModel):
    """One candidate configuration to be trained."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    candidate_id: UUID = Field(default_factory=uuid4)

    family: ModelFamily

    role: CandidateRole

    parameters: tuple[ModelParameter, ...] = ()

    @property
    def parameter_map(self) -> dict[str, ParameterValue]:
        return {
            parameter.name: parameter.value
            for parameter in self.parameters
        }

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        names = [
            parameter.name
            for parameter in self.parameters
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Model parameter names must be unique."
            )

        is_baseline = self.family in BASELINE_FAMILIES

        if is_baseline and self.role != CandidateRole.BASELINE:
            raise ValueError(
                "Baseline families require the baseline role."
            )

        if not is_baseline and self.role != CandidateRole.CHALLENGER:
            raise ValueError(
                "Candidate family requires the challenger role."
            )

        return self


class TrainingBudget(BaseModel):
    """Resource constraints for one training execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_trials: int = Field(default=20, ge=1)

    max_training_seconds: int = Field(
        default=600,
        ge=1,
    )

    max_parallel_jobs: int = Field(
        default=1,
        ge=1,
    )

    max_memory_mb: int = Field(
        default=4096,
        ge=1,
    )


class TrainingConfiguration(BaseModel):
    """Controlled configuration for a regression training run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    configuration_id: UUID = Field(
        default_factory=uuid4,
    )

    task_type: TaskType = TaskType.REGRESSION

    primary_metric: RegressionMetric = RegressionMetric.MAE

    secondary_metrics: tuple[RegressionMetric, ...] = (
        RegressionMetric.RMSE,
        RegressionMetric.WAPE,
        RegressionMetric.R2,
    )

    candidates: tuple[CandidateConfiguration, ...] = Field(
        min_length=2,
    )

    budget: TrainingBudget = Field(
        default_factory=TrainingBudget,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
    )

    justification: NonEmptyString

    @property
    def baselines(self) -> tuple[CandidateConfiguration, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.role == CandidateRole.BASELINE
        )

    @property
    def challengers(self) -> tuple[CandidateConfiguration, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.role == CandidateRole.CHALLENGER
        )

    def get_candidate(
        self,
        candidate_id: UUID,
    ) -> CandidateConfiguration:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate

        raise KeyError(
            f"Unknown candidate '{candidate_id}'."
        )

    def validate_contract(
        self,
        contract: DatasetContract,
    ) -> None:
        if contract.target.task_type != self.task_type:
            raise ValueError(
                "Training task type does not match target task type."
            )

        if not contract.target.is_confirmed:
            raise ValueError(
                "Production training requires a confirmed target."
            )

        if any(
            candidate.family == ModelFamily.HISTORICAL_RATE
            for candidate in self.candidates
        ):
            if contract.target.derived_rate is None:
                raise ValueError(
                    "Historical rate baseline requires a rate definition."
                )

            denominator = contract.get_feature(
                contract.target.derived_rate.denominator
            )

            if not contract.is_available_at_prediction(denominator):
                raise ValueError(
                    "Historical rate denominator is unavailable "
                    "at prediction time."
                )

            if denominator.name not in contract.predictor_names:
                raise ValueError(
                    "Historical rate denominator must be an approved predictor."
                )

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.task_type != TaskType.REGRESSION:
            raise ValueError(
                "Current training configuration supports regression only."
            )

        if self.primary_metric != RegressionMetric.MAE:
            raise ValueError(
                "MAE is the primary regression metric in V1."
            )

        if self.primary_metric in self.secondary_metrics:
            raise ValueError(
                "Primary metric cannot also be a secondary metric."
            )

        if len(self.secondary_metrics) != len(
            set(self.secondary_metrics)
        ):
            raise ValueError(
                "Secondary metrics must be unique."
            )

        candidate_ids = [
            candidate.candidate_id
            for candidate in self.candidates
        ]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                "Candidate identifiers must be unique."
            )

        if not any(
            candidate.family == ModelFamily.DUMMY_REGRESSOR
            for candidate in self.baselines
        ):
            raise ValueError(
                "A dummy regression baseline is mandatory."
            )

        if not self.challengers:
            raise ValueError(
                "At least one challenger model is required."
            )

        if len(self.candidates) > self.budget.max_trials:
            raise ValueError(
                "Candidate count exceeds training trial budget."
            )

        return self


class DevelopmentSplit(BaseModel):
    """Restricted split view for training and model selection.

    Protected test record identifiers are intentionally omitted.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    split_id: UUID

    context: PipelineContext

    dataset_revision_id: UUID

    train: SplitPartition

    validation: SplitPartition

    @model_validator(mode="after")
    def validate_development_split(self) -> Self:
        if self.context.purpose != DatasetPurpose.TRAINING:
            raise ValueError(
                "Development split requires a training context."
            )

        if self.train.role != SplitRole.TRAIN:
            raise ValueError(
                "Invalid development training partition."
            )

        if self.validation.role != SplitRole.VALIDATION:
            raise ValueError(
                "Invalid development validation partition."
            )

        if set(self.train.record_ids) & set(self.validation.record_ids):
            raise ValueError(
                "Development partitions contain overlapping records."
            )

        return self

    @classmethod
    def from_manifest(
        cls,
        manifest: SplitManifest,
    ) -> Self:
        return cls(
            split_id=manifest.split_id,
            context=manifest.context,
            dataset_revision_id=manifest.dataset_revision_id,
            train=manifest.train,
            validation=manifest.validation,
        )