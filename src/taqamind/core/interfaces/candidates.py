"""Model candidate results for controlled regression training."""

from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.feature import NonEmptyString
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.results import (
    ExecutionError,
    StageTiming,
)
from taqamind.core.interfaces.training import (
    CandidateRole,
    DevelopmentSplit,
    RegressionMetric,
    TrainingConfiguration,
)


class CandidateStatus(StrEnum):
    """Final status of an individual model training trial."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationMetric(BaseModel):
    """A regression metric calculated on validation data."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    metric: RegressionMetric

    value: float | None

    undefined_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        if self.value is None:
            if self.undefined_reason is None:
                raise ValueError(
                    "Undefined metrics require a reason."
                )

            return self

        if self.undefined_reason is not None:
            raise ValueError(
                "Defined metrics cannot have an undefined reason."
            )

        if self.metric in {
            RegressionMetric.MAE,
            RegressionMetric.RMSE,
            RegressionMetric.WAPE,
        }:
            if self.value < 0:
                raise ValueError(
                    "Error metrics cannot be negative."
                )

        if self.metric == RegressionMetric.R2:
            if self.value > 1:
                raise ValueError(
                    "R2 cannot exceed one."
                )

        return self


class CandidateModelReference(BaseModel):
    """Reference to a fitted candidate model bundle."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_id: UUID = Field(default_factory=uuid4)

    factory_id: UUID

    execution_id: UUID

    candidate_id: UUID

    content_sha256: SHA256Digest

    size_bytes: int = Field(gt=0)

    @property
    def storage_key(self) -> str:
        return (
            f"factories/{self.factory_id}/"
            f"training/{self.execution_id}/"
            f"candidates/{self.candidate_id}/"
            f"{self.artifact_id}.bundle"
        )


class CandidateResult(BaseModel):
    """Outcome of a single model training trial."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    result_id: UUID = Field(default_factory=uuid4)

    candidate_id: UUID

    status: CandidateStatus

    timing: StageTiming

    validation_metrics: tuple[ValidationMetric, ...] = ()

    evaluated_row_count: int | None = Field(
        default=None,
        gt=0,
    )

    model_reference: CandidateModelReference | None = None

    error: ExecutionError | None = None

    skip_reason: NonEmptyString | None = None

    def get_metric(
        self,
        metric: RegressionMetric,
    ) -> float | None:
        for item in self.validation_metrics:
            if item.metric == metric:
                return item.value

        return None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        metrics = [
            item.metric
            for item in self.validation_metrics
        ]

        if len(metrics) != len(set(metrics)):
            raise ValueError(
                "Validation metrics must be unique."
            )

        if self.status == CandidateStatus.SUCCEEDED:
            if self.model_reference is None:
                raise ValueError(
                    "Successful candidates require a model reference."
                )

            if self.model_reference.candidate_id != self.candidate_id:
                raise ValueError(
                    "Candidate model reference mismatch."
                )

            if self.evaluated_row_count is None:
                raise ValueError(
                    "Successful candidates require evaluation row count."
                )

            if self.get_metric(RegressionMetric.MAE) is None:
                raise ValueError(
                    "Successful candidates require a defined MAE."
                )

            if self.error is not None or self.skip_reason is not None:
                raise ValueError(
                    "Successful candidate cannot contain failure information."
                )

        else:
            if (
                self.model_reference is not None
                or self.validation_metrics
                or self.evaluated_row_count is not None
            ):
                raise ValueError(
                    "Unsuccessful candidates cannot expose usable model results."
                )

            if self.status == CandidateStatus.FAILED:
                if self.error is None:
                    raise ValueError(
                        "Failed candidates require error information."
                    )

                if self.skip_reason is not None:
                    raise ValueError(
                        "Failed candidates cannot contain a skip reason."
                    )

            elif self.status == CandidateStatus.SKIPPED:
                if self.skip_reason is None:
                    raise ValueError(
                        "Skipped candidates require a reason."
                    )

                if self.error is not None:
                    raise ValueError(
                        "Skipped candidates cannot contain execution errors."
                    )

        return self


class TrainingRunResult(BaseModel):
    """Development results of a controlled training execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    training_run_id: UUID = Field(default_factory=uuid4)

    configuration: TrainingConfiguration

    development_split: DevelopmentSplit

    results: tuple[CandidateResult, ...]

    @property
    def successful_results(self) -> tuple[CandidateResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.status == CandidateStatus.SUCCEEDED
        )

    def get_best_result(
        self,
        role: CandidateRole,
    ) -> CandidateResult | None:
        candidates = [
            result
            for result in self.successful_results
            if self.configuration.get_candidate(
                result.candidate_id
            ).role == role
        ]

        if not candidates:
            return None

        def ranking_key(result: CandidateResult) -> tuple[float, str]:
            value = result.get_metric(
                self.configuration.primary_metric
            )

            if value is None:
                raise ValueError(
                    "Successful candidate has no primary metric."
                )

            return value, str(result.candidate_id)

        return min(
            candidates,
            key=ranking_key,
        )

    @property
    def best_baseline(self) -> CandidateResult | None:
        return self.get_best_result(
            CandidateRole.BASELINE
        )

    @property
    def best_challenger(self) -> CandidateResult | None:
        return self.get_best_result(
            CandidateRole.CHALLENGER
        )

    @property
    def validation_improvement_pct(self) -> float | None:
        baseline = self.best_baseline
        challenger = self.best_challenger

        if baseline is None or challenger is None:
            return None

        baseline_mae = baseline.get_metric(
            RegressionMetric.MAE
        )

        challenger_mae = challenger.get_metric(
            RegressionMetric.MAE
        )

        if baseline_mae is None or challenger_mae is None:
            return None

        if baseline_mae == 0:
            return None

        return (
            (baseline_mae - challenger_mae)
            / baseline_mae
        ) * 100

    @property
    def eligible_for_final_evaluation(self) -> bool:
        baseline = self.best_baseline
        challenger = self.best_challenger

        if baseline is None or challenger is None:
            return False

        dummy_succeeded = any(
            self.configuration.get_candidate(
                result.candidate_id
            ).family.value == "dummy_regressor"
            for result in self.successful_results
        )

        if not dummy_succeeded:
            return False

        baseline_mae = baseline.get_metric(
            RegressionMetric.MAE
        )

        challenger_mae = challenger.get_metric(
            RegressionMetric.MAE
        )

        return (
            baseline_mae is not None
            and challenger_mae is not None
            and challenger_mae < baseline_mae
        )

    @model_validator(mode="after")
    def validate_training_run(self) -> Self:
        expected_ids = {
            candidate.candidate_id
            for candidate in self.configuration.candidates
        }

        actual_ids = [
            result.candidate_id
            for result in self.results
        ]

        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError(
                "Candidate results must be unique."
            )

        if set(actual_ids) != expected_ids:
            raise ValueError(
                "Training results must account for every candidate."
            )

        context = self.development_split.context

        expected_rows = (
            self.development_split.validation.row_count
        )

        expected_metrics = {
            self.configuration.primary_metric,
            *self.configuration.secondary_metrics,
        }

        artifact_ids = set()

        for result in self.results:
            if result.timing.started_at < context.created_at:
                raise ValueError(
                    "Candidate training precedes execution creation."
                )

            if result.status != CandidateStatus.SUCCEEDED:
                continue

            if result.evaluated_row_count != expected_rows:
                raise ValueError(
                    "Candidate validation row count mismatch."
                )

            actual_metrics = {
                metric.metric
                for metric in result.validation_metrics
            }

            if actual_metrics != expected_metrics:
                raise ValueError(
                    "Candidate evaluation metrics do not match configuration."
                )

            reference = result.model_reference

            if reference is None:
                raise ValueError(
                    "Missing candidate model reference."
                )

            if reference.factory_id != context.factory_id:
                raise ValueError(
                    "Candidate model factory mismatch."
                )

            if reference.execution_id != context.execution_id:
                raise ValueError(
                    "Candidate model execution mismatch."
                )

            if reference.artifact_id in artifact_ids:
                raise ValueError(
                    "Candidate model artifacts must be unique."
                )

            artifact_ids.add(reference.artifact_id)

        return self