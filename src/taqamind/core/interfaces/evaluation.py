"""Protected final evaluation contracts for industrial regression."""

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

from taqamind.core.contracts.feature import NonEmptyString
from taqamind.core.interfaces.candidates import (
    TrainingRunResult,
    ValidationMetric,
)
from taqamind.core.interfaces.results import StageTiming
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.training import (
    DevelopmentSplit,
    RegressionMetric,
)


class EvaluationDecision(StrEnum):
    """Final model evaluation decision."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AcceptancePolicy(BaseModel):
    """Predefined acceptance requirements for regression models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    policy_id: UUID = Field(default_factory=uuid4)

    minimum_improvement_pct: float = Field(
        gt=0,
        le=100,
    )

    maximum_test_mae: float | None = Field(
        default=None,
        gt=0,
    )

    justification: NonEmptyString

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Policy timestamp must be timezone-aware."
            )

        return self


class FrozenModelSelection(BaseModel):
    """Model selection fixed before protected test evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    selection_id: UUID = Field(default_factory=uuid4)

    training_run: TrainingRunResult

    policy: AcceptancePolicy

    baseline_candidate_id: UUID

    selected_candidate_id: UUID

    selected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @classmethod
    def from_training_run(
        cls,
        *,
        training_run: TrainingRunResult,
        policy: AcceptancePolicy,
    ) -> Self:
        if not training_run.eligible_for_final_evaluation:
            raise ValueError(
                "Training run is not eligible for final evaluation."
            )

        baseline = training_run.best_baseline
        challenger = training_run.best_challenger

        if baseline is None or challenger is None:
            raise ValueError(
                "Model selection requires a baseline and challenger."
            )

        return cls(
            training_run=training_run,
            policy=policy,
            baseline_candidate_id=baseline.candidate_id,
            selected_candidate_id=challenger.candidate_id,
        )

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selected_at.utcoffset() is None:
            raise ValueError(
                "Selection timestamp must be timezone-aware."
            )

        run = self.training_run

        if not run.eligible_for_final_evaluation:
            raise ValueError(
                "Training run is not eligible for final evaluation."
            )

        baseline = run.best_baseline
        challenger = run.best_challenger

        if baseline is None or challenger is None:
            raise ValueError(
                "Model selection requires a baseline and challenger."
            )

        if self.baseline_candidate_id != baseline.candidate_id:
            raise ValueError(
                "Selected baseline does not match training results."
            )

        if self.selected_candidate_id != challenger.candidate_id:
            raise ValueError(
                "Selected challenger does not match training results."
            )

        context = run.development_split.context

        if self.policy.created_at > context.created_at:
            raise ValueError(
                "Acceptance policy must be defined before training."
            )

        for result in run.results:
            if result.timing.finished_at > self.selected_at:
                raise ValueError(
                    "Model selection cannot precede training completion."
                )

        return self


class TestMetricResult(BaseModel):
    """Regression metrics measured on the protected test partition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    candidate_id: UUID

    model_artifact_id: UUID

    record_ids: tuple[UUID, ...] = Field(
        min_length=1,
    )

    metrics: tuple[ValidationMetric, ...] = Field(
        min_length=1,
    )

    @property
    def evaluated_row_count(self) -> int:
        return len(self.record_ids)

    def get_metric(
        self,
        metric: RegressionMetric,
    ) -> float | None:
        for item in self.metrics:
            if item.metric == metric:
                return item.value

        return None

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError(
                "Test record identifiers must be unique."
            )

        metric_names = [
            item.metric
            for item in self.metrics
        ]

        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "Test metrics must be unique."
            )

        if self.get_metric(RegressionMetric.MAE) is None:
            raise ValueError(
                "Final evaluation requires a defined MAE."
            )

        return self


class FinalEvaluationResult(BaseModel):
    """Final evaluation of a frozen model selection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evaluation_id: UUID = Field(default_factory=uuid4)

    selection: FrozenModelSelection

    split_manifest: SplitManifest

    baseline_result: TestMetricResult

    selected_result: TestMetricResult

    timing: StageTiming

    @property
    def baseline_mae(self) -> float:
        value = self.baseline_result.get_metric(
            RegressionMetric.MAE
        )

        if value is None:
            raise ValueError("Baseline MAE is undefined.")

        return value

    @property
    def selected_mae(self) -> float:
        value = self.selected_result.get_metric(
            RegressionMetric.MAE
        )

        if value is None:
            raise ValueError("Selected model MAE is undefined.")

        return value

    @property
    def test_improvement_pct(self) -> float | None:
        if self.baseline_mae == 0:
            return None

        return (
            (self.baseline_mae - self.selected_mae)
            / self.baseline_mae
        ) * 100

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        reasons = []

        improvement = self.test_improvement_pct
        policy = self.selection.policy

        if improvement is None:
            reasons.append("ZERO_BASELINE_MAE")

        elif improvement < policy.minimum_improvement_pct:
            reasons.append("INSUFFICIENT_BASELINE_IMPROVEMENT")

        if (
            policy.maximum_test_mae is not None
            and self.selected_mae > policy.maximum_test_mae
        ):
            reasons.append("MAXIMUM_TEST_MAE_EXCEEDED")

        return tuple(reasons)

    @property
    def decision(self) -> EvaluationDecision:
        if self.rejection_reasons:
            return EvaluationDecision.REJECTED

        return EvaluationDecision.ACCEPTED

    @property
    def eligible_for_registry_validation(self) -> bool:
        return self.decision == EvaluationDecision.ACCEPTED

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        selection = self.selection
        run = selection.training_run

        expected_development = DevelopmentSplit.from_manifest(
            self.split_manifest
        )

        if expected_development != run.development_split:
            raise ValueError(
                "Final evaluation split does not match training split."
            )

        if self.timing.started_at < selection.selected_at:
            raise ValueError(
                "Final evaluation cannot precede model selection."
            )

        expected_records = set(
            self.split_manifest.test.record_ids
        )

        expected_metrics = {
            run.configuration.primary_metric,
            *run.configuration.secondary_metrics,
        }

        evaluation_pairs = (
            (
                self.baseline_result,
                selection.baseline_candidate_id,
            ),
            (
                self.selected_result,
                selection.selected_candidate_id,
            ),
        )

        training_results = {
            result.candidate_id: result
            for result in run.successful_results
        }

        for evaluation, candidate_id in evaluation_pairs:
            if evaluation.candidate_id != candidate_id:
                raise ValueError(
                    "Test candidate does not match frozen selection."
                )

            if set(evaluation.record_ids) != expected_records:
                raise ValueError(
                    "Evaluation must use the exact protected test records."
                )

            actual_metrics = {
                item.metric
                for item in evaluation.metrics
            }

            if actual_metrics != expected_metrics:
                raise ValueError(
                    "Final evaluation metrics do not match configuration."
                )

            reference = training_results[candidate_id].model_reference

            if reference is None:
                raise ValueError(
                    "Evaluated model reference is missing."
                )

            if evaluation.model_artifact_id != reference.artifact_id:
                raise ValueError(
                    "Evaluated model does not match selected model artifact."
                )

        return self