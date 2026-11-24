"""Factory-scoped model registry and lifecycle contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.feature import FeatureName, NonEmptyString
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.candidates import CandidateModelReference
from taqamind.core.interfaces.evaluation import (
    EvaluationDecision,
    FinalEvaluationResult,
    FrozenModelSelection,
)
from taqamind.core.interfaces.prediction import PredictionModelIdentity
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    RegressionMetric,
)


class ModelStatus(StrEnum):
    """Lifecycle status of a registered model."""

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ModelScope(BaseModel):
    """Factory and industrial task owning a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factory_id: UUID

    domain: FeatureName

    process: FeatureName

    task_id: FeatureName

    @classmethod
    def from_identity(
        cls,
        identity: PredictionModelIdentity,
    ) -> Self:
        return cls(
            factory_id=identity.factory_id,
            domain=identity.domain,
            process=identity.process,
            task_id=identity.task_id,
        )

    def validate_identity(
        self,
        identity: PredictionModelIdentity,
    ) -> None:
        if self != self.from_identity(identity):
            raise ValueError("Model scope mismatch.")


class ModelFeatureSpecification(BaseModel):
    """Ordered input features expected by a fitted model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_names: tuple[FeatureName, ...] = Field(min_length=1)

    target_name: FeatureName

    target_unit: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Model features must be unique.")

        if self.target_name in self.feature_names:
            raise ValueError("Target cannot be a model predictor.")

        return self

    def validate_contract(
        self,
        contract: DatasetContract,
    ) -> None:
        if not contract.target.is_confirmed:
            raise ValueError("Model requires a confirmed target.")

        if self.target_name != contract.target.feature.name:
            raise ValueError("Model target name mismatch.")

        if self.target_unit != contract.target.feature.unit:
            raise ValueError("Model target unit mismatch.")

        approved = set(contract.predictor_names)

        if not set(self.feature_names) <= approved:
            raise ValueError("Model contains unapproved predictors.")

        required = {
            feature.name
            for feature in contract.features
            if feature.name in approved and feature.required
        }

        if not required <= set(self.feature_names):
            raise ValueError("Model is missing required predictors.")

        for name in self.feature_names:
            feature = contract.get_feature(name)

            if not contract.is_available_at_prediction(feature):
                raise ValueError(
                    f"Predictor '{name}' is unavailable at prediction time."
                )


class ModelVersion(BaseModel):
    """Immutable description of one trained model version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: PredictionModelIdentity

    version_number: int = Field(ge=1)

    training_run_id: UUID

    selection_id: UUID

    training_dataset_revision_id: UUID

    split_id: UUID

    candidate: CandidateConfiguration

    artifact: CandidateModelReference

    feature_specification: ModelFeatureSpecification

    contract_sha256: SHA256Digest

    implementation_sha256: SHA256Digest

    previous_model_version_id: UUID | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def model_version_id(self) -> UUID:
        return self.identity.model_version_id

    @property
    def scope(self) -> ModelScope:
        return ModelScope.from_identity(self.identity)

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError("Model timestamp must be timezone-aware.")

        if self.artifact.factory_id != self.identity.factory_id:
            raise ValueError("Model artifact factory mismatch.")

        if self.artifact.candidate_id != self.candidate.candidate_id:
            raise ValueError("Model artifact candidate mismatch.")

        if self.previous_model_version_id == self.model_version_id:
            raise ValueError("Model cannot reference itself.")

        return self

    def validate_selection(
        self,
        selection: FrozenModelSelection,
    ) -> None:
        run = selection.training_run
        development = run.development_split
        context = development.context

        expected_scope = ModelScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
        )

        expected_scope.validate_identity(self.identity)

        fields = (
            "dataset_id",
            "contract_version",
            "task_version",
            "target_version",
        )

        for field in fields:
            if getattr(self.identity, field) != getattr(context, field):
                raise ValueError(f"Model contract mismatch for '{field}'.")

        if self.training_run_id != run.training_run_id:
            raise ValueError("Model training run mismatch.")

        if self.selection_id != selection.selection_id:
            raise ValueError("Model selection mismatch.")

        if self.split_id != development.split_id:
            raise ValueError("Model split mismatch.")

        if (
            self.training_dataset_revision_id
            != development.dataset_revision_id
        ):
            raise ValueError("Model training dataset mismatch.")

        if self.candidate.candidate_id != selection.selected_candidate_id:
            raise ValueError("Model candidate does not match selection.")

        selected_configuration = run.configuration.get_candidate(
            selection.selected_candidate_id
        )

        if self.candidate != selected_configuration:
            raise ValueError("Model candidate configuration mismatch.")

        selected_result = run.best_challenger

        if selected_result is None:
            raise ValueError("Selected training result is unavailable.")

        if self.artifact != selected_result.model_reference:
            raise ValueError("Model artifact does not match training result.")

        if self.created_at < selection.selected_at:
            raise ValueError("Model registration precedes selection.")


class ModelRegistryEntry(BaseModel):
    """Registered model version with its current lifecycle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: ModelVersion

    status: ModelStatus = ModelStatus.CANDIDATE

    evaluation_id: UUID | None = None

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        if self.updated_at.utcoffset() is None:
            raise ValueError("Registry timestamp must be timezone-aware.")

        if self.updated_at < self.version.created_at:
            raise ValueError("Registry update precedes model creation.")

        if self.status == ModelStatus.CANDIDATE:
            if self.evaluation_id is not None:
                raise ValueError(
                    "Candidate cannot contain a final evaluation."
                )

        elif self.evaluation_id is None:
            raise ValueError(
                "Non-candidate model requires a final evaluation."
            )

        return self

    def validate_evaluation(
        self,
        evaluation: FinalEvaluationResult,
    ) -> None:
        self.version.validate_selection(evaluation.selection)

        if self.evaluation_id != evaluation.evaluation_id:
            raise ValueError("Model evaluation identifier mismatch.")

        if self.updated_at < evaluation.timing.finished_at:
            raise ValueError("Registry update precedes final evaluation.")

        expected_decision = (
            EvaluationDecision.REJECTED
            if self.status == ModelStatus.REJECTED
            else EvaluationDecision.ACCEPTED
        )

        if evaluation.decision != expected_decision:
            raise ValueError(
                "Model status conflicts with final evaluation decision."
            )

    def apply_evaluation(
        self,
        evaluation: FinalEvaluationResult,
    ) -> Self:
        if self.status != ModelStatus.CANDIDATE:
            raise ValueError(
                "Only candidate models can receive final evaluation."
            )

        self.version.validate_selection(evaluation.selection)

        status = (
            ModelStatus.VALIDATED
            if evaluation.decision == EvaluationDecision.ACCEPTED
            else ModelStatus.REJECTED
        )

        entry = self.__class__(
            version=self.version,
            status=status,
            evaluation_id=evaluation.evaluation_id,
        )

        entry.validate_evaluation(evaluation)

        return entry


class ModelTransitionAction(StrEnum):
    """Supported production lifecycle operations."""

    FIRST_ACTIVATION = "first_activation"
    PROMOTION = "promotion"
    ROLLBACK = "rollback"


class PromotionComparison(BaseModel):
    """Performance comparison on a common independent dataset."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    comparison_id: UUID = Field(default_factory=uuid4)

    scope: ModelScope

    active_model_version_id: UUID

    challenger_model_version_id: UUID

    evaluation_dataset_revision_id: UUID

    evaluation_record_ids: tuple[UUID, ...] = Field(min_length=1)

    metric: RegressionMetric = RegressionMetric.MAE

    active_mae: float = Field(ge=0)

    challenger_mae: float = Field(ge=0)

    minimum_improvement_pct: float = Field(gt=0, le=100)

    @property
    def improvement_pct(self) -> float | None:
        if self.active_mae == 0:
            return None

        return (
            (self.active_mae - self.challenger_mae)
            / self.active_mae
        ) * 100

    @property
    def passes(self) -> bool:
        improvement = self.improvement_pct

        return (
            improvement is not None
            and improvement >= self.minimum_improvement_pct
        )

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.metric != RegressionMetric.MAE:
            raise ValueError("Promotion comparison requires MAE.")

        if self.active_model_version_id == self.challenger_model_version_id:
            raise ValueError("Promotion requires different model versions.")

        if len(self.evaluation_record_ids) != len(
            set(self.evaluation_record_ids)
        ):
            raise ValueError("Comparison record IDs must be unique.")

        return self


class ModelTransitionRequest(BaseModel):
    """Explicit request to change the active model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID = Field(default_factory=uuid4)

    scope: ModelScope

    action: ModelTransitionAction

    target_model_version_id: UUID

    expected_active_model_version_id: UUID | None = None

    requested_by: NonEmptyString

    reason: NonEmptyString

    comparison: PromotionComparison | None = None

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.action == ModelTransitionAction.FIRST_ACTIVATION:
            if self.expected_active_model_version_id is not None:
                raise ValueError(
                    "First activation requires no existing active model."
                )

        elif self.expected_active_model_version_id is None:
            raise ValueError(
                "Promotion and rollback require the current active model."
            )

        if (
            self.target_model_version_id
            == self.expected_active_model_version_id
        ):
            raise ValueError("Target model is already active.")

        if self.action == ModelTransitionAction.PROMOTION:
            if self.comparison is None:
                raise ValueError("Promotion requires a model comparison.")

            if self.comparison.scope != self.scope:
                raise ValueError("Promotion comparison scope mismatch.")

            if (
                self.comparison.active_model_version_id
                != self.expected_active_model_version_id
            ):
                raise ValueError("Comparison active model mismatch.")

            if (
                self.comparison.challenger_model_version_id
                != self.target_model_version_id
            ):
                raise ValueError("Comparison challenger model mismatch.")

            if not self.comparison.passes:
                raise ValueError(
                    "Challenger does not meet promotion requirements."
                )

        elif self.comparison is not None:
            raise ValueError(
                "Comparison is only supported for promotion requests."
            )

        return self


def validate_model_transition(
    *,
    request: ModelTransitionRequest,
    target: ModelRegistryEntry,
    active: ModelRegistryEntry | None,
) -> None:
    """Validate a lifecycle transition against current registry state."""

    request.scope.validate_identity(target.version.identity)

    if target.version.model_version_id != request.target_model_version_id:
        raise ValueError("Transition target model mismatch.")

    if active is not None:
        request.scope.validate_identity(active.version.identity)

        if active.status != ModelStatus.ACTIVE:
            raise ValueError("Current model must have active status.")

    actual_active_id = (
        active.version.model_version_id
        if active is not None
        else None
    )

    if actual_active_id != request.expected_active_model_version_id:
        raise ValueError("Active model changed since request creation.")

    expected_target_status = (
        ModelStatus.ARCHIVED
        if request.action == ModelTransitionAction.ROLLBACK
        else ModelStatus.VALIDATED
    )

    if target.status != expected_target_status:
        raise ValueError("Target model has an invalid lifecycle status.")

    if active is not None:
        fields = (
            "dataset_id",
            "contract_version",
            "task_version",
            "target_version",
        )

        for field in fields:
            if (
                getattr(active.version.identity, field)
                != getattr(target.version.identity, field)
            ):
                raise ValueError(
                    f"Model transition requires compatible '{field}'."
                )

        if target.version.contract_sha256 != active.version.contract_sha256:
            raise ValueError(
                "Model transition requires compatible contract content."
            )


@runtime_checkable
class ModelRegistry(Protocol):
    """Registry for immutable model versions and active model state."""

    def register_candidate(
        self,
        *,
        scope: ModelScope,
        version: ModelVersion,
    ) -> ModelRegistryEntry:
        """Verify selection, artifact integrity and version uniqueness."""
        ...

    def apply_evaluation(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> ModelRegistryEntry:
        """Apply a verified registered final evaluation."""
        ...

    def get(
        self,
        *,
        scope: ModelScope,
        model_version_id: UUID,
    ) -> ModelRegistryEntry:
        ...

    def get_active(
        self,
        *,
        scope: ModelScope,
    ) -> ModelRegistryEntry | None:
        """Return the active model without triggering training."""
        ...

    def list_versions(
        self,
        *,
        scope: ModelScope,
    ) -> tuple[ModelRegistryEntry, ...]:
        ...

    def transition(
        self,
        *,
        request: ModelTransitionRequest,
    ) -> ModelRegistryEntry:
        """Atomically activate the target and archive any active model.

        Implementations must verify authorization, evaluation evidence,
        artifact integrity, compatibility and current registry state.

        Lifecycle changes must be recorded in an audit history.
        """
        ...