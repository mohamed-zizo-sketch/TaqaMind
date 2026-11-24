"""Shared machine learning protocols for industrial regression."""

from typing import Protocol, runtime_checkable

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DatasetPurpose
from taqamind.core.interfaces.candidates import TrainingRunResult
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetState,
)
from taqamind.core.interfaces.evaluation import (
    AcceptancePolicy,
    FinalEvaluationResult,
    FrozenModelSelection,
)
from taqamind.core.interfaces.leakage import (
    LeakageValidationResult,
    TrainingAuthorization,
)
from taqamind.core.interfaces.prediction import (
    PredictionBatchResult,
    PredictionModelIdentity,
)
from taqamind.core.interfaces.processing import (
    validate_processing_input,
)
from taqamind.core.interfaces.readiness import (
    TrainingReadinessResult,
)
from taqamind.core.interfaces.splitting import (
    SplitConfiguration,
    SplitManifest,
)
from taqamind.core.interfaces.training import (
    TrainingConfiguration,
)


def validate_training_request(
    *,
    authorization: TrainingAuthorization,
    contract: DatasetContract,
    configuration: TrainingConfiguration,
    policy: AcceptancePolicy,
) -> None:
    """Validate common training request requirements."""

    context = authorization.development_split.context

    if context.purpose != DatasetPurpose.TRAINING:
        raise ValueError(
            "Model training requires a training context."
        )

    context.validate_dataset_contract(contract)

    configuration.validate_contract(contract)

    if policy.created_at > context.created_at:
        raise ValueError(
            "Acceptance policy must be defined before training."
        )


def validate_prediction_request(
    *,
    context: PipelineContext,
    artifact: DatasetArtifact,
    contract: DatasetContract,
    model: PredictionModelIdentity,
) -> None:
    """Validate common prediction request requirements."""

    model.validate_context(context)

    validate_processing_input(
        context=context,
        artifact=artifact,
        contract=contract,
        expected_state=DatasetState.ENGINEERED,
    )

    if not contract.target.is_confirmed:
        raise ValueError(
            "Prediction requires a confirmed target."
        )

    if artifact.metadata.row_count == 0:
        raise ValueError(
            "Prediction dataset cannot be empty."
        )

    observed_columns = {
        column.name
        for column in artifact.metadata.columns
    }

    if contract.target.feature.name in observed_columns:
        raise ValueError(
            "Prediction artifact cannot contain the target."
        )

    required = set(contract.predictor_names)

    missing = required - observed_columns

    if missing:
        raise ValueError(
            f"Prediction predictors are missing: {sorted(missing)}"
        )


@runtime_checkable
class TrainingReadinessChecker(Protocol):
    """Assess whether a prepared dataset can proceed to splitting."""

    def assess(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> TrainingReadinessResult:
        ...


@runtime_checkable
class DatasetSplitter(Protocol):
    """Create a registered split from eligible industrial data."""

    def split(
        self,
        *,
        readiness: TrainingReadinessResult,
        configuration: SplitConfiguration,
    ) -> SplitManifest:
        ...


@runtime_checkable
class LeakageValidator(Protocol):
    """Validate record identity, split integrity and historical usage."""

    def validate(
        self,
        *,
        readiness: TrainingReadinessResult,
        manifest: SplitManifest,
    ) -> LeakageValidationResult:
        ...


@runtime_checkable
class ModelTrainer(Protocol):
    """Fit approved candidates using development data only."""

    def train(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
        configuration: TrainingConfiguration,
        policy: AcceptancePolicy,
    ) -> TrainingRunResult:
        ...


@runtime_checkable
class FinalModelEvaluator(Protocol):
    """Evaluate frozen model selection on a protected test partition."""

    def evaluate(
        self,
        *,
        selection: FrozenModelSelection,
        manifest: SplitManifest,
    ) -> FinalEvaluationResult:
        ...


@runtime_checkable
class PredictionEngine(Protocol):
    """Generate predictions using an existing registered model."""

    def predict(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        model: PredictionModelIdentity,
    ) -> PredictionBatchResult:
        ...