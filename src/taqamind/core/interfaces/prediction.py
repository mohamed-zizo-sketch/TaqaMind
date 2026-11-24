"""Factory-scoped regression prediction result contracts."""

from enum import StrEnum
from math import isclose, isfinite
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    TaskType,
)
from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetState,
)
from taqamind.core.interfaces.results import StageTiming


class PredictionStatus(StrEnum):
    """Status of an individual prediction."""

    SUCCEEDED = "succeeded"
    REJECTED = "rejected"


class PredictionBatchStatus(StrEnum):
    """Overall batch prediction status."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    REJECTED = "rejected"


class PredictionModelIdentity(BaseModel):
    """Expected identity of a registered prediction model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_version_id: UUID

    factory_id: UUID

    domain: FeatureName

    process: FeatureName

    task_id: FeatureName

    dataset_id: FeatureName

    contract_version: ContractVersion

    task_version: ContractVersion

    target_version: ContractVersion

    def validate_context(
        self,
        context: PipelineContext,
    ) -> None:
        if context.purpose != DatasetPurpose.PREDICTION:
            raise ValueError(
                "Model identity requires a prediction context."
            )

        fields = (
            "model_version_id",
            "factory_id",
            "domain",
            "process",
            "task_id",
            "dataset_id",
            "contract_version",
            "task_version",
            "target_version",
        )

        for field in fields:
            if getattr(self, field) != getattr(context, field):
                raise ValueError(
                    f"Prediction model mismatch for '{field}'."
                )


class DerivedRatePrediction(BaseModel):
    """Optional rate derived from a predicted target quantity."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    value: float | None

    denominator_value: float | None = None

    unit: NonEmptyString

    undefined_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.value is None:
            if self.undefined_reason is None:
                raise ValueError(
                    "Undefined prediction rate requires a reason."
                )

        else:
            if (
                self.denominator_value is None
                or self.denominator_value <= 0
            ):
                raise ValueError(
                    "Defined rate requires a positive denominator."
                )

            if self.undefined_reason is not None:
                raise ValueError(
                    "Defined rate cannot have an undefined reason."
                )

            if self.value < 0:
                raise ValueError(
                    "Predicted waste rate cannot be negative."
                )

        return self


class RecordPrediction(BaseModel):
    """Prediction result for one registered production record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    record_id: UUID

    status: PredictionStatus

    predicted_value: float | None = None

    derived_rate: DerivedRatePrediction | None = None

    warnings: tuple[NonEmptyString, ...] = ()

    rejection_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> Self:
        if self.status == PredictionStatus.SUCCEEDED:
            if self.predicted_value is None:
                raise ValueError(
                    "Successful prediction requires a value."
                )

            if self.rejection_reason is not None:
                raise ValueError(
                    "Successful prediction cannot have a rejection reason."
                )

        else:
            if self.predicted_value is not None:
                raise ValueError(
                    "Rejected prediction cannot expose a predicted value."
                )

            if self.derived_rate is not None:
                raise ValueError(
                    "Rejected prediction cannot expose a derived rate."
                )

            if self.rejection_reason is None:
                raise ValueError(
                    "Rejected prediction requires a reason."
                )

        return self


class PredictionBatchResult(BaseModel):
    """Batch of predictions generated by one registered model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    prediction_id: UUID = Field(default_factory=uuid4)

    context: PipelineContext

    model: PredictionModelIdentity

    contract: DatasetContract

    input_artifact: DatasetArtifact

    predictions: tuple[RecordPrediction, ...] = Field(
        min_length=1,
    )

    timing: StageTiming

    @property
    def successful_count(self) -> int:
        return sum(
            prediction.status == PredictionStatus.SUCCEEDED
            for prediction in self.predictions
        )

    @property
    def rejected_count(self) -> int:
        return len(self.predictions) - self.successful_count

    @property
    def status(self) -> PredictionBatchStatus:
        if self.successful_count == 0:
            return PredictionBatchStatus.REJECTED

        if self.rejected_count > 0:
            return PredictionBatchStatus.PARTIAL

        return PredictionBatchStatus.SUCCEEDED

    @property
    def target_name(self) -> str:
        return self.contract.target.feature.name

    @property
    def target_unit(self) -> str | None:
        return self.contract.target.feature.unit

    def validate_record_coverage(
        self,
        expected_record_ids: tuple[UUID, ...],
    ) -> None:
        expected = set(expected_record_ids)

        if len(expected) != len(expected_record_ids):
            raise ValueError(
                "Expected prediction record IDs must be unique."
            )

        actual = {
            prediction.record_id
            for prediction in self.predictions
        }

        if actual != expected:
            raise ValueError(
                "Prediction results do not cover the expected records."
            )

    def validate_target_value(
        self,
        value: float,
    ) -> None:
        constraints = self.contract.target.feature.constraints

        minimum = constraints.minimum
        maximum = constraints.maximum

        if minimum is not None:
            if value < minimum:
                raise ValueError(
                    "Prediction is below the target minimum."
                )

            if value == minimum and not constraints.minimum_inclusive:
                raise ValueError(
                    "Prediction violates exclusive target minimum."
                )

        if maximum is not None:
            if value > maximum:
                raise ValueError(
                    "Prediction exceeds the target maximum."
                )

            if value == maximum and not constraints.maximum_inclusive:
                raise ValueError(
                    "Prediction violates exclusive target maximum."
                )

    def validate_derived_rate(
        self,
        prediction: RecordPrediction,
    ) -> None:
        rate = prediction.derived_rate

        if rate is None:
            return

        definition = self.contract.target.derived_rate

        if definition is None:
            raise ValueError(
                "Derived rate requires a target rate definition."
            )

        if rate.unit != definition.output_unit:
            raise ValueError(
                "Derived rate unit mismatch."
            )

        if rate.value is None:
            return

        denominator = rate.denominator_value
        numerator = prediction.predicted_value

        if denominator is None or numerator is None:
            raise ValueError(
                "Derived rate requires numerator and denominator."
            )

        denominator_feature = self.contract.get_feature(
            definition.denominator
        )

        if not self.contract.is_available_at_prediction(
            denominator_feature
        ):
            raise ValueError(
                "Rate denominator is unavailable at prediction time."
            )

        expected = (
            numerator / denominator
        ) * definition.multiplier

        if not isfinite(expected):
            raise ValueError(
                "Calculated prediction rate is not finite."
            )

        if not isclose(
            rate.value,
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Derived rate does not match predicted quantity."
            )

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        context = self.context
        metadata = self.input_artifact.metadata

        self.model.validate_context(context)

        context.validate_dataset_contract(
            self.contract
        )

        metadata.validate_scope(context)

        if self.contract.target.task_type != TaskType.REGRESSION:
            raise ValueError(
                "Current prediction contract supports regression only."
            )

        if not self.contract.target.is_confirmed:
            raise ValueError(
                "Prediction requires a confirmed target."
            )

        if metadata.state != DatasetState.ENGINEERED:
            raise ValueError(
                "Prediction requires an engineered dataset."
            )

        if self.timing.started_at < context.created_at:
            raise ValueError(
                "Prediction cannot precede execution creation."
            )

        actual_columns = {
            column.name
            for column in metadata.columns
        }

        if self.contract.target.feature.name in actual_columns:
            raise ValueError(
                "Prediction artifact cannot contain the target."
            )

        required_predictors = set(
            self.contract.predictor_names
        )

        if not required_predictors <= actual_columns:
            raise ValueError(
                "Prediction artifact is missing model predictors."
            )

        record_ids = [
            prediction.record_id
            for prediction in self.predictions
        ]

        if len(record_ids) != len(set(record_ids)):
            raise ValueError(
                "Prediction record identifiers must be unique."
            )

        if len(self.predictions) != metadata.row_count:
            raise ValueError(
                "Prediction result count does not match input rows."
            )

        for prediction in self.predictions:
            if prediction.status != PredictionStatus.SUCCEEDED:
                continue

            value = prediction.predicted_value

            if value is None:
                raise ValueError(
                    "Successful prediction is missing its value."
                )

            self.validate_target_value(value)
            self.validate_derived_rate(prediction)

        return self