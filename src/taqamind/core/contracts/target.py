"""Target contracts for industrial prediction tasks."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from taqamind.core.contracts.enums import (
    ConfirmationStatus,
    DataType,
    FeatureAvailability,
    FeatureRole,
    MissingValuePolicy,
    TargetKind,
    TaskType,
)
from taqamind.core.contracts.feature import (
    FeatureContract,
    FeatureName,
    NonEmptyString,
)

ContractVersion = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    ),
]


class TargetDefinition(BaseModel):
    """Industrial meaning and scope of a target."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    definition: NonEmptyString

    granularity: NonEmptyString

    measurement_method: NonEmptyString | None = None

    included_components: tuple[NonEmptyString, ...] = ()

    excluded_components: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_components(self) -> Self:
        included = [
            component.casefold()
            for component in self.included_components
        ]

        excluded = [
            component.casefold()
            for component in self.excluded_components
        ]

        if len(included) != len(set(included)):
            raise ValueError(
                "Included target components must be unique."
            )

        if len(excluded) != len(set(excluded)):
            raise ValueError(
                "Excluded target components must be unique."
            )

        overlap = set(included) & set(excluded)

        if overlap:
            raise ValueError(
                "A target component cannot be both included "
                "and excluded."
            )

        return self


class TargetConfirmation(BaseModel):
    """Confirmation of target business semantics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: ConfirmationStatus = (
        ConfirmationStatus.UNCONFIRMED
    )

    confirmed_by: NonEmptyString | None = None

    confirmed_at: datetime | None = None

    notes: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_confirmation(self) -> Self:
        if self.status == ConfirmationStatus.CONFIRMED:
            if self.confirmed_by is None:
                raise ValueError(
                    "A confirmed target requires a confirmer."
                )

            if self.confirmed_at is None:
                raise ValueError(
                    "A confirmed target requires a timestamp."
                )

            if self.confirmed_at.utcoffset() is None:
                raise ValueError(
                    "Confirmation timestamp must be timezone-aware."
                )

        else:
            if (
                self.confirmed_by is not None
                or self.confirmed_at is not None
            ):
                raise ValueError(
                    "Unconfirmed targets cannot contain "
                    "confirmation metadata."
                )

        return self


class RateDefinition(BaseModel):
    """Definition of a derived industrial rate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: FeatureName

    numerator: FeatureName

    denominator: FeatureName

    granularity: NonEmptyString

    output_unit: NonEmptyString = "%"

    multiplier: float = Field(
        default=100.0,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.numerator == self.denominator:
            raise ValueError(
                "Rate numerator and denominator must differ."
            )

        if self.name in {
            self.numerator,
            self.denominator,
        }:
            raise ValueError(
                "Rate output cannot reuse an input name."
            )

        if self.output_unit not in {"%", "ratio"}:
            raise ValueError(
                "Only percentage and dimensionless ratio "
                "rates are currently supported."
            )

        if self.output_unit == "%" and self.multiplier != 100:
            raise ValueError(
                "Percentage rates require a multiplier of 100."
            )

        if self.output_unit == "ratio" and self.multiplier != 1:
            raise ValueError(
                "Ratio rates require a multiplier of 1."
            )

        return self


class TargetContract(BaseModel):
    """Complete definition of an industrial prediction target."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    target_id: FeatureName

    version: ContractVersion = "1.0.0"

    feature: FeatureContract

    task_type: TaskType

    target_kind: TargetKind

    definition: TargetDefinition

    confirmation: TargetConfirmation = Field(
        default_factory=TargetConfirmation
    )

    derived_rate: RateDefinition | None = None

    @property
    def is_confirmed(self) -> bool:
        """Return whether the target semantics are confirmed."""

        return (
            self.confirmation.status
            == ConfirmationStatus.CONFIRMED
        )

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.feature.role != FeatureRole.TARGET:
            raise ValueError(
                "Target feature must have the TARGET role."
            )

        if not self.feature.required:
            raise ValueError(
                "Target column must be required in labeled datasets."
            )

        if self.feature.missing_policy == MissingValuePolicy.IMPUTE:
            raise ValueError(
                "Target imputation is not allowed."
            )

        if self.feature.availability == FeatureAvailability.UNKNOWN:
            if self.is_confirmed:
                raise ValueError(
                    "Confirmed targets require known availability."
                )

        numeric_types = {
            DataType.INTEGER,
            DataType.FLOAT,
        }

        numeric_kinds = {
            TargetKind.QUANTITY,
            TargetKind.RATE,
            TargetKind.CONTINUOUS,
            TargetKind.COUNT,
        }

        if self.target_kind in numeric_kinds:
            if self.feature.data_type not in numeric_types:
                raise ValueError(
                    "Numeric targets require a numeric data type."
                )

        if self.task_type in {
            TaskType.REGRESSION,
            TaskType.FORECASTING,
        }:
            if self.feature.data_type not in numeric_types:
                raise ValueError(
                    "Regression and forecasting require numeric targets."
                )

        elif self.task_type == TaskType.CLASSIFICATION:
            if self.target_kind != TargetKind.CATEGORY:
                raise ValueError(
                    "Classification requires a categorical target."
                )

        else:
            raise ValueError(
                "This target contract currently supports regression, "
                "classification, and numeric forecasting only."
            )

        if self.target_kind == TargetKind.CATEGORY:
            if self.feature.data_type not in {
                DataType.STRING,
                DataType.CATEGORY,
                DataType.BOOLEAN,
                DataType.INTEGER,
            }:
                raise ValueError(
                    "Categorical targets require a valid label type."
                )

            if self.task_type != TaskType.CLASSIFICATION:
                raise ValueError(
                    "Categorical targets require classification."
                )

        if self.target_kind == TargetKind.COUNT:
            if self.feature.data_type != DataType.INTEGER:
                raise ValueError(
                    "Count targets require integer values."
                )

            minimum = self.feature.constraints.minimum

            if minimum is None or minimum < 0:
                raise ValueError(
                    "Count targets require a non-negative minimum."
                )

        if self.target_kind == TargetKind.QUANTITY:
            if self.feature.unit is None:
                raise ValueError(
                    "Quantity targets require a measurement unit."
                )

        if self.target_kind == TargetKind.RATE:
            if self.feature.unit not in {"%", "ratio"}:
                raise ValueError(
                    "Rate targets require a percentage or ratio unit."
                )

        if self.derived_rate is not None:
            if self.target_kind != TargetKind.QUANTITY:
                raise ValueError(
                    "A derived rate requires a quantity target."
                )

            if self.derived_rate.numerator != self.feature.name:
                raise ValueError(
                    "Derived rate numerator must reference "
                    "the target feature."
                )

            if (
                self.derived_rate.granularity
                != self.definition.granularity
            ):
                raise ValueError(
                    "Target and derived rate must have "
                    "the same granularity."
                )

        return self