"""Shared feature contracts for industrial datasets."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from taqamind.core.contracts.enums import (
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
)

FeatureName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]

NonEmptyString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]


class FeatureConstraints(BaseModel):
    """Declarative constraints for a feature."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    minimum: int | float | None = None
    maximum: int | float | None = None

    minimum_inclusive: bool = True
    maximum_inclusive: bool = True

    allowed_values: tuple[NonEmptyString, ...] | None = None

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        """Validate internal constraint consistency."""

        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Minimum cannot be greater than maximum.")

            if self.minimum == self.maximum and not (
                self.minimum_inclusive and self.maximum_inclusive
            ):
                raise ValueError("Exclusive equal bounds produce an empty range.")

        if self.minimum is None and not self.minimum_inclusive:
            raise ValueError("An exclusive minimum requires a minimum value.")

        if self.maximum is None and not self.maximum_inclusive:
            raise ValueError("An exclusive maximum requires a maximum value.")

        if self.allowed_values is not None:
            if not self.allowed_values:
                raise ValueError("Allowed values cannot be empty.")

            if len(self.allowed_values) != len(set(self.allowed_values)):
                raise ValueError("Allowed values must be unique.")

        return self


class FeatureContract(BaseModel):
    """Standardized definition of an industrial feature."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    name: FeatureName

    display_name: NonEmptyString

    description: NonEmptyString

    data_type: DataType

    role: FeatureRole

    availability: FeatureAvailability = FeatureAvailability.UNKNOWN

    origin: FeatureOrigin = FeatureOrigin.UNKNOWN

    unit: NonEmptyString | None = None

    required: bool = True

    nullable: bool = False

    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT

    aliases: tuple[NonEmptyString, ...] = ()

    source_system: NonEmptyString | None = None

    constraints: FeatureConstraints = Field(default_factory=FeatureConstraints)

    derived_from: tuple[FeatureName, ...] = ()

    @field_validator("aliases")
    @classmethod
    def validate_aliases(
        cls,
        aliases: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Reject duplicate aliases after normalization."""

        normalized = [alias.strip().casefold() for alias in aliases]

        if len(normalized) != len(set(normalized)):
            raise ValueError("Feature aliases must be unique.")

        return aliases

    @model_validator(mode="after")
    def validate_feature(self) -> Self:
        """Validate feature-level semantic consistency."""

        numeric_types = {
            DataType.INTEGER,
            DataType.FLOAT,
        }

        categorical_types = {
            DataType.STRING,
            DataType.CATEGORY,
        }

        has_numeric_constraints = (
            self.constraints.minimum is not None or self.constraints.maximum is not None
        )

        if has_numeric_constraints and self.data_type not in numeric_types:
            raise ValueError("Numeric constraints require a numeric data type.")

        if self.constraints.allowed_values is not None and self.data_type not in categorical_types:
            raise ValueError("Allowed values require a string or category data type.")

        if self.nullable and self.missing_policy == MissingValuePolicy.REJECT:
            raise ValueError("Nullable features cannot use the reject missing policy.")

        if not self.nullable and self.missing_policy != MissingValuePolicy.REJECT:
            raise ValueError("Non-nullable features must reject missing values.")

        if self.role == FeatureRole.TARGET and self.missing_policy == MissingValuePolicy.IMPUTE:
            raise ValueError("Target values cannot use feature imputation.")

        if self.origin == FeatureOrigin.DERIVED:
            if not self.derived_from:
                raise ValueError("Derived features must declare their dependencies.")

        elif self.derived_from:
            raise ValueError("Only derived features may declare dependencies.")

        if self.name in self.derived_from:
            raise ValueError("A feature cannot depend on itself.")

        if len(self.derived_from) != len(set(self.derived_from)):
            raise ValueError("Derived feature dependencies must be unique.")

        if self.name.casefold() in {alias.casefold() for alias in self.aliases}:
            raise ValueError("An alias cannot duplicate the canonical feature name.")

        return self
