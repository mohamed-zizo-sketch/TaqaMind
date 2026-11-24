"""Dataset contracts and structural leakage prevention."""

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
)
from taqamind.core.contracts.feature import (
    FeatureContract,
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import (
    ContractVersion,
    TargetContract,
)

AVAILABILITY_ORDER = {
    FeatureAvailability.STATIC: 0,
    FeatureAvailability.PRE_PRODUCTION: 1,
    FeatureAvailability.IN_PRODUCTION: 2,
    FeatureAvailability.POST_PRODUCTION: 3,
}


class PredictionContext(BaseModel):
    """Defines when a prediction is made."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    prediction_stage: FeatureAvailability = (
        FeatureAvailability.PRE_PRODUCTION
    )

    horizon_seconds: int | None = Field(
        default=None,
        ge=0,
    )

    description: NonEmptyString

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if self.prediction_stage not in {
            FeatureAvailability.PRE_PRODUCTION,
            FeatureAvailability.IN_PRODUCTION,
            FeatureAvailability.POST_PRODUCTION,
        }:
            raise ValueError(
                "Prediction stage must be an explicit "
                "production stage."
            )

        return self


class DatasetContract(BaseModel):
    """Complete structural contract for an industrial dataset."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_id: FeatureName

    version: ContractVersion = "1.0.0"

    domain: FeatureName

    process: FeatureName

    description: NonEmptyString

    granularity: NonEmptyString

    features: tuple[FeatureContract, ...] = Field(
        min_length=1
    )

    target: TargetContract

    prediction_context: PredictionContext

    business_keys: tuple[FeatureName, ...] = ()

    grouping_columns: tuple[FeatureName, ...] = ()

    time_column: FeatureName | None = None

    @property
    def all_columns(self) -> tuple[FeatureContract, ...]:
        """Return every declared column, including the target."""

        return (*self.features, self.target.feature)

    @property
    def column_map(self) -> dict[str, FeatureContract]:
        """Return canonical column definitions."""

        return {
            column.name: column
            for column in self.all_columns
        }

    @property
    def predictor_names(self) -> tuple[str, ...]:
        """Return candidate model inputs."""

        return tuple(
            feature.name
            for feature in self.features
            if feature.role == FeatureRole.PREDICTOR
        )

    def get_feature(self, name: str) -> FeatureContract:
        """Retrieve a column by its canonical name."""

        if name not in self.column_map:
            raise KeyError(f"Unknown feature: {name}")

        return self.column_map[name]

    def is_available_at_prediction(
        self,
        feature: FeatureContract,
    ) -> bool:
        """Check declared availability against prediction timing."""

        if feature.availability == FeatureAvailability.UNKNOWN:
            return False

        return (
            AVAILABILITY_ORDER[feature.availability]
            <= AVAILABILITY_ORDER[
                self.prediction_context.prediction_stage
            ]
        )

    def get_dependencies(
        self,
        name: str,
    ) -> tuple[str, ...]:
        """Return all direct and indirect dependencies."""

        columns = self.column_map

        if name not in columns:
            raise KeyError(f"Unknown feature: {name}")

        visited: set[str] = set()
        pending = list(columns[name].derived_from)

        while pending:
            dependency = pending.pop()

            if dependency in visited:
                continue

            visited.add(dependency)
            pending.extend(
                columns[dependency].derived_from
            )

        return tuple(sorted(visited))

    def required_columns(
        self,
        purpose: DatasetPurpose,
    ) -> tuple[str, ...]:
        """Return required raw input columns for a dataset purpose."""

        if purpose not in {
            DatasetPurpose.TRAINING,
            DatasetPurpose.PREDICTION,
            DatasetPurpose.EVALUATION,
        }:
            raise ValueError(
                f"Required columns are not defined for {purpose}."
            )

        required: set[str] = set()
        prediction = purpose == DatasetPurpose.PREDICTION

        if prediction:
            candidates = [
                feature
                for feature in self.features
                if feature.role in {
                    FeatureRole.PREDICTOR,
                    FeatureRole.IDENTIFIER,
                    FeatureRole.TIMESTAMP,
                    FeatureRole.METADATA,
                }
                and self.is_available_at_prediction(feature)
            ]
        else:
            candidates = list(self.all_columns)

        def require_raw_column(name: str) -> None:
            feature = self.get_feature(name)

            if feature.origin == FeatureOrigin.DERIVED:
                for dependency in feature.derived_from:
                    require_raw_column(dependency)
            else:
                required.add(name)

        for feature in candidates:
            if feature.required:
                require_raw_column(feature.name)

        for name in self.business_keys:
            if not prediction or self.is_available_at_prediction(
                self.get_feature(name)
            ):
                require_raw_column(name)

        if not prediction:
            for name in self.grouping_columns:
                require_raw_column(name)

            if self.time_column is not None:
                require_raw_column(self.time_column)

        return tuple(
            column.name
            for column in self.all_columns
            if column.name in required
        )

    def validate_column_names(self) -> None:
        """Reject duplicate names and ambiguous aliases."""

        names = [
            column.name
            for column in self.all_columns
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Dataset column names must be unique."
            )

        registered_names = {
            name.casefold(): name
            for name in names
        }

        for column in self.all_columns:
            for alias in column.aliases:
                normalized = alias.strip().casefold()

                if normalized in registered_names:
                    owner = registered_names[normalized]

                    raise ValueError(
                        f"Alias '{alias}' conflicts with "
                        f"column '{owner}'."
                    )

                registered_names[normalized] = column.name

    def validate_key_columns(self) -> None:
        """Validate business keys, grouping and temporal metadata."""

        columns = self.column_map

        key_roles = {
            FeatureRole.IDENTIFIER,
            FeatureRole.METADATA,
        }

        for label, names in (
            ("business_keys", self.business_keys),
            ("grouping_columns", self.grouping_columns),
        ):
            if len(names) != len(set(names)):
                raise ValueError(
                    f"{label} must contain unique names."
                )

            for name in names:
                if name not in columns:
                    raise ValueError(
                        f"Unknown {label} column: {name}"
                    )

                feature = columns[name]

                if feature.role not in key_roles:
                    raise ValueError(
                        f"Invalid role for {label}: {name}"
                    )

                if feature.nullable or not feature.required:
                    raise ValueError(
                        f"{label} column '{name}' must be "
                        "required and non-nullable."
                    )

                if feature.origin == FeatureOrigin.DERIVED:
                    raise ValueError(
                        f"{label} column '{name}' must be "
                        "a directly supplied column."
                    )

        if self.time_column is not None:
            if self.time_column not in columns:
                raise ValueError(
                    "Time column does not exist."
                )

            time_feature = columns[self.time_column]

            if time_feature.role != FeatureRole.TIMESTAMP:
                raise ValueError(
                    "Time column must have the TIMESTAMP role."
                )

            if time_feature.data_type not in {
                DataType.DATE,
                DataType.DATETIME,
            }:
                raise ValueError(
                    "Time column must be date or datetime."
                )

            if time_feature.nullable or not time_feature.required:
                raise ValueError(
                    "Time column must be required and non-nullable."
                )

            if time_feature.origin == FeatureOrigin.DERIVED:
                raise ValueError(
                    "Time column must be directly supplied."
                )

    def validate_dependency_graph(self) -> None:
        """Reject unknown dependencies and dependency cycles."""

        columns = self.column_map

        for feature in self.all_columns:
            for dependency in feature.derived_from:
                if dependency not in columns:
                    raise ValueError(
                        f"Feature '{feature.name}' depends on "
                        f"unknown column '{dependency}'."
                    )

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(
                    f"Circular dependency detected at '{name}'."
                )

            if name in visited:
                return

            visiting.add(name)

            for dependency in columns[name].derived_from:
                visit(dependency)

            visiting.remove(name)
            visited.add(name)

        for name in columns:
            visit(name)

    def validate_dependency_availability(self) -> None:
        """Reject inconsistent availability declarations."""

        columns = self.column_map

        for feature in self.all_columns:
            if feature.origin != FeatureOrigin.DERIVED:
                continue

            if feature.availability == FeatureAvailability.UNKNOWN:
                continue

            for dependency in feature.derived_from:
                source = columns[dependency]

                if source.availability == FeatureAvailability.UNKNOWN:
                    raise ValueError(
                        f"Feature '{feature.name}' has a dependency "
                        f"with unknown availability: '{dependency}'."
                    )

                if (
                    AVAILABILITY_ORDER[source.availability]
                    > AVAILABILITY_ORDER[feature.availability]
                ):
                    raise ValueError(
                        f"Feature '{feature.name}' cannot become "
                        f"available before dependency '{dependency}'."
                    )

    def validate_prediction_features(self) -> None:
        """Reject structurally unsafe prediction inputs."""

        target_name = self.target.feature.name

        if not self.predictor_names:
            raise ValueError(
                "Dataset must contain at least one predictor."
            )

        prediction_inputs = [
            feature
            for feature in self.features
            if feature.role == FeatureRole.PREDICTOR
            or (
                feature.role in {
                    FeatureRole.IDENTIFIER,
                    FeatureRole.TIMESTAMP,
                    FeatureRole.METADATA,
                }
                and self.is_available_at_prediction(feature)
            )
        ]

        for feature in prediction_inputs:
            if not self.is_available_at_prediction(feature):
                raise ValueError(
                    f"Predictor '{feature.name}' is not available "
                    "at prediction time."
                )

            dependencies = self.get_dependencies(feature.name)

            if target_name in dependencies:
                raise ValueError(
                    f"Feature '{feature.name}' depends on "
                    "the target column."
                )

            for dependency in dependencies:
                source = self.get_feature(dependency)

                if not self.is_available_at_prediction(source):
                    raise ValueError(
                        f"Feature '{feature.name}' depends on "
                        f"unavailable column '{dependency}'."
                    )

    def validate_rate_definition(self) -> None:
        """Validate structural compatibility of derived rates."""

        rate = self.target.derived_rate

        if rate is None:
            return

        columns = self.column_map

        if rate.name in columns:
            raise ValueError(
                "Derived rate output conflicts with a dataset column."
            )

        if rate.denominator not in columns:
            raise ValueError(
                "Rate denominator does not exist."
            )

        denominator = columns[rate.denominator]

        if denominator.data_type not in {
            DataType.INTEGER,
            DataType.FLOAT,
        }:
            raise ValueError(
                "Rate denominator must be numeric."
            )

        if denominator.unit != self.target.feature.unit:
            raise ValueError(
                "Rate denominator and target must use "
                "the same canonical unit."
            )

        if self.target.feature.name in self.get_dependencies(
            denominator.name
        ):
            raise ValueError(
                "Rate denominator cannot depend on the target."
            )

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        """Validate the complete dataset contract."""

        if self.granularity != self.target.definition.granularity:
            raise ValueError(
                "Dataset and target granularity must match."
            )

        for feature in self.features:
            if feature.role == FeatureRole.TARGET:
                raise ValueError(
                    "Target must be declared only "
                    "inside TargetContract."
                )

        self.validate_column_names()
        self.validate_key_columns()
        self.validate_dependency_graph()
        self.validate_dependency_availability()
        self.validate_prediction_features()
        self.validate_rate_definition()

        return self