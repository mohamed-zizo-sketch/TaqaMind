"""Tests for dataset contracts and structural leakage prevention."""

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    PredictionContext,
    RateDefinition,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)


def make_feature(name, **overrides):
    data = {
        "name": name,
        "display_name": name,
        "description": f"Test definition for {name}.",
        "data_type": DataType.FLOAT,
        "role": FeatureRole.PREDICTOR,
        "availability": FeatureAvailability.PRE_PRODUCTION,
        "origin": FeatureOrigin.RECORDED,
    }

    data.update(overrides)

    return FeatureContract(**data)


def make_target(**overrides):
    data = {
        "target_id": "waste_prediction",
        "feature": make_feature(
            "actual_waste_kg",
            role=FeatureRole.TARGET,
            availability=FeatureAvailability.POST_PRODUCTION,
            unit="kg",
            constraints=FeatureConstraints(minimum=0),
        ),
        "task_type": TaskType.REGRESSION,
        "target_kind": TargetKind.QUANTITY,
        "definition": TargetDefinition(
            definition="Measured production waste.",
            granularity="production_run",
        ),
    }

    data.update(overrides)

    return TargetContract(**data)


def make_dataset(**overrides):
    data = {
        "dataset_id": "textile_waste_dataset",
        "domain": "textile",
        "process": "knitting",
        "description": "Synthetic textile dataset contract.",
        "granularity": "production_run",
        "features": (
            make_feature("planned_quantity_kg", unit="kg"),
            make_feature(
                "run_id",
                data_type=DataType.STRING,
                role=FeatureRole.IDENTIFIER,
            ),
            make_feature(
                "order_id",
                data_type=DataType.STRING,
                role=FeatureRole.IDENTIFIER,
            ),
        ),
        "target": make_target(),
        "prediction_context": PredictionContext(
            description="Predict waste before production.",
        ),
        "business_keys": ("run_id",),
        "grouping_columns": ("order_id",),
    }

    data.update(overrides)

    return DatasetContract(**data)


def test_valid_dataset():
    dataset = make_dataset()

    assert dataset.domain == "textile"
    assert dataset.process == "knitting"
    assert len(dataset.all_columns) == 4


def test_predictor_names_exclude_identifiers_and_target():
    dataset = make_dataset()

    assert dataset.predictor_names == (
        "planned_quantity_kg",
    )


def test_training_requires_target():
    dataset = make_dataset()

    assert "actual_waste_kg" in dataset.required_columns(
        DatasetPurpose.TRAINING
    )


def test_prediction_excludes_target():
    dataset = make_dataset()

    assert "actual_waste_kg" not in dataset.required_columns(
        DatasetPurpose.PREDICTION
    )


def test_evaluation_requires_target():
    dataset = make_dataset()

    assert "actual_waste_kg" in dataset.required_columns(
        DatasetPurpose.EVALUATION
    )


def test_prediction_requires_identifiers():
    dataset = make_dataset()

    required = dataset.required_columns(
        DatasetPurpose.PREDICTION
    )

    assert "run_id" in required
    assert "order_id" in required


def test_unknown_feature_lookup():
    dataset = make_dataset()

    with pytest.raises(KeyError):
        dataset.get_feature("unknown_feature")


def test_duplicate_column_names():
    dataset = make_dataset()

    with pytest.raises(
        ValidationError,
        match="column names must be unique",
    ):
        make_dataset(
            features=(
                *dataset.features,
                make_feature("planned_quantity_kg"),
            )
        )


def test_target_cannot_be_declared_twice():
    dataset = make_dataset()

    with pytest.raises(
        ValidationError,
        match="Target must be declared only",
    ):
        make_dataset(
            features=(
                *dataset.features,
                dataset.target.feature,
            )
        )


def test_alias_conflicts_with_column_name():
    dataset = make_dataset()

    with pytest.raises(
        ValidationError,
        match="conflicts with column",
    ):
        make_dataset(
            features=(
                *dataset.features,
                make_feature(
                    "machine_speed",
                    aliases=("order_id",),
                ),
            )
        )


def test_alias_conflicts_between_features():
    with pytest.raises(
        ValidationError,
        match="conflicts with column",
    ):
        make_dataset(
            features=(
                make_feature("feature_a", aliases=("Quantity",)),
                make_feature("feature_b", aliases=("quantity",)),
            ),
            business_keys=(),
            grouping_columns=(),
        )


def test_target_granularity_mismatch():
    with pytest.raises(
        ValidationError,
        match="granularity must match",
    ):
        make_dataset(granularity="machine_shift")


def test_unknown_business_key():
    with pytest.raises(
        ValidationError,
        match="Unknown business_keys",
    ):
        make_dataset(
            business_keys=("unknown_id",),
        )


def test_predictor_cannot_be_business_key():
    with pytest.raises(
        ValidationError,
        match="Invalid role",
    ):
        make_dataset(
            business_keys=("planned_quantity_kg",),
        )


def test_duplicate_grouping_columns():
    with pytest.raises(
        ValidationError,
        match="unique names",
    ):
        make_dataset(
            grouping_columns=("order_id", "order_id"),
        )


def test_unknown_time_column():
    with pytest.raises(
        ValidationError,
        match="Time column does not exist",
    ):
        make_dataset(time_column="unknown_date")


def test_valid_time_column():
    dataset = make_dataset()

    result = make_dataset(
        features=(
            *dataset.features,
            make_feature(
                "planned_start",
                data_type=DataType.DATETIME,
                role=FeatureRole.TIMESTAMP,
            ),
        ),
        time_column="planned_start",
    )

    assert result.time_column == "planned_start"

    assert "planned_start" in result.required_columns(
        DatasetPurpose.TRAINING
    )


def test_invalid_time_column_role():
    with pytest.raises(
        ValidationError,
        match="TIMESTAMP role",
    ):
        make_dataset(
            time_column="planned_quantity_kg",
        )


@pytest.mark.parametrize(
    "availability",
    [
        FeatureAvailability.IN_PRODUCTION,
        FeatureAvailability.POST_PRODUCTION,
        FeatureAvailability.UNKNOWN,
    ],
)
def test_unavailable_predictor_is_rejected(availability):
    dataset = make_dataset()

    with pytest.raises(
        ValidationError,
        match="not available at prediction time",
    ):
        make_dataset(
            features=(
                *dataset.features,
                make_feature(
                    "unavailable_feature",
                    availability=availability,
                ),
            )
        )


def test_static_predictor_is_allowed():
    dataset = make_dataset(
        features=(
            make_feature(
                "machine_capacity",
                availability=FeatureAvailability.STATIC,
            ),
        ),
        business_keys=(),
        grouping_columns=(),
    )

    assert dataset.predictor_names == (
        "machine_capacity",
    )


def test_in_production_predictor_for_in_production_task():
    dataset = make_dataset(
        features=(
            make_feature(
                "actual_speed",
                availability=FeatureAvailability.IN_PRODUCTION,
            ),
        ),
        business_keys=(),
        grouping_columns=(),
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.IN_PRODUCTION,
            description="Predict during production.",
        ),
    )

    assert dataset.predictor_names == (
        "actual_speed",
    )


def test_missing_dependency():
    dataset = make_dataset()

    with pytest.raises(
        ValidationError,
        match="unknown column",
    ):
        make_dataset(
            features=(
                *dataset.features,
                make_feature(
                    "derived_feature",
                    origin=FeatureOrigin.DERIVED,
                    derived_from=("unknown_feature",),
                ),
            )
        )


def test_circular_dependency():
    with pytest.raises(
        ValidationError,
        match="Circular dependency",
    ):
        make_dataset(
            features=(
                make_feature(
                    "feature_a",
                    origin=FeatureOrigin.DERIVED,
                    derived_from=("feature_b",),
                ),
                make_feature(
                    "feature_b",
                    origin=FeatureOrigin.DERIVED,
                    derived_from=("feature_c",),
                ),
                make_feature(
                    "feature_c",
                    origin=FeatureOrigin.DERIVED,
                    derived_from=("feature_a",),
                ),
            ),
            business_keys=(),
            grouping_columns=(),
        )


def test_valid_derived_feature():
    dataset = make_dataset()

    result = make_dataset(
        features=(
            *dataset.features,
            make_feature(
                "planned_weight_tonnes",
                origin=FeatureOrigin.DERIVED,
                derived_from=("planned_quantity_kg",),
                unit="t",
            ),
        )
    )

    assert result.get_dependencies(
        "planned_weight_tonnes"
    ) == ("planned_quantity_kg",)

    assert "planned_weight_tonnes" in result.predictor_names

    assert (
        "planned_weight_tonnes"
        not in result.required_columns(
            DatasetPurpose.PREDICTION
        )
    )


def test_required_derived_feature_requires_optional_source():
    dataset = make_dataset(
        features=(
            make_feature(
                "raw_quantity",
                required=False,
            ),
            make_feature(
                "derived_quantity",
                origin=FeatureOrigin.DERIVED,
                derived_from=("raw_quantity",),
            ),
        ),
        business_keys=(),
        grouping_columns=(),
    )

    assert "raw_quantity" in dataset.required_columns(
        DatasetPurpose.PREDICTION
    )


def test_derived_feature_cannot_hide_future_dependency():
    with pytest.raises(
        ValidationError,
        match="cannot become available before dependency",
    ):
        make_dataset(
            features=(
                make_feature(
                    "future_measurement",
                    role=FeatureRole.METADATA,
                    availability=FeatureAvailability.POST_PRODUCTION,
                ),
                make_feature(
                    "derived_feature",
                    origin=FeatureOrigin.DERIVED,
                    derived_from=("future_measurement",),
                ),
            ),
            business_keys=(),
            grouping_columns=(),
        )


def test_direct_target_dependency():
    with pytest.raises(
        ValidationError,
        match="depends on the target column",
    ):
        make_dataset(
            features=(
                make_feature(
                    "leaked_feature",
                    origin=FeatureOrigin.DERIVED,
                    availability=FeatureAvailability.POST_PRODUCTION,
                    derived_from=("actual_waste_kg",),
                ),
            ),
            business_keys=(),
            grouping_columns=(),
            prediction_context=PredictionContext(
                prediction_stage=FeatureAvailability.POST_PRODUCTION,
                description="Test target dependency protection.",
            ),
        )


def test_indirect_target_dependency():
    with pytest.raises(
        ValidationError,
        match="depends on the target column",
    ):
        make_dataset(
            features=(
                make_feature(
                    "intermediate",
                    role=FeatureRole.METADATA,
                    origin=FeatureOrigin.DERIVED,
                    availability=FeatureAvailability.POST_PRODUCTION,
                    derived_from=("actual_waste_kg",),
                ),
                make_feature(
                    "leaked_predictor",
                    origin=FeatureOrigin.DERIVED,
                    availability=FeatureAvailability.POST_PRODUCTION,
                    derived_from=("intermediate",),
                ),
            ),
            business_keys=(),
            grouping_columns=(),
            prediction_context=PredictionContext(
                prediction_stage=FeatureAvailability.POST_PRODUCTION,
                description="Test indirect target leakage.",
            ),
        )


def make_rate(**overrides):
    data = {
        "name": "waste_rate_pct",
        "numerator": "actual_waste_kg",
        "denominator": "planned_quantity_kg",
        "granularity": "production_run",
    }

    data.update(overrides)

    return RateDefinition(**data)


def test_valid_rate_definition():
    dataset = make_dataset(
        target=make_target(
            derived_rate=make_rate(),
        ),
    )

    assert dataset.target.derived_rate.denominator == (
        "planned_quantity_kg"
    )


def test_missing_rate_denominator():
    with pytest.raises(
        ValidationError,
        match="Rate denominator does not exist",
    ):
        make_dataset(
            target=make_target(
                derived_rate=make_rate(
                    denominator="unknown_quantity",
                ),
            ),
        )


def test_non_numeric_rate_denominator():
    with pytest.raises(
        ValidationError,
        match="Rate denominator must be numeric",
    ):
        make_dataset(
            target=make_target(
                derived_rate=make_rate(
                    denominator="order_id",
                ),
            ),
        )


def test_rate_unit_mismatch():
    with pytest.raises(
        ValidationError,
        match="same canonical unit",
    ):
        make_dataset(
            features=(
                make_feature(
                    "planned_quantity_kg",
                    unit="m",
                ),
            ),
            business_keys=(),
            grouping_columns=(),
            target=make_target(
                derived_rate=make_rate(),
            ),
        )


def test_no_predictors():
    with pytest.raises(
        ValidationError,
        match="at least one predictor",
    ):
        make_dataset(
            features=(
                make_feature(
                    "run_id",
                    role=FeatureRole.IDENTIFIER,
                ),
            ),
            grouping_columns=(),
        )


@pytest.mark.parametrize(
    "stage",
    [
        FeatureAvailability.UNKNOWN,
        FeatureAvailability.STATIC,
    ],
)
def test_invalid_prediction_stage(stage):
    with pytest.raises(ValidationError):
        PredictionContext(
            prediction_stage=stage,
            description="Invalid prediction context.",
        )


def test_negative_prediction_horizon():
    with pytest.raises(ValidationError):
        PredictionContext(
            horizon_seconds=-1,
            description="Invalid horizon.",
        )


def test_dataset_is_frozen():
    dataset = make_dataset()

    with pytest.raises(ValidationError):
        dataset.domain = "food_beverage"


def test_dataset_json_round_trip():
    dataset = make_dataset()

    restored = DatasetContract.model_validate_json(
        dataset.model_dump_json()
    )

    assert restored == dataset


def test_research_required_columns_not_supported():
    dataset = make_dataset()

    with pytest.raises(ValueError):
        dataset.required_columns(DatasetPurpose.RESEARCH)