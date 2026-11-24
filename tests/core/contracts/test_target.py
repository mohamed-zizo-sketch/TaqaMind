"""Tests for industrial target contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    ConfirmationStatus,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureRole,
    MissingValuePolicy,
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)


def make_target_feature(**overrides) -> FeatureContract:
    data = {
        "name": "waste_quantity_kg",
        "display_name": "Waste Quantity",
        "description": "Measured production waste.",
        "data_type": DataType.FLOAT,
        "role": FeatureRole.TARGET,
        "availability": FeatureAvailability.POST_PRODUCTION,
        "unit": "kg",
        "constraints": FeatureConstraints(minimum=0),
    }

    data.update(overrides)

    return FeatureContract(**data)


def make_target(**overrides) -> TargetContract:
    data = {
        "target_id": "waste_prediction",
        "feature": make_target_feature(),
        "task_type": TaskType.REGRESSION,
        "target_kind": TargetKind.QUANTITY,
        "definition": TargetDefinition(
            definition="Total measured production waste.",
            granularity="production_run",
        ),
    }

    data.update(overrides)

    return TargetContract(**data)


def make_rate(**overrides) -> RateDefinition:
    data = {
        "name": "waste_rate_pct",
        "numerator": "waste_quantity_kg",
        "denominator": "planned_quantity_kg",
        "granularity": "production_run",
    }

    data.update(overrides)

    return RateDefinition(**data)


def test_valid_target():
    target = make_target()

    assert target.target_id == "waste_prediction"
    assert target.version == "1.0.0"
    assert not target.is_confirmed


def test_target_requires_target_role():
    with pytest.raises(ValidationError):
        make_target(
            feature=make_target_feature(
                role=FeatureRole.PREDICTOR,
            )
        )


def test_target_column_is_required():
    with pytest.raises(ValidationError):
        make_target(
            feature=make_target_feature(
                required=False,
            )
        )


def test_target_cannot_be_imputed():
    with pytest.raises(ValidationError):
        make_target_feature(
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
        )


def test_quantity_requires_unit():
    with pytest.raises(ValidationError):
        make_target(
            feature=make_target_feature(unit=None),
        )


def test_regression_requires_numeric_target():
    with pytest.raises(ValidationError):
        make_target(
            feature=make_target_feature(
                data_type=DataType.STRING,
                constraints=FeatureConstraints(),
            )
        )


def test_valid_classification_target():
    target = make_target(
        target_id="quality_classification",
        target_kind=TargetKind.CATEGORY,
        task_type=TaskType.CLASSIFICATION,
        feature=make_target_feature(
            name="quality_class",
            data_type=DataType.CATEGORY,
            unit=None,
            constraints=FeatureConstraints(
                allowed_values=("pass", "fail"),
            ),
        ),
    )

    assert target.task_type == TaskType.CLASSIFICATION


def test_classification_requires_categorical_target():
    with pytest.raises(ValidationError):
        make_target(
            task_type=TaskType.CLASSIFICATION,
        )


def test_valid_count_target():
    target = make_target(
        target_kind=TargetKind.COUNT,
        feature=make_target_feature(
            data_type=DataType.INTEGER,
            unit=None,
        ),
    )

    assert target.target_kind == TargetKind.COUNT


def test_count_requires_integer():
    with pytest.raises(ValidationError):
        make_target(
            target_kind=TargetKind.COUNT,
        )


def test_count_requires_non_negative_minimum():
    with pytest.raises(ValidationError):
        make_target(
            target_kind=TargetKind.COUNT,
            feature=make_target_feature(
                data_type=DataType.INTEGER,
                constraints=FeatureConstraints(minimum=-1),
            ),
        )


def test_valid_target_components():
    definition = TargetDefinition(
        definition="Production waste.",
        granularity="production_run",
        included_components=("setup_scrap", "defective_fabric"),
        excluded_components=("cutting_waste",),
    )

    assert len(definition.included_components) == 2


def test_overlapping_target_components():
    with pytest.raises(ValidationError):
        TargetDefinition(
            definition="Production waste.",
            granularity="production_run",
            included_components=("setup_scrap",),
            excluded_components=("Setup_Scrap",),
        )


def test_duplicate_target_components():
    with pytest.raises(ValidationError):
        TargetDefinition(
            definition="Production waste.",
            granularity="production_run",
            included_components=("scrap", "SCRAP"),
        )


def test_default_confirmation():
    confirmation = TargetConfirmation()

    assert confirmation.status == ConfirmationStatus.UNCONFIRMED


def test_valid_confirmation():
    target = make_target(
        confirmation=TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="synthetic_factory_engineer",
            confirmed_at=datetime.now(UTC),
        )
    )

    assert target.is_confirmed


def test_confirmation_requires_confirmer():
    with pytest.raises(ValidationError):
        TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_at=datetime.now(UTC),
        )


def test_confirmation_requires_timestamp():
    with pytest.raises(ValidationError):
        TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="synthetic_factory_engineer",
        )


def test_confirmation_requires_timezone():
    with pytest.raises(ValidationError):
        TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="synthetic_factory_engineer",
            confirmed_at=datetime(2026, 1, 1),
        )


def test_unconfirmed_target_rejects_confirmation_metadata():
    with pytest.raises(ValidationError):
        TargetConfirmation(
            confirmed_by="synthetic_factory_engineer",
        )


def test_confirmed_target_requires_known_availability():
    with pytest.raises(ValidationError):
        make_target(
            feature=make_target_feature(
                availability=FeatureAvailability.UNKNOWN,
            ),
            confirmation=TargetConfirmation(
                status=ConfirmationStatus.CONFIRMED,
                confirmed_by="synthetic_factory_engineer",
                confirmed_at=datetime.now(UTC),
            ),
        )


def test_valid_derived_rate():
    target = make_target(
        derived_rate=make_rate(),
    )

    assert target.derived_rate.multiplier == 100


def test_ratio_rate():
    rate = make_rate(
        output_unit="ratio",
        multiplier=1,
    )

    assert rate.multiplier == 1


def test_rate_rejects_identical_inputs():
    with pytest.raises(ValidationError):
        make_rate(
            denominator="waste_quantity_kg",
        )


def test_rate_rejects_invalid_numerator():
    with pytest.raises(ValidationError):
        make_target(
            derived_rate=make_rate(
                numerator="another_target",
            )
        )


def test_rate_rejects_granularity_mismatch():
    with pytest.raises(ValidationError):
        make_target(
            derived_rate=make_rate(
                granularity="machine_shift",
            )
        )


@pytest.mark.parametrize(
    "multiplier",
    [0, -1, float("nan"), float("inf")],
)
def test_invalid_rate_multiplier(multiplier):
    with pytest.raises(ValidationError):
        make_rate(multiplier=multiplier)


def test_invalid_percentage_multiplier():
    with pytest.raises(ValidationError):
        make_rate(multiplier=1)


def test_rate_output_cannot_reuse_input():
    with pytest.raises(ValidationError):
        make_rate(name="waste_quantity_kg")


def test_invalid_contract_version():
    with pytest.raises(ValidationError):
        make_target(version="invalid")


def test_target_contract_is_frozen():
    target = make_target()

    with pytest.raises(ValidationError):
        target.version = "2.0.0"


def test_unknown_target_fields():
    with pytest.raises(ValidationError):
        make_target(unknown_field=True)


def test_target_json_round_trip():
    target = make_target(
        derived_rate=make_rate(),
    )

    restored = TargetContract.model_validate_json(
        target.model_dump_json()
    )

    assert restored == target


@pytest.mark.parametrize(
    "task_type",
    [
        TaskType.OPTIMIZATION,
        TaskType.ANOMALY_DETECTION,
        TaskType.DESCRIPTIVE_ANALYTICS,
    ],
)
def test_unsupported_supervised_target_tasks(task_type):
    with pytest.raises(ValidationError):
        make_target(task_type=task_type)