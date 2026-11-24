"""Tests for industrial feature contracts."""

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
)


def make_feature(**overrides) -> FeatureContract:
    """Create a valid feature for testing."""

    data = {
        "name": "planned_quantity_kg",
        "display_name": "Planned Quantity",
        "description": "Planned production quantity.",
        "data_type": DataType.FLOAT,
        "role": FeatureRole.PREDICTOR,
        "availability": FeatureAvailability.PRE_PRODUCTION,
        "unit": "kg",
    }

    data.update(overrides)

    return FeatureContract(**data)


def test_valid_feature():
    feature = make_feature()

    assert feature.name == "planned_quantity_kg"
    assert feature.data_type == DataType.FLOAT
    assert feature.required is True
    assert feature.nullable is False


def test_valid_numeric_constraints():
    constraints = FeatureConstraints(
        minimum=0,
        maximum=1000,
        minimum_inclusive=False,
    )

    assert constraints.minimum == 0
    assert constraints.maximum == 1000


@pytest.mark.parametrize(
    "name",
    ["Invalid Name", "1quantity", "quantity-kg", "", "UPPERCASE"],
)
def test_invalid_feature_name(name):
    with pytest.raises(ValidationError):
        make_feature(name=name)


def test_invalid_numeric_range():
    with pytest.raises(ValidationError):
        FeatureConstraints(
            minimum=100,
            maximum=10,
        )


def test_empty_exclusive_range():
    with pytest.raises(ValidationError):
        FeatureConstraints(
            minimum=10,
            maximum=10,
            minimum_inclusive=False,
        )


def test_numeric_constraint_on_string():
    with pytest.raises(ValidationError):
        make_feature(
            data_type=DataType.STRING,
            constraints=FeatureConstraints(minimum=0),
        )


def test_valid_categorical_feature():
    feature = make_feature(
        name="material_type",
        data_type=DataType.CATEGORY,
        unit=None,
        constraints=FeatureConstraints(
            allowed_values=("cotton", "polyester"),
        ),
    )

    assert feature.constraints.allowed_values == (
        "cotton",
        "polyester",
    )


def test_duplicate_allowed_values():
    with pytest.raises(ValidationError):
        FeatureConstraints(
            allowed_values=("cotton", "cotton"),
        )


def test_empty_allowed_values():
    with pytest.raises(ValidationError):
        FeatureConstraints(allowed_values=())


def test_numeric_feature_with_string_categories():
    with pytest.raises(ValidationError):
        make_feature(
            constraints=FeatureConstraints(
                allowed_values=("low", "high"),
            ),
        )


def test_nullable_feature_with_valid_policy():
    feature = make_feature(
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
    )

    assert feature.nullable is True


def test_nullable_feature_with_reject_policy():
    with pytest.raises(ValidationError):
        make_feature(
            nullable=True,
            missing_policy=MissingValuePolicy.REJECT,
        )


def test_non_nullable_feature_with_imputation():
    with pytest.raises(ValidationError):
        make_feature(
            nullable=False,
            missing_policy=MissingValuePolicy.IMPUTE,
        )


def test_target_imputation_is_rejected():
    with pytest.raises(ValidationError):
        make_feature(
            role=FeatureRole.TARGET,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
        )


def test_optional_non_nullable_feature():
    feature = make_feature(
        required=False,
        nullable=False,
    )

    assert feature.required is False
    assert feature.nullable is False


def test_duplicate_aliases():
    with pytest.raises(ValidationError):
        make_feature(
            aliases=("Planned Qty", " planned qty "),
        )


def test_alias_matching_canonical_name():
    with pytest.raises(ValidationError):
        make_feature(
            aliases=("PLANNED_QUANTITY_KG",),
        )


def test_valid_derived_feature():
    feature = make_feature(
        name="planned_area_m2",
        unit="m2",
        origin=FeatureOrigin.DERIVED,
        derived_from=(
            "planned_length_m",
            "fabric_width_m",
        ),
    )

    assert len(feature.derived_from) == 2


def test_derived_feature_without_dependencies():
    with pytest.raises(ValidationError):
        make_feature(
            origin=FeatureOrigin.DERIVED,
        )


def test_non_derived_feature_with_dependencies():
    with pytest.raises(ValidationError):
        make_feature(
            origin=FeatureOrigin.MEASURED,
            derived_from=("another_feature",),
        )


def test_feature_cannot_depend_on_itself():
    with pytest.raises(ValidationError):
        make_feature(
            origin=FeatureOrigin.DERIVED,
            derived_from=("planned_quantity_kg",),
        )


def test_duplicate_dependencies():
    with pytest.raises(ValidationError):
        make_feature(
            origin=FeatureOrigin.DERIVED,
            derived_from=("length", "length"),
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_feature(
            unknown_field="unexpected",
        )


def test_feature_contract_is_frozen():
    feature = make_feature()

    with pytest.raises(ValidationError):
        feature.unit = "meter"


def test_json_round_trip():
    feature = make_feature(
        aliases=("Planned Qty",),
        constraints=FeatureConstraints(minimum=0),
    )

    serialized = feature.model_dump_json()

    restored = FeatureContract.model_validate_json(serialized)

    assert restored == feature


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_constraints_are_rejected(value):
    with pytest.raises(ValidationError):
        FeatureConstraints(minimum=value)


def test_exclusive_minimum_without_bound():
    with pytest.raises(ValidationError):
        FeatureConstraints(minimum_inclusive=False)


def test_exclusive_maximum_without_bound():
    with pytest.raises(ValidationError):
        FeatureConstraints(maximum_inclusive=False)