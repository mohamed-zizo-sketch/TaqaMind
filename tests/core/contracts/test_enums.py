"""Tests for shared contract enumerations."""

import json
from enum import StrEnum

import pytest

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
    ReadinessStatus,
    TargetKind,
    TaskType,
    ValidationSeverity,
)

ENUMS = [
    DataType,
    FeatureRole,
    FeatureAvailability,
    FeatureOrigin,
    MissingValuePolicy,
    TaskType,
    ValidationSeverity,
    DatasetPurpose,
    ReadinessStatus,
    TargetKind,
    ConfirmationStatus,
]


@pytest.mark.parametrize("enum_class", ENUMS)
def test_enum_inherits_from_str_enum(enum_class):
    assert issubclass(enum_class, StrEnum)


@pytest.mark.parametrize("enum_class", ENUMS)
def test_enum_values_are_unique(enum_class):
    values = [member.value for member in enum_class.__members__.values()]

    assert len(values) == len(set(values))


@pytest.mark.parametrize("enum_class", ENUMS)
def test_enum_values_are_lowercase(enum_class):
    for member in enum_class:
        assert member.value == member.value.lower()


@pytest.mark.parametrize("enum_class", ENUMS)
def test_enum_serialization(enum_class):
    for member in enum_class:
        serialized = json.dumps(member)

        assert json.loads(serialized) == member.value


@pytest.mark.parametrize("enum_class", ENUMS)
def test_enum_round_trip(enum_class):
    for member in enum_class:
        assert enum_class(member.value) is member


@pytest.mark.parametrize("enum_class", ENUMS)
def test_invalid_enum_value_is_rejected(enum_class):
    with pytest.raises(ValueError):
        enum_class("invalid_value")


def test_feature_availability_values():
    assert FeatureAvailability.STATIC.value == "static"
    assert FeatureAvailability.PRE_PRODUCTION.value == "pre_production"
    assert FeatureAvailability.IN_PRODUCTION.value == "in_production"
    assert FeatureAvailability.POST_PRODUCTION.value == "post_production"
    assert FeatureAvailability.UNKNOWN.value == "unknown"


def test_readiness_status_values():
    assert ReadinessStatus.READY.value == "ready"
    assert ReadinessStatus.LIMITED.value == "limited"
    assert ReadinessStatus.BLOCKED.value == "blocked"