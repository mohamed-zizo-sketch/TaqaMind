"""Tests for shared pipeline execution context."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetPurpose,
)
from taqamind.core.interfaces import PipelineContext


def make_context(**overrides) -> PipelineContext:
    """Create a synthetic execution context."""

    data = {
        "factory_id": uuid4(),
        "dataset_revision_id": uuid4(),
        "domain": "textile",
        "process": "knitting",
        "task_id": "waste_prediction",
        "dataset_id": "textile_knitting_waste",
        "contract_version": "1.0.0",
        "task_version": "1.0.0",
        "target_version": "1.0.0",
        "purpose": DatasetPurpose.TRAINING,
    }

    data.update(overrides)

    return PipelineContext(**data)


def test_valid_training_context():
    context = make_context()

    assert context.domain == "textile"
    assert context.process == "knitting"
    assert context.purpose == DatasetPurpose.TRAINING


def test_execution_ids_are_unique():
    first = make_context()
    second = make_context()

    assert first.execution_id != second.execution_id


def test_task_key():
    context = make_context()

    assert context.task_key == (
        "textile",
        "knitting",
        "waste_prediction",
    )


def test_factory_task_key():
    context = make_context()

    assert context.factory_task_key == (
        context.factory_id,
        "textile",
        "knitting",
        "waste_prediction",
    )


def test_factory_scopes_are_independent():
    first = make_context()
    second = make_context()

    assert first.task_key == second.task_key

    assert (
        first.factory_task_key
        != second.factory_task_key
    )


def test_same_dataset_can_have_multiple_executions():
    revision_id = uuid4()
    factory_id = uuid4()

    first = make_context(
        factory_id=factory_id,
        dataset_revision_id=revision_id,
    )

    second = make_context(
        factory_id=factory_id,
        dataset_revision_id=revision_id,
    )

    assert (
        first.dataset_revision_id
        == second.dataset_revision_id
    )

    assert first.execution_id != second.execution_id


def test_training_does_not_require_model_version():
    context = make_context()

    assert context.model_version_id is None


def test_prediction_requires_model_version():
    with pytest.raises(
        ValidationError,
        match="Prediction requires a registered model",
    ):
        make_context(
            purpose=DatasetPurpose.PREDICTION,
        )


def test_valid_prediction_context():
    model_version_id = uuid4()

    context = make_context(
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=model_version_id,
    )

    assert context.model_version_id == model_version_id


def test_timestamp_is_timezone_aware():
    context = make_context()

    assert context.created_at.utcoffset() is not None


def test_naive_timestamp_is_rejected():
    with pytest.raises(
        ValidationError,
        match="timezone-aware",
    ):
        make_context(
            created_at=datetime(2026, 1, 1),
        )


@pytest.mark.parametrize(
    "field",
    [
        "domain",
        "process",
        "task_id",
        "dataset_id",
    ],
)
def test_invalid_identifiers(field):
    with pytest.raises(ValidationError):
        make_context(
            **{field: "Invalid Identifier"}
        )


@pytest.mark.parametrize(
    "field",
    [
        "contract_version",
        "task_version",
        "target_version",
    ],
)
def test_invalid_versions(field):
    with pytest.raises(ValidationError):
        make_context(
            **{field: "invalid"}
        )


def test_invalid_factory_id():
    with pytest.raises(ValidationError):
        make_context(factory_id="invalid")


def test_invalid_dataset_revision_id():
    with pytest.raises(ValidationError):
        make_context(dataset_revision_id="invalid")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_context(unknown_field=True)


def test_context_is_frozen():
    context = make_context()

    with pytest.raises(ValidationError):
        context.factory_id = uuid4()


def test_json_round_trip():
    context = make_context()

    restored = PipelineContext.model_validate_json(
        context.model_dump_json()
    )

    assert restored == context


@pytest.mark.parametrize(
    "domain",
    [
        "textile",
        "food_beverage",
        "iron_manufacturing",
    ],
)
def test_multi_domain_compatibility(domain):
    context = make_context(domain=domain)

    assert context.domain == domain