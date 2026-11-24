"""Tests for immutable industrial split manifests."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    PipelineContext,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
    TrainingReadinessResult,
)
from tests.core.interfaces.test_readiness import (
    make_contract,
    make_stage,
)


def make_configuration(**overrides):
    data = {
        "strategy": SplitStrategy.GROUP_TEMPORAL,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "justification": "Synthetic group-temporal configuration.",
    }

    data.update(overrides)

    return SplitConfiguration(**data)


def make_context():
    return PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain="textile",
        process="knitting",
        task_id="waste_prediction",
        dataset_id="knitting_waste",
        contract_version="1.0.0",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
    )


def make_manifest(**overrides):
    data = {
        "context": make_context(),
        "dataset_revision_id": uuid4(),
        "configuration": make_configuration(),
        "train": SplitPartition(
            role=SplitRole.TRAIN,
            record_ids=tuple(
                uuid4() for _ in range(70)
            ),
        ),
        "validation": SplitPartition(
            role=SplitRole.VALIDATION,
            record_ids=tuple(
                uuid4() for _ in range(15)
            ),
        ),
        "test": SplitPartition(
            role=SplitRole.TEST,
            record_ids=tuple(
                uuid4() for _ in range(15)
            ),
        ),
    }

    data.update(overrides)

    return SplitManifest(**data)


def test_valid_split_configuration():
    config = make_configuration()

    assert config.train_ratio == 0.70
    assert config.random_seed == 42


def test_invalid_split_ratios():
    with pytest.raises(
        ValidationError,
        match="must sum to 1",
    ):
        make_configuration(train_ratio=0.80)


@pytest.mark.parametrize(
    "ratio",
    [0, -0.1, 1, float("nan")],
)
def test_invalid_individual_ratios(ratio):
    with pytest.raises(ValidationError):
        make_configuration(train_ratio=ratio)


def test_valid_split_manifest():
    manifest = make_manifest()

    assert manifest.total_rows == 100
    assert manifest.accounted_rows == 100

    assert manifest.actual_ratios == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }


def test_partition_cannot_be_empty():
    with pytest.raises(ValidationError):
        SplitPartition(
            role=SplitRole.TRAIN,
            record_ids=(),
        )


def test_duplicate_records_inside_partition():
    record_id = uuid4()

    with pytest.raises(ValidationError):
        SplitPartition(
            role=SplitRole.TRAIN,
            record_ids=(record_id, record_id),
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ],
)
def test_cross_partition_leakage(first, second):
    manifest = make_manifest()

    payload = manifest.model_dump()

    shared_record = payload[first]["record_ids"][0]

    payload[second]["record_ids"] = (
        shared_record,
        *payload[second]["record_ids"][1:],
    )

    with pytest.raises(
        ValidationError,
        match="Record leakage",
    ):
        SplitManifest.model_validate(payload)


def test_invalid_partition_role():
    with pytest.raises(
        ValidationError,
        match="Invalid partition role",
    ):
        make_manifest(
            train=SplitPartition(
                role=SplitRole.TEST,
                record_ids=(uuid4(),),
            ),
        )


def test_valid_record_coverage():
    manifest = make_manifest()

    expected = (
        *manifest.train.record_ids,
        *manifest.validation.record_ids,
        *manifest.test.record_ids,
    )

    manifest.validate_record_coverage(expected)


def test_missing_source_record():
    manifest = make_manifest()

    expected = (
        *manifest.train.record_ids,
        *manifest.validation.record_ids,
        *manifest.test.record_ids,
        uuid4(),
    )

    with pytest.raises(
        ValueError,
        match="expected records",
    ):
        manifest.validate_record_coverage(expected)


def test_excluded_records_require_reason():
    with pytest.raises(
        ValidationError,
        match="exclusion reason",
    ):
        make_manifest(
            excluded_record_ids=(uuid4(),),
        )


def test_excluded_records_cannot_be_in_split():
    manifest = make_manifest()

    payload = manifest.model_dump()

    payload["excluded_record_ids"] = (
        manifest.train.record_ids[0],
    )

    payload["exclusion_reason"] = "Synthetic exclusion."

    with pytest.raises(
        ValidationError,
        match="cannot belong to a partition",
    ):
        SplitManifest.model_validate(payload)


def test_excluded_records_are_accounted_for():
    manifest = make_manifest(
        excluded_record_ids=(uuid4(),),
        exclusion_reason="Synthetic boundary exclusion.",
    )

    assert manifest.total_rows == 100
    assert manifest.accounted_rows == 101

    manifest.validate_record_coverage(
        (
            *manifest.train.record_ids,
            *manifest.validation.record_ids,
            *manifest.test.record_ids,
            *manifest.excluded_record_ids,
        )
    )


def test_actual_ratios_can_differ_from_requested():
    manifest = make_manifest(
        train=SplitPartition(
            role=SplitRole.TRAIN,
            record_ids=tuple(
                uuid4() for _ in range(80)
            ),
        ),
    )

    assert manifest.actual_ratios["train"] != (
        manifest.configuration.train_ratio
    )


def test_manifest_is_frozen():
    manifest = make_manifest()

    with pytest.raises(ValidationError):
        manifest.split_id = uuid4()


def test_manifest_json_round_trip():
    manifest = make_manifest()

    restored = SplitManifest.model_validate_json(
        manifest.model_dump_json()
    )

    assert restored == manifest


def test_temporal_split_requires_time_column():
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(contract),
    )

    manifest = make_manifest(
        context=readiness.result.context,
        dataset_revision_id=(
            readiness.result.input_artifact.metadata.revision_id
        ),
        configuration=make_configuration(
            strategy=SplitStrategy.TEMPORAL,
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires a time column",
    ):
        manifest.validate_readiness(readiness)


def test_group_split_requires_grouping_columns():
    contract = make_contract()

    readiness = TrainingReadinessResult(
        contract=contract,
        result=make_stage(contract),
    )

    manifest = make_manifest(
        context=readiness.result.context,
        dataset_revision_id=(
            readiness.result.input_artifact.metadata.revision_id
        ),
        configuration=make_configuration(
            strategy=SplitStrategy.GROUP,
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires grouping columns",
    ):
        manifest.validate_readiness(readiness)