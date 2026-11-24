"""Tests for restricted industrial ML data access."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    ActiveModelProvider,
    DataAccessDeniedError,
    DevelopmentDataAccess,
    DevelopmentDataProvider,
    EvaluationReservation,
    ModelRegistryEntry,
    ModelStatus,
    PartitionDataReference,
    ProtectedTestDataAccess,
    ProtectedTestDataProvider,
    SplitRole,
    TrainingAuthorization,
    validate_active_model_access,
)
from tests.core.interfaces.test_evaluation import make_evaluation
from tests.core.interfaces.test_leakage import make_validation
from tests.core.interfaces.test_model_registry import make_validated
from tests.core.interfaces.test_prediction import make_prediction_batch
from tests.core.interfaces.test_readiness import make_contract
from tests.core.interfaces.test_training_registry import make_scope

LEARNING_COLUMNS = (
    "planned_quantity_kg",
    "actual_waste_kg",
)


def make_partition_reference(
    *,
    context,
    split_id,
    dataset_revision_id,
    partition,
    **overrides,
):
    data = {
        "partition_artifact_id": uuid4(),
        "factory_id": context.factory_id,
        "dataset_revision_id": dataset_revision_id,
        "split_id": split_id,
        "role": partition.role,
        "record_ids": partition.record_ids,
        "columns": LEARNING_COLUMNS,
        "content_sha256": "a" * 64,
        "size_bytes": 1024,
    }

    data.update(overrides)

    return PartitionDataReference(**data)


def make_development_access():
    leakage = make_validation()

    authorization = TrainingAuthorization.from_validation(
        leakage
    )

    development = authorization.development_split

    train = make_partition_reference(
        context=development.context,
        split_id=development.split_id,
        dataset_revision_id=development.dataset_revision_id,
        partition=development.train,
    )

    validation = make_partition_reference(
        context=development.context,
        split_id=development.split_id,
        dataset_revision_id=development.dataset_revision_id,
        partition=development.validation,
    )

    return DevelopmentDataAccess(
        authorization=authorization,
        train=train,
        validation=validation,
    )


def make_test_access():
    evaluation = make_evaluation()

    selection = evaluation.selection

    manifest = evaluation.split_manifest

    reservation = EvaluationReservation.from_selection(
        scope=make_scope(manifest),
        selection=selection,
        manifest=manifest,
    )

    test = make_partition_reference(
        context=manifest.context,
        split_id=manifest.split_id,
        dataset_revision_id=manifest.dataset_revision_id,
        partition=manifest.test,
    )

    return ProtectedTestDataAccess(
        reservation=reservation,
        selection=selection,
        test=test,
    )


def test_valid_development_access():
    access = make_development_access()

    assert access.train.role == SplitRole.TRAIN

    assert access.validation.role == SplitRole.VALIDATION

    assert not hasattr(access, "test")


def test_development_access_matches_contract():
    access = make_development_access()

    access.validate_contract(make_contract())


def test_training_partition_cannot_have_test_role():
    access = make_development_access()

    payload = access.model_dump()

    payload["train"]["role"] = SplitRole.TEST

    with pytest.raises(
        DataAccessDeniedError,
        match="Partition role mismatch",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_cross_factory_partition_rejected():
    access = make_development_access()

    payload = access.model_dump()

    payload["train"]["factory_id"] = uuid4()

    with pytest.raises(
        DataAccessDeniedError,
        match="factory mismatch",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_wrong_split_rejected():
    access = make_development_access()

    payload = access.model_dump()

    payload["train"]["split_id"] = uuid4()

    with pytest.raises(
        DataAccessDeniedError,
        match="split mismatch",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_wrong_dataset_revision_rejected():
    access = make_development_access()

    payload = access.model_dump()

    payload["validation"]["dataset_revision_id"] = uuid4()

    with pytest.raises(
        DataAccessDeniedError,
        match="dataset revision mismatch",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_wrong_record_coverage_rejected():
    access = make_development_access()

    payload = access.model_dump()

    payload["train"]["record_ids"] = (
        uuid4(),
        *access.train.record_ids[1:],
    )

    with pytest.raises(
        DataAccessDeniedError,
        match="record coverage mismatch",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_development_artifacts_must_be_separate():
    access = make_development_access()

    payload = access.model_dump()

    payload["validation"]["partition_artifact_id"] = (
        access.train.partition_artifact_id
    )

    with pytest.raises(
        DataAccessDeniedError,
        match="separate artifacts",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_development_schemas_must_match():
    access = make_development_access()

    payload = access.model_dump()

    payload["validation"]["columns"] = (
        "actual_waste_kg",
        "planned_quantity_kg",
    )

    with pytest.raises(
        DataAccessDeniedError,
        match="schemas must match",
    ):
        DevelopmentDataAccess.model_validate(payload)


def test_duplicate_partition_records_rejected():
    access = make_development_access()

    payload = access.train.model_dump()

    payload["record_ids"] = (
        access.train.record_ids[0],
        access.train.record_ids[0],
    )

    with pytest.raises(ValidationError):
        PartitionDataReference.model_validate(payload)


def test_unapproved_learning_column_rejected():
    access = make_development_access()

    payload = access.train.model_dump()

    payload["columns"] = (
        *LEARNING_COLUMNS,
        "unapproved_post_production_feature",
    )

    reference = PartitionDataReference.model_validate(
        payload
    )

    with pytest.raises(
        DataAccessDeniedError,
        match="unapproved learning columns",
    ):
        reference.validate_columns(make_contract())


def test_missing_target_rejected():
    access = make_development_access()

    payload = access.train.model_dump()

    payload["columns"] = (
        "planned_quantity_kg",
    )

    reference = PartitionDataReference.model_validate(
        payload
    )

    with pytest.raises(
        DataAccessDeniedError,
        match="missing required learning columns",
    ):
        reference.validate_columns(make_contract())


def test_valid_protected_test_access():
    access = make_test_access()

    assert access.test.role == SplitRole.TEST

    assert access.test.record_ids == (
        access.reservation.test_record_ids
    )


def test_protected_test_wrong_selection_rejected():
    access = make_test_access()

    payload = access.model_dump()

    payload["reservation"]["selection_id"] = uuid4()

    with pytest.raises(
        DataAccessDeniedError,
        match="selection mismatch",
    ):
        ProtectedTestDataAccess.model_validate(payload)


def test_protected_test_wrong_execution_rejected():
    access = make_test_access()

    payload = access.model_dump()

    payload["reservation"]["execution_id"] = uuid4()

    with pytest.raises(
        DataAccessDeniedError,
        match="execution mismatch",
    ):
        ProtectedTestDataAccess.model_validate(payload)


def test_protected_test_cannot_include_training_records():
    access = make_test_access()

    payload = access.model_dump()

    development = (
        access.selection.training_run.development_split
    )

    contaminated_ids = (
        development.train.record_ids[0],
        *access.test.record_ids[1:],
    )

    payload["reservation"]["test_record_ids"] = (
        contaminated_ids
    )

    payload["test"]["record_ids"] = contaminated_ids

    with pytest.raises(
        DataAccessDeniedError,
        match="overlaps development records",
    ):
        ProtectedTestDataAccess.model_validate(payload)


def test_protected_test_wrong_role_rejected():
    access = make_test_access()

    payload = access.model_dump()

    payload["test"]["role"] = SplitRole.TRAIN

    with pytest.raises(
        DataAccessDeniedError,
        match="Partition role mismatch",
    ):
        ProtectedTestDataAccess.model_validate(payload)


def test_development_access_serialization():
    access = make_development_access()

    restored = DevelopmentDataAccess.model_validate_json(
        access.model_dump_json()
    )

    assert restored == access


def test_protected_test_access_serialization():
    access = make_test_access()

    restored = ProtectedTestDataAccess.model_validate_json(
        access.model_dump_json()
    )

    assert restored == access


def test_inactive_model_access_rejected():
    batch = make_prediction_batch()

    entry = make_validated()

    assert entry.status == ModelStatus.VALIDATED

    with pytest.raises(
        DataAccessDeniedError,
        match="requires an active registered model",
    ):
        validate_active_model_access(
            context=batch.context,
            entry=entry,
            contract=batch.contract,
        )


def make_active_prediction_inputs():
    """Construct compatible synthetic prediction metadata."""

    batch = make_prediction_batch()

    validated = make_validated()

    version_payload = validated.version.model_dump()

    version_payload["identity"] = batch.model.model_dump()

    version_payload["artifact"]["factory_id"] = (
        batch.context.factory_id
    )

    entry = ModelRegistryEntry.model_validate(
        {
            "version": version_payload,
            "status": ModelStatus.ACTIVE,
            "evaluation_id": validated.evaluation_id,
        }
    )

    return batch, entry


def test_active_model_access():
    batch, entry = make_active_prediction_inputs()

    validate_active_model_access(
        context=batch.context,
        entry=entry,
        contract=batch.contract,
    )


def test_active_model_wrong_version_rejected():
    batch, entry = make_active_prediction_inputs()

    payload = entry.model_dump()

    payload["version"]["identity"]["model_version_id"] = (
        uuid4()
    )

    updated = ModelRegistryEntry.model_validate(payload)

    with pytest.raises(ValueError, match="model_version_id"):
        validate_active_model_access(
            context=batch.context,
            entry=updated,
            contract=batch.contract,
        )


@pytest.mark.parametrize(
    "protocol",
    [
        DevelopmentDataProvider,
        ProtectedTestDataProvider,
        ActiveModelProvider,
    ],
)
def test_empty_object_is_not_provider(protocol):
    assert not isinstance(object(), protocol)