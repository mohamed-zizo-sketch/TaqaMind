"""Tests for training reuse and historical record exposure."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    AcceptancePolicy,
    DatasetMembershipManifest,
    DatasetRecordMembership,
    EvaluationRegistry,
    EvaluationReservation,
    RecordScope,
    RecordUsage,
    RecordUsageRegistry,
    RecordUsageRole,
    SplitRegistry,
    TrainingIntent,
    TrainingMode,
    TrainingRegistry,
    TrainingReuseKey,
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
    find_previously_exposed_test_records,
)
from tests.core.interfaces.test_evaluation import make_evaluation
from tests.core.interfaces.test_splitting import make_manifest
from tests.core.interfaces.test_training import make_configuration


def make_policy(**overrides):
    data = {
        "minimum_improvement_pct": 10.0,
        "justification": "Synthetic acceptance policy.",
        "created_at": datetime.now(UTC) - timedelta(seconds=10),
    }

    data.update(overrides)

    return AcceptancePolicy(**data)


def make_intent(**overrides):
    manifest = make_manifest()

    data = {
        "context": manifest.context,
        "mode": TrainingMode.INITIAL,
        "requested_by": "synthetic_factory_admin",
        "reason": "Initial factory training.",
    }

    data.update(overrides)

    return TrainingIntent(**data)


def make_reuse_key(**overrides):
    intent = make_intent()

    key = TrainingReuseKey.from_intent(
        intent=intent,
        dataset_content_sha256="a" * 64,
        contract_sha256="b" * 64,
        split_sha256="c" * 64,
        training_configuration_sha256="d" * 64,
        implementation_sha256="e" * 64,
    )

    payload = key.model_dump()
    payload.update(overrides)

    return TrainingReuseKey.model_validate(payload)


def make_scope(manifest):
    context = manifest.context

    return RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace="production_run",
    )


def make_membership(manifest):
    scope = make_scope(manifest)

    record_ids = (
        *manifest.train.record_ids,
        *manifest.validation.record_ids,
        *manifest.test.record_ids,
        *manifest.excluded_record_ids,
    )

    return DatasetMembershipManifest(
        dataset_revision_id=manifest.dataset_revision_id,
        scope=scope,
        memberships=tuple(
            DatasetRecordMembership(
                dataset_revision_id=manifest.dataset_revision_id,
                scope=scope,
                row_position=position,
                record_id=record_id,
                record_version_id=uuid4(),
            )
            for position, record_id in enumerate(record_ids)
        ),
    )


def make_usage(scope, record_id, **overrides):
    data = {
        "scope": scope,
        "record_id": record_id,
        "record_version_id": uuid4(),
        "execution_id": uuid4(),
        "split_id": uuid4(),
        "role": RecordUsageRole.TRAIN,
    }

    data.update(overrides)

    return RecordUsage(**data)


def test_initial_training_intent():
    intent = make_intent()

    assert intent.mode == TrainingMode.INITIAL
    assert intent.previous_model_version_id is None


def test_retraining_requires_previous_model():
    with pytest.raises(
        ValidationError,
        match="requires a previous model",
    ):
        make_intent(mode=TrainingMode.RETRAINING)


def test_valid_retraining_intent():
    intent = make_intent(
        mode=TrainingMode.RETRAINING,
        previous_model_version_id=uuid4(),
    )

    assert intent.previous_model_version_id is not None


def test_initial_training_rejects_previous_model():
    with pytest.raises(ValidationError):
        make_intent(previous_model_version_id=uuid4())


def test_prediction_cannot_create_training_intent():
    context = make_manifest().context

    prediction_context = context.__class__.model_validate(
        {
            **context.model_dump(),
            "purpose": DatasetPurpose.PREDICTION,
            "model_version_id": uuid4(),
        }
    )

    with pytest.raises(
        ValidationError,
        match="requires a training context",
    ):
        make_intent(context=prediction_context)


def test_equivalent_configurations_have_equal_fingerprints():
    first = make_configuration()
    second = make_configuration()

    assert first.configuration_id != second.configuration_id

    assert (
        calculate_training_configuration_fingerprint(
            first,
            make_policy(),
        )
        == calculate_training_configuration_fingerprint(
            second,
            make_policy(),
        )
    )


def test_changed_random_seed_changes_fingerprint():
    first = make_configuration(random_seed=42)
    second = make_configuration(random_seed=100)

    policy = make_policy()

    assert (
        calculate_training_configuration_fingerprint(first, policy)
        != calculate_training_configuration_fingerprint(second, policy)
    )


def test_changed_policy_changes_fingerprint():
    configuration = make_configuration()

    assert (
        calculate_training_configuration_fingerprint(
            configuration,
            make_policy(minimum_improvement_pct=10),
        )
        != calculate_training_configuration_fingerprint(
            configuration,
            make_policy(minimum_improvement_pct=20),
        )
    )


def test_reuse_key_is_deterministic():
    key = make_reuse_key()

    restored = TrainingReuseKey.model_validate_json(
        key.model_dump_json()
    )

    assert key.fingerprint == restored.fingerprint


@pytest.mark.parametrize(
    "field",
    [
        "dataset_content_sha256",
        "contract_sha256",
        "split_sha256",
        "training_configuration_sha256",
        "implementation_sha256",
    ],
)
def test_changed_content_invalidates_reuse_key(field):
    key = make_reuse_key()

    payload = key.model_dump()
    payload[field] = "f" * 64

    updated = TrainingReuseKey.model_validate(payload)

    assert key.fingerprint != updated.fingerprint


def test_different_factory_invalidates_reuse():
    key = make_reuse_key()

    payload = key.model_dump()
    payload["factory_id"] = uuid4()

    updated = TrainingReuseKey.model_validate(payload)

    assert key.fingerprint != updated.fingerprint


def test_split_fingerprint_is_deterministic():
    manifest = make_manifest()
    membership = make_membership(manifest)

    assert (
        calculate_split_fingerprint(manifest, membership)
        == calculate_split_fingerprint(manifest, membership)
    )


def test_record_version_change_invalidates_split_fingerprint():
    manifest = make_manifest()
    membership = make_membership(manifest)

    payload = membership.model_dump()
    payload["memberships"][0]["record_version_id"] = uuid4()

    updated = DatasetMembershipManifest.model_validate(payload)

    assert (
        calculate_split_fingerprint(manifest, membership)
        != calculate_split_fingerprint(manifest, updated)
    )


def test_membership_wrong_revision_rejected():
    manifest = make_manifest()
    other_manifest = make_manifest()

    membership = make_membership(other_manifest)

    with pytest.raises(
        ValueError,
        match="dataset revisions must match",
    ):
        calculate_split_fingerprint(manifest, membership)


@pytest.mark.parametrize(
    "role",
    [
        RecordUsageRole.TRAIN,
        RecordUsageRole.VALIDATION,
        RecordUsageRole.PROTECTED_TEST,
    ],
)
def test_previous_exposure_detected(role):
    manifest = make_manifest()
    scope = make_scope(manifest)

    record_id = manifest.test.record_ids[0]

    usage = make_usage(
        scope,
        record_id,
        role=role,
    )

    exposed = find_previously_exposed_test_records(
        manifest=manifest,
        scope=scope,
        history=(usage,),
    )

    assert exposed == (record_id,)


def test_old_training_records_are_not_automatically_blocked():
    manifest = make_manifest()
    scope = make_scope(manifest)

    usage = make_usage(
        scope,
        manifest.train.record_ids[0],
    )

    exposed = find_previously_exposed_test_records(
        manifest=manifest,
        scope=scope,
        history=(usage,),
    )

    assert exposed == ()


def test_cross_factory_history_rejected():
    manifest = make_manifest()
    scope = make_scope(manifest)

    other_scope = RecordScope.model_validate(
        {
            **scope.model_dump(),
            "factory_id": uuid4(),
        }
    )

    usage = make_usage(
        other_scope,
        manifest.test.record_ids[0],
    )

    with pytest.raises(ValueError, match="incompatible scope"):
        find_previously_exposed_test_records(
            manifest=manifest,
            scope=scope,
            history=(usage,),
        )


def test_evaluation_reservation():
    evaluation = make_evaluation()

    manifest = evaluation.split_manifest
    scope = make_scope(manifest)

    reservation = EvaluationReservation.from_selection(
        scope=scope,
        selection=evaluation.selection,
        manifest=manifest,
    )

    assert reservation.split_id == manifest.split_id
    assert reservation.test_record_ids == manifest.test.record_ids

    assert not (
        set(reservation.test_record_ids)
        & set(manifest.train.record_ids)
    )


def test_reservation_rejects_different_split():
    evaluation = make_evaluation()
    manifest = make_manifest()

    with pytest.raises(ValueError, match="split mismatch"):
        EvaluationReservation.from_selection(
            scope=make_scope(manifest),
            selection=evaluation.selection,
            manifest=manifest,
        )


def test_record_usage_serialization():
    manifest = make_manifest()

    usage = make_usage(
        make_scope(manifest),
        manifest.train.record_ids[0],
    )

    restored = RecordUsage.model_validate_json(
        usage.model_dump_json()
    )

    assert restored == usage


@pytest.mark.parametrize(
    "protocol",
    [
        SplitRegistry,
        RecordUsageRegistry,
        TrainingRegistry,
        EvaluationRegistry,
    ],
)
def test_empty_object_is_not_registry(protocol):
    assert not isinstance(object(), protocol)