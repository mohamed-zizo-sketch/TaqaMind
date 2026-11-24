"""Tests for industrial record identity and version lineage."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    DatasetRecordMembership,
    FingerprintField,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchResult,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    calculate_record_fingerprint,
)


def make_scope(**overrides):
    data = {
        "factory_id": uuid4(),
        "domain": "textile",
        "process": "knitting",
        "dataset_id": "knitting_waste",
        "identity_namespace": "production_run",
    }

    data.update(overrides)

    return RecordScope(**data)


def make_identity(**overrides):
    data = {
        "scope": make_scope(),
        "business_key_sha256": "a" * 64,
        "status": RecordIdentityStatus.CONFIRMED,
    }

    data.update(overrides)

    return RecordIdentity(**data)


def make_version(**overrides):
    data = {
        "identity": make_identity(),
        "content_sha256": "b" * 64,
        "source_dataset_revision_id": uuid4(),
    }

    data.update(overrides)

    return RecordVersion(**data)


def make_updated_version(previous, **overrides):
    data = {
        "identity": previous.identity,
        "content_sha256": "c" * 64,
        "source_dataset_revision_id": uuid4(),
        "version_number": previous.version_number + 1,
        "previous_version_id": previous.record_version_id,
        "revision_reason": "Confirmed synthetic correction.",
        "confirmed_by": "synthetic_reviewer",
    }

    data.update(overrides)

    return RecordVersion(**data)


def test_fingerprint_is_deterministic():
    fields = (
        FingerprintField(name="run_id", value="R001"),
    )

    first = calculate_record_fingerprint(
        fields,
        namespace="business_key",
    )

    second = calculate_record_fingerprint(
        fields,
        namespace="business_key",
    )

    assert first == second
    assert len(first) == 64


def test_field_order_does_not_change_fingerprint():
    fields = (
        FingerprintField(name="quantity", value=1000),
        FingerprintField(name="waste", value=30),
    )

    first = calculate_record_fingerprint(
        fields,
        namespace="record_content",
    )

    second = calculate_record_fingerprint(
        tuple(reversed(fields)),
        namespace="record_content",
    )

    assert first == second


def test_changed_content_changes_fingerprint():
    first = calculate_record_fingerprint(
        (FingerprintField(name="waste", value=30),),
        namespace="record_content",
    )

    second = calculate_record_fingerprint(
        (FingerprintField(name="waste", value=35),),
        namespace="record_content",
    )

    assert first != second


def test_fingerprint_namespaces_are_distinct():
    fields = (
        FingerprintField(name="run_id", value="R001"),
    )

    first = calculate_record_fingerprint(
        fields,
        namespace="business_key",
    )

    second = calculate_record_fingerprint(
        fields,
        namespace="record_content",
    )

    assert first != second


def test_fingerprint_version_changes_hash():
    fields = (
        FingerprintField(name="run_id", value="R001"),
    )

    first = calculate_record_fingerprint(
        fields,
        namespace="business_key",
        version="1.0.0",
    )

    second = calculate_record_fingerprint(
        fields,
        namespace="business_key",
        version="2.0.0",
    )

    assert first != second


def test_empty_fingerprint_rejected():
    with pytest.raises(ValueError):
        calculate_record_fingerprint(
            (),
            namespace="record_content",
        )


def test_duplicate_fingerprint_fields_rejected():
    field = FingerprintField(
        name="run_id",
        value="R001",
    )

    with pytest.raises(ValueError, match="must be unique"):
        calculate_record_fingerprint(
            (field, field),
            namespace="business_key",
        )


def test_nonfinite_fingerprint_values_rejected():
    with pytest.raises(ValidationError):
        FingerprintField(
            name="waste",
            value=float("nan"),
        )


def test_confirmed_identity():
    identity = make_identity()

    assert identity.eligible_for_automatic_matching


def test_confirmed_identity_requires_business_key():
    with pytest.raises(ValidationError):
        make_identity(
            business_key_sha256=None,
        )


def test_unverified_identity_is_not_automatically_matched():
    identity = make_identity(
        status=RecordIdentityStatus.UNVERIFIED,
        business_key_sha256=None,
    )

    assert not identity.eligible_for_automatic_matching


def test_valid_record_version():
    version = make_version()

    assert version.version_number == 1
    assert version.previous_version_id is None


def test_valid_record_update():
    previous = make_version()

    updated = make_updated_version(previous)

    updated.validate_previous(previous)

    assert updated.identity == previous.identity

    assert updated.record_version_id != previous.record_version_id


def test_record_update_requires_confirmation():
    previous = make_version()

    with pytest.raises(ValidationError):
        make_updated_version(
            previous,
            confirmed_by=None,
        )


def test_unchanged_content_reuses_existing_version():
    previous = make_version()

    updated = make_updated_version(
        previous,
        content_sha256=previous.content_sha256,
    )

    with pytest.raises(ValueError, match="Unchanged content"):
        updated.validate_previous(previous)


def test_record_update_cannot_change_identity():
    previous = make_version()

    updated = make_updated_version(
        previous,
        identity=make_identity(),
    )

    with pytest.raises(ValueError, match="identity cannot change"):
        updated.validate_previous(previous)


def test_record_version_numbers_must_be_sequential():
    previous = make_version()

    updated = make_updated_version(
        previous,
        version_number=5,
    )

    with pytest.raises(ValueError, match="advance sequentially"):
        updated.validate_previous(previous)


def test_record_update_requires_correct_parent():
    previous = make_version()

    updated = make_updated_version(
        previous,
        previous_version_id=uuid4(),
    )

    with pytest.raises(ValueError, match="Previous record version"):
        updated.validate_previous(previous)


def test_fingerprint_migration_requires_explicit_handling():
    previous = make_version()

    updated = make_updated_version(
        previous,
        content_fingerprint_version="2.0.0",
    )

    with pytest.raises(ValueError, match="explicit migration"):
        updated.validate_previous(previous)


def test_exact_match_requires_existing_version():
    with pytest.raises(ValidationError):
        RecordMatchResult(
            status=RecordMatchStatus.EXACT_MATCH,
        )


def test_exact_match_does_not_require_review():
    result = RecordMatchResult(
        status=RecordMatchStatus.EXACT_MATCH,
        matched_version=make_version(),
    )

    assert not result.requires_review


def test_changed_content_requires_review():
    result = RecordMatchResult(
        status=RecordMatchStatus.CONTENT_CHANGED,
        matched_version=make_version(),
        reason="Target value changed.",
    )

    assert result.requires_review


@pytest.mark.parametrize(
    "status",
    [
        RecordMatchStatus.AMBIGUOUS,
        RecordMatchStatus.UNVERIFIED,
    ],
)
def test_unreliable_match_requires_review(status):
    result = RecordMatchResult(
        status=status,
        reason="Record identity requires verification.",
    )

    assert result.requires_review


def test_review_requires_reason():
    with pytest.raises(ValidationError):
        RecordMatchResult(
            status=RecordMatchStatus.AMBIGUOUS,
        )


def test_valid_record_membership():
    version = make_version()

    membership = DatasetRecordMembership(
        dataset_revision_id=uuid4(),
        scope=version.identity.scope,
        row_position=0,
        record_id=version.identity.record_id,
        record_version_id=version.record_version_id,
    )

    membership.validate_record_version(version)


def test_membership_rejects_wrong_record_version():
    version = make_version()

    membership = DatasetRecordMembership(
        dataset_revision_id=uuid4(),
        scope=version.identity.scope,
        row_position=0,
        record_id=version.identity.record_id,
        record_version_id=uuid4(),
    )

    with pytest.raises(ValueError, match="version mismatch"):
        membership.validate_record_version(version)


def test_record_version_serialization():
    version = make_version()

    restored = RecordVersion.model_validate_json(
        version.model_dump_json()
    )

    assert restored == version


def test_record_identity_is_frozen():
    identity = make_identity()

    with pytest.raises(ValidationError):
        identity.record_id = uuid4()