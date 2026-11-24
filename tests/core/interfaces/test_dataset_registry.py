"""Tests for dataset registry and membership contracts."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMembershipManifest,
    DatasetRecordMembership,
    DatasetRegistrationResult,
    DatasetRegistrationStatus,
    DatasetRegistry,
    DatasetState,
    RecordLineageRegistry,
    RecordScope,
    RegistryConflictError,
    RegistryScopeError,
    validate_dataset_parent,
)


def make_membership_manifest(artifact, repeated=False):
    metadata = artifact.metadata

    scope = RecordScope(
        factory_id=metadata.factory_id,
        domain=metadata.domain,
        process=metadata.process,
        dataset_id=metadata.dataset_id,
        identity_namespace="production_run",
    )

    shared_record_id = uuid4()
    shared_version_id = uuid4()

    memberships = tuple(
        DatasetRecordMembership(
            dataset_revision_id=metadata.revision_id,
            scope=scope,
            row_position=position,
            record_id=(
                shared_record_id
                if repeated
                else uuid4()
            ),
            record_version_id=(
                shared_version_id
                if repeated
                else uuid4()
            ),
        )
        for position in range(metadata.row_count or 0)
    )

    return DatasetMembershipManifest(
        dataset_revision_id=metadata.revision_id,
        scope=scope,
        memberships=memberships,
    )


def test_valid_dataset_registration(raw_artifact):
    result = DatasetRegistrationResult(
        status=DatasetRegistrationStatus.REGISTERED,
        artifact=raw_artifact,
    )

    assert not result.has_matching_source


def test_matching_source_detected(raw_artifact):
    result = DatasetRegistrationResult(
        status=DatasetRegistrationStatus.REGISTERED,
        artifact=raw_artifact,
        matching_source_revision_ids=(uuid4(),),
    )

    assert result.has_matching_source


def test_dataset_cannot_match_itself(raw_artifact):
    with pytest.raises(ValidationError):
        DatasetRegistrationResult(
            status=DatasetRegistrationStatus.REGISTERED,
            artifact=raw_artifact,
            matching_source_revision_ids=(
                raw_artifact.metadata.revision_id,
            ),
        )


def test_valid_dataset_parent(raw_artifact, ingested_artifact):
    validate_dataset_parent(
        parent=raw_artifact,
        child=ingested_artifact,
    )


def test_wrong_parent_revision(raw_artifact, ingested_artifact):
    payload = ingested_artifact.model_dump()

    payload["metadata"]["transformation"][
        "parent_revision_id"
    ] = uuid4()

    child = DatasetArtifact.model_validate(payload)

    with pytest.raises(
        RegistryConflictError,
        match="parent revision mismatch",
    ):
        validate_dataset_parent(
            parent=raw_artifact,
            child=child,
        )


def test_wrong_root_revision(raw_artifact, ingested_artifact):
    payload = ingested_artifact.model_dump()

    payload["metadata"]["root_revision_id"] = uuid4()

    child = DatasetArtifact.model_validate(payload)

    with pytest.raises(
        RegistryConflictError,
        match="root revision mismatch",
    ):
        validate_dataset_parent(
            parent=raw_artifact,
            child=child,
        )


def test_cross_factory_parent_rejected(
    raw_artifact,
    ingested_artifact,
):
    factory_id = uuid4()

    payload = ingested_artifact.model_dump()

    payload["metadata"]["factory_id"] = factory_id
    payload["reference"]["factory_id"] = factory_id

    child = DatasetArtifact.model_validate(payload)

    with pytest.raises(
        RegistryScopeError,
        match="factory_id",
    ):
        validate_dataset_parent(
            parent=raw_artifact,
            child=child,
        )


def test_child_must_preserve_source_hash(
    raw_artifact,
    ingested_artifact,
):
    payload = ingested_artifact.model_dump()

    payload["metadata"]["source_file_sha256"] = "c" * 64

    child = DatasetArtifact.model_validate(payload)

    with pytest.raises(
        RegistryConflictError,
        match="source hash mismatch",
    ):
        validate_dataset_parent(
            parent=raw_artifact,
            child=child,
        )


def test_complete_dataset_lineage(
    context,
    raw_artifact,
    ingested_artifact,
):
    metadata = ingested_artifact.metadata.derive(
        context=context,
        state=DatasetState.MAPPED,
        row_count=1000,
        columns=ingested_artifact.metadata.columns,
        operation="synthetic_mapping",
        implementation_version="1.0.0",
        description="Synthetic mapping.",
    )

    mapped = DatasetArtifact(
        metadata=metadata,
        reference=ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256="c" * 64,
            size_bytes=2048,
        ),
    )

    validate_dataset_parent(
        parent=raw_artifact,
        child=ingested_artifact,
    )

    validate_dataset_parent(
        parent=ingested_artifact,
        child=mapped,
    )


def test_valid_dataset_membership(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
    )

    manifest.validate_artifact(ingested_artifact)

    assert manifest.row_count == 1000
    assert manifest.unique_record_count == 1000
    assert not manifest.has_repeated_record_ids


def test_repeated_records_are_representable(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
        repeated=True,
    )

    assert manifest.row_count == 1000
    assert manifest.unique_record_count == 1
    assert manifest.has_repeated_record_ids


def test_duplicate_row_positions_rejected(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
    )

    payload = manifest.model_dump()

    payload["memberships"][1]["row_position"] = 0

    with pytest.raises(
        ValidationError,
        match="positions must be unique",
    ):
        DatasetMembershipManifest.model_validate(payload)


def test_incomplete_row_membership(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
    )

    payload = manifest.model_dump()

    payload["memberships"] = payload["memberships"][:-1]

    updated = DatasetMembershipManifest.model_validate(
        payload
    )

    with pytest.raises(ValueError, match="row count"):
        updated.validate_artifact(ingested_artifact)


def test_membership_wrong_revision(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
    )

    payload = manifest.model_dump()

    payload["memberships"][0]["dataset_revision_id"] = uuid4()

    with pytest.raises(ValidationError):
        DatasetMembershipManifest.model_validate(payload)


def test_membership_serialization(ingested_artifact):
    manifest = make_membership_manifest(
        ingested_artifact,
    )

    restored = DatasetMembershipManifest.model_validate_json(
        manifest.model_dump_json()
    )

    assert restored == manifest


@pytest.mark.parametrize(
    "protocol",
    [
        DatasetRegistry,
        RecordLineageRegistry,
    ],
)
def test_empty_object_is_not_registry(protocol):
    assert not isinstance(object(), protocol)