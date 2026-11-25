"""Tests for LocalDatasetRegistry and LocalRecordLineageRegistry."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMembershipManifest,
    DatasetMetadata,
    DatasetRecordMembership,
    DatasetRegistrationStatus,
    DatasetRegistry,
    DatasetState,
    FingerprintField,
    PipelineContext,
    RecordIdentity,
    RecordIdentityStatus,
    RecordLineageRegistry,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    calculate_record_fingerprint,
)
from taqamind.core.lineage import LocalRecordLineageRegistry
from taqamind.core.registry import LocalDatasetArtifactStorage, LocalDatasetRegistry


def make_context(factory_id=None) -> PipelineContext:
    return PipelineContext(
        factory_id=factory_id or uuid4(),
        dataset_revision_id=uuid4(),
        domain="textile",
        process="weaving",
        dataset_id="weaving_batches",
        contract_version="1.0.0",
        task_id="waste_regression",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
    )


def store_and_make_raw(
    storage: LocalDatasetArtifactStorage,
    context: PipelineContext,
    payload: bytes,
) -> DatasetArtifact:
    ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=context.factory_id, reference=ref, stream=BytesIO(payload))
    return DatasetArtifact(
        metadata=DatasetMetadata.create_raw(
            context=context,
            source_file_sha256=ref.content_sha256,
        ),
        reference=ref,
    )


def store_and_derive_ingested(
    storage: LocalDatasetArtifactStorage,
    context: PipelineContext,
    raw_artifact: DatasetArtifact,
    payload: bytes,
) -> DatasetArtifact:
    ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=context.factory_id, reference=ref, stream=BytesIO(payload))
    metadata = raw_artifact.metadata.derive(
        context=context,
        state=DatasetState.INGESTED,
        row_count=2,
        columns=(
            DatasetColumnSchema(name="batch_id", physical_dtype="object"),
            DatasetColumnSchema(name="waste_qty", physical_dtype="float64"),
        ),
        operation="ingest_csv",
        implementation_version="0.1.0",
        description="Synthetic ingested revision.",
    )
    return DatasetArtifact(metadata=metadata, reference=ref)


def test_registries_satisfy_protocols(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    ds_reg = LocalDatasetRegistry(tmp_path / "registry", storage)
    lin_reg = LocalRecordLineageRegistry(tmp_path / "registry", dataset_registry=ds_reg)
    assert isinstance(ds_reg, DatasetRegistry)
    assert isinstance(lin_reg, RecordLineageRegistry)


def test_dataset_registry_registration_idempotency_and_lineage(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    registry = LocalDatasetRegistry(tmp_path / "registry", storage)

    ctx1 = make_context()
    csv_bytes = b"batch_id,waste_qty\nB1,5.0\nB2,6.0\n"
    raw1 = store_and_make_raw(storage, ctx1, csv_bytes)

    res1 = registry.register(factory_id=ctx1.factory_id, artifact=raw1)
    assert res1.status == DatasetRegistrationStatus.REGISTERED
    assert not res1.has_matching_source

    # Idempotent re-registration succeeds without overwriting.
    res1_again = registry.register(factory_id=ctx1.factory_id, artifact=raw1)
    assert res1_again.status == DatasetRegistrationStatus.ALREADY_REGISTERED

    # Second raw upload with identical source bytes detects matching source revision.
    ctx2 = make_context(factory_id=ctx1.factory_id)
    raw2 = store_and_make_raw(storage, ctx2, csv_bytes)
    res2 = registry.register(factory_id=ctx2.factory_id, artifact=raw2)
    assert res2.has_matching_source
    assert raw1.metadata.revision_id in res2.matching_source_revision_ids

    # Child registration requires parent to be registered first.
    ingested1 = store_and_derive_ingested(storage, ctx1, raw1, csv_bytes)
    reg_ing = registry.register(factory_id=ctx1.factory_id, artifact=ingested1)
    assert reg_ing.status == DatasetRegistrationStatus.REGISTERED

    chain = registry.get_lineage(
        factory_id=ctx1.factory_id,
        dataset_revision_id=ingested1.metadata.revision_id,
    )
    assert [item.metadata.state for item in chain] == [
        DatasetState.RAW,
        DatasetState.INGESTED,
    ]


def test_dataset_registry_rejects_cross_factory_and_missing_parent(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    registry = LocalDatasetRegistry(tmp_path / "registry", storage)
    ctx = make_context()
    csv_bytes = b"batch_id,waste_qty\nB1,5.0\nB2,6.0\n"
    raw = store_and_make_raw(storage, ctx, csv_bytes)
    ingested = store_and_derive_ingested(storage, ctx, raw, csv_bytes)

    with pytest.raises(RegistryScopeError):
        registry.register(factory_id=uuid4(), artifact=raw)

    with pytest.raises(RegistryNotFoundError):
        registry.register(factory_id=ctx.factory_id, artifact=ingested)


def test_record_lineage_matching_versioning_and_membership(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    ds_reg = LocalDatasetRegistry(tmp_path / "registry", storage)
    lin_reg = LocalRecordLineageRegistry(tmp_path / "registry", dataset_registry=ds_reg)

    ctx = make_context()
    csv_bytes = b"batch_id,waste_qty\nB1,5.0\nB2,6.0\n"
    raw = store_and_make_raw(storage, ctx, csv_bytes)
    ingested = store_and_derive_ingested(storage, ctx, raw, csv_bytes)
    ds_reg.register(factory_id=ctx.factory_id, artifact=raw)
    ds_reg.register(factory_id=ctx.factory_id, artifact=ingested)

    scope = RecordScope(
        factory_id=ctx.factory_id,
        domain=ctx.domain,
        process=ctx.process,
        dataset_id=ctx.dataset_id,
        identity_namespace="production_batch",
    )

    key1 = calculate_record_fingerprint(
        (FingerprintField(name="batch_id", value="B1"),),
        namespace="business_key",
    )
    content1_v1 = calculate_record_fingerprint(
        (
            FingerprintField(name="batch_id", value="B1"),
            FingerprintField(name="waste_qty", value=5.0),
        ),
        namespace="record_content",
    )

    # Unknown business key -> NEW_RECORD; missing key -> UNVERIFIED.
    assert (
        lin_reg.match(
            scope=scope,
            business_key_sha256=None,
            content_sha256=content1_v1,
            content_fingerprint_version="1.0.0",
        ).status
        == RecordMatchStatus.UNVERIFIED
    )
    assert (
        lin_reg.match(
            scope=scope,
            business_key_sha256=key1,
            content_sha256=content1_v1,
            content_fingerprint_version="1.0.0",
        ).status
        == RecordMatchStatus.NEW_RECORD
    )

    id1 = RecordIdentity(
        scope=scope,
        business_key_sha256=key1,
        status=RecordIdentityStatus.CONFIRMED,
    )
    lin_reg.register_identity(scope=scope, identity=id1)

    v1 = RecordVersion(
        identity=id1,
        content_sha256=content1_v1,
        source_dataset_revision_id=ingested.metadata.revision_id,
    )
    lin_reg.register_version(scope=scope, version=v1)

    exact = lin_reg.match(
        scope=scope,
        business_key_sha256=key1,
        content_sha256=content1_v1,
        content_fingerprint_version="1.0.0",
    )
    assert exact.status == RecordMatchStatus.EXACT_MATCH
    assert exact.matched_version == v1

    content1_v2 = calculate_record_fingerprint(
        (
            FingerprintField(name="batch_id", value="B1"),
            FingerprintField(name="waste_qty", value=7.5),
        ),
        namespace="record_content",
    )
    changed = lin_reg.match(
        scope=scope,
        business_key_sha256=key1,
        content_sha256=content1_v2,
        content_fingerprint_version="1.0.0",
    )
    assert changed.status == RecordMatchStatus.CONTENT_CHANGED
    assert changed.requires_review

    v2 = RecordVersion(
        identity=id1,
        content_sha256=content1_v2,
        source_dataset_revision_id=uuid4(),
        version_number=2,
        previous_version_id=v1.record_version_id,
        revision_reason="Corrected scale reading.",
        confirmed_by="plant_supervisor",
    )
    lin_reg.register_version(scope=scope, version=v2)
    assert len(lin_reg.get_versions(scope=scope, record_id=id1.record_id)) == 2

    # Register second record and build full dataset membership manifest.
    key2 = calculate_record_fingerprint(
        (FingerprintField(name="batch_id", value="B2"),),
        namespace="business_key",
    )
    content2 = calculate_record_fingerprint(
        (
            FingerprintField(name="batch_id", value="B2"),
            FingerprintField(name="waste_qty", value=6.0),
        ),
        namespace="record_content",
    )
    id2 = RecordIdentity(
        scope=scope,
        business_key_sha256=key2,
        status=RecordIdentityStatus.CONFIRMED,
    )
    lin_reg.register_identity(scope=scope, identity=id2)
    v_b2 = RecordVersion(
        identity=id2,
        content_sha256=content2,
        source_dataset_revision_id=ingested.metadata.revision_id,
    )
    lin_reg.register_version(scope=scope, version=v_b2)

    manifest = DatasetMembershipManifest(
        dataset_revision_id=ingested.metadata.revision_id,
        scope=scope,
        memberships=(
            DatasetRecordMembership(
                dataset_revision_id=ingested.metadata.revision_id,
                scope=scope,
                row_position=0,
                record_id=id1.record_id,
                record_version_id=v1.record_version_id,
            ),
            DatasetRecordMembership(
                dataset_revision_id=ingested.metadata.revision_id,
                scope=scope,
                row_position=1,
                record_id=id2.record_id,
                record_version_id=v_b2.record_version_id,
            ),
        ),
    )
    lin_reg.register_membership(scope=scope, manifest=manifest)
    loaded = lin_reg.get_membership(
        scope=scope,
        dataset_revision_id=ingested.metadata.revision_id,
    )
    assert loaded == manifest

    # Duplicate confirmed business key is rejected.
    dup_id = RecordIdentity(
        scope=scope,
        business_key_sha256=key1,
        status=RecordIdentityStatus.CONFIRMED,
    )
    with pytest.raises(RegistryConflictError):
        lin_reg.register_identity(scope=scope, identity=dup_id)
