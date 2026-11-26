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


def test_indexed_lineage_lookup_scales_for_large_factory_history(tmp_path: Path) -> None:
    import time

    lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
    scope = RecordScope(
        factory_id=uuid4(),
        domain="textile",
        process="weaving",
        dataset_id="weaving_batches",
        identity_namespace="production_record",
    )

    total_records = 300
    keys: list[str] = []
    for idx in range(total_records):
        bkey = sha256(f"REC-{idx:05d}".encode()).hexdigest()
        csha = sha256(f"CONTENT-{idx:05d}".encode()).hexdigest()
        keys.append(bkey)
        ident = RecordIdentity(
            scope=scope,
            business_key_sha256=bkey,
            status=RecordIdentityStatus.CONFIRMED,
        )
        lin_reg.register_identity(scope=scope, identity=ident)
        ver = RecordVersion(
            identity=ident,
            content_sha256=csha,
            source_dataset_revision_id=uuid4(),
        )
        lin_reg.register_version(scope=scope, version=ver)

    started = time.perf_counter()
    for idx in range(total_records):
        res = lin_reg.match(
            scope=scope,
            business_key_sha256=keys[idx],
            content_sha256=sha256(f"CONTENT-{idx:05d}".encode()).hexdigest(),
            content_fingerprint_version="1.0.0",
        )
        assert res.status == RecordMatchStatus.EXACT_MATCH
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0


def test_record_membership_builder_distinguishes_new_exact_corrected_and_distinct_keys(
    tmp_path: Path,
) -> None:
    from taqamind.core.contracts import (
        DatasetContract,
        DataType,
        FeatureAvailability,
        FeatureConstraints,
        FeatureContract,
        FeatureOrigin,
        FeatureRole,
        MissingValuePolicy,
        PredictionContext,
        TargetContract,
        TargetDefinition,
        TargetKind,
        TaskType,
    )
    from taqamind.core.ingestion import CsvDataIngestion
    from taqamind.core.interfaces import ApprovedMapping
    from taqamind.core.lineage import RecordMembershipBuilder
    from taqamind.core.mapping import DeterministicColumnMapper

    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    ds_reg = LocalDatasetRegistry(tmp_path / "reg", storage)
    lin_reg = LocalRecordLineageRegistry(tmp_path / "reg", dataset_registry=ds_reg)
    builder = RecordMembershipBuilder(storage, lin_reg)

    factory_id = uuid4()
    batch_id = FeatureContract(
        name="batch_id",
        display_name="Batch ID",
        description="Key",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    qty = FeatureContract(
        name="qty",
        display_name="Qty",
        description="Planned",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
    )
    waste = FeatureContract(
        name="waste",
        display_name="Waste",
        description="Target",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    contract = DatasetContract(
        dataset_id="ds_mem",
        version="1.0.0",
        domain="textile",
        process="weaving",
        description="Membership test",
        granularity="batch",
        features=(batch_id, qty),
        target=TargetContract(
            target_id="t1",
            feature=waste,
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(definition="Waste", granularity="batch"),
        ),
        prediction_context=PredictionContext(description="Pred"),
        business_keys=("batch_id",),
    )
    scope = RecordScope(
        factory_id=factory_id,
        domain="textile",
        process="weaving",
        dataset_id="ds_mem",
        identity_namespace="production_record",
    )

    def _prepare(csv_bytes: bytes) -> DatasetArtifact:
        rev_id = uuid4()
        ctx = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=rev_id,
            domain="textile",
            process="weaving",
            task_id="t1",
            dataset_id="ds_mem",
            contract_version="1.0.0",
            task_version="1.0.0",
            target_version="1.0.0",
            purpose=DatasetPurpose.TRAINING,
        )
        ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=sha256(csv_bytes).hexdigest(),
            size_bytes=len(csv_bytes),
        )
        storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))
        raw = DatasetArtifact(
            metadata=DatasetMetadata(
                revision_id=rev_id,
                root_revision_id=rev_id,
                factory_id=factory_id,
                domain="textile",
                process="weaving",
                dataset_id="ds_mem",
                contract_version="1.0.0",
                task_id="t1",
                task_version="1.0.0",
                target_version="1.0.0",
                source_file_sha256=ref.content_sha256,
                state=DatasetState.RAW,
            ),
            reference=ref,
        )
        ds_reg.register(factory_id=factory_id, artifact=raw)
        ing = CsvDataIngestion(storage).ingest(
            context=ctx, artifact=raw, contract=contract
        ).output_artifact
        assert ing is not None
        ds_reg.register(factory_id=factory_id, artifact=ing)
        mapper = DeterministicColumnMapper(storage)
        prop = mapper.propose_mapping(context=ctx, artifact=ing, contract=contract)
        entries = tuple(
            e.model_copy(update={"source_unit": e.canonical_unit})
            if e.canonical_unit is not None
            else e
            for e in prop.entries
        )
        mapped = mapper.apply_mapping(
            context=ctx,
            artifact=ing,
            contract=contract,
            mapping=ApprovedMapping(
                proposal=prop.model_copy(update={"entries": entries}),
                approved_by="test",
                confirmed_source_columns=tuple(e.source_column for e in entries),
            ),
        ).output_artifact
        assert mapped is not None
        ds_reg.register(factory_id=factory_id, artifact=mapped)
        return mapped

    # 1. Initial upload: 2 distinct business keys with identical predictor/target values.
    art1 = _prepare(b"batch_id,qty,waste\n001,100.0,2.0\n002,100.0,2.0\n")
    s1 = builder.build_and_register(scope=scope, artifact=art1, contract=contract)
    assert s1.new_records == 2
    assert s1.manifest.memberships[0].record_id != s1.manifest.memberships[1].record_id

    # 2. Re-upload 2 known + 1 new record.
    art2 = _prepare(b"batch_id,qty,waste\n001,100.0,2.0\n002,100.0,2.0\n003,150.0,3.0\n")
    s2 = builder.build_and_register(scope=scope, artifact=art2, contract=contract)
    assert s2.exact_matches == 2
    assert s2.new_records == 1

    # 3. Unconfirmed correction is rejected; confirmed correction increments version.
    art3 = _prepare(b"batch_id,qty,waste\n001,100.0,5.0\n")
    with pytest.raises(ValueError, match="without explicit confirmation"):
        builder.build_and_register(scope=scope, artifact=art3, contract=contract)
    s3 = builder.build_and_register(
        scope=scope,
        artifact=art3,
        contract=contract,
        confirm_updates_by="qa_lead",
        update_reason="Scale recalibration",
    )
    assert s3.content_changed == 1
    versions = lin_reg.get_versions(
        scope=scope, record_id=s1.manifest.memberships[0].record_id
    )
    assert len(versions) == 2
