"""Concurrency regression tests for storage and registries (AUD-005)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from taqamind.core.interfaces import (
    ArtifactConflictError,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    RecordIdentity,
    RecordIdentityStatus,
    RecordScope,
    RegistryConflictError,
)
from taqamind.core.lineage import LocalRecordLineageRegistry
from taqamind.core.registry import LocalDatasetArtifactStorage, LocalDatasetRegistry


def test_concurrent_artifact_put_prevents_overwrite(tmp_path: Path) -> None:
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    factory_id = uuid4()
    artifact_id = uuid4()

    payload_a = b"col1,col2\n1,2\n"
    payload_b = b"col1,col2\n9,9\n"
    ref_a = ArtifactReference(
        artifact_id=artifact_id,
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload_a).hexdigest(),
        size_bytes=len(payload_a),
    )
    ref_b = ArtifactReference(
        artifact_id=artifact_id,
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload_b).hexdigest(),
        size_bytes=len(payload_b),
    )

    def _put(ref: ArtifactReference, data: bytes) -> str:
        try:
            storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(data))
            return "ok"
        except ArtifactConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                _put,
                ref_a if i % 2 == 0 else ref_b,
                payload_a if i % 2 == 0 else payload_b,
            )
            for i in range(12)
        ]
        outcomes = [f.result() for f in futures]

    assert "ok" in outcomes
    assert "conflict" in outcomes


def test_concurrent_dataset_registry_registration(tmp_path: Path) -> None:
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    registry = LocalDatasetRegistry(tmp_path / "reg", storage)
    factory_id = uuid4()
    rev_id = uuid4()
    payload = b"id,val\nA,1\n"
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(payload))
    meta = DatasetMetadata(
        revision_id=rev_id,
        root_revision_id=rev_id,
        factory_id=factory_id,
        domain="textile",
        process="weaving",
        dataset_id="ds",
        contract_version="1.0.0",
        task_id="task",
        task_version="1.0.0",
        target_version="1.0.0",
        source_file_sha256=ref.content_sha256,
        state=DatasetState.RAW,
    )
    artifact = DatasetArtifact(metadata=meta, reference=ref)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: registry.register(factory_id=factory_id, artifact=artifact),
                range(8),
            )
        )

    assert len(results) == 8
    assert registry.get(factory_id=factory_id, dataset_revision_id=rev_id) == artifact


def test_concurrent_lineage_identity_prevents_duplicate_allocation(tmp_path: Path) -> None:
    lineage = LocalRecordLineageRegistry(tmp_path / "reg")
    scope = RecordScope(
        factory_id=uuid4(),
        domain="textile",
        process="weaving",
        dataset_id="ds",
        identity_namespace="production_record",
    )
    biz_sha = sha256(b"ORDER-001").hexdigest()

    def _register_new(_: int) -> str:
        identity = RecordIdentity(
            scope=scope,
            business_key_sha256=biz_sha,
            status=RecordIdentityStatus.CONFIRMED,
        )
        try:
            lineage.register_identity(scope=scope, identity=identity)
            return "created"
        except RegistryConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_register_new, range(10)))

    assert outcomes.count("created") == 1
    assert outcomes.count("conflict") == 9
