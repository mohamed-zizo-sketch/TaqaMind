"""Tests for local immutable dataset and model artifact storage."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactReference,
    ArtifactScopeError,
    ArtifactSizeLimitError,
    CandidateModelReference,
    DatasetArtifactStorage,
    ModelArtifactStorage,
)
from taqamind.core.interfaces.storage import ArtifactConflictError
from taqamind.core.registry import (
    LocalDatasetArtifactStorage,
    LocalModelArtifactStorage,
)

CSV_BYTES = b"planned_quantity,waste_kg\n1000,25.5\n"
MODEL_BYTES = b"serialized-model-bundle-bytes"


def make_dataset_ref(
    content: bytes = CSV_BYTES,
    *,
    factory_id=None,
    artifact_id=None,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id or uuid4(),
        factory_id=factory_id or uuid4(),
        format=ArtifactFormat.CSV,
        content_sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def make_model_ref(
    content: bytes = MODEL_BYTES,
    *,
    factory_id=None,
    execution_id=None,
    candidate_id=None,
) -> CandidateModelReference:
    return CandidateModelReference(
        factory_id=factory_id or uuid4(),
        execution_id=execution_id or uuid4(),
        candidate_id=candidate_id or uuid4(),
        content_sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def test_storage_implementations_satisfy_protocols(tmp_path: Path):
    dataset_store = LocalDatasetArtifactStorage(tmp_path)
    model_store = LocalModelArtifactStorage(tmp_path)

    assert isinstance(dataset_store, DatasetArtifactStorage)
    assert isinstance(model_store, ModelArtifactStorage)


def test_dataset_put_exists_and_open_verified(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref()

    assert not store.exists(factory_id=ref.factory_id, reference=ref)

    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))

    assert store.exists(factory_id=ref.factory_id, reference=ref)
    with store.open_verified(factory_id=ref.factory_id, reference=ref) as stream:
        assert stream.read() == CSV_BYTES


def test_model_put_exists_and_open_verified(tmp_path: Path):
    store = LocalModelArtifactStorage(tmp_path)
    ref = make_model_ref()

    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(MODEL_BYTES))

    assert store.exists(factory_id=ref.factory_id, reference=ref)
    with store.open_verified(factory_id=ref.factory_id, reference=ref) as stream:
        assert stream.read() == MODEL_BYTES


def test_idempotent_put_with_identical_bytes_succeeds(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref()

    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))
    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))

    with store.open_verified(factory_id=ref.factory_id, reference=ref) as stream:
        assert stream.read() == CSV_BYTES


def test_conflict_when_existing_file_differs(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref()
    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))

    target = tmp_path / ref.storage_key
    target.write_bytes(b"corrupted-or-different-bytes")

    with pytest.raises(ArtifactConflictError):
        store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))


def test_put_rejects_hash_mismatch_and_leaves_no_artifact(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref(CSV_BYTES)
    wrong_same_size = b"x" * len(CSV_BYTES)

    with pytest.raises(ArtifactIntegrityError):
        store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(wrong_same_size))

    assert not store.exists(factory_id=ref.factory_id, reference=ref)
    assert not (tmp_path / ref.storage_key).exists()


def test_put_rejects_size_limit_exceeded(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path, max_size_bytes=10)
    ref = make_dataset_ref(CSV_BYTES)

    with pytest.raises(ArtifactSizeLimitError):
        store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))

    assert not store.exists(factory_id=ref.factory_id, reference=ref)


def test_factory_scope_isolation_enforced(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref()
    other_factory = uuid4()

    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))

    with pytest.raises(ArtifactScopeError):
        store.exists(factory_id=other_factory, reference=ref)

    with pytest.raises(ArtifactScopeError):
        with store.open_verified(factory_id=other_factory, reference=ref):
            pass


def test_open_verified_detects_missing_and_corrupted_artifacts(tmp_path: Path):
    store = LocalDatasetArtifactStorage(tmp_path)
    ref = make_dataset_ref()

    with pytest.raises(ArtifactNotFoundError):
        with store.open_verified(factory_id=ref.factory_id, reference=ref):
            pass

    store.put(factory_id=ref.factory_id, reference=ref, stream=BytesIO(CSV_BYTES))
    (tmp_path / ref.storage_key).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError):
        with store.open_verified(factory_id=ref.factory_id, reference=ref):
            pass