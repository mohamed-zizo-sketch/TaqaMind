"""Tests for shared artifact storage interfaces."""

from hashlib import sha256
from io import BytesIO
from uuid import uuid4

import pytest

from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactIntegrity,
    ArtifactIntegrityError,
    ArtifactReference,
    ArtifactScopeError,
    ArtifactSizeLimitError,
    CandidateModelReference,
    DatasetArtifactStorage,
    ModelArtifactStorage,
    calculate_stream_integrity,
    validate_artifact_scope,
    verify_artifact_integrity,
)

CONTENT = b"planned_quantity,waste\n1000,30\n"


def make_reference(content=CONTENT, **overrides):
    data = {
        "factory_id": uuid4(),
        "format": ArtifactFormat.CSV,
        "content_sha256": sha256(content).hexdigest(),
        "size_bytes": len(content),
    }

    data.update(overrides)

    return ArtifactReference(**data)


def test_stream_integrity():
    stream = BytesIO(CONTENT)

    integrity = calculate_stream_integrity(stream)

    assert integrity.content_sha256 == (
        sha256(CONTENT).hexdigest()
    )

    assert integrity.size_bytes == len(CONTENT)


def test_empty_stream_integrity():
    integrity = calculate_stream_integrity(
        BytesIO(b"")
    )

    assert integrity.size_bytes == 0

    assert integrity.content_sha256 == (
        sha256(b"").hexdigest()
    )


def test_multiple_chunks():
    content = b"industrial production data" * 10000

    integrity = calculate_stream_integrity(
        BytesIO(content),
        chunk_size=128,
    )

    assert integrity.size_bytes == len(content)

    assert integrity.content_sha256 == (
        sha256(content).hexdigest()
    )


def test_chunk_size_does_not_change_hash():
    first = calculate_stream_integrity(
        BytesIO(CONTENT),
        chunk_size=1,
    )

    second = calculate_stream_integrity(
        BytesIO(CONTENT),
        chunk_size=1024,
    )

    assert first == second


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(
        ValueError,
        match="Chunk size must be positive",
    ):
        calculate_stream_integrity(
            BytesIO(CONTENT),
            chunk_size=chunk_size,
        )


def test_stream_position_is_consumed():
    stream = BytesIO(CONTENT)

    calculate_stream_integrity(stream)

    assert stream.tell() == len(CONTENT)


def test_stream_remains_open():
    stream = BytesIO(CONTENT)

    calculate_stream_integrity(stream)

    assert not stream.closed


def test_integrity_starts_at_current_position():
    stream = BytesIO(b"abcdef")

    stream.seek(3)

    integrity = calculate_stream_integrity(stream)

    assert integrity.size_bytes == 3

    assert integrity.content_sha256 == (
        sha256(b"def").hexdigest()
    )


def test_size_limit_is_enforced():
    with pytest.raises(ArtifactSizeLimitError):
        calculate_stream_integrity(
            BytesIO(CONTENT),
            chunk_size=4,
            max_size_bytes=10,
        )


def test_exact_size_limit_is_allowed():
    integrity = calculate_stream_integrity(
        BytesIO(CONTENT),
        max_size_bytes=len(CONTENT),
    )

    assert integrity.size_bytes == len(CONTENT)


def test_negative_size_limit_rejected():
    with pytest.raises(ValueError):
        calculate_stream_integrity(
            BytesIO(CONTENT),
            max_size_bytes=-1,
        )


def test_valid_artifact_scope():
    reference = make_reference()

    validate_artifact_scope(
        factory_id=reference.factory_id,
        reference=reference,
    )


def test_invalid_factory_scope():
    reference = make_reference()

    with pytest.raises(ArtifactScopeError):
        validate_artifact_scope(
            factory_id=uuid4(),
            reference=reference,
        )


def test_valid_artifact_integrity():
    reference = make_reference()

    integrity = calculate_stream_integrity(
        BytesIO(CONTENT)
    )

    verify_artifact_integrity(
        reference=reference,
        integrity=integrity,
    )


def test_size_mismatch_rejected():
    reference = make_reference(
        size_bytes=len(CONTENT) + 1,
    )

    integrity = calculate_stream_integrity(
        BytesIO(CONTENT)
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="size",
    ):
        verify_artifact_integrity(
            reference=reference,
            integrity=integrity,
        )


def test_hash_mismatch_rejected():
    reference = make_reference(
        content_sha256="a" * 64,
    )

    integrity = calculate_stream_integrity(
        BytesIO(CONTENT)
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="SHA-256",
    ):
        verify_artifact_integrity(
            reference=reference,
            integrity=integrity,
        )


def test_modified_content_with_same_size_is_rejected():
    original = b"1000,30"
    modified = b"1000,90"

    reference = make_reference(content=original)

    integrity = calculate_stream_integrity(
        BytesIO(modified)
    )

    assert reference.size_bytes == integrity.size_bytes

    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_integrity(
            reference=reference,
            integrity=integrity,
        )


def test_model_artifact_integrity():
    content = b"synthetic model bundle"

    reference = CandidateModelReference(
        factory_id=uuid4(),
        execution_id=uuid4(),
        candidate_id=uuid4(),
        content_sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
    )

    validate_artifact_scope(
        factory_id=reference.factory_id,
        reference=reference,
    )

    verify_artifact_integrity(
        reference=reference,
        integrity=calculate_stream_integrity(
            BytesIO(content)
        ),
    )


def test_model_artifact_factory_mismatch():
    content = b"synthetic model bundle"

    reference = CandidateModelReference(
        factory_id=uuid4(),
        execution_id=uuid4(),
        candidate_id=uuid4(),
        content_sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
    )

    with pytest.raises(ArtifactScopeError):
        validate_artifact_scope(
            factory_id=uuid4(),
            reference=reference,
        )


def test_integrity_serialization():
    integrity = calculate_stream_integrity(
        BytesIO(CONTENT)
    )

    restored = ArtifactIntegrity.model_validate_json(
        integrity.model_dump_json()
    )

    assert restored == integrity


class SyntheticStorage:
    """Structural protocol test double only."""

    def put(self, *, factory_id, reference, stream):
        raise NotImplementedError

    def open_verified(self, *, factory_id, reference):
        raise NotImplementedError

    def exists(self, *, factory_id, reference):
        raise NotImplementedError


@pytest.mark.parametrize(
    "protocol",
    [
        DatasetArtifactStorage,
        ModelArtifactStorage,
    ],
)
def test_storage_protocol_compatibility(protocol):
    assert isinstance(
        SyntheticStorage(),
        protocol,
    )


@pytest.mark.parametrize(
    "protocol",
    [
        DatasetArtifactStorage,
        ModelArtifactStorage,
    ],
)
def test_empty_object_is_not_compatible(protocol):
    assert not isinstance(
        object(),
        protocol,
    )