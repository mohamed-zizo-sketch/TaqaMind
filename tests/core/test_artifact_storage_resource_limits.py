"""Tests for artifact storage resource limit enforcement."""

from hashlib import sha256
from io import BytesIO
from uuid import uuid4

import pytest

from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
)
from taqamind.core.interfaces.storage import (
    ArtifactIntegrityError,
    ArtifactSizeLimitError,
)
from taqamind.core.registry import LocalDatasetArtifactStorage


def make_reference(payload, factory_id):
    return ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def test_storage_rejects_oversized_reference_before_read(tmp_path):
    storage = LocalDatasetArtifactStorage(
        tmp_path,
        max_size_bytes=5,
    )

    factory_id = uuid4()
    payload = b"123456"
    reference = make_reference(payload, factory_id)

    class ForbiddenReadStream(BytesIO):
        def read(self, size=-1):
            pytest.fail("Oversized reference must not be read.")

    with ForbiddenReadStream(payload) as stream:
        with pytest.raises(ArtifactSizeLimitError):
            storage.put(
                factory_id=factory_id,
                reference=reference,
                stream=stream,
            )


def test_storage_rejects_oversized_existing_artifact(tmp_path):
    factory_id = uuid4()
    payload = b"123456"

    reference = make_reference(payload, factory_id)

    writer = LocalDatasetArtifactStorage(tmp_path)

    writer.put(
        factory_id=factory_id,
        reference=reference,
        stream=BytesIO(payload),
    )

    restricted = LocalDatasetArtifactStorage(
        tmp_path,
        max_size_bytes=5,
    )

    with pytest.raises(ArtifactSizeLimitError):
        with restricted.open_verified(
            factory_id=factory_id,
            reference=reference,
        ):
            pytest.fail("Oversized artifact was exposed.")


def test_storage_verifies_actual_size_not_only_reference(tmp_path):
    factory_id = uuid4()
    payload = b"12345"

    reference = make_reference(payload, factory_id)

    storage = LocalDatasetArtifactStorage(
        tmp_path,
        max_size_bytes=5,
    )

    storage.put(
        factory_id=factory_id,
        reference=reference,
        stream=BytesIO(payload),
    )

    artifact_path = tmp_path / reference.storage_key
    artifact_path.write_bytes(b"123456")

    with pytest.raises(ArtifactSizeLimitError):
        with storage.open_verified(
            factory_id=factory_id,
            reference=reference,
        ):
            pytest.fail("Oversized content was exposed.")


def test_storage_accepts_exact_size_limit(tmp_path):
    factory_id = uuid4()
    payload = b"12345"

    reference = make_reference(payload, factory_id)

    storage = LocalDatasetArtifactStorage(
        tmp_path,
        max_size_bytes=len(payload),
    )

    storage.put(
        factory_id=factory_id,
        reference=reference,
        stream=BytesIO(payload),
    )

    with storage.open_verified(
        factory_id=factory_id,
        reference=reference,
    ) as stream:
        assert stream.read() == payload

    assert stream.closed


def test_storage_still_rejects_corrupted_content(tmp_path):
    factory_id = uuid4()
    payload = b"12345"

    reference = make_reference(payload, factory_id)

    storage = LocalDatasetArtifactStorage(
        tmp_path,
        max_size_bytes=100,
    )

    storage.put(
        factory_id=factory_id,
        reference=reference,
        stream=BytesIO(payload),
    )

    artifact_path = tmp_path / reference.storage_key
    artifact_path.write_bytes(b"54321")

    with pytest.raises(ArtifactIntegrityError):
        with storage.open_verified(
            factory_id=factory_id,
            reference=reference,
        ):
            pytest.fail("Corrupted artifact was exposed.")
