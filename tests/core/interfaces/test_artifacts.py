"""Tests for dataset artifact references and hashing."""

from hashlib import sha256
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
    calculate_file_sha256,
)


def make_artifact(**overrides):
    data = {
        "factory_id": uuid4(),
        "format": ArtifactFormat.CSV,
        "content_sha256": "a" * 64,
        "size_bytes": 1024,
    }

    data.update(overrides)

    return ArtifactReference(**data)


def test_bytes_hash():
    content = b"sample industrial data"

    assert calculate_bytes_sha256(content) == (
        sha256(content).hexdigest()
    )


def test_same_content_has_same_hash():
    assert calculate_bytes_sha256(b"abc") == (
        calculate_bytes_sha256(b"abc")
    )


def test_different_content_has_different_hash():
    assert calculate_bytes_sha256(b"abc") != (
        calculate_bytes_sha256(b"abd")
    )


def test_file_hash(tmp_path):
    file = tmp_path / "dataset.csv"
    content = b"quantity,waste\n1000,30\n"

    file.write_bytes(content)

    assert calculate_file_sha256(file) == (
        calculate_bytes_sha256(content)
    )


def test_large_file_hash(tmp_path):
    file = tmp_path / "large_dataset.csv"
    content = b"industrial-data\n" * 200_000

    file.write_bytes(content)

    assert calculate_file_sha256(file) == (
        sha256(content).hexdigest()
    )


def test_empty_file_hash(tmp_path):
    file = tmp_path / "empty.csv"

    file.write_bytes(b"")

    assert calculate_file_sha256(file) == (
        sha256(b"").hexdigest()
    )


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        calculate_file_sha256(
            "nonexistent_dataset.csv"
        )


def test_valid_artifact():
    artifact = make_artifact()

    assert artifact.format == ArtifactFormat.CSV
    assert artifact.size_bytes == 1024


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "abc",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "A" * 64,
    ],
)
def test_invalid_hash_rejected(digest):
    with pytest.raises(ValidationError):
        make_artifact(content_sha256=digest)


def test_negative_file_size():
    with pytest.raises(ValidationError):
        make_artifact(size_bytes=-1)


def test_invalid_artifact_format():
    with pytest.raises(ValidationError):
        make_artifact(format="exe")


def test_storage_key_is_factory_scoped():
    artifact = make_artifact()

    assert artifact.storage_key == (
        f"factories/{artifact.factory_id}/"
        f"datasets/{artifact.artifact_id}.csv"
    )


def test_different_artifacts_have_unique_storage_keys():
    factory_id = uuid4()

    first = make_artifact(factory_id=factory_id)
    second = make_artifact(factory_id=factory_id)

    assert first.storage_key != second.storage_key


def test_different_factories_have_different_storage_keys():
    artifact_id = uuid4()

    first = make_artifact(
        artifact_id=artifact_id,
        factory_id=uuid4(),
    )

    second = make_artifact(
        artifact_id=artifact_id,
        factory_id=uuid4(),
    )

    assert first.storage_key != second.storage_key


def test_artifact_is_frozen():
    artifact = make_artifact()

    with pytest.raises(ValidationError):
        artifact.size_bytes = 500


def test_artifact_json_round_trip():
    artifact = make_artifact()

    restored = ArtifactReference.model_validate_json(
        artifact.model_dump_json()
    )

    assert restored == artifact