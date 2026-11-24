"""Storage interfaces for immutable industrial artifacts."""

from contextlib import AbstractContextManager
from hashlib import sha256
from typing import BinaryIO, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from taqamind.core.interfaces.artifacts import (
    ArtifactReference,
    SHA256Digest,
)
from taqamind.core.interfaces.candidates import (
    CandidateModelReference,
)

DEFAULT_CHUNK_SIZE = 1024 * 1024

StorageReference = ArtifactReference | CandidateModelReference


class StorageError(Exception):
    """Base exception for artifact storage failures."""


class ArtifactNotFoundError(StorageError):
    """Requested artifact does not exist."""


class ArtifactConflictError(StorageError):
    """An existing artifact conflicts with the requested content."""


class ArtifactIntegrityError(StorageError):
    """Stored artifact content failed integrity verification."""


class ArtifactScopeError(StorageError):
    """Artifact does not belong to the requested factory."""


class ArtifactSizeLimitError(StorageError):
    """Artifact exceeds the configured storage size limit."""


class ArtifactIntegrity(BaseModel):
    """Observed integrity information for binary content."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    content_sha256: SHA256Digest

    size_bytes: int = Field(ge=0)


def validate_artifact_scope(
    *,
    factory_id: UUID,
    reference: StorageReference,
) -> None:
    """Verify factory compatibility before storage operations."""

    if reference.factory_id != factory_id:
        raise ArtifactScopeError(
            "Artifact does not belong to the requested factory."
        )


def calculate_stream_integrity(
    stream: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_size_bytes: int | None = None,
) -> ArtifactIntegrity:
    """Calculate integrity from the current position to EOF.

    The stream is consumed but is not closed by this function.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    if max_size_bytes is not None and max_size_bytes < 0:
        raise ValueError(
            "Maximum artifact size cannot be negative."
        )

    digest = sha256()
    size_bytes = 0

    while chunk := stream.read(chunk_size):
        size_bytes += len(chunk)

        if (
            max_size_bytes is not None
            and size_bytes > max_size_bytes
        ):
            raise ArtifactSizeLimitError(
                "Artifact exceeds the configured size limit."
            )

        digest.update(chunk)

    return ArtifactIntegrity(
        content_sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def verify_artifact_integrity(
    *,
    reference: StorageReference,
    integrity: ArtifactIntegrity,
) -> None:
    """Compare observed content with the registered reference."""

    if reference.size_bytes != integrity.size_bytes:
        raise ArtifactIntegrityError(
            "Artifact size does not match the registered reference."
        )

    if reference.content_sha256 != integrity.content_sha256:
        raise ArtifactIntegrityError(
            "Artifact SHA-256 does not match the registered reference."
        )


@runtime_checkable
class DatasetArtifactStorage(Protocol):
    """Storage operations for immutable dataset artifacts."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
        stream: BinaryIO,
    ) -> None:
        """Store verified dataset content.

        Implementations must verify scope, size and hash before
        publishing the artifact.

        Existing content must never be overwritten with
        different bytes.

        Identical existing content may be reused.

        The caller owns the input stream.
        """
        ...

    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> AbstractContextManager[BinaryIO]:
        """Open a verified dataset stream.

        Implementations must validate scope and verify the
        complete content before exposing it to the caller.

        The returned context manager owns the output stream.
        """
        ...

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> bool:
        """Check whether an artifact exists in the requested scope.

        Existence alone does not establish content integrity.
        """
        ...


@runtime_checkable
class ModelArtifactStorage(Protocol):
    """Storage operations for internally generated model bundles."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
        stream: BinaryIO,
    ) -> None:
        """Store a trusted internally generated model bundle.

        Implementations must verify scope, integrity and
        immutability before publishing the artifact.

        The caller owns the input stream.
        """
        ...

    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> AbstractContextManager[BinaryIO]:
        """Open a model bundle after full integrity verification.

        Integrity verification does not authorize deserialization
        of an untrusted model.
        """
        ...

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> bool:
        """Check whether the referenced model artifact exists."""
        ...