"""Local filesystem storage adapters implementing Shared Core storage protocols.

Migration note:
    These adapters are temporary local implementations of `DatasetArtifactStorage`
    and `ModelArtifactStorage`. They enforce factory scope, SHA-256 integrity,
    size limits, and artifact immutability, and are designed for direct migration
    to `taqamind.core.storage` or replacement via dependency injection.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from taqamind.core.interfaces.artifacts import ArtifactReference
from taqamind.core.interfaces.candidates import CandidateModelReference
from taqamind.core.interfaces.storage import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    ArtifactScopeError,
    StorageReference,
    calculate_stream_integrity,
    validate_artifact_scope,
    verify_artifact_integrity,
)


class _BaseLocalArtifactStorage:
    """Reusable local filesystem artifact store with integrity and immutability checks."""

    def __init__(self, root: Path | str, *, max_size_bytes: int | None = None) -> None:
        self._root = Path(root).resolve()
        self._max_size_bytes = max_size_bytes
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve_path(self, reference: StorageReference) -> Path:
        target = (self._root / reference.storage_key).resolve()
        try:
            target.relative_to(self._root)
        except ValueError as exc:
            raise ArtifactScopeError("Resolved artifact path escapes storage root.") from exc
        if os.name == "nt":
            raw_str = str(target)
            if not raw_str.startswith("\\\\?\\"):
                return Path(f"\\\\?\\{raw_str}")
        return target

    def _put_reference(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
        stream: BinaryIO,
    ) -> None:
        validate_artifact_scope(factory_id=factory_id, reference=reference)
        data = stream.read()
        integrity = calculate_stream_integrity(
            io.BytesIO(data),
            max_size_bytes=self._max_size_bytes,
        )
        verify_artifact_integrity(reference=reference, integrity=integrity)

        target = self._resolve_path(reference)
        if target.exists():
            existing_bytes = target.read_bytes()
            existing_integrity = calculate_stream_integrity(io.BytesIO(existing_bytes))
            if (
                existing_integrity.content_sha256 == reference.content_sha256
                and existing_integrity.size_bytes == reference.size_bytes
            ):
                return
            raise ArtifactConflictError(
                f"Existing immutable artifact at '{reference.storage_key}' "
                "conflicts with incoming content."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp_art_")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    @contextmanager
    def _open_reference(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
    ) -> Iterator[BinaryIO]:
        validate_artifact_scope(factory_id=factory_id, reference=reference)
        target = self._resolve_path(reference)
        if not target.exists():
            raise ArtifactNotFoundError(f"Artifact not found: '{reference.storage_key}'.")

        raw_bytes = target.read_bytes()
        integrity = calculate_stream_integrity(
            io.BytesIO(raw_bytes),
            max_size_bytes=self._max_size_bytes,
        )
        verify_artifact_integrity(reference=reference, integrity=integrity)

        buffer = io.BytesIO(raw_bytes)
        try:
            yield buffer
        finally:
            buffer.close()

    def _exists_reference(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
    ) -> bool:
        validate_artifact_scope(factory_id=factory_id, reference=reference)
        return self._resolve_path(reference).exists()


class LocalDatasetArtifactStorage(_BaseLocalArtifactStorage):
    """Local filesystem implementation of Shared Core `DatasetArtifactStorage`."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
        stream: BinaryIO,
    ) -> None:
        self._put_reference(factory_id=factory_id, reference=reference, stream=stream)

    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> AbstractContextManager[BinaryIO]:
        return self._open_reference(factory_id=factory_id, reference=reference)

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> bool:
        return self._exists_reference(factory_id=factory_id, reference=reference)


class LocalModelArtifactStorage(_BaseLocalArtifactStorage):
    """Local filesystem implementation of Shared Core `ModelArtifactStorage`."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
        stream: BinaryIO,
    ) -> None:
        self._put_reference(factory_id=factory_id, reference=reference, stream=stream)

    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> AbstractContextManager[BinaryIO]:
        return self._open_reference(factory_id=factory_id, reference=reference)

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> bool:
        return self._exists_reference(factory_id=factory_id, reference=reference)
