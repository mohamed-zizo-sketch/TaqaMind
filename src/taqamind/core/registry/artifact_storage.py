"""Local filesystem implementations for immutable artifact storage."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from taqamind.core.interfaces.artifacts import ArtifactReference
from taqamind.core.interfaces.candidates import CandidateModelReference
from taqamind.core.interfaces.storage import (
    DEFAULT_CHUNK_SIZE,
    ArtifactConflictError,
    ArtifactIntegrity,
    ArtifactNotFoundError,
    ArtifactSizeLimitError,
    DatasetArtifactStorage,
    ModelArtifactStorage,
    StorageReference,
    calculate_stream_integrity,
    validate_artifact_scope,
    verify_artifact_integrity,
)


class _LocalArtifactStore:
    """Shared local filesystem store for immutable factory-scoped artifacts."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_size_bytes: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if max_size_bytes is not None and max_size_bytes < 0:
            raise ValueError("Maximum artifact size cannot be negative.")

        self._base_dir = Path(base_dir).resolve()
        self._chunk_size = chunk_size
        self._max_size_bytes = max_size_bytes

    @property
    def base_dir(self) -> Path:
        """Return the resolved root directory for stored artifacts."""
        return self._base_dir

    def _resolve_target(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
    ) -> Path:
        validate_artifact_scope(factory_id=factory_id, reference=reference)
        target = (self._base_dir / reference.storage_key).resolve()
        try:
            target.relative_to(self._base_dir)
        except ValueError as exc:
            raise ValueError("Resolved artifact path escapes storage root.") from exc
        return target

    def _stream_to_temp(
        self,
        *,
        parent_dir: Path,
        stream: BinaryIO,
    ) -> tuple[Path, ArtifactIntegrity]:
        parent_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=parent_dir)
        tmp_path = Path(raw_tmp_path)
        digest = sha256()
        size_bytes = 0

        try:
            with os.fdopen(fd, "wb") as tmp_file:
                while chunk := stream.read(self._chunk_size):
                    size_bytes += len(chunk)
                    if self._max_size_bytes is not None and size_bytes > self._max_size_bytes:
                        raise ArtifactSizeLimitError(
                            "Artifact exceeds the configured size limit."
                        )
                    digest.update(chunk)
                    tmp_file.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        integrity = ArtifactIntegrity(
            content_sha256=digest.hexdigest(),
            size_bytes=size_bytes,
        )
        return tmp_path, integrity

    def put(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
        stream: BinaryIO,
    ) -> None:
        target = self._resolve_target(factory_id=factory_id, reference=reference)
        tmp_path, integrity = self._stream_to_temp(
            parent_dir=target.parent,
            stream=stream,
        )

        try:
            verify_artifact_integrity(reference=reference, integrity=integrity)

            if target.exists():
                with target.open("rb") as existing_file:
                    existing_integrity = calculate_stream_integrity(
                        existing_file,
                        chunk_size=self._chunk_size,
                    )
                if existing_integrity != integrity:
                    raise ArtifactConflictError(
                        "Existing artifact content differs from the new content."
                    )
                tmp_path.unlink(missing_ok=True)
                return

            os.replace(tmp_path, target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
    ) -> Iterator[BinaryIO]:
        target = self._resolve_target(factory_id=factory_id, reference=reference)
        if not target.is_file():
            raise ArtifactNotFoundError(
                f"Artifact '{reference.storage_key}' does not exist."
            )

        file_obj = target.open("rb")
        try:
            integrity = calculate_stream_integrity(
                file_obj,
                chunk_size=self._chunk_size,
            )
            verify_artifact_integrity(reference=reference, integrity=integrity)
            file_obj.seek(0)
        except BaseException:
            file_obj.close()
            raise

        try:
            yield file_obj
        finally:
            file_obj.close()

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: StorageReference,
    ) -> bool:
        target = self._resolve_target(factory_id=factory_id, reference=reference)
        return target.is_file()


class LocalDatasetArtifactStorage(_LocalArtifactStore, DatasetArtifactStorage):
    """Local filesystem storage for immutable dataset artifacts."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
        stream: BinaryIO,
    ) -> None:
        super().put(factory_id=factory_id, reference=reference, stream=stream)

    @contextmanager
    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> Iterator[BinaryIO]:
        with super().open_verified(factory_id=factory_id, reference=reference) as stream:
            yield stream

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: ArtifactReference,
    ) -> bool:
        return super().exists(factory_id=factory_id, reference=reference)


class LocalModelArtifactStorage(_LocalArtifactStore, ModelArtifactStorage):
    """Local filesystem storage for trusted internally generated model bundles."""

    def put(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
        stream: BinaryIO,
    ) -> None:
        super().put(factory_id=factory_id, reference=reference, stream=stream)

    @contextmanager
    def open_verified(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> Iterator[BinaryIO]:
        with super().open_verified(factory_id=factory_id, reference=reference) as stream:
            yield stream

    def exists(
        self,
        *,
        factory_id: UUID,
        reference: CandidateModelReference,
    ) -> bool:
        return super().exists(factory_id=factory_id, reference=reference)
