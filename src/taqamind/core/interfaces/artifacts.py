"""Shared artifact references and integrity utilities."""

from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

SHA256Digest = Annotated[
    str,
    StringConstraints(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    ),
]


class ArtifactFormat(StrEnum):
    """Supported dataset artifact formats."""

    CSV = "csv"
    PARQUET = "parquet"


def calculate_bytes_sha256(content: bytes) -> str:
    """Calculate the SHA-256 digest of binary content."""

    return sha256(content).hexdigest()


def calculate_file_sha256(path: str | Path) -> str:
    """Calculate a file digest using bounded memory."""

    digest = sha256()

    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class ArtifactReference(BaseModel):
    """Immutable reference to a stored dataset artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_id: UUID = Field(default_factory=uuid4)

    factory_id: UUID

    format: ArtifactFormat

    content_sha256: SHA256Digest

    size_bytes: int = Field(ge=0)

    @property
    def storage_key(self) -> str:
        """Return a deterministic factory-scoped storage key."""

        return (
            f"factories/{self.factory_id}/"
            f"datasets/{self.artifact_id}.{self.format.value}"
        )