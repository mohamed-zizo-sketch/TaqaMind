"""Shared bounded reading of verified CSV artifacts."""

import json
from dataclasses import asdict
from uuid import UUID

import pandas as pd

from taqamind.core.ingestion.resource_policy import CsvResourcePolicy
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
from taqamind.core.interfaces.artifacts import ArtifactReference, calculate_bytes_sha256
from taqamind.core.interfaces.storage import (
    ArtifactSizeLimitError,
    DatasetArtifactStorage,
    validate_artifact_scope,
)


def read_verified_csv(
    *,
    storage: DatasetArtifactStorage,
    factory_id: UUID,
    reference: ArtifactReference,
    reader: SafeCsvReader,
    policy: CsvResourcePolicy,
) -> pd.DataFrame:
    """Check scope and declared size before opening, then bound actual reading."""
    validate_artifact_scope(factory_id=factory_id, reference=reference)
    if reference.size_bytes > policy.max_bytes:
        raise SafeCsvError("CSV_BYTE_LIMIT_EXCEEDED", "CSV byte limit exceeded.")
    try:
        with storage.open_verified(factory_id=factory_id, reference=reference) as stream:
            return reader.read_stream(stream)
    except ArtifactSizeLimitError as exc:
        raise SafeCsvError(
            "ARTIFACT_SIZE_LIMIT_EXCEEDED",
            "Artifact exceeds the configured storage size limit.",
        ) from exc


def csv_configuration_sha256(
    policy: CsvResourcePolicy,
    *,
    delimiter: str = ",",
    extra: dict | None = None,
) -> str:
    """Fingerprint effective CSV configuration and optional stage configuration."""
    payload = {
        "resource_policy": asdict(policy),
        "delimiter": delimiter,
        "reader_version": SafeCsvReader.VERSION,
        "stage_configuration": extra,
    }
    return calculate_bytes_sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
