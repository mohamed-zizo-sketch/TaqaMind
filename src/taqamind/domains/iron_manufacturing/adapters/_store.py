"""Internal local JSON persistence and verified CSV helpers for P1.6-P1.8 adapters."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from taqamind.core.interfaces.dataset import DatasetArtifact
from taqamind.core.interfaces.storage import DatasetArtifactStorage

_VALIDATION_REPORT_REQUIRED_KEYS = frozenset(
    {"dataset_id", "contract_version", "completed_stages", "issues"}
)
_VALIDATION_REPORT_COMPUTED_KEYS = frozenset({"status", "summary", "has_blocking_issues"})


def strip_computed_fields(payload: Any) -> Any:
    """Recursively strip ValidationReport @computed_field keys before JSON persistence.

    Pydantic v2 models with `extra='forbid'` reject `@computed_field` keys
    (`status`, `summary`, `has_blocking_issues`) on strict `model_validate_json()`.
    Stripping only dictionaries matching `ValidationReport` structure preserves
    legitimate `status` fields on `StageResult`, `CandidateResult`, and
    `ModelRegistryEntry`.
    """
    if isinstance(payload, dict):
        is_report = _VALIDATION_REPORT_REQUIRED_KEYS.issubset(payload.keys())
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if is_report and key in _VALIDATION_REPORT_COMPUTED_KEYS:
                continue
            cleaned[key] = strip_computed_fields(value)
        return cleaned
    if isinstance(payload, list):
        return [strip_computed_fields(item) for item in payload]
    if isinstance(payload, tuple):
        return [strip_computed_fields(item) for item in payload]
    return payload


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically after stripping ValidationReport computed fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = strip_computed_fields(payload)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(sanitized, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any) -> Any:
    """Read a local JSON file or return a default value if it does not exist."""
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame deterministically to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def read_artifact_df(
    storage: DatasetArtifactStorage,
    factory_id: UUID,
    artifact: DatasetArtifact,
) -> pd.DataFrame:
    """Read a verified CSV dataset artifact from storage into a DataFrame."""
    with storage.open_verified(factory_id=factory_id, reference=artifact.reference) as stream:
        return pd.read_csv(stream, keep_default_na=True, low_memory=False)
