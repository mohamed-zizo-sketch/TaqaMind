"""Deterministic SHA-256 hashing helpers for contracts, configs, and implementations."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pydantic import BaseModel

from taqamind.core.interfaces.artifacts import calculate_bytes_sha256
from taqamind.core.interfaces.serialization import serialize_contract
from taqamind.domains.iron_manufacturing.config import (
    DOMAIN_ID,
    DOMAIN_VERSION,
    IMPLEMENTATION_VERSION,
)


def calculate_config_sha256(payload: dict[str, Any]) -> str:
    """Return a canonical SHA-256 hex digest of a JSON-serializable dictionary."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return calculate_bytes_sha256(canonical)


def calculate_contract_sha256(contract: BaseModel) -> str:
    """Return a canonical SHA-256 hex digest of a Pydantic contract model."""
    return calculate_config_sha256(serialize_contract(contract))


def collect_runtime_dependency_versions() -> dict[str, str]:
    """Capture installed runtime dependency versions for reproducibility."""
    packages = (
        "numpy",
        "pandas",
        "pydantic",
        "scikit-learn",
        "xgboost",
        "joblib",
        "PyYAML",
    )
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def calculate_implementation_sha256(extra: dict[str, Any] | None = None) -> str:
    """Calculate a deterministic fingerprint of domain implementation and dependencies."""
    payload: dict[str, Any] = {
        "domain_id": DOMAIN_ID,
        "domain_version": DOMAIN_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "dependencies": collect_runtime_dependency_versions(),
    }
    if extra:
        payload["extra"] = extra
    return calculate_config_sha256(payload)
