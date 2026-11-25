"""Configuration constants and defaults for the iron_manufacturing domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DOMAIN_ID = "iron_manufacturing"
DOMAIN_VERSION = "1.0.0"
IMPLEMENTATION_VERSION = "1.0.0"
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_TRAINING_ROWS = 30
DEFAULT_MIN_TEST_ROWS = 5
DEFAULT_MIN_VALIDATION_ROWS = 5
DEFAULT_MIN_POSITIVE_TARGET_ROWS = 5
ZERO_INFLATION_HURDLE_THRESHOLD = 0.15

# Bare ambiguous column headers that must never be silently mapped by name similarity.
AMBIGUOUS_RAW_HEADERS: frozenset[str] = frozenset(
    {
        "weight",
        "mass",
        "energy",
        "power",
        "temp",
        "temperature",
        "waste",
        "scrap",
        "yield",
        "cost",
        "output",
        "input",
        "quantity",
        "amount",
        "الوزن",
        "الكميه",
        "الطاقه",
        "الحراره",
        "الهالك",
        "التكلفه",
    }
)


@dataclass(frozen=True)
class IronDomainPaths:
    """Configurable local paths for domain storage, artifacts, and reports."""

    storage_root: Path = Path("artifacts/iron_manufacturing/store")
    templates_root: Path = Path("contracts/domains/iron_manufacturing/templates")
    reports_root: Path = Path("reports/iron_manufacturing")
    data_root: Path = Path("data/iron_manufacturing")

    def ensure_runtime_dirs(self) -> None:
        """Create local runtime directories without touching repository root."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
