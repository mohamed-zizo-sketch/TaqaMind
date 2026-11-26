"""Configuration constants and paths for the Food & Beverage domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DOMAIN_ID = "food_beverage"
DOMAIN_VERSION = "1.0.0"

SUPPORTED_PROCESSES = ("bakery", "dairy", "beverage")

UNIT_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    ("g", "kg"): 0.001,
    ("kg", "kg"): 1.0,
    ("ton", "kg"): 1000.0,
    ("tonne", "kg"): 1000.0,
    ("ml", "L"): 0.001,
    ("l", "L"): 1.0,
    ("L", "L"): 1.0,
    ("hL", "L"): 100.0,
    ("hl", "L"): 100.0,
    ("m3", "L"): 1000.0,
    ("h", "min"): 60.0,
    ("hr", "min"): 60.0,
    ("min", "min"): 1.0,
    ("s", "min"): 1.0 / 60.0,
    ("degC", "degC"): 1.0,
    ("C", "degC"): 1.0,
}


@dataclass(frozen=True)
class FoodBeverageStorageConfig:
    """Local filesystem storage layout for factory-isolated artifacts."""

    root_dir: Path = Path("artifacts/food_beverage")

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "datasets" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "datasets" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def registries_dir(self) -> Path:
        return self.root_dir / "registries"

    @property
    def predictions_dir(self) -> Path:
        return self.root_dir / "predictions"

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.registries_dir,
            self.predictions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
