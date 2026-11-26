"""Deterministic synthetic fixtures for Food & Beverage domain tests."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pandas as pd
import pytest

from taqamind.domains.food_beverage.config import FoodBeverageStorageConfig
from taqamind.domains.food_beverage.domain import make_synthetic_target_confirmation
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline

FACTORY_A = uuid5(UUID("00000000-0000-0000-0000-000000000001"), "eg_cairo_bakery_01")
FACTORY_B = uuid5(UUID("00000000-0000-0000-0000-000000000001"), "eg_alex_bakery_02")


def make_synthetic_bakery_dataframe(
    *,
    factory_id: UUID = FACTORY_A,
    n_rows: int = 80,
    seed: int = 42,
    include_target: bool = True,
    batch_offset: int = 0,
) -> pd.DataFrame:
    """Create deterministic synthetic bakery batch records with learnable waste signal."""

    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 6, 0, tzinfo=UTC) + timedelta(hours=batch_offset * 4)
    timestamps = [
        (start + timedelta(hours=4 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n_rows)
    ]
    lines = rng.choice(["OVEN_LINE_1", "OVEN_LINE_2"], size=n_rows)
    products = rng.choice(["white_bread", "pastry", "biscuit"], size=n_rows)
    qty = rng.uniform(500.0, 2500.0, size=n_rows).round(2)
    raw_mat = (qty * rng.uniform(1.06, 1.12, size=n_rows)).round(2)
    temp = rng.uniform(170.0, 240.0, size=n_rows).round(1)
    dur = rng.uniform(15.0, 50.0, size=n_rows).round(1)
    shifts = rng.choice(["morning", "afternoon", "night"], size=n_rows)

    data: dict[str, object] = {
        "factory_id": [str(factory_id)] * n_rows,
        "batch_id": [f"BATCH-{batch_offset + i:04d}" for i in range(n_rows)],
        "production_timestamp": timestamps,
        "production_line_id": lines,
        "product_category": products,
        "planned_batch_quantity_kg": qty,
        "planned_raw_material_kg": raw_mat,
        "planned_processing_temp_c": temp,
        "planned_processing_duration_min": dur,
        "planned_shift": shifts,
    }
    if include_target:
        # Realistic physical signal: base 1.5% + thermal deviation + line factor + small noise
        line_factor = np.where(lines == "OVEN_LINE_2", 0.008, 0.0)
        temp_dev = np.abs(temp - 195.0) * 0.0004
        waste = qty * (0.015 + line_factor + temp_dev) + rng.normal(0.0, 1.0, size=n_rows)
        data["waste_quantity_kg"] = np.clip(waste, 0.0, None).round(3)

    return pd.DataFrame(data)


@pytest.fixture
def short_tmp_dir() -> Path:
    """Short temporary directory to avoid Windows MAX_PATH limits in nested lineage stores."""
    path = Path(tempfile.mkdtemp(prefix="fb_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tmp_pipeline(short_tmp_dir: Path) -> FoodBeveragePipeline:
    cfg = FoodBeverageStorageConfig(root_dir=short_tmp_dir / "s")
    return FoodBeveragePipeline(storage_config=cfg)


@pytest.fixture
def synthetic_confirmation():
    return make_synthetic_target_confirmation()


@pytest.fixture
def bakery_train_csv(tmp_path: Path) -> Path:
    df = make_synthetic_bakery_dataframe(n_rows=80, seed=42, include_target=True)
    path = tmp_path / "bakery_train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def bakery_predict_csv(tmp_path: Path) -> Path:
    df = make_synthetic_bakery_dataframe(n_rows=10, seed=99, include_target=False, batch_offset=200)
    path = tmp_path / "bakery_predict.csv"
    df.to_csv(path, index=False)
    return path
