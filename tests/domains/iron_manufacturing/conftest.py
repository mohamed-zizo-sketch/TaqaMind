"""Deterministic offline fixtures for iron_manufacturing domain tests."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from taqamind.domains.iron_manufacturing import (
    IronManufacturingPipeline,
    make_synthetic_confirmation,
)
from taqamind.domains.iron_manufacturing.common.synthetic import (
    generate_synthetic_eaf_dataset,
    generate_synthetic_rebar_dataset,
    generate_synthetic_scrap_yard_dataset,
)

FACTORY_A_ID = UUID("11111111-1111-4111-8111-111111111111")
FACTORY_B_ID = UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture()
def storage_root(tmp_path: Path) -> Path:
    return tmp_path / "iron_store"


@pytest.fixture()
def pipeline(storage_root: Path) -> IronManufacturingPipeline:
    return IronManufacturingPipeline(storage_root=storage_root)


@pytest.fixture()
def synthetic_confirmation():
    return make_synthetic_confirmation(
        confirmed_by="pytest_synthetic_fixture",
        notes="Explicit synthetic confirmation for offline unit/integration tests.",
    )


@pytest.fixture()
def eaf_train_csv(tmp_path: Path) -> Path:
    df = generate_synthetic_eaf_dataset(n_rows=90, seed=42, include_target=True)
    path = tmp_path / "eaf_train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def eaf_predict_csv(tmp_path: Path) -> Path:
    df = generate_synthetic_eaf_dataset(
        n_rows=12,
        seed=99,
        include_target=False,
        start_index=501,
        start_date="2026-03-01T06:00:00",
    )
    path = tmp_path / "eaf_predict.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def rebar_train_csv(tmp_path: Path) -> Path:
    df = generate_synthetic_rebar_dataset(n_rows=90, seed=42, include_target=True)
    path = tmp_path / "rebar_train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def scrap_yard_train_csv(tmp_path: Path) -> Path:
    df = generate_synthetic_scrap_yard_dataset(
        n_rows=90, seed=42, include_target=True, zero_inflated=True
    )
    path = tmp_path / "scrap_yard_train.csv"
    df.to_csv(path, index=False)
    return path
