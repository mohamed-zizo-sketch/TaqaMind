"""Textile Energy V2 scientific validation and bounded-read regression tests."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.domains.textile.common.formulas import compute_theoretical_nameplate_energy_kwh
from taqamind.domains.textile.energy_pipeline import TextileEnergyPipelineService
from taqamind.domains.textile.knitting.contracts import KNITTING_PROCESS_ID
from taqamind.domains.textile.knitting.energy_contracts import (
    KNITTING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS,
    build_knitting_energy_dataset_contract,
    build_synthetic_knitting_energy_confirmation,
)
from taqamind.domains.textile.weaving.contracts import WEAVING_PROCESS_ID
from taqamind.domains.textile.weaving.energy_contracts import (
    WEAVING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS,
    build_synthetic_weaving_energy_confirmation,
    build_weaving_energy_dataset_contract,
)


def test_energy_v2_contracts_exclude_post_production_leakage_columns() -> None:
    """Task 09: Post-production actuals must never be declared as predictors."""
    weaving_contract = build_weaving_energy_dataset_contract(
        confirmation=build_synthetic_weaving_energy_confirmation()
    )
    knitting_contract = build_knitting_energy_dataset_contract(
        confirmation=build_synthetic_knitting_energy_confirmation()
    )
    assert set(WEAVING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS).isdisjoint(
        set(weaving_contract.predictor_names)
    )
    assert set(KNITTING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS).isdisjoint(
        set(knitting_contract.predictor_names)
    )
    assert weaving_contract.target.feature.name == "total_electrical_energy_kwh"
    assert weaving_contract.target.feature.unit == "kWh"
    assert knitting_contract.target.feature.name == "total_electrical_energy_kwh"
    assert knitting_contract.target.feature.unit == "kWh"


def test_theoretical_nameplate_energy_formula_is_deterministic() -> None:
    """Task 10: Nameplate energy calculation is row-independent and rejects <= 0 inputs."""
    kw = pd.Series([15.0, 20.0, 0.0, -5.0])
    hours = pd.Series([8.0, 10.0, 8.0, 8.0])
    util = pd.Series([80.0, 100.0, 80.0, 80.0])
    res = compute_theoretical_nameplate_energy_kwh(kw, hours, util)
    assert res.iloc[0] == pytest.approx(15.0 * 8.0 * 0.80)
    assert res.iloc[1] == pytest.approx(20.0 * 10.0 * 1.00)
    assert pd.isna(res.iloc[2])
    assert pd.isna(res.iloc[3])

    rev = compute_theoretical_nameplate_energy_kwh(kw.iloc[::-1], hours.iloc[::-1], util.iloc[::-1])
    pd.testing.assert_series_equal(res, rev.iloc[::-1])


def test_energy_pipeline_rejects_empty_csv_before_storage(tmp_path: Path) -> None:
    """Task 10 / TI-002: Empty CSV must be rejected by bounded file-size check."""
    service = TextileEnergyPipelineService(tmp_path / "store")
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_bytes(b"")

    for process_id in (WEAVING_PROCESS_ID, KNITTING_PROCESS_ID):
        with pytest.raises(ValueError, match="Energy CSV size"):
            service.ingest_and_prepare(
                factory_id=uuid4(),
                process_id=process_id,
                csv_path=empty_csv,
                purpose=DatasetPurpose.TRAINING,
                contract=(
                    build_weaving_energy_dataset_contract(
                        confirmation=build_synthetic_weaving_energy_confirmation()
                    )
                    if process_id == WEAVING_PROCESS_ID
                    else build_knitting_energy_dataset_contract(
                        confirmation=build_synthetic_knitting_energy_confirmation()
                    )
                ),
            )
