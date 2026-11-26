"""Tests for textile common unit conversions and domain engineering formulas."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from taqamind.domains.textile.common.formulas import (
    OperatingRegime,
    classify_setup_scrap_regime,
    compute_knitted_fabric_gsm,
    compute_knitting_tightness_factor,
    compute_weaving_thread_density_sum,
)
from taqamind.domains.textile.common.units import (
    UnsupportedUnitConversionError,
    convert_unit_value,
    get_unit_conversion_factor,
    recover_excel_yarn_count,
)


def test_unit_conversion_factors_and_values() -> None:
    assert math.isclose(get_unit_conversion_factor("m", "yd"), 1.0 / 0.9144)
    assert math.isclose(get_unit_conversion_factor("g", "kg"), 0.001)
    assert math.isclose(get_unit_conversion_factor("in", "cm"), 2.54)
    assert math.isclose(get_unit_conversion_factor("yd", "yd"), 1.0)
    assert math.isclose(convert_unit_value(100.0, "m", "yd"), 100.0 / 0.9144)

    with pytest.raises(UnsupportedUnitConversionError):
        get_unit_conversion_factor("kg", "yd")


def test_excel_yarn_count_recovery() -> None:
    rec = recover_excel_yarn_count(44591)
    assert rec.recovered_value == "30/1"
    assert rec.provenance == "EXCEL_DATE_DEOBFUSCATED"
    assert rec.requires_confirmation

    txt = recover_excel_yarn_count("30/1")
    assert txt.recovered_value == "30/1"
    assert txt.provenance == "ORIGINAL_TEXT"
    assert not txt.requires_confirmation


def test_knitting_and_weaving_formulas() -> None:
    tightness = compute_knitting_tightness_factor(pd.Series([25.0]), pd.Series([2.5]))
    # sqrt(25) / (2.5 / 10) = 5 / 0.25 = 20.0
    assert math.isclose(float(tightness.iloc[0]), 20.0)

    gsm_cm = compute_knitted_fabric_gsm(
        course_density=16.0,
        wale_density=12.0,
        stitch_length_mm=2.8,
        yarn_count_tex=20.0,
        density_unit="cm",
    )
    assert math.isclose(gsm_cm, (16.0 * 12.0 * 2.8 * 20.0) / 100.0)

    gsm_inch = compute_knitted_fabric_gsm(
        course_density=40.0,
        wale_density=30.0,
        stitch_length_mm=2.8,
        yarn_count_tex=20.0,
        density_unit="inch",
    )
    assert math.isclose(gsm_inch, (40.0 * 30.0 * 2.8 * 20.0) / 645.16)

    density = compute_weaving_thread_density_sum(pd.Series([80.0]), pd.Series([60.0]))
    assert float(density.iloc[0]) == 140.0

    assert (
        classify_setup_scrap_regime(50.0, setup_threshold=100.0)
        == OperatingRegime.SETUP_CHANGEOVER
    )
    assert (
        classify_setup_scrap_regime(200.0, setup_threshold=100.0)
        == OperatingRegime.RUNNING_STEADY
    )
