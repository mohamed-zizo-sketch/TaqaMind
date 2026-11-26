"""Tests for deterministic industrial unit conversion."""

import numpy as np
import pandas as pd
import pytest

from taqamind.core.mapping.unit_conversion import (
    IncompatibleUnitError,
    InvalidConversionValueError,
    UnitConverter,
    UnknownUnitError,
)


@pytest.mark.parametrize(
    ("source", "target", "value", "expected"),
    [
        ("g", "kg", 5000.0, 5.0),
        ("kg", "g", 5.0, 5000.0),
        ("tonne", "kg", 2.0, 2000.0),
        ("mm", "m", 1500.0, 1.5),
        ("cm", "m", 250.0, 2.5),
        ("cm2", "m2", 10000.0, 1.0),
        ("mL", "L", 500.0, 0.5),
        ("m³", "L", 2.0, 2000.0),
        ("Wh", "kWh", 3500.0, 3.5),
        ("MWh", "kWh", 2.0, 2000.0),
        ("W", "kW", 1500.0, 1.5),
        ("h", "s", 2.0, 7200.0),
        ("bar", "kPa", 2.0, 200.0),
        ("°C", "K", 0.0, 273.15),
        ("°F", "°C", 32.0, 0.0),
        ("°C", "°F", 100.0, 212.0),
        ("K", "°C", 273.15, 0.0),
        ("kg", "kg", 15.0, 15.0),
        ("mW", "MW", 1.0, 1e-9),
    ],
)
def test_supported_conversions(
    source: str,
    target: str,
    value: float,
    expected: float,
):
    converter = UnitConverter()

    result = converter.convert_series(
        pd.Series([value]),
        source_unit=source,
        canonical_unit=target,
    )

    assert result.iloc[0] == pytest.approx(
        expected,
        rel=1e-9,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("kg", "L"),
        ("kW", "kWh"),
        ("m", "m2"),
        ("kg", "bar"),
        ("s", "kg"),
    ],
)
def test_incompatible_dimensions_are_rejected(
    source: str,
    target: str,
):
    converter = UnitConverter()

    with pytest.raises(IncompatibleUnitError):
        converter.resolve(
            source_unit=source,
            canonical_unit=target,
        )


@pytest.mark.parametrize(
    "unit",
    [
        "ton",
        "KG",
        "unknown",
        "pieces",
        "%",
        "",
    ],
)
def test_unknown_or_ambiguous_units_are_rejected(unit: str):
    converter = UnitConverter()

    with pytest.raises(UnknownUnitError):
        converter.resolve(
            source_unit=unit,
            canonical_unit="kg",
        )


def test_missing_values_zero_and_index_are_preserved():
    converter = UnitConverter()

    original = pd.Series(
        [0.0, 5000.0, np.nan],
        index=[10, 20, 30],
        name="waste_qty",
    )

    result = converter.convert_series(
        original,
        source_unit="g",
        canonical_unit="kg",
    )

    assert result.index.equals(original.index)
    assert result.name == original.name
    assert result.loc[10] == 0.0
    assert result.loc[20] == 5.0
    assert pd.isna(result.loc[30])

    # The original data must remain unchanged.
    assert original.loc[20] == 5000.0


@pytest.mark.parametrize(
    "values",
    [
        [100.0, "invalid"],
        [True, False],
        [100.0, np.inf],
        [100.0, -np.inf],
        [100.0, ""],
    ],
)
def test_invalid_measurements_are_rejected(values: list):
    converter = UnitConverter()

    with pytest.raises(InvalidConversionValueError):
        converter.convert_series(
            pd.Series(values),
            source_unit="g",
            canonical_unit="kg",
        )


def test_conversion_overflow_is_rejected():
    converter = UnitConverter()

    with pytest.raises(InvalidConversionValueError):
        converter.convert_series(
            pd.Series([1e308]),
            source_unit="kg",
            canonical_unit="g",
        )


def test_identity_conversion():
    converter = UnitConverter()

    conversion = converter.resolve(
        source_unit="kilogram",
        canonical_unit="kg",
    )

    assert conversion.is_identity
    assert conversion.dimension == "mass"
    assert conversion.scale == 1.0
    assert conversion.offset == 0.0


@pytest.mark.parametrize(
    ("source", "target", "value", "expected"),
    [("g", "kg", "2000", 2), ("degF", "degC", "32", 0),
     ("kg", "g", "0.001", 1), ("degC", "degF", "100", 212)],
)
def test_exact_integer_conversion(source, target, value, expected):
    series = pd.Series([value, None], index=[4, 8], name="quantity")
    result = UnitConverter().convert_integer_series(
        series, source_unit=source, canonical_unit=target
    )
    assert result.loc[4] == expected
    assert pd.isna(result.loc[8])
    assert str(result.dtype) == "Int64"
    assert result.index.equals(series.index)
    assert result.name == series.name


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [("g", "kg", "1500"), ("g", "kg", "9007199254740993"),
     ("kg", "g", "9007199254741"), ("g", "kg", "2000.00000000000001")],
)
def test_integer_conversion_rejects_unsafe_values(source, target, value):
    with pytest.raises(InvalidConversionValueError):
        UnitConverter().convert_integer_series(
            pd.Series([value]), source_unit=source, canonical_unit=target
        )
