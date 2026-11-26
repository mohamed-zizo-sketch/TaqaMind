"""Verified deterministic physical unit conversions for iron_manufacturing.

Only dimensionally compatible SI / industrial conversions with exact factors
are permitted. Never invent material density values or convert mass <-> volume.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class UnitConversionError(ValueError):
    """Raised when a requested unit conversion is unsupported or incompatible."""


# Canonical unit aliases normalized to a standard key.
_UNIT_ALIASES: Mapping[str, str] = {
    "t": "metric_ton",
    "ton": "metric_ton",
    "tons": "metric_ton",
    "mt": "metric_ton",
    "metric_ton": "metric_ton",
    "metric_tons": "metric_ton",
    "tonne": "metric_ton",
    "tonnes": "metric_ton",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "degc": "degC",
    "c": "degC",
    "celsius": "degC",
    "kwh": "kWh",
    "mwh": "MWh",
    "kw": "kW",
    "mw": "MW",
    "nm3": "Nm3",
    "%": "%",
    "pct": "%",
    "percent": "%",
    "ratio": "ratio",
    "min": "min",
    "minutes": "min",
    "hr": "hr",
    "hours": "hr",
    "sec": "s",
    "s": "s",
    "heats": "heats",
    "strands": "strands",
    "billets": "billets",
    "m/s": "m/s",
    "mps": "m/s",
}

# Verified exact multiplicative factors: target_value = source_value * factor
SUPPORTED_UNIT_CONVERSIONS: Mapping[tuple[str, str], float] = {
    ("kg", "metric_ton"): 0.001,
    ("metric_ton", "kg"): 1000.0,
    ("g", "kg"): 0.001,
    ("kg", "g"): 1000.0,
    ("g", "metric_ton"): 1e-6,
    ("metric_ton", "g"): 1e6,
    ("mm", "m"): 0.001,
    ("m", "mm"): 1000.0,
    ("cm", "m"): 0.01,
    ("m", "cm"): 100.0,
    ("mm", "cm"): 0.1,
    ("cm", "mm"): 10.0,
    ("MWh", "kWh"): 1000.0,
    ("kWh", "MWh"): 0.001,
    ("MW", "kW"): 1000.0,
    ("kW", "MW"): 0.001,
    ("%", "ratio"): 0.01,
    ("ratio", "%"): 100.0,
    ("s", "min"): 1.0 / 60.0,
    ("min", "s"): 60.0,
    ("hr", "min"): 60.0,
    ("min", "hr"): 1.0 / 60.0,
}


def normalize_unit_name(unit: str) -> str:
    """Normalize common unit spellings without guessing incompatible units."""
    cleaned = unit.strip()
    return _UNIT_ALIASES.get(cleaned.casefold(), cleaned)


def is_conversion_supported(source_unit: str, target_unit: str) -> bool:
    """Return True when an exact verified conversion factor exists."""
    src = normalize_unit_name(source_unit)
    dst = normalize_unit_name(target_unit)
    return src == dst or (src, dst) in SUPPORTED_UNIT_CONVERSIONS


def get_conversion_factor(source_unit: str, target_unit: str) -> float:
    """Return the verified multiplicative factor from source_unit to target_unit."""
    src = normalize_unit_name(source_unit)
    dst = normalize_unit_name(target_unit)
    if src == dst:
        return 1.0
    factor = SUPPORTED_UNIT_CONVERSIONS.get((src, dst))
    if factor is None:
        raise UnitConversionError(
            f"Unsupported or dimensionally incompatible unit conversion: "
            f"'{source_unit}' -> '{target_unit}'."
        )
    return factor


def convert_series_unit(
    series: pd.Series,
    source_unit: str,
    target_unit: str,
) -> pd.Series:
    """Convert a numeric Series using an exact verified conversion factor."""
    factor = get_conversion_factor(source_unit, target_unit)
    numeric = pd.to_numeric(series, errors="coerce")
    if factor == 1.0:
        return numeric
    return numeric * factor
