"""Verified unit conversions and Excel yarn-count de-obfuscation for textile data."""

from __future__ import annotations

from dataclasses import dataclass

# Exact verified SI conversion factors.
YARDS_TO_METERS = 0.9144
METERS_TO_YARDS = 1.0 / YARDS_TO_METERS
GRAMS_TO_KG = 0.001
KG_TO_GRAMS = 1000.0
INCHES_TO_CM = 2.54
NE_TO_TEX_COTTON = 590.5

# Verified Excel Jan-2022 date serial -> Ne cotton count table (knitting raw Excel only).
VERIFIED_EXCEL_SERIAL_YARN_COUNTS: dict[int, str] = {
    44573: "12/1",
    44577: "16/1",
    44581: "20/1",
    44585: "24/1",
    44587: "26/1",
    44588: "27/1",
    44589: "28/1",
    44591: "30/1",
}

_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    ("yd", "yd"): 1.0,
    ("m", "m"): 1.0,
    ("m", "yd"): METERS_TO_YARDS,
    ("yd", "m"): YARDS_TO_METERS,
    ("kg", "kg"): 1.0,
    ("g", "g"): 1.0,
    ("g", "kg"): GRAMS_TO_KG,
    ("kg", "g"): KG_TO_GRAMS,
    ("in", "in"): 1.0,
    ("cm", "cm"): 1.0,
    ("in", "cm"): INCHES_TO_CM,
    ("cm", "in"): 1.0 / INCHES_TO_CM,
    ("CPI", "CPC"): 1.0 / INCHES_TO_CM,
    ("CPC", "CPI"): INCHES_TO_CM,
    ("WPI", "WPC"): 1.0 / INCHES_TO_CM,
    ("WPC", "WPI"): INCHES_TO_CM,
}


class UnsupportedUnitConversionError(ValueError):
    """Raised when no verified dimensional conversion factor exists."""


@dataclass(frozen=True, slots=True)
class YarnCountRecoveryResult:
    """Result of checking a raw knitting yarn count token for Excel date corruption."""

    original_value: str
    recovered_value: str
    provenance: str
    requires_confirmation: bool


def convert_unit_value(value: float, source_unit: str, target_unit: str) -> float:
    """Convert a numeric quantity using verified dimensional conversion factors only."""
    if source_unit == target_unit:
        return float(value)
    factor = _CONVERSION_FACTORS.get((source_unit, target_unit))
    if factor is None:
        raise UnsupportedUnitConversionError(
            f"No verified conversion factor from '{source_unit}' to '{target_unit}'."
        )
    return float(value) * factor


def get_unit_conversion_factor(source_unit: str, target_unit: str) -> float:
    """Return the exact conversion factor or raise UnsupportedUnitConversionError."""
    if source_unit == target_unit:
        return 1.0
    factor = _CONVERSION_FACTORS.get((source_unit, target_unit))
    if factor is None:
        raise UnsupportedUnitConversionError(
            f"No verified conversion factor from '{source_unit}' to '{target_unit}'."
        )
    return factor


def recover_excel_yarn_count(raw_value: int | str) -> YarnCountRecoveryResult:
    """Recover verified Jan-2022 Excel date-serial yarn counts with explicit provenance."""
    original_str = str(raw_value).strip()
    try:
        serial = int(float(original_str))
    except ValueError:
        return YarnCountRecoveryResult(
            original_value=original_str,
            recovered_value=original_str,
            provenance="ORIGINAL_TEXT",
            requires_confirmation=False,
        )

    if serial in VERIFIED_EXCEL_SERIAL_YARN_COUNTS:
        return YarnCountRecoveryResult(
            original_value=original_str,
            recovered_value=VERIFIED_EXCEL_SERIAL_YARN_COUNTS[serial],
            provenance="EXCEL_DATE_DEOBFUSCATED",
            requires_confirmation=True,
        )

    return YarnCountRecoveryResult(
        original_value=original_str,
        recovered_value=original_str,
        provenance="UNRESOLVED_NUMERIC_TOKEN" if serial >= 40000 else "ORIGINAL_NUMERIC",
        requires_confirmation=serial >= 40000,
    )
