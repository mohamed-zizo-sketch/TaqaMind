"""Deterministic and domain-independent industrial unit conversions."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pandas as pd


class UnitConversionError(ValueError):
    """Base error for invalid unit conversions."""


class UnknownUnitError(UnitConversionError):
    """Raised when a unit is not supported."""


class IncompatibleUnitError(UnitConversionError):
    """Raised when units belong to different dimensions."""


class InvalidConversionValueError(UnitConversionError):
    """Raised when values cannot be converted safely."""


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    """Definition of a unit relative to its base unit."""

    symbol: str
    dimension: str
    scale: float
    offset: float = 0.0


_UNIT_DEFINITIONS = MappingProxyType({
    # Mass — base: kg
    "g": UnitDefinition("g", "mass", 0.001),
    "kg": UnitDefinition("kg", "mass", 1.0),
    "t": UnitDefinition("t", "mass", 1000.0),

    # Length — base: m
    "mm": UnitDefinition("mm", "length", 0.001),
    "cm": UnitDefinition("cm", "length", 0.01),
    "m": UnitDefinition("m", "length", 1.0),
    "km": UnitDefinition("km", "length", 1000.0),

    # Area — base: m2
    "mm2": UnitDefinition("mm2", "area", 0.000001),
    "cm2": UnitDefinition("cm2", "area", 0.0001),
    "m2": UnitDefinition("m2", "area", 1.0),

    # Volume — base: L
    "mL": UnitDefinition("mL", "volume", 0.001),
    "L": UnitDefinition("L", "volume", 1.0),
    "m3": UnitDefinition("m3", "volume", 1000.0),

    # Energy — base: kWh
    "Wh": UnitDefinition("Wh", "energy", 0.001),
    "kWh": UnitDefinition("kWh", "energy", 1.0),
    "MWh": UnitDefinition("MWh", "energy", 1000.0),

    # Power — base: kW
    "mW": UnitDefinition("mW", "power", 0.000001),
    "W": UnitDefinition("W", "power", 0.001),
    "kW": UnitDefinition("kW", "power", 1.0),
    "MW": UnitDefinition("MW", "power", 1000.0),

    # Time — base: s
    "ms": UnitDefinition("ms", "time", 0.001),
    "s": UnitDefinition("s", "time", 1.0),
    "min": UnitDefinition("min", "time", 60.0),
    "h": UnitDefinition("h", "time", 3600.0),

    # Pressure — base: Pa
    "Pa": UnitDefinition("Pa", "pressure", 1.0),
    "kPa": UnitDefinition("kPa", "pressure", 1000.0),
    "MPa": UnitDefinition("MPa", "pressure", 1000000.0),
    "bar": UnitDefinition("bar", "pressure", 100000.0),

    # Absolute temperature — base: K
    "K": UnitDefinition("K", "temperature", 1.0),
    "degC": UnitDefinition(
        "degC", "temperature", 1.0, 273.15
    ),
    "degF": UnitDefinition(
        "degF",
        "temperature",
        5.0 / 9.0,
        273.15 - 32.0 * (5.0 / 9.0),
    ),
})


_UNIT_ALIASES = MappingProxyType({
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "tonne": "t",
    "metric_ton": "t",
    "metric ton": "t",
    "mm²": "mm2",
    "cm²": "cm2",
    "m²": "m2",
    "m³": "m3",
    "ml": "mL",
    "l": "L",
    "liter": "L",
    "litre": "L",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "°C": "degC",
    "Celsius": "degC",
    "°F": "degF",
    "Fahrenheit": "degF",
    "Kelvin": "K",
})


@dataclass(frozen=True, slots=True)
class UnitConversion:
    """Resolved affine conversion between compatible units."""

    source_unit: str
    canonical_unit: str
    dimension: str
    scale: float
    offset: float

    @property
    def is_identity(self) -> bool:
        """Return whether numeric transformation is unnecessary."""

        return self.scale == 1.0 and self.offset == 0.0


class UnitConverter:
    """Resolve and execute explicitly supported unit conversions."""

    VERSION = "1.1.0"

    @staticmethod
    def normalize_unit(unit: str) -> str:
        """Normalize explicitly registered aliases without case folding."""

        cleaned = unit.strip()

        if not cleaned:
            raise UnknownUnitError("Unit cannot be empty.")

        normalized = _UNIT_ALIASES.get(cleaned, cleaned)

        if normalized not in _UNIT_DEFINITIONS:
            raise UnknownUnitError(
                f"Unsupported or ambiguous unit: '{unit}'."
            )

        return normalized

    def resolve(
        self,
        *,
        source_unit: str,
        canonical_unit: str,
    ) -> UnitConversion:
        """Resolve a compatible unit conversion."""

        source_symbol = self.normalize_unit(source_unit)
        canonical_symbol = self.normalize_unit(canonical_unit)

        source = _UNIT_DEFINITIONS[source_symbol]
        target = _UNIT_DEFINITIONS[canonical_symbol]

        if source.dimension != target.dimension:
            raise IncompatibleUnitError(
                f"Cannot convert '{source_unit}' ({source.dimension}) "
                f"to '{canonical_unit}' ({target.dimension})."
            )

        return UnitConversion(
            source_unit=source_symbol,
            canonical_unit=canonical_symbol,
            dimension=source.dimension,
            scale=source.scale / target.scale,
            offset=(source.offset - target.offset) / target.scale,
        )

    def convert_series(
        self,
        series: pd.Series,
        *,
        source_unit: str,
        canonical_unit: str,
    ) -> pd.Series:
        """Convert numeric values while preserving missing values and index."""

        conversion = self.resolve(
            source_unit=source_unit,
            canonical_unit=canonical_unit,
        )

        missing_mask = series.isna()
        observed = series.loc[~missing_mask]

        boolean_mask = observed.map(
            lambda value: isinstance(value, (bool, np.bool_))
        )

        if bool(boolean_mask.any()):
            raise InvalidConversionValueError(
                "Boolean values cannot be converted as numeric measurements."
            )

        numeric = pd.to_numeric(series, errors="coerce")

        invalid_mask = ~missing_mask & numeric.isna()

        if bool(invalid_mask.any()):
            raise InvalidConversionValueError(
                "Unit conversion received non-numeric values."
            )

        values = numeric.to_numpy(dtype=float, na_value=np.nan)

        if bool(np.isinf(values).any()):
            raise InvalidConversionValueError(
                "Unit conversion received non-finite values."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            converted = (
                values * conversion.scale + conversion.offset
            )

        if bool((~np.isfinite(converted) & ~missing_mask.to_numpy()).any()):
            raise InvalidConversionValueError(
                "Unit conversion produced non-finite values."
            )

        return pd.Series(
            converted,
            index=series.index,
            name=series.name,
        )

    def convert_integer_series(
        self,
        series: pd.Series,
        *,
        source_unit: str,
        canonical_unit: str,
    ) -> pd.Series:
        """Convert to nullable integers without rounding fractional results."""
        conversion = self.resolve(source_unit=source_unit, canonical_unit=canonical_unit)

        def coefficients(symbol: str) -> tuple[Fraction, Fraction]:
            if symbol == "degF":
                return Fraction(5, 9), Fraction("273.15") - 32 * Fraction(5, 9)
            definition = _UNIT_DEFINITIONS[symbol]
            return Fraction(str(definition.scale)), Fraction(str(definition.offset))

        source_scale, source_offset = coefficients(conversion.source_unit)
        target_scale, target_offset = coefficients(conversion.canonical_unit)
        output = []
        for value in series:
            if pd.isna(value):
                output.append(pd.NA)
                continue
            if isinstance(value, (bool, np.bool_)):
                raise InvalidConversionValueError("Boolean measurement is not numeric.")
            try:
                number = Decimal(str(value))
            except (InvalidOperation, ValueError) as exc:
                raise InvalidConversionValueError("Invalid integer measurement.") from exc
            if not number.is_finite():
                raise InvalidConversionValueError("Non-finite integer measurement.")
            # Conservative bounds also prevent huge rational allocations.
            if number.copy_abs() >= 2**53 or abs(number.as_tuple().exponent) > 100:
                raise InvalidConversionValueError(
                    "Integer source exceeds supported precision range."
                )
            result = (
                Fraction(number) * source_scale + source_offset - target_offset
            ) / target_scale
            if result.denominator != 1:
                raise InvalidConversionValueError("Unit conversion produced fractional values.")
            if abs(result) >= 2**53:
                raise InvalidConversionValueError(
                    "Integer result exceeds supported precision range."
                )
            output.append(result.numerator)
        return pd.Series(output, dtype="Int64", index=series.index, name=series.name)
