"""Contract-directed scalar conversion with explicit invalid-row tracking."""

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

from taqamind.core.contracts.enums import DataType

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1
_TRUE = {"true", "yes", "1", "1.0"}
_FALSE = {"false", "no", "0", "0.0"}


@dataclass(frozen=True, slots=True)
class TypeConversionResult:
    """Converted values and zero-based positions of invalid observations."""

    values: pd.Series
    invalid_positions: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_positions


class ContractTypeConverter:
    """Convert lexical values without treating invalid tokens as genuine missing data."""

    VERSION = "1.0.0"

    def convert(self, series: pd.Series, *, data_type: DataType) -> TypeConversionResult:
        """Return typed values; callers must block on invalid_positions before use."""
        if data_type not in {
            DataType.STRING, DataType.CATEGORY, DataType.INTEGER, DataType.FLOAT,
            DataType.BOOLEAN,
        }:
            raise ValueError(f"Type conversion not supported yet: {data_type}.")

        values: list = []
        invalid: list[int] = []
        for position, value in enumerate(series):
            if pd.isna(value):
                values.append(pd.NA)
                continue
            try:
                values.append(self._convert_value(value, data_type))
            except (ValueError, InvalidOperation, OverflowError):
                invalid.append(position)
                values.append(pd.NA)

        dtype = {
            DataType.STRING: "string",
            DataType.CATEGORY: "string",
            DataType.INTEGER: "Int64",
            DataType.FLOAT: "Float64",
            DataType.BOOLEAN: "boolean",
        }[data_type]
        converted = pd.Series(values, index=series.index, name=series.name, dtype=dtype)
        return TypeConversionResult(converted, tuple(invalid))

    @staticmethod
    def _convert_value(value: object, data_type: DataType) -> object:
        if data_type in {DataType.STRING, DataType.CATEGORY}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Expected non-blank text.")
            return value

        if data_type == DataType.BOOLEAN:
            token = str(value).strip().casefold()
            if token in _TRUE:
                return True
            if token in _FALSE:
                return False
            raise ValueError("Invalid boolean literal.")

        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Boolean is not a numeric measurement.")
        token = str(value).strip()
        if not _NUMBER.fullmatch(token):
            raise ValueError("Invalid numeric literal.")
        number = Decimal(token)
        if data_type == DataType.INTEGER:
            # Reject float inputs outside their exact-integer range; original digits
            # cannot be recovered after upstream floating-point conversion.
            if isinstance(value, (float, np.floating)):
                precision = np.finfo(type(value)).nmant + 1
                if abs(value) >= 2**precision:
                    raise ValueError("Integer source precision cannot be verified.")
            if not _INT_MIN <= number <= _INT_MAX:
                raise ValueError("Integer outside signed 64-bit range.")
            if number != number.to_integral_value():
                raise ValueError("Fractional integer value.")
            return int(number)

        result = float(number)
        if not math.isfinite(result) or (result == 0.0 and number != 0):
            raise ValueError("Float overflow or underflow.")
        return result
