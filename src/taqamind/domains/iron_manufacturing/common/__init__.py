"""Shared domain-local utilities for iron_manufacturing."""

from taqamind.domains.iron_manufacturing.common.hashing import (
    calculate_config_sha256,
    calculate_contract_sha256,
    calculate_implementation_sha256,
)
from taqamind.domains.iron_manufacturing.common.normalization import (
    normalize_header,
    safe_export_csv,
    safe_read_csv,
    sanitize_spreadsheet_cell,
)
from taqamind.domains.iron_manufacturing.common.units import (
    SUPPORTED_UNIT_CONVERSIONS,
    UnitConversionError,
    convert_series_unit,
    is_conversion_supported,
)

__all__ = [
    "SUPPORTED_UNIT_CONVERSIONS",
    "UnitConversionError",
    "calculate_config_sha256",
    "calculate_contract_sha256",
    "calculate_implementation_sha256",
    "convert_series_unit",
    "is_conversion_supported",
    "normalize_header",
    "safe_export_csv",
    "safe_read_csv",
    "sanitize_spreadsheet_cell",
]
