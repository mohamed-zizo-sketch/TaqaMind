"""Shared utilities and verified engineering logic for the textile domain."""

from taqamind.domains.textile.common.formulas import (
    OperatingRegime,
    classify_setup_scrap_regime,
    compute_knitted_fabric_gsm,
    compute_knitting_tightness_factor,
    compute_weaving_thread_density_sum,
)
from taqamind.domains.textile.common.units import (
    VERIFIED_EXCEL_SERIAL_YARN_COUNTS,
    UnsupportedUnitConversionError,
    YarnCountRecoveryResult,
    convert_unit_value,
    get_unit_conversion_factor,
    recover_excel_yarn_count,
)

__all__ = [
    "OperatingRegime",
    "UnsupportedUnitConversionError",
    "VERIFIED_EXCEL_SERIAL_YARN_COUNTS",
    "YarnCountRecoveryResult",
    "classify_setup_scrap_regime",
    "compute_knitted_fabric_gsm",
    "compute_knitting_tightness_factor",
    "compute_weaving_thread_density_sum",
    "convert_unit_value",
    "get_unit_conversion_factor",
    "recover_excel_yarn_count",
]
