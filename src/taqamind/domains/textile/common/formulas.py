"""Verified deterministic textile engineering formulas and regime rules."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import pandas as pd


class OperatingRegime(StrEnum):
    """Operational production regime for scrap/waste diagnostics."""

    RUNNING_STEADY = "running_steady"
    SETUP_CHANGEOVER = "setup_changeover"


def compute_knitting_tightness_factor(
    yarn_count_tex: pd.Series,
    stitch_length_mm: pd.Series,
) -> pd.Series:
    """Compute Munden knitting tightness factor K = sqrt(tex) / (l_mm / 10).

    Preconditions: yarn_count_tex > 0 and stitch_length_mm > 0. Invalid/missing
    inputs yield NaN rather than fabricating values.
    """
    tex = pd.to_numeric(yarn_count_tex, errors="coerce")
    stitch = pd.to_numeric(stitch_length_mm, errors="coerce")
    valid = (tex > 0) & (stitch > 0) & tex.notna() & stitch.notna()
    out = pd.Series(np.nan, index=yarn_count_tex.index, dtype="float64")
    if bool(valid.any()):
        out.loc[valid] = np.sqrt(tex.loc[valid]) / (stitch.loc[valid] / 10.0)
    return out


def compute_knitted_fabric_gsm(
    course_density: float,
    wale_density: float,
    stitch_length_mm: float,
    yarn_count_tex: float,
    *,
    density_unit: str = "cm",
    loop_multiplicity: float = 1.0,
) -> float:
    """Compute theoretical knitted GSM with explicit metric vs imperial density guard.

    Supports density_unit='cm' (CPC, WPC) and density_unit='inch' (CPI, WPI).
    """
    if course_density <= 0 or wale_density <= 0 or stitch_length_mm <= 0 or yarn_count_tex <= 0:
        raise ValueError("GSM inputs must be strictly positive.")
    if loop_multiplicity <= 0:
        raise ValueError("Loop multiplicity must be strictly positive.")
    if density_unit == "cm":
        divisor = 100.0
    elif density_unit == "inch":
        divisor = 645.16
    else:
        raise ValueError("density_unit must be 'cm' or 'inch'.")
    return (
        course_density * wale_density * stitch_length_mm * yarn_count_tex * loop_multiplicity
    ) / divisor


def compute_weaving_thread_density_sum(
    warp_ends_per_inch: pd.Series,
    weft_picks_per_inch: pd.Series,
) -> pd.Series:
    """Compute total woven thread density (EPI + PPI) from pre-production inputs."""
    epi = pd.to_numeric(warp_ends_per_inch, errors="coerce")
    ppi = pd.to_numeric(weft_picks_per_inch, errors="coerce")
    valid = (epi > 0) & (ppi > 0) & epi.notna() & ppi.notna()
    out = pd.Series(np.nan, index=warp_ends_per_inch.index, dtype="float64")
    if bool(valid.any()):
        out.loc[valid] = epi.loc[valid] + ppi.loc[valid]
    return out


def classify_setup_scrap_regime(
    production_base: float | None,
    *,
    setup_threshold: float,
) -> OperatingRegime:
    """Tag changeover/setup runs without deleting legitimate scrap events."""
    if setup_threshold < 0:
        raise ValueError("setup_threshold cannot be negative.")
    if production_base is None or math.isnan(production_base) or production_base <= setup_threshold:
        return OperatingRegime.SETUP_CHANGEOVER
    return OperatingRegime.RUNNING_STEADY
