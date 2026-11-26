"""Deterministic physics-informed formulas for Iron Manufacturing V2 Energy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_theoretical_nameplate_energy_kwh(
    rated_kw: pd.Series,
    planned_duration_hr: pd.Series,
    planned_utilization_pct: pd.Series | None = None,
) -> pd.Series:
    """Compute pre-production theoretical nameplate electrical energy envelope (kWh).

    Formula:
        theoretical_nameplate_energy_kwh = (
            rated_kw * planned_duration_hr * (planned_utilization_pct / 100)
        )
    """
    kw = pd.to_numeric(rated_kw, errors="coerce")
    duration = pd.to_numeric(planned_duration_hr, errors="coerce")
    if planned_utilization_pct is None:
        util_factor = 1.0
    else:
        util = pd.to_numeric(planned_utilization_pct, errors="coerce").fillna(100.0)
        util_factor = util / 100.0
    return kw * duration * util_factor


def compute_specific_electrical_energy_kwh_per_ton(
    electrical_energy_kwh: pd.Series,
    mass_ton: pd.Series,
) -> pd.Series:
    """Compute secondary KPI: specific electrical energy consumption (kWh/t)."""
    energy = pd.to_numeric(electrical_energy_kwh, errors="coerce")
    mass = pd.to_numeric(mass_ton, errors="coerce")
    safe_mass = mass.where(mass > 0.0, np.nan)
    return energy / safe_mass
