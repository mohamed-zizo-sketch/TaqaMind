# TaqaMind V2 — Phase 2: Energy Measurement & Target Definition Report

**Status:** Approved for V2 design baseline  
**Task Scope:** Pre-Production Electrical Energy Consumption Prediction (`eaf_meltshop`, `rebar_hot_rolling`, `scrap_yard_reception`)

---

## 1. Energy Quantities & Unit Standardization

| Concept | Canonical Quantity | Canonical Unit | Notes |
| :--- | :--- | :--- | :--- |
| **Electrical Power (Instantaneous)** | Active electrical power | `kW` (or `W`, `MW`) | Instantaneous demand rate (`rated_kw` nameplate or forbidden post-prod `active_power_kw`); **never** equated to `kWh`. |
| **Electrical Energy (Consumed)** | Cumulative electrical work | **`kWh`** (or `Wh`, `MWh`) | $\text{Energy (kWh)} = \int P(t)\,dt$; primary V2 regression target (`total_electrical_energy_kwh`). |
| **Specific Energy Consumption (KPI)** | Energy per unit output | **`kWh/t`** (`kWh/metric_ton`) | Post-prediction KPI via `compute_specific_electrical_energy_kwh_per_ton(kwh, mass_ton)` (`common/energy_formulas.py`). |
| **Nameplate Utilization Ratio (KPI)** | Actual vs. theoretical envelope | **`ratio`** (`multiplier=1.0`) | Contract `RateDefinition`: `electrical_energy_to_nameplate_ratio` (`total_electrical_energy_kwh / theoretical_nameplate_energy_kwh`). |
| **Thermal Energy (Out of V2 Scope)** | Natural gas / steam / chemical heat | `MJ` or `GJ` (or `Nm³` gas) | Strictly separated from electrical `kWh` (e.g., EAF oxy-fuel burners, DRI shaft gas, rolling reheating furnace gas). |
| **Reactive Power / Power Factor** | Inductive reactive demand | `kvar` / `ratio` | Post-production telemetry (EgyptERA `0.92` PF compliance); strictly excluded from pre-production predictors. |

> **Core Guardrail:** Equipment nameplate **rated power (`rated_kw`)** is a static pre-production specification, **not** consumed electrical energy (`kWh`).

---

## 2. Measurement Boundaries (Per-Process Sub-Metering)

TaqaMind V2 prohibits mixing whole-plant utility bills, thermal fuels, or cross-process feeders into a single regression target:

1. **EAF Primary Meltshop (`eaf_electrical_energy_regression` — `one_eaf_heat`):**
   - **Included:** `eaf_main_transformer`, `eaf_dedusting_fan_feeder` (active `kWh` integrated over one tap-to-tap heat boundary).
   - **Excluded:** `natural_gas_oxy_fuel_burners`, `oxygen_plant_air_separation`, `office_hvac`, upstream DRI shaft gas, downstream Ladle Furnace (LF) / Continuous Caster (`CC`).
2. **Rebar Hot Rolling Mill (`rebar_electrical_energy_regression` — `one_rolling_campaign_lot`):**
   - **Included:** `mill_stand_main_drives`, `reheating_furnace_electrical_auxiliaries`, `water_cooling_pumps` (descaling/QST/stand cooling).
   - **Excluded:** `reheating_furnace_natural_gas_thermal` (`GJ` / `reheat_gas_nm3`), `meltshop_eaf_feeders`.
3. **Scrap Yard Reception (`scrap_yard_electrical_energy_regression` — `one_scrap_reception_lot`):**
   - **Included:** `magnetic_cranes`, `hydraulic_shears`, `scrap_balers` (sub-metered scrap yard substation feeder).
   - **Excluded:** `diesel_mobile_loaders`, `meltshop_eaf_transformer`.

---

## 3. Approved Target & Derived Rate Contract

- **Primary Target (All 3 Processes):**
  - `total_electrical_energy_kwh` (`kWh`, `DataType.FLOAT`, `minimum >= 0.0`, `FeatureAvailability.POST_PRODUCTION`, `FeatureOrigin.MEASURED`, `TargetKind.QUANTITY`, `TaskType.REGRESSION`).
- **Contractual `RateDefinition` (`energy_contracts.py`):**
  - `name="electrical_energy_to_nameplate_ratio"`
  - `numerator="total_electrical_energy_kwh"`, `denominator="theoretical_nameplate_energy_kwh"`
  - `output_unit="ratio"`, `multiplier=1.0`
- **Secondary Mass-Based KPI (`common/energy_formulas.py`):**
  - `compute_specific_electrical_energy_kwh_per_ton(electrical_energy_kwh, mass_ton)` -> `kWh/t` (evaluated against `planned_tap_mass_t`, `planned_billet_mass_t`, or `planned_scrap_mass_t`).

All targets ship as `ConfirmationStatus.UNCONFIRMED` by default until the partner factory signs off on sub-meter boundaries, `kWh` integration windows, and unit provenance.
