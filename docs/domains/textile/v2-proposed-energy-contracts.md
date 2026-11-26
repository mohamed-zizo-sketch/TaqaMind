# TaqaMind V2 — Phase 7: Proposed Energy Contracts & Shared Core Integration

**Status:** Proposed for Architecture Approval before Phase 8 implementation  
**Domain:** `textile` (`register_textile`)  
**Module:** `PlatformModule.ENERGY_EFFICIENCY` (completely isolated from `PlatformModule.WASTE_MANAGEMENT` V1 tasks)

---

## 1. Task Separation & Identities

| Process | V1 Waste Task (Preserved Untouched) | V2 Energy Task (New) | Dataset ID (V2) | Granularity |
| :--- | :--- | :--- | :--- | :--- |
| `weaving` | `weaving_waste_regression` | `weaving_energy_regression` | `weaving_energy_batches` | `beam_batch` |
| `knitting` | `knitting_waste_regression` | `knitting_energy_regression` | `knitting_energy_runs` | `shift_run` |

---

## 2. Proposed Weaving Energy Contract (`weaving_energy_regression`)

- **Business Keys:** `("batch_record_id",)` | **Grouping:** `("order_id",)` | **Time Column:** `"scheduled_date"`
- **Pre-Production Predictors (`FeatureAvailability.PRE_PRODUCTION`):**
  - Identifiers / Timing: `batch_record_id` (`STRING`), `order_id` (`STRING`), `scheduled_date` (`DATE`), `machine_id` (`STRING`).
  - Equipment & Auxiliary: `loom_type` (`CATEGORY`: `air_jet`, `rapier`, `water_jet`), `machine_rated_power_kw` (`FLOAT`, `kW`), `fabric_width_cm` (`FLOAT`, `cm`), `compressed_air_pressure_bar` (`FLOAT`, `bar`, optional/impute), `equipment_age_years` (`FLOAT`, `yr`, optional/impute).
  - Production & Fabric Spec (reusing V1 semantics): `planned_beam_length_m` (`FLOAT`, `m`), `planned_operating_duration_hr` (`FLOAT`, `h`), `planned_loom_speed_rpm` (`FLOAT`, `rpm`, optional/impute), `planned_utilization_pct` (`FLOAT`, `%`, optional/impute), `warp_count_ne` (`FLOAT`, `Ne`), `weft_count_ne` (`FLOAT`, `Ne`), `warp_ends_per_inch` (`FLOAT`, `ends/in`), `weft_picks_per_inch` (`FLOAT`, `picks/in`), `yarn_material` (`CATEGORY`, optional/impute).
  - Environment: `ambient_temp_c` (`FLOAT`, `degC`, optional/impute), `relative_humidity_pct` (`FLOAT`, `%`, optional/impute).
  - Derived Pre-Production (`Origin.DERIVED`, deterministic):
    - `total_thread_density_per_inch` = `warp_ends_per_inch + weft_picks_per_inch` (reused from V1 `common/formulas.py`)
    - `theoretical_nameplate_energy_kwh` = `machine_rated_power_kw * planned_operating_duration_hr * (planned_utilization_pct / 100.0)`
- **Target (`FeatureAvailability.POST_PRODUCTION`):** `total_electrical_energy_kwh` (`kWh`, `minimum >= 0.0`, `TargetConfirmation.UNCONFIRMED` by default).
- **Derived KPI (`RateDefinition`):** `weaving_energy_intensity_kwh_per_m` (`numerator="total_electrical_energy_kwh"`, `denominator="planned_beam_length_m"`, `output_unit="kWh/m"`, `multiplier=1.0`).

---

## 3. Proposed Knitting Energy Contract (`knitting_energy_regression`)

- **Business Keys:** `("run_id",)` | **Grouping:** `("order_lot_id",)` | **Time Column:** `"production_date"`
- **Pre-Production Predictors (`FeatureAvailability.PRE_PRODUCTION`):**
  - Identifiers / Timing: `run_id` (`STRING`), `order_lot_id` (`STRING`), `production_date` (`DATE`), `machine_id` (`STRING`).
  - Equipment (reusing V1 specs + V2 energy fields): `knitting_machine_type` (`CATEGORY`: `circular`, `flat`), `machine_rated_power_kw` (`FLOAT`, `kW`), `machine_diameter_in` (`FLOAT`, `in`), `machine_gauge_e` (`FLOAT`, `E`), `number_of_feeders` (`INTEGER`), `equipment_age_years` (`FLOAT`, `yr`, optional/impute).
  - Production & Yarn Spec: `planned_qty_kg` (`FLOAT`, `kg`), `planned_operating_duration_hr` (`FLOAT`, `h`), `planned_machine_speed_rpm` (`FLOAT`, `rpm`, optional/impute), `planned_utilization_pct` (`FLOAT`, `%`, optional/impute), `yarn_count_tex_1` (`FLOAT`, `tex`), `stitch_length_mm` (`FLOAT`, `mm`), `target_gsm` (`FLOAT`, `g/m2`, optional/impute), `fabric_structure` (`CATEGORY`, optional/impute), `yarn_material` (`CATEGORY`, optional/impute).
  - Environment: `ambient_temp_c` (`FLOAT`, `degC`, optional/impute), `relative_humidity_pct` (`FLOAT`, `%`, optional/impute).
  - Derived Pre-Production (`Origin.DERIVED`, deterministic):
    - `knitting_tightness_factor` = `sqrt(yarn_count_tex_1) / (stitch_length_mm / 10.0)` (reused from V1 `common/formulas.py`)
    - `theoretical_nameplate_energy_kwh` = `machine_rated_power_kw * planned_operating_duration_hr * (planned_utilization_pct / 100.0)`
- **Target (`FeatureAvailability.POST_PRODUCTION`):** `total_electrical_energy_kwh` (`kWh`, `minimum >= 0.0`, `TargetConfirmation.UNCONFIRMED` by default).
- **Derived KPI (`RateDefinition`):** `knitting_energy_intensity_kwh_per_kg` (`numerator="total_electrical_energy_kwh"`, `denominator="planned_qty_kg"`, `output_unit="kWh/kg"`, `multiplier=1.0`).

---

## 4. Shared Core Integration Requirements Assessment

- **Result:** **Zero Shared Core changes required!**
- `CorePlatformServices` (`src/taqamind/core/pipeline.py`), `PlatformModule.ENERGY_EFFICIENCY`, and `UnitConverter` (`Wh`, `kWh`, `MWh`, `W`, `kW`, `MW`, `h`, `bar`, `degC`, `m`, `cm`, `kg`) already support every V2 Energy contract, feature engineering, readiness, splitting, leakage, AutoML, protected evaluation, model registry, and zero-retraining prediction requirement.
- V2 Energy will wire directly to `CorePlatformServices.create_local(...)` with **zero** duplicated adapters in `src/taqamind/domains/textile/adapters/`.
