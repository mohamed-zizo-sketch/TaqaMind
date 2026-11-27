# TaqaMind V2 — Phase 7: Proposed Energy Contracts (Food & Beverage)

**Module:** `PlatformModule.ENERGY_EFFICIENCY`  
**Processes:** `bakery`, `dairy`, `beverage`  
**Task Type:** `TaskType.REGRESSION`  
**Default Target Confirmation:** `ConfirmationStatus.UNCONFIRMED`

---

## 1. Task & Dataset Identifiers

| Process | Task ID | Dataset ID | Granularity |
|---------|---------|------------|-------------|
| `bakery` | `bakery_energy_regression` | `bakery_energy_batches` | `batch` |
| `dairy` | `dairy_energy_regression` | `dairy_energy_batches` | `batch` |
| `beverage` | `beverage_energy_regression` | `beverage_energy_batches` | `batch` |

---

## 2. Shared Core Invariants & Contract Structure

- **Target:** `total_electrical_energy_kwh` (`DataType.FLOAT`, `unit="kWh"`, `FeatureRole.TARGET`, `FeatureAvailability.POST_PRODUCTION`, `FeatureConstraints(minimum=0.0)`, `MissingValuePolicy.REJECT`).
- **Derived Feature:** `theoretical_nameplate_energy_kwh = machine_rated_power_kw * planned_operating_duration_hr * (planned_utilization_pct / 100.0)` (`FeatureOrigin.DERIVED`, `FeatureAvailability.PRE_PRODUCTION`), computed by `DeterministicFeatureEngineer(storage, derived_specs=...)`.
- **Derived Rate (`RateDefinition`):**
  - `name`: `<process>_energy_to_nameplate_ratio`
  - `numerator`: `total_electrical_energy_kwh`
  - `denominator`: `theoretical_nameplate_energy_kwh`
  - `output_unit`: `"ratio"`
  - `multiplier`: `1.0`
- **Business Keys & Grouping:**
  - `business_keys=("batch_id",)`
  - `grouping_columns=("production_line_id",)`
  - `time_column="production_timestamp"`

---

## 3. Process-Specific Pre-Production Predictors

- **Bakery (`bakery_energy_regression`):** `production_timestamp`, `production_line_id`, `machine_id`, `product_category`, `oven_heating_type` (`electric` vs `gas_fired`), `planned_batch_quantity_kg`, `planned_raw_material_kg`, `planned_processing_temp_c`, `planned_processing_duration_min`, `machine_rated_power_kw`, `planned_operating_duration_hr`, `planned_utilization_pct`, `ambient_temp_c`, `theoretical_nameplate_energy_kwh`.
- **Dairy (`dairy_energy_regression`):** `production_timestamp`, `production_line_id`, `machine_id`, `product_category`, `chiller_stage_type` (`pasteurization_chiller`, `uht_line`, `cold_storage_holding`), `planned_batch_quantity_kg`, `planned_pasteurization_temp_c`, `planned_processing_duration_min`, `machine_rated_power_kw`, `planned_operating_duration_hr`, `planned_utilization_pct`, `ambient_temp_c`, `theoretical_nameplate_energy_kwh`.
- **Beverage (`beverage_energy_regression`):** `production_timestamp`, `production_line_id`, `machine_id`, `product_category`, `packaging_line_type` (`pet_bottling`, `canning`, `glass_bottling`), `planned_batch_quantity_kg`, `planned_line_speed_bph`, `planned_processing_duration_min`, `machine_rated_power_kw`, `planned_operating_duration_hr`, `planned_utilization_pct`, `ambient_temp_c`, `theoretical_nameplate_energy_kwh`.
