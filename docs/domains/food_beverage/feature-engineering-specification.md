# TaqaMind — Food & Beverage Feature Engineering Specification

**Domain ID:** `food_beverage`  
**Implementation:** `DomainFeatureEngineer` (`src/taqamind/domains/food_beverage/adapters/validation.py`)

---

## 1. Dimensional Consistency & Anti-Leakage Rules

- Derived features must never depend on post-production targets (`waste_quantity_kg`, `waste_quantity_l`) or post-production KPIs (`actual_yield_pct`, `brewhouse_efficiency_pct`).
- Compatible physical units only; ratios require positive denominators.

---

## 2. Module 1 (Waste) — Pre-Production & Post-Production Derived Features

| Feature | Formula | Inputs | Unit | Timing | Implemented in V1? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `planned_month` | `month(production_timestamp)` | `production_timestamp` | `1..12` | `pre_production` | **Yes** |
| `planned_dayofweek` | `dayofweek(production_timestamp)` | `production_timestamp` | `0..6` | `pre_production` | **Yes** |
| `waste_rate_pct` | `(waste_quantity / planned_quantity) * 100` | Target + planned qty | `%` | `post_production` (KPI / Prediction output rate only) | **Yes** (Output `DerivedRatePrediction` only) |
| `production_yield_pct` | `(actual_accepted_kg / planned_raw_material_kg) * 100` | Actual accepted + raw mat | `%` | `post_production` KPI | Documented (Future) |
| `packaging_loss_rate_pct` | `(packaging_scrap_kg / planned_packaging_kg) * 100` | Packaging scrap + planned | `%` | `post_production` KPI | Documented (Future) |

---

## 3. Modules 2, 3, and 4 — Future Extension Formulas

- **Module 2 (Energy):**
  - `specific_energy_consumption_wh_l = (total_thermal_kwh + total_electric_kwh) * 1000 / processed_volume_l`
  - `idle_energy_ratio = idle_boiler_kwh / total_boiler_kwh`
- **Module 3 (Business):**
  - `material_loss_cost = waste_quantity_kg * unit_material_cost`
  - `unit_production_cost = (material_cost + energy_cost + labor_cost) / actual_accepted_units`
- **Module 4 (Operations Research):**
  - `machine_utilization_pct = (sum(duration_PT) / shift_available_min) * 100`
  - `remaining_capacity_min = shift_available_min - scheduled_load_min`
