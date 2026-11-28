# Textile Business Analytics & Cost Optimization Requirements (Task 13)

- **Module:** TaqaMind Module 3 — Business Analytics and Cost Optimization
- **Domain Scope:** `textile` (`weaving`, `knitting`)
- **Status:** Requirements research only (no universal BA engine or final DB schema implemented)

---

## 1. Cost Classification Framework

To prevent financial misrepresentation in Egyptian textile factories (`Training-per-Factory` & factory-configured tariffs), all cost items must be tagged across two orthogonal axes:

1. **Behavior Axis:**
   - **Variable Costs:** Yarn/raw material consumed, direct electrical energy (`kWh` × tariff), compressed air (`Nm^3` for air-jet looms), direct piece-rate/overtime, recoverable vs irrecoverable scrap loss.
   - **Fixed / Step-Fixed Costs:** Depreciation, salaried supervision, baseline facility HVAC/lighting, scheduled preventive maintenance contracts (not proportional to batch quantity).
2. **Epistemic / Time Axis:**
   - **Historical Actual Cost:** Computed from confirmed post-production ERP/meter/payroll actuals.
   - **Estimated Cost:** Allocated from standard unit rates when sub-metering or batch-level invoices are unavailable.
   - **Predicted Future Cost:** Derived from Waste V1 (`rejected_fabric_yds`, `knitting_waste_kg`) and Energy V2 (`total_electrical_energy_kwh`) pre-production forecasts.
   - **Avoidable Cost:** Portion of predicted waste/energy loss above factory-confirmed best-practice baseline (never reported as *realized* savings).
   - **Recoverable Waste Value:** Resale/recycling credit for recoverable yarn/selvedge/cotton waste, subtracted only when factory-confirmed.

---

## 2. Textile Cost Categories

| Category | Weaving (`beam_batch`) | Knitting (`shift_run`) | Behavior | Primary Source |
|---|---|---|---|---|
| **Raw Material / Yarn Cost** | Warp + weft yarn mass × yarn price (`EGP/kg`) | Input yarn mass × yarn price (`EGP/kg`) | Variable | ERP / BOM / Yarn Issue |
| **Electricity Cost** | Loom + auxiliary `kWh` × TOU tariff (`EGP/kWh`) | Knitting machine `kWh` × TOU tariff (`EGP/kWh`) | Variable (+ demand charge) | Sub-meter / Energy V2 |
| **Auxiliary Utility Cost** | Compressed air (`Nm^3` for air-jet), sizing steam/water | HVAC / humidification / creel blower | Semi-variable | Utility meter / Allocation |
| **Waste Financial Loss** | `waste_yds × fabric_value_per_yd - salvage_credit` | `waste_kg × yarn_cost_per_kg - salvage_credit` | Variable (Avoidable) | Waste V1 / QC Rejection |
| **Rework & Quality Rejection** | Mending, re-beaming, second-quality markdown | Re-knitting, dyeing shade correction, downgrade | Variable | QC / Inspection |
| **Labor, Maintenance & Downtime** | Weaver/fixer wages, spare parts, lost contribution during stops | Operator wages, needle/sinker wear, downtime loss | Fixed / Step-variable | HR / CMMS / MES |

---

## 3. Core Textile Operational & Financial KPIs

| KPI | Business Objective | Mathematical Formula | Unit | Grain | Actual vs Predicted | Factory Config Required |
|---|---|---|---|---|---|---|
| **Waste Cost** | Quantify financial loss from scrap/rejection | `waste_qty * unit_material_cost - recoverable_qty * salvage_price` | `EGP` | `beam_batch` / `shift_run` | **Both** (Forecast via Waste V1; Actual via QC) | `unit_material_cost`, `salvage_price` |
| **Energy Cost** | Quantify electrical cost per run | `total_electrical_energy_kwh * tariff_egp_per_kwh` | `EGP` | `beam_batch` / `shift_run` | **Both** (Forecast via Energy V2; Actual via meter) | `tariff_egp_per_kwh` (TOU brackets) |
| **Cost per kg** | Unit manufacturing cost per mass | `total_run_cost_egp / produced_mass_kg` | `EGP/kg` | `shift_run` / `beam_batch` | **Both** (uses `planned_qty_kg` when forecast) | Overhead & labor allocation rules |
| **Cost per Meter / Yard** | Unit weaving manufacturing cost | `total_batch_cost_egp / produced_length_m` | `EGP/m` | `beam_batch` | **Both** | Overhead & sizing allocation rules |
| **Specific Energy Consumption (SEC)** | Track energy intensity per mass | `total_electrical_energy_kwh / mass_kg` | `kWh/kg` | `beam_batch` / `shift_run` | **Both** (Forecast uses `planned_qty_kg > 0`) | Mass conversion if weaving length-based |
| **Material Utilization / Yield** | Measure yarn-to-good-fabric efficiency | `(good_output_mass_kg / input_yarn_mass_kg) * 100` | `%` | `beam_batch` / `shift_run` | **Actual** (Forecast approximation via `1 - predicted_waste_rate`) | BOM yarn weight |
| **Machine Utilization & OEE** | Measure asset availability, performance, quality | `Availability * Performance * Quality` | `%` | `shift_run` / `machine` | **Actual** | Rated speed (`rpm`), planned shift hours |
| **Defect / Rejection Rate** | Track quality loss ratio | `(rejected_qty / total_produced_qty) * 100` | `%` | `beam_batch` / `shift_run` | **Both** | Consistent numerator/denominator grain |

---

## 4. Integration with Waste V1 & Energy V2

- **Waste V1 -> Module 3:** Predicted `waste_length_yds` (Weaving) and `knitting_waste_kg` (Knitting) provide pre-production **Predicted Waste Cost** and **Avoidable Loss Alerts** prior to beam mounting or shift start.
- **Energy V2 -> Module 3:** Predicted `total_electrical_energy_kwh` provides pre-production **Predicted Energy Cost** and **Predicted SEC** (`kWh/kg` using `planned_qty_kg > 0`), enabling cost-aware shift/tariff scheduling in Module 4 (Operations Research).
- **Governance Rule:** Predicted cost reductions are always labeled *Estimated Avoidable Cost*, never *Realized Financial Savings*, until confirmed by post-production actuals.
