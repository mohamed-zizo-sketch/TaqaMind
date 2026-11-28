# Textile Cross-Module Data Requirements Matrix (Task 15)

- **Scope:** Connecting all four TaqaMind modules for `textile` (`M1: Waste V1`, `M2: Energy V2`, `M3: Business Analytics`, `M4: Operations Research`).
- **Status:** Candidate cross-module requirements only. **Not** a finalized database schema, ERD, or migration (final schema design is deferred until all domain and module requirements are consolidated).

---

## 1. Shared Candidate Entities

1. `Factory` (tariff schedule, shift calendar, substation `kW` cap, currency `EGP`)
2. `Machine` (`machine_id`, `loom_type` / `knitting_machine_type`, `machine_rated_power_kw`, OEM `min/max rpm`, width/gauge/feeders)
3. `ProductionProcess` (`weaving`, `knitting`)
4. `ProductionOrder` (`order_id`, `order_lot_id`, customer deadline, target fabric spec, ordered quantity)
5. `Batch` / `ShiftRun` (`batch_record_id` at `beam_batch` grain; `run_id` at `shift_run` grain)
6. `Shift` (shift code, start/end timestamps, TOU tariff bracket, operator crew)
7. `RawMaterial` (`yarn_lot_id`, `yarn_count_ne` / `yarn_count_tex_1`, inventory kg, unit cost `EGP/kg`)
8. `WasteObservation` (measured `waste_yds` / `knitting_waste_kg`, waste category, recoverable flag)
9. `EnergyMeasurement` (sub-metered `total_electrical_energy_kwh`, compressed air `Nm^3`)
10. `CostObservation` (actual vs estimated material, energy, labor, rework, salvage credit)
11. `Prediction` (`model_version_id`, `prediction_dataset_revision_id`, `observation_id`, predicted waste/energy + uncertainty)
12. `OptimizationScenario` (solver run ID, clamped bounds, selected assignments, feasibility check status)

---

## 2. Cross-Module Field Matrix

| Field Name | Definition | Type | Unit | Source System | Grain | Availability | Modules | Origin Kind | Retention & Validation |
|---|---|---|---|---|---|---|---|---|---|
| `factory_id` | Authoritative tenant/plant UUID | `UUID` | — | Platform Config | `Factory` | `STATIC` | M1, M2, M3, M4 | Configured | Permanent; strict tenant isolation |
| `machine_id` | Unique loom / knitting machine ID | `STRING` | — | CMMS / MES | `Machine` | `STATIC` | M1, M2, M3, M4 | Configured | Strip whitespace, preserve leading zeros |
| `machine_rated_power_kw` | Nameplate electrical power | `FLOAT` | `kW` | OEM Spec | `Machine` | `STATIC` | M2, M3, M4 | Configured | `> 0`, finite `Float64` |
| `rpm_min_max` | Approved OEM speed envelope | `FLOAT` | `rpm` | OEM Spec | `Machine` | `STATIC` | M4 | Configured | `0 < min <= max`; hard OR clamp |
| `order_id` / `order_lot_id` | Production planning order key | `STRING` | — | ERP / MES | `ProductionOrder` | `PRE_PRODUCTION` | M1, M2, M3, M4 | Recorded | Lexical key preserved; group-split key |
| `batch_record_id` / `run_id` | Unique beam batch / shift run key | `STRING` | — | MES | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M1, M2, M3 | Recorded | Unique business key per `RecordScope` |
| `scheduled_date` / `production_date` | Planned run date/timestamp | `DATETIME` | UTC ISO | MES / Planner | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M1, M2, M3, M4 | Planned | UTC nanosecond precision; no `NaT` |
| `planned_qty_kg` / `planned_beam_length_yds` | Planned production quantity | `FLOAT` | `kg` / `yd` | ERP / MES | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M1, M2, M3, M4 | Planned | `> 0`; valid denominator for forecast KPIs |
| `planned_operating_duration_hr` | Planned machine run hours | `FLOAT` | `h` | Planner | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M2, M3, M4 | Planned | `> 0`; bounded by shift length |
| `yarn_unit_cost_egp_per_kg` | Yarn purchase/standard cost | `FLOAT` | `EGP/kg` | ERP Finance | `RawMaterial` | `PRE_PRODUCTION` | M3, M4 | Configured | `>= 0`; versioned by effective date |
| `tariff_egp_per_kwh` | Electricity TOU rate bracket | `FLOAT` | `EGP/kWh` | Utility / Finance | `Shift` | `STATIC` | M3, M4 | Configured | `>= 0`; linked to shift time window |
| `waste_qty` (actual) | Confirmed post-run waste | `FLOAT` | `yd` / `kg` | QC / MES | `Batch` / `ShiftRun` | `POST_PRODUCTION` | M1 (target), M3 | Measured | `>= 0`; blocked as pre-prod predictor |
| `total_electrical_energy_kwh` (actual) | Sub-metered electrical energy | `FLOAT` | `kWh` | Sub-meter / SCADA | `Batch` / `ShiftRun` | `POST_PRODUCTION` | M2 (target), M3 | Measured | `>= 0`; blocked as pre-prod predictor |
| `predicted_waste` / `predicted_energy_kwh` | Model inference output | `FLOAT` | `yd`/`kg`/`kWh` | TaqaMind M1/M2 | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M3, M4 | Predicted | Bound to `model_version_id` & `observation_id` |
| `estimated_avoidable_cost_egp` | Forecasted avoidable waste + energy cost | `FLOAT` | `EGP` | TaqaMind M3 | `Batch` / `ShiftRun` | `PRE_PRODUCTION` | M3, M4 | Derived | Tagged *Estimated*, never *Realized* |
