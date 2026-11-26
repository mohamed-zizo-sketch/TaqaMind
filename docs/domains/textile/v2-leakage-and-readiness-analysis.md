# TaqaMind V2 — Phase 6: Problem Formulation & Leakage Prevention Analysis

**Task Scope:** Pre-Production Electrical Energy Consumption Prediction (`PlatformModule.ENERGY_EFFICIENCY`) for `weaving` and `knitting`.

---

## 1. Exact Problem Formulation

- **What is being predicted:** Total electrical energy consumption (`total_electrical_energy_kwh`, in `kWh`) for a single planned production event (`weaving_energy_regression` at `beam_batch` / `machine_run` granularity; `knitting_energy_regression` at `shift_run` / `batch` granularity).
- **When prediction occurs:** **Before production starts** (at production scheduling / order release time), using only static machine specifications, planned order/loom/knitting parameters, planned operating duration, and strictly historical (`t < t_run`) machine energy baselines.
- **Measurement Boundary:** Machine-run electrical boundary (main drive + machine auxiliary electronics + allocated compressed air for air-jet looms when sub-metered). Excludes whole-factory HVAC/lighting/thermal steam unless contractually sub-metered.

---

## 2. Prohibited Future Information & Indirect/Proxy Leakage

1. **Direct Post-Production Outcome Variables (Strictly Excluded):**
   - `total_electrical_energy_kwh` (Target)
   - `actual_operating_duration_hr` (actual run hours reveal downtime & total kWh)
   - `actual_produced_qty_kg` / `actual_produced_length_m` (actual output reveals stops/efficiency)
   - `actual_machine_efficiency_pct` / `actual_stop_count` / `actual_downtime_hr`
   - `actual_compressed_air_nm3` (measured compressed air volume during run)
   - `actual_energy_intensity_kwh_per_kg` / `actual_energy_intensity_kwh_per_m` (computed directly from target)
2. **Indirect & Proxy Leakage Controls:**
   - **Historical Machine SEC (`historical_machine_sec_kwh_per_kg`):** Must be computed strictly from prior completed runs (`t < t_run`) — never using rolling windows that include the current or future shifts, and never leaking test-partition targets into training/validation.
   - **Temporal & Group Splitting:**
     - **Weaving Energy:** `SplitStrategy.GROUP_TEMPORAL` when `scheduled_date` is present (grouping by `order_id`), falling back to `SplitStrategy.GROUP` (`order_id`) if calendar dates are absent.
     - **Knitting Energy:** `SplitStrategy.GROUP_TEMPORAL` (`order_lot_id`, `production_date`).
   - **Shared Core Enforcement:** `CoreLeakageValidator` enforces all 8 leakage gates and `CorePlatformServices` enforces `TargetConfirmation.CONFIRMED` before splitting or training.
