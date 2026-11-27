# TaqaMind V2 — Phase 6: Leakage & Readiness Analysis (Food & Beverage)

**Task Scope:** Pre-Production Electrical Energy Consumption Prediction (`PlatformModule.ENERGY_EFFICIENCY`) for `bakery`, `dairy`, and `beverage`.

---

## 1. Exact Problem Formulation

- **Target:** `total_electrical_energy_kwh` (`kWh`) at the batch / shift-run machine-line electrical boundary.
- **Prediction Timing:** **Pre-production** (at batch/shift scheduling time before equipment run starts), using only machine nameplate ratings, planned operating parameters, planned batch quantities, ambient conditions, and deterministic `theoretical_nameplate_energy_kwh`.
- **Thermal Separation:** Thermal fuel/steam inputs (`MJ` / `GJ` for gas-fired ovens, pasteurization steam boilers, and CIP hot-water heating) are strictly separated from electrical `kWh`.

---

## 2. Prohibited Post-Production & Leakage Controls

1. **Direct Post-Production Outcome Variables (Strictly Excluded):**
   - `total_electrical_energy_kwh` (Target; rejected if present in `DatasetPurpose.PREDICTION` CSVs)
   - `actual_operating_duration_hr` (reveals unplanned stops and total kWh)
   - `actual_produced_quantity_kg` / `actual_produced_volume_l` (actual output reveals yield/stops)
   - `actual_downtime_min` / `actual_stop_count`
   - `actual_thermal_energy_mj` / `actual_steam_kg`
   - `actual_sec_kwh_per_kg` / `actual_sec_kwh_per_l` (computed directly from target)
2. **Splitting & Readiness Controls:**
   - **Temporal / Group Splitting:** `SplitStrategy.GROUP_TEMPORAL` using `production_line_id` as grouping column and `production_timestamp` as time column so batches from the same line/time window do not leak across train/validation/test partitions.
   - **Target Confirmation Gate:** All three energy contracts default to `ConfirmationStatus.UNCONFIRMED`. `CorePlatformServices` (`CoreTrainingReadinessChecker`) blocks splitting and training until an explicit `TargetConfirmation(status=ConfirmationStatus.CONFIRMED, ...)` is supplied.
   - **8-Gate Leakage Validation:** `CoreLeakageValidator` verifies target exclusion, partition disjointness, feature availability (`PRE_PRODUCTION` only), and temporal ordering prior to issuing `TrainingAuthorization`.
