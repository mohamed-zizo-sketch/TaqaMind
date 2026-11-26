# TaqaMind V2 — Phase 5: Universal Iron Manufacturing Energy Feature Registry

**Design Principle:** Reuse V1 Waste feature semantics, units, and derived specifications wherever pre-production availability (`t <= t0`) matches; standardize electrical rating, planned duration, planned utilization, and physics-informed nameplate envelope across all 3 Iron Manufacturing processes with strict post-production leakage exclusion.

---

## 1. Features Reused from Iron Manufacturing V1

- **EAF Primary Meltshop (`eaf_meltshop`):**
  - `heat_id` (`IDENTIFIER`), `production_order_id` (`METADATA`, group key), `planned_start_ts` (`TIMESTAMP`, `DATETIME`), `furnace_id` (`CATEGORY`, `STATIC`), `steel_grade_code` (`CATEGORY`), `planned_scrap_mass_t` (`FLOAT`, `metric_ton`), `planned_dri_mass_t` (`FLOAT`, `metric_ton`), `planned_tap_mass_t` (`FLOAT`, `metric_ton`), `target_tap_temp_c` (`FLOAT`, `degC`).
  - Reused V1 Derived Specs (`build_eaf_derived_feature_specs()`): `planned_total_metallic_charge_t` (`metric_ton`), `planned_dri_share_pct` (`%`).
- **Rebar Hot Rolling (`rebar_hot_rolling`):**
  - `rolling_lot_id` (`IDENTIFIER`), `production_order_id` (`METADATA`, group key), `planned_start_ts` (`TIMESTAMP`, `DATETIME`), `mill_line_id` (`CATEGORY`, `STATIC`).
  - V1-aligned pre-production quantities: `planned_billet_mass_t` (`FLOAT`, `metric_ton`, aligned to V1 `planned_input_billet_mass_t`), `target_rebar_diameter_mm` (`FLOAT`, `mm`, aligned to V1 `rebar_diameter_mm`).
- **Scrap Yard Reception (`scrap_yard_reception`):**
  - `planned_scrap_mass_t` (`FLOAT`, `metric_ton`), paired with batch-level identifiers (`reception_lot_id`, `delivery_batch_id`, `planned_start_ts`).

---

## 2. Shared V2 Electrical Envelope Features (All 3 Processes)

| Canonical Name | Process | Category | Type | Unit | Timing Class | Missing Policy | Constraints / Leakage Risk & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rated_kw` | All 3 | Equipment | `FLOAT` | `kW` | `PRE_PRODUCTION` (`CONFIGURED`) | `REJECT` | `[1.0, 250000.0]`; nameplate active power rating (`aliases`: `nameplate_kw`, `rated_power_kw`). **Never** treated as `kWh`. |
| `planned_duration_hr` | All 3 | Planning | `FLOAT` | `hr` | `PRE_PRODUCTION` (`RECORDED`) | `REJECT` | `[0.05, 48.0]`; scheduled tap-to-tap / rolling / handling hours (`aliases`: `scheduled_duration_hr`, `planned_hours`). Must not use actual runtime. |
| `planned_utilization_pct` | All 3 | Planning | `FLOAT` | `%` | `PRE_PRODUCTION` (`CONFIGURED`) | `IMPUTE` | `[5.0, 100.0]`; expected power utilization (`aliases`: `utilization_pct`, `planned_load_pct`). Defaults to `100.0%` inside derived formula if absent. |
| `theoretical_nameplate_energy_kwh` | All 3 | Derived (Physics) | `FLOAT` | `kWh` | `PRE_PRODUCTION` (`DERIVED`) | `IMPUTE` | `>= 0.0`; `rated_kw * planned_duration_hr * (planned_utilization_pct / 100)` via `compute_theoretical_nameplate_energy_kwh`. |

---

## 3. Process-Specific V2 Identifiers & Predictors

| Canonical Name | Process | Role | Type | Unit | Timing Class | Missing Policy | Constraints & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rolling_lot_id` | `rebar_hot_rolling` | `IDENTIFIER` | `STRING` | — | `PRE_PRODUCTION` | `REJECT` | Business key for one billet-to-rebar rolling campaign lot. |
| `planned_billet_mass_t` | `rebar_hot_rolling` | `PREDICTOR` | `FLOAT` | `metric_ton` | `PRE_PRODUCTION` | `REJECT` | `[0.1, 5000.0]`; scheduled billet charge mass prior to reheating. |
| `target_rebar_diameter_mm` | `rebar_hot_rolling` | `PREDICTOR` | `FLOAT` | `mm` | `PRE_PRODUCTION` | `REJECT` | `[6.0, 50.0]`; nominal deformed bar diameter (drives stand pass/elongation work). |
| `reception_lot_id` | `scrap_yard_reception` | `IDENTIFIER` | `STRING` | — | `PRE_PRODUCTION` | `REJECT` | Business key for one scrap yard processing batch. |
| `delivery_batch_id` | `scrap_yard_reception` | `METADATA` | `STRING` | — | `PRE_PRODUCTION` | `REJECT` | Grouping column for supplier/delivery campaign safe splitting. |
| `planned_start_ts` | All 3 | `TIMESTAMP` | `DATETIME` | ISO-8601 | `PRE_PRODUCTION` | `REJECT` | Scheduled start timestamp prior to operation (`t0`). |
| `planned_scrap_mass_t` | `eaf_meltshop`, `scrap_yard_reception` | `PREDICTOR` | `FLOAT` | `metric_ton` | `PRE_PRODUCTION` | `REJECT` | `[0.0, 300.0]` (EAF) / `[0.1, 2000.0]` (Scrap Yard). |

---

## 4. Strictly Prohibited Post-Production / Leakage Columns (`IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS`)

Any CSV column matching (case-insensitively) the excluded post-production set is rejected immediately during `IronEnergyPipelineService.ingest_and_prepare()` for both `TRAINING` and `PREDICTION`:

- **Target (`POST_PRODUCTION`):** `total_electrical_energy_kwh` (allowed on `TRAINING` / `EVALUATION`, strictly rejected on `PREDICTION`).
- **Measured Electrical / Power Telemetry:** `active_power_kw`, `reactive_power_kvar`, `actual_peak_demand_kw`, `actual_power_factor`, `in_process_arc_power_mw`.
- **Actual Execution Timings & Temperatures:** `actual_tap_to_tap_min`, `actual_power_on_min`, `actual_power_off_delay_min`, `actual_final_temp_c`.
- **Actual Mass Balances:** `actual_liquid_steel_tapped_t`, `actual_good_rebar_output_t`.
- **Thermal Fuels, Emissions & Post-Hoc Intensities:** `natural_gas_nm3`, `reheat_gas_nm3`, `thermal_energy_gj`, `co2_emissions_ton`, `actual_specific_energy_kwh_per_t`.
