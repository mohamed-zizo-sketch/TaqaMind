# TaqaMind V2 — Phase 4: Dataset Inspection, Schema Audit & EDA Summary

**Machine-Readable Summary:** [v2_eda_reference_summary.json](file:///E:/TaqaMind/research/textile/v2_eda_reference_summary.json)

---

## 1. Reference V1 Schema Inspection (for V2 Feature Reuse & Gap Analysis)

### A. Weaving Reference (`weaving_beam_batches`)
- **Available Pre-Production Predictors Reusable in V2:** `batch_record_id`, `order_id`, `planned_beam_length_yds`, `planned_finish_fabric_yds`, `planned_grey_fabric_yds`, `fabric_allowance_pct`, `shrink_allowance_pct`, `warp_count_ne`, `weft_count_ne`, `warp_ends_per_inch`, `weft_picks_per_inch`, `fabric_construction`, `total_thread_density_per_inch`.
- **Critical Energy Gaps in V1 Schema:** Missing `loom_type` (`air_jet`, `rapier`, `water_jet`), `loom_rated_power_kw`, `planned_loom_speed_rpm`, `fabric_width_cm`, `planned_operating_duration_hr`, `compressed_air_pressure_bar`, `ambient_temp_c`, `relative_humidity_pct`, and target `total_electrical_energy_kwh`.
- **Splitting Implication:** Real V1 weaving logs lacked full calendar dates; V2 Weaving contracts include optional `scheduled_date` so `GROUP_TEMPORAL` is used when dates exist and `GROUP` (`order_id`) when dates are absent.

### B. Knitting Reference (`knitting_shift_runs`)
- **Available Pre-Production Predictors Reusable in V2:** `run_id`, `order_lot_id`, `production_date`, `planned_qty_kg`, `machine_brand`, `machine_diameter_in`, `machine_gauge_e`, `yarn_count_tex_1`, `stitch_length_mm`, `target_gsm`, `number_of_feeders`, `knitting_tightness_factor`.
- **Critical Energy Gaps in V1 Schema:** Missing `knitting_machine_type` (`circular`, `flat`), `machine_rated_power_kw`, `planned_machine_speed_rpm`, `planned_operating_duration_hr`, `fabric_structure`, `ambient_temp_c`, `relative_humidity_pct`, and target `total_electrical_energy_kwh`.
- **Splitting Implication:** `SplitStrategy.GROUP_TEMPORAL` (`order_lot_id`, `production_date`) remains the default for Knitting.

---

## 2. Physics-Calibrated Synthetic Verification Dataset Design (Non-Production)

To verify V2 contracts, leakage gates, feature engineering, baselines, and `CorePlatformServices` end-to-end before real factory meter CSVs arrive:
- **Weaving Energy Verification Frame:**
  - Models `air_jet` (`3.5–6.0 kW` eq. + compressed air penalty), `rapier` (`2.0–4.5 kW`), and `water_jet` (`1.5–3.5 kW`) across reed widths, RPM, EPI/PPI, planned beam length, planned duration, ambient RH/temp, and equipment age.
  - Yields `total_electrical_energy_kwh` consistent with `1.58 – 5.80 kWh/kg` reference benchmarks.
- **Knitting Energy Verification Frame:**
  - Models `circular` (`3.7–7.5 kW`, `0.5–1.5 kWh/kg`) and `flat` (`1.5–3.0 kW`, `1.16–4.59 kWh/kg`) across gauge `E`, cylinder diameter, feeders, RPM, yarn count `tex`, planned GSM, planned qty `kg`, and planned duration.
- **Explicit Disclaimer:** All synthetic energy datasets are strictly for **engineering verification** and **must never** be presented as real-world factory prediction accuracy.
