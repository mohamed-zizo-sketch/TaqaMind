# TaqaMind V2 — Phase 4: Dataset Inspection, Schema Audit & EDA Summary

**Machine-Readable Summary:** [v2_eda_reference_summary.json](../../../research/iron_manufacturing/v2_eda_reference_summary.json)  
**Prior Raw Audit Script:** [audit_raw_datasets.py](../../../research/iron_manufacturing/audit_raw_datasets.py)

---

## 1. Supplied Public & V1 Reference Inspection

### A. UCI Steel Industry Energy Consumption (`Steel_industry_data.csv`, ID 851)
- **Dimensions & Integrity:** `35,040` rows × `11` columns (`9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`), `0` missing values, `0` exact duplicates, spanning 15-minute intervals across 2018.
- **Forensic Findings:** `59.9%` exact zero `CO2(tCO2)` readings; **zero** production tonnage, heat/lot IDs, furnace/mill ratings, or pre-production recipe parameters. Confirms why 15-minute plant telemetry cannot support V2 pre-production heat/lot regression.

### B. V1 Iron Manufacturing Schema Reuse & V2 Energy Gap Analysis
1. **EAF Primary Meltshop (`eaf_meltshop`):**
   - **Reusable V1 Pre-Production Fields:** `heat_id`, `production_order_id`, `planned_start_ts`, `furnace_id`, `steel_grade_code`, `planned_scrap_mass_t`, `planned_dri_mass_t`, `planned_tap_mass_t`, `target_tap_temp_c`, plus deterministic derived features `planned_total_metallic_charge_t` and `planned_dri_share_pct`.
   - **Required V2 Energy Additions:** `rated_kw`, `planned_duration_hr`, `planned_utilization_pct`, `theoretical_nameplate_energy_kwh`, and target `total_electrical_energy_kwh` (`kWh`).
2. **Rebar Hot Rolling Mill (`rebar_hot_rolling`):**
   - **Reusable V1 Pre-Production Concepts:** `rolling_lot_id`, `production_order_id`, `planned_start_ts`, `mill_line_id`, billet mass (`planned_billet_mass_t`), and diameter (`target_rebar_diameter_mm`).
   - **Required V2 Energy Additions:** `rated_kw`, `planned_duration_hr`, `planned_utilization_pct`, `theoretical_nameplate_energy_kwh`, and target `total_electrical_energy_kwh` (`kWh`).
3. **Scrap Yard Reception (`scrap_yard_reception`):**
   - **Reference V1 Context (`ALCHIMIA` `scrap_A.csv` / `scrap_C.csv`):** `16,369` & `9,399` rows proved `supplier_group_id` dominance (top supplier `45.53%` in Plant A), requiring group-temporal splitting (`delivery_batch_id` + `planned_start_ts` in V2).
   - **Required V2 Energy Additions:** `reception_lot_id`, `delivery_batch_id`, `planned_start_ts`, `planned_scrap_mass_t`, `rated_kw`, `planned_duration_hr`, `planned_utilization_pct`, `theoretical_nameplate_energy_kwh`, and target `total_electrical_energy_kwh` (`kWh`).

---

## 2. Physics-Calibrated Synthetic Verification Dataset Design (Non-Production)

To exercise all `CorePlatformServices` stages end-to-end (`common/energy_synthetic.py`):
- **EAF Meltshop (`generate_synthetic_eaf_energy_dataset`):** `rated_kw` in `{65,000, 75,000}` kW, `planned_duration_hr` in `[0.75, 1.15]` hr, `planned_utilization_pct` in `[76, 92]%`, `planned_scrap_mass_t` in `[45, 75]` t, `planned_dri_mass_t` in `[35, 65]` t, `target_tap_temp_c` in `[1610, 1665]` °C; `total_electrical_energy_kwh` calibrated around nameplate envelope + DRI/scrap melting + superheat terms.
- **Rebar Hot Rolling (`generate_synthetic_rebar_energy_dataset`):** `rated_kw = 18,000` kW, `planned_duration_hr` in `[1.5, 3.2]` hr, `planned_billet_mass_t` in `[120, 260]` t, `target_rebar_diameter_mm` in `{10, 12, 16, 25}` mm (higher `kWh/t` for smaller diameters).
- **Scrap Yard Reception (`generate_synthetic_scrap_yard_energy_dataset`):** `rated_kw` in `{1,200, 1,800}` kW, `planned_duration_hr` in `[2.0, 6.0]` hr, `planned_scrap_mass_t` in `[80, 450]` t.
- **Explicit Non-Production Disclaimer:** These fixtures verify software contracts, leakage checks, splitting, AutoML, and zero-retraining inference **only**; they are **not** real Egyptian steel factory measurements.
