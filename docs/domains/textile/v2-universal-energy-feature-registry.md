# TaqaMind V2 — Phase 5: Universal Textile Energy Feature Registry

**Design Principle:** Reuse V1 Waste feature definitions wherever semantics, units, and pre-production availability match; introduce new V2 Energy features with strict availability classification (`PRE_PRODUCTION` vs `HISTORICAL` vs `POST_PRODUCTION`) and zero target leakage.

---

## 1. Features Reused Directly from Textile V1

- **Weaving V1 Reused:** `batch_record_id` (`IDENTIFIER`), `order_id` (`IDENTIFIER`/`GROUP`), `planned_beam_length_yds` (`yd`), `planned_finish_fabric_yds` (`yd`), `planned_grey_fabric_yds` (`yd`), `fabric_allowance_pct` (`%`), `shrink_allowance_pct` (`%`), `warp_count_ne` (`Ne`), `weft_count_ne` (`Ne`), `warp_ends_per_inch` (`ends/in`), `weft_picks_per_inch` (`picks/in`), `fabric_construction` (`CATEGORY`), `total_thread_density_per_inch` (`threads/in`, derived).
- **Knitting V1 Reused:** `run_id` (`IDENTIFIER`), `order_lot_id` (`IDENTIFIER`/`GROUP`), `production_date` (`DATE`, `TIMESTAMP`), `planned_qty_kg` (`kg`), `machine_brand` (`CATEGORY`), `machine_diameter_in` (`in`), `machine_gauge_e` (`E`), `yarn_count_tex_1` (`tex`), `stitch_length_mm` (`mm`), `target_gsm` (`g/m2`), `number_of_feeders` (`INTEGER`), `knitting_tightness_factor` (`tex^0.5/cm`, derived).

---

## 2. New V2 Energy Candidate Feature Registry

| Canonical Name | Process | Category | Type | Unit | Timing Class | Missing Policy | Leakage Risk & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `machine_id` | Both | Equipment | `STRING` | — | Static (`PRE_PRODUCTION`) | `REJECT` | Machine identity; enables machine-level error analysis & historical lookups. |
| `loom_type` | Weaving | Weaving Equip. | `CATEGORY` | — | Static (`PRE_PRODUCTION`) | `REJECT` | `air_jet`, `rapier`, `water_jet` (dictates compressed air penalty). |
| `knitting_machine_type` | Knitting | Knitting Equip. | `CATEGORY` | — | Static (`PRE_PRODUCTION`) | `REJECT` | `circular`, `flat` (dictates kW & kWh/kg regime). |
| `machine_rated_power_kw` | Both | Equipment | `FLOAT` | `kW` | Static (`PRE_PRODUCTION`) | `REJECT` | Nameplate motor rating (> 0); **never** treated as consumed `kWh`. |
| `planned_operating_duration_hr` | Both | Planning | `FLOAT` | `h` | Planned (`PRE_PRODUCTION`) | `REJECT` | Scheduled run duration (> 0); **must not** use actual completed duration. |
| `planned_machine_speed_rpm` | Both | Planning | `FLOAT` | `rpm` | Planned (`PRE_PRODUCTION`) | `IMPUTE` | Setpoint RPM before run starts. |
| `fabric_width_cm` | Weaving | Weaving Equip. | `FLOAT` | `cm` | Planned (`PRE_PRODUCTION`) | `IMPUTE` | Reed/working width; drives relay nozzle count on air-jet looms. |
| `yarn_material` | Both | Raw Materials | `CATEGORY` | — | Planned (`PRE_PRODUCTION`) | `IMPUTE` | `cotton`, `polyester`, `poly_cotton`, `viscose`. |
| `fabric_structure` | Knitting | Knitting Equip. | `CATEGORY` | — | Planned (`PRE_PRODUCTION`) | `IMPUTE` | `single_jersey`, `interlock`, `rib`, `fleece`, `flat_panel`. |
| `compressed_air_pressure_bar` | Weaving | Auxiliary | `FLOAT` | `bar` | Planned (`PRE_PRODUCTION`) | `IMPUTE` | Header/nozzle pressure setpoint (typical 4.0–6.5 bar for air-jet; 0 for rapier/water-jet). |
| `planned_utilization_pct` | Both | Planning | `FLOAT` | `%` | Planned (`PRE_PRODUCTION`) | `IMPUTE` | Expected loom/machine efficiency (`0 < u <= 100`), **not** actual run efficiency. |
| `ambient_temp_c` | Both | Environment | `FLOAT` | `C` | Planned/Forecast (`PRE_PRODUCTION`) | `IMPUTE` | Shed setpoint or pre-shift ambient temperature. |
| `relative_humidity_pct` | Both | Environment | `FLOAT` | `%` | Planned/Forecast (`PRE_PRODUCTION`) | `IMPUTE` | Shed RH setpoint (typically 55–80% in weaving/knitting). |
| `equipment_age_years` | Both | Maintenance | `FLOAT` | `yr` | Static (`PRE_PRODUCTION`) | `IMPUTE` | Older motors in Egyptian SMEs correlate with higher losses. |
| `historical_machine_sec_kwh_per_kg` | Both | Maintenance | `FLOAT` | `kWh/kg` | Historical (`PRE_PRODUCTION`) | `IMPUTE` | Computed strictly from prior completed runs (`t < t_run`) — zero current-event leakage. |
| `theoretical_nameplate_energy_kwh` | Both | Derived (Physics) | `FLOAT` | `kWh` | Derived (`PRE_PRODUCTION`) | `IMPUTE` | `machine_rated_power_kw * planned_operating_duration_hr * (planned_utilization_pct / 100)`. |

---

## 3. Strictly Prohibited Post-Production / Leakage Features (`POST_PRODUCTION`)

Any column measuring actual event outcomes **must be excluded** from `predictor_names` and blocked in semantic validation:
- `total_electrical_energy_kwh` (Target)
- `actual_operating_duration_hr`, `actual_produced_qty_kg`, `actual_produced_length_yds`
- `actual_machine_efficiency_pct`, `actual_compressed_air_nm3`, `actual_energy_intensity_kwh_per_kg`
- `actual_stop_count`, `actual_downtime_hr`
