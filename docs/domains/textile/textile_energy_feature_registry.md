# Textile Energy V2 — Feature Registry & Leakage Review (Tasks 08–09)

- **Domain:** `textile` (`weaving`, `knitting`)
- **Note on Provenance:** Where Egyptian factory-level empirical distributions are unavailable, ranges are explicitly tagged as *International Literature*, *Manufacturer Spec*, or *Engineering / Synthetic Assumption*.

---

## 1. Weaving Energy Feature Classification (`weaving_energy_batches`)

| Feature | Unit | Category | Evidence Source | Pre-Production Safety |
|---|---|---|---|---|
| `loom_type` (`air_jet`, `rapier`, `water_jet`) | category | Machine Characteristics | Manufacturer / International | **Safe for pre-production** |
| `machine_rated_power_kw` | `kW` | Machine Characteristics | Equipment Nameplate | **Safe for pre-production** (not consumed `kWh`) |
| `planned_beam_length_m` | `m` | Production Settings | Factory Plan | **Safe for pre-production** |
| `planned_operating_duration_hr` | `h` | Production Settings | Factory Schedule | **Safe for pre-production** |
| `fabric_width_cm`, `planned_loom_speed_rpm`, `planned_utilization_pct` | `cm`, `rpm`, `%` | Production Settings | Factory Plan / Machine Spec | **Safe for pre-production** |
| `compressed_air_pressure_bar` | `bar` | Auxiliary Systems | Air-Jet Spec / Literature | **Safe for pre-production** (if set-point); requires factory confirmation if measured post-run |
| `warp_count_ne`, `weft_count_ne`, `warp_ends_per_inch`, `weft_picks_per_inch` | `Ne`, `1/inch` | Fabric Spec | Factory Spec | **Safe for pre-production** |
| `ambient_temp_c`, `relative_humidity_pct` | `C`, `%` | Environmental | HVAC / Mill Sensor | **Requires factory confirmation** (forecasted/set-point vs actual post-run average) |
| `total_thread_density_per_inch`, `theoretical_nameplate_energy_kwh` | `1/inch`, `kWh` | Derived | Deterministic | **Safe for pre-production** |
| `actual_operating_duration_hr`, `actual_produced_length_m`, `actual_machine_efficiency_pct`, `actual_stop_count`, `actual_downtime_hr`, `actual_compressed_air_nm3`, `actual_energy_intensity_kwh_per_m` | various | Post-Production | Excluded (`WEAVING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS`) | **Potentially leaking / Blocked** |

---

## 2. Knitting Energy Feature Classification (`knitting_energy_batches`)

| Feature | Unit | Category | Evidence Source | Pre-Production Safety |
|---|---|---|---|---|
| `knitting_machine_type`, `machine_rated_power_kw`, `machine_diameter_in`, `machine_gauge_e`, `number_of_feeders` | category, `kW`, `inch`, `E`, count | Machine Characteristics | Equipment Nameplate | **Safe for pre-production** |
| `planned_qty_kg`, `planned_operating_duration_hr`, `planned_machine_speed_rpm`, `planned_utilization_pct` | `kg`, `h`, `rpm`, `%` | Production Settings | Factory Schedule | **Safe for pre-production** |
| `yarn_count_tex_1`, `stitch_length_mm`, `target_gsm` | `tex`, `mm`, `g/m2` | Fabric Spec | Factory Spec | **Safe for pre-production** |
| `ambient_temp_c` | `C` | Environmental | Mill Sensor | **Requires factory confirmation** |
| `knitting_tightness_factor`, `theoretical_nameplate_energy_kwh` | `tex^0.5/cm`, `kWh` | Derived | Deterministic | **Safe for pre-production** |
