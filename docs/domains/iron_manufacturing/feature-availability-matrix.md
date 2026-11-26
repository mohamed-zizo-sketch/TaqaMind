# Feature Availability Matrix — `iron_manufacturing`

Legend:
- **`DIR`** = Directly present in raw source at the required granularity.
- **`ENG`** = Deterministically derivable from pre-production inputs (`<= t0`).
- **`REC`** = Recorded in Egyptian factory MES/ERP/LIMS systems (requires onboarding export).
- **`GAP`** = Absent in dataset / unavailable.
- **`BLOCKED`** = In-process or post-production variable strictly forbidden as a pre-production predictor.

| Canonical Feature | Process | Availability Stage | ALCHIMIA EU | Yandex LF (Quarantined) | UCI 851 (Energy) | Target Egyptian Pilot |
|---|---|---|---|---|---|---|
| `heat_id` / `planned_start_ts` | `eaf_meltshop` | `PRE_PRODUCTION` | `GAP` | `DIR` (LF grain only) | `GAP` | `REC` (MES) |
| `production_order_id` | `eaf_meltshop` / `rebar` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (ERP) |
| `planned_scrap_mass_t` / `planned_dri_mass_t` | `eaf_meltshop` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (MES/Scales) |
| `dri_metallization_pct` / `dri_gangue_pct` | `eaf_meltshop` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (DRP LIMS) |
| `planned_total_metallic_charge_t` | `eaf_meltshop` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `ENG` |
| `metallic_waste_loss_t` (Target) | `eaf_meltshop` | `POST_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (Post-Tap) |
| `actual_power_on_min` / `in_process_arc_power_mw` | `eaf_meltshop` | `IN_PRODUCTION` | `GAP` | `BLOCKED` | `GAP` | `BLOCKED` |
| `slag_mass_t` / `actual_liquid_steel_tapped_t` | `eaf_meltshop` | `POST_PRODUCTION` | `GAP` | `GAP` | `GAP` | `BLOCKED` |
| `rolling_lot_id` / `rebar_diameter_mm` | `rebar_hot_rolling` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (Rolling MES) |
| `planned_input_billet_mass_t` / `slitting_strands_count` | `rebar_hot_rolling` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (Mill MES) |
| `planned_input_billet_mass_kg` | `rebar_hot_rolling` | `PRE_PRODUCTION` | `GAP` | `GAP` | `GAP` | `ENG` |
| `total_rolling_waste_kg` (Target) | `rebar_hot_rolling` | `POST_PRODUCTION` | `GAP` | `GAP` | `GAP` | `REC` (Scrap Scales) |
| `crop_ends_kg` / `cobble_scrap_kg` / `mill_scale_kg` | `rebar_hot_rolling` | `POST_PRODUCTION` | `GAP` | `GAP` | `GAP` | `BLOCKED` |
| `scrap_delivery_id` / `entry_ts` / `supplier_code` | `scrap_yard_reception` | `PRE_PRODUCTION` | `DIR` | `GAP` | `GAP` | `REC` (Weighbridge) |
| `scrap_grade_code` / `gross_delivery_mass_kg` | `scrap_yard_reception` | `PRE_PRODUCTION` | `DIR` (confirm gross/net) | `GAP` | `GAP` | `REC` (Weighbridge) |
| `total_contaminant_kg` (Target) | `scrap_yard_reception` | `POST_PRODUCTION` | `ENG` (`Soil+Water+Other`) | `GAP` | `GAP` | `REC` (Yard Inspection) |
| `ExitDate` / `Difference` (`stay_duration_min`) | `scrap_yard_reception` | `POST_PRODUCTION` | `BLOCKED` | `GAP` | `GAP` | `BLOCKED` |
