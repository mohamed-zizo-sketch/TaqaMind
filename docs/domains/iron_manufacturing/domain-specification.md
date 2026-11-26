# Industrial Domain Specification — `iron_manufacturing`

- **Domain ID:** `iron_manufacturing`
- **Contract Version:** `1.0.0`
- **Target Market:** Egyptian Iron & Steel Manufacturing Facilities (100% EAF steelmaking route — integrated Midrex DRI+EAF meltshops, scrap-based mini-mills, and billet-to-rebar hot rolling mills).
- **Executable ML Scope (Current Phase):** Pre-production waste and scrap quantity regression (`PlatformModule.WASTE_MANAGEMENT`) trained **per factory**.

## 1. Supported Processes (Process-Specific Schemas)

Each manufacturing process has its own dedicated `ProcessContract`, `IndustrialTaskContract`, `DatasetContract`, business keys, grouping columns, target semantics, and deterministic feature engineering — never forced into a single universal table.

| Process ID | Task ID | Row Granularity | Business Key | Grouping Column | Time Column | Target & Unit | Default Confirmation |
|---|---|---|---|---|---|---|---|
| `eaf_meltshop` | `eaf_metallic_waste_loss_prediction` | `one_eaf_heat` | `heat_id` | `production_order_id` | `planned_start_ts` | `metallic_waste_loss_t` (`metric_ton`) | `UNCONFIRMED` |
| `rebar_hot_rolling` | `rebar_rolling_waste_quantity_prediction` | `one_rolling_campaign_lot` | `rolling_lot_id` | `production_order_id` | `planned_start_ts` | `total_rolling_waste_kg` (`kg`) | `UNCONFIRMED` |
| `scrap_yard_reception` | `scrap_delivery_contaminant_prediction` | `one_scrap_delivery_vehicle` | `scrap_delivery_id` | `supplier_group_id` | `entry_ts` | `total_contaminant_kg` (`kg`) | `UNCONFIRMED` |

### Quarantined / Non-Executable Process
- **`ladle_furnace_secondary_metallurgy`:** Documented during forensic audit (Yandex LF dataset) and explicitly **excluded** from executable waste prediction because no valid waste target, tonnage, or pre-production predictor set exists in supplied evidence.

## 2. Process Boundaries & Waste Semantics

### 2.1 EAF Primary Meltshop (`eaf_meltshop`)
- **Boundary:** Starts at `t0` (before scrap basket charging / DRI feeding) and ends at liquid steel tapping into the ladle.
- **Included Target Components:** Iron oxidation to slag (`FeO`), EAF baghouse dust, non-recoverable metallic splash/carryover, and non-metallic scrap contaminants.
- **Excluded Output Categories:** Uncharged scrap returned to yard inventory, credited reusable ladle/tundish skulls, and hot heel retained in the EAF vessel.
- **Critical Rule:** Do not assume raw `input - output == waste` without factory sign-off on hot-heel accounting and skull recovery.

### 2.2 Rebar Hot Rolling Mill (`rebar_hot_rolling`)
- **Boundary:** Starts at `t0` (before billet lot enters the reheating furnace) and ends after flying shear, cooling bed, and quality inspection.
- **Included Target Components:** Crop ends (`crop_ends_kg`), cobble/jam scrap (`cobble_scrap_kg`), furnace/stand mill scale oxidation (`mill_scale_kg`), and dimensional/mechanical quality reject bars (`quality_reject_bars_kg`).
- **Excluded Output Categories:** Unrolled billets returned to billet yard and prime short bars sold as commercial secondary product.

### 2.3 Scrap Yard Reception (`scrap_yard_reception`)
- **Boundary:** Starts at `t0` (entry weighbridge arrival) and ends after unloading inspection.
- **Included Target Components:** Deducted `soil_kg + water_kg + other_kg`.
- **Excluded Output Categories:** Recoverable ferrous scrap mass and full truckloads rejected and returned to the supplier.

## 3. Future Platform Modules (Documented Scope Only — Not Implemented in V1)
- **Energy Efficiency (`ENERGY_EFFICIENCY`):** Requires dedicated furnace/mill kWh meters synced to `heat_id`/`rolling_lot_id`, EgyptERA `0.92` power-factor compliance tracking, and rolling natural gas meters.
- **Business Analytics & Cost Optimization (`BUSINESS_ANALYTICS`):** Requires verified ERP scrap/DRI/billet unit costs and EgyptERA voltage-tier billing schedules.
- **Operations Research (`PRODUCTION_OPTIMIZATION`):** Covers EgyptERA TOU heat scheduling (`OR-1`) and DRI/scrap charge-mix blending under tramp copper/sulfur constraints (`OR-2`). OR outputs are optimization decision variables, never supervised ML targets.
