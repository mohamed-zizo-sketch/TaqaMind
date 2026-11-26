# TaqaMind — Food & Beverage Conceptual Industrial Schema

**Scope:** Logical Entity-Relationship Architecture (No Production Database Implementation)  
**Supported Ingestion:** CSV (V1 Executable), with structural readiness for Excel, ERP, MES, SCADA, and IoT streams.

---

## 1. Core Shared Entities

1. **Factory (`factory_id` PK)**
   - Attributes: `factory_name`, `country` (e.g., `EG`), `governorate`, `subsector` (`bakery`, `dairy`, `beverage`, `fruit_vegetable`), `timezone` (`Africa/Cairo`).
2. **ProductionLine (`line_id` PK, `factory_id` FK)**
   - Attributes: `line_name`, `process_type`, `nominal_throughput`, `throughput_unit`.
3. **Machine (`machine_id` PK, `line_id` FK, `factory_id` FK)**
   - Attributes: `machine_name`, `machine_category` (e.g., `mixer`, `oven`, `pasteurizer`, `filler`), `rated_power_kw`, `capacity_units_hr`.
4. **Product (`product_id` PK, `factory_id` FK)**
   - Attributes: `product_name`, `product_category`, `standard_unit` (`kg`, `L`, `units`), `shelf_life_days`.
5. **Recipe (`recipe_id` PK, `product_id` FK, `version`)**
   - Attributes: `planned_batch_size`, `planned_processing_temp_c`, `planned_processing_duration_min`, `target_ph_level`, `expected_evaporation_pct`.
6. **ProductionOrder (`order_id` PK, `factory_id` FK, `product_id` FK)**
   - Attributes: `planned_start_ts`, `planned_quantity`, `quantity_unit`, `priority`.
7. **ProductionBatch (`batch_id` PK, `order_id` FK, `factory_id` FK, `line_id` FK, `recipe_id` FK)**
   - Attributes: `planned_shift`, `planned_raw_material_kg`, `planned_batch_quantity_kg`, `planned_batch_volume_l`, `actual_start_ts`, `actual_end_ts`.
8. **WasteEvent (`waste_event_id` PK, `batch_id` FK, `factory_id` FK, `machine_id` FK)**
   - Attributes: `waste_category` (`material_waste`, `product_reject`, `packaging_waste`), `waste_quantity`, `waste_unit` (`kg`, `L`, `units`), `is_recoverable_byproduct` (bool), `is_rework` (bool), `recorded_ts`.
9. **EnergyMeasurement (`energy_id` PK, `factory_id` FK, `machine_id` FK, `batch_id` FK nullable)**
   - Attributes: `timestamp_utc`, `thermal_load_kw`, `electric_load_kw`, `boiler_kw`, `water_m3`.
10. **QualityInspection (`qc_id` PK, `batch_id` FK, `factory_id` FK)**
    - Attributes: `inspection_timing` (`pre_production`, `in_process`, `post_production`), `qc_status`, `rejected_unit_count`, `actual_ph`.
11. **ProductionCost (`cost_id` PK, `batch_id` FK, `factory_id` FK)**
    - Attributes: `raw_material_cost`, `energy_cost`, `labor_cost`, `waste_disposal_cost`, `currency` (e.g., `EGP`, `USD`).

---

## 2. Process-Specific Extensions

- **Bakery Extension:** `dough_hydration_pct`, `proofing_humidity_pct`, `oven_zone_count`, `expected_bake_evaporation_pct` (separating moisture loss from scrap).
- **Dairy Extension:** `raw_milk_fat_pct`, `raw_milk_protein_pct`, `cip_cycle_count`, `whey_byproduct_kg`.
- **Beverage Extension:** `carbonation_vol_co2`, `brix_target`, `container_material` (`PET`, `glass`, `can`), `filler_head_count`.

---

## 3. Timestamp & Missing-Value Conventions

- **Timestamps:** Stored in ISO-8601 with explicit timezone offset or UTC (`YYYY-MM-DDTHH:MM:SSZ`).
- **Missing Values:** Never coerce missing waste targets (`NULL`) to `0.0`. Zero waste must be explicitly recorded as `0.0`.
- **Multi-Tenant Isolation:** Every entity carries `factory_id` as a mandatory partition key.
