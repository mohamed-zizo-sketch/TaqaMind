# TaqaMind — Food & Beverage Feature Availability Matrix

Comparison of scientifically required features against actually observed columns in the audited datasets.

| Canonical Feature | Module | `brewery_data.csv` (M1 Benchmark) | `dairy_energy` (M2) | `ITAC_NAICS_311_312` (M2/M3) | `BK15/BK50` (M4) | Recovery / Collection Action Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `factory_id` | All | Proxy (`Location`) | Single plant (implicit) | `ID` (plant audit) | Single bakery | Require explicit `factory_id` in customer CSV |
| `batch_id` | M1, M4 | `Batch_ID` | Absent (time-series) | Absent (annual) | `batches` (group count) | Require ERP `batch_id` |
| `production_timestamp` | M1, M2 | `Brew_Date` (date only) | `timestamp` (1-min/1-hr) | `FY` (fiscal year) | `shift_start` (min) | Require ISO-8601 timestamp |
| `planned_batch_quantity_kg` | M1, M3, M4 | **Absent** (only `Volume_Produced` post-prod) | Absent (`Processed milk [m^3]` monthly) | `PRODLEVEL` (annual) | `batches` count | **Critical:** Collect pre-prod planned mass/volume |
| `planned_processing_temp_c` | M1, M2 | `Temperature` (unit/timing unconfirmed) | Absent | Absent | Absent | Collect recipe setpoint temperature (`degC`) |
| `planned_processing_duration_min` | M1, M4 | `Fermentation_Time` (days/hours unconfirmed) | Absent | `PRODHOURS` (annual) | `duration_PT` (min) | Collect stage/batch planned duration |
| `product_category` | All | `Beer_Style`, `SKU` | `yoghurt`, `melk` (aggregate) | `PRODUCTS`, `NAICS` | `pr_group`, `product_name` | Standardize SKU & product family |
| `waste_quantity_kg` / `waste_quantity_l` | M1 | `Loss_During_*` (**unit unconfirmed**) | **Absent** | **Absent** | **Absent** | **Critical:** Collect unit-verified avoidable waste |
| `thermal_load_kw` / `electric_boiler_kw` | M2 | Absent | **Present** (`kW`) | Annual usage (`EC_plant_usage`) | Absent | Available via SCADA / energy meters |
| `unit_material_cost` / `energy_cost` | M3 | `Total_Sales` (revenue only) | Absent | `EC_plant_cost`, `IMPCOST` | Absent | Collect ERP BOM unit costs |
| `machine_capacity_units_hr` | M4 | Absent | Absent | Absent | `machine_category`, `ME` | Collect OEM machine rates & changeover matrix |
