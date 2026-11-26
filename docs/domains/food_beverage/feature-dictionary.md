# TaqaMind — Food & Beverage Industrial Feature Dictionary

**Domain ID:** `food_beverage`  
**Modules Covered:**
1. Waste Management and Prediction (`waste`)
2. Energy Efficiency and Consumption Analysis (`energy`)
3. Business Analytics and Cost Optimization (`business`)
4. Operations Research and Production Optimization (`operations_research`)

---

## 1. Feature Categories & Leakage Policy

Every feature is classified by:
- **Category:** `shared`, `process_specific`, `derived`, `optional`, `critical`, `target`, `identifier`, `timestamp`, `metadata`, `constraint`, `business_kpi`.
- **Availability at Prediction Time:** `pre_production` (known before batch start), `in_process` (recorded during run — **prohibited** for pre-production prediction), `post_production` (outcome/KPI — **strictly prohibited** as predictor).

---

## 2. Canonical Feature Inventory

| # | Canonical Name (`snake_case`) | Human Name | Process | Module | Type | Unit | Granularity | Origin / Collection | Status | Availability | Leakage Risk | Range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `factory_id` | Factory Identifier | All | All | string | dimensionless | factory | Configured / ERP | Required | `pre_production` | None (grouping/isolation key) | Non-empty |
| 2 | `production_line_id` | Production Line ID | All | All | string | dimensionless | line | Configured / MES | Recommended | `pre_production` | None (group key) | Non-empty |
| 3 | `machine_id` | Machine Identifier | All | All | string | dimensionless | machine | Configured / MES | Optional | `pre_production` | None | Non-empty |
| 4 | `batch_id` | Production Batch ID | All | All | string | dimensionless | batch | Recorded / ERP | Required | `pre_production` | Identifier only (never model feature) | Unique per factory |
| 5 | `production_timestamp` | Planned Production Timestamp | All | All | datetime | ISO-8601 UTC | batch | Recorded / ERP | Required | `pre_production` | Temporal split key | Valid ISO-8601 |
| 6 | `product_id` | Product / SKU Code | All | All | categorical | dimensionless | batch | Configured / ERP | Required | `pre_production` | High cardinality if ungrouped | Non-empty |
| 7 | `product_category` | Product Family / Style | All | All | categorical | dimensionless | batch | Configured / ERP | Required | `pre_production` | None | Non-empty |
| 8 | `planned_shift` | Planned Production Shift | All | All | categorical | dimensionless | batch | Recorded / MES | Optional | `pre_production` | None | `{morning, afternoon, night}` |
| 9 | `planned_batch_quantity_kg` | Planned Batch Quantity (kg) | Bakery, Dairy, Fruit/Veg | Waste, Business, OR | float | `kg` | batch | Recorded / ERP | Required | `pre_production` | None (core scale predictor) | `> 0` |
| 10 | `planned_batch_volume_l` | Planned Batch Volume (L) | Beverage, Dairy | Waste, Energy | float | `L` | batch | Recorded / ERP | Required (Liquid) | `pre_production` | None | `> 0` |
| 11 | `planned_raw_material_kg` | Planned Raw Material Mass | Bakery, Fruit/Veg | Waste, Business | float | `kg` | batch | Recorded / ERP | Recommended | `pre_production` | None | `> 0` |
| 12 | `planned_ingredient_count` | Planned Recipe Ingredient Count | All | Waste | integer | `count` | batch | Derived / Recipe | Optional | `pre_production` | None | `>= 1` |
| 13 | `planned_processing_temp_c` | Planned Processing Temperature | Bakery, Dairy, Beverage | Waste, Energy | float | `degC` | batch | Configured / Recipe | Recommended | `pre_production` | None (do not confuse with actual sensor temp) | `-10` to `300` |
| 14 | `planned_processing_duration_min` | Planned Processing Duration | All | Waste, OR | float | `min` | stage/batch | Configured / Recipe | Recommended | `pre_production` | None | `> 0` |
| 15 | `planned_fermentation_hours` | Planned Fermentation Duration | Bakery, Dairy, Brewing | Waste | float | `h` | batch | Configured / Recipe | Optional | `pre_production` | None | `>= 0` |
| 16 | `target_ph_level` | Target Recipe pH | Dairy, Beverage | Waste | float | `pH` | batch | Configured / Spec | Optional | `pre_production` | None (must be spec, not post-batch LIMS) | `2.0` to `9.0` |
| 17 | `packaging_type` | Packaging Format / SKU | All | Waste, Business | categorical | dimensionless | batch | Configured / BOM | Optional | `pre_production` | None | Non-empty |
| 18 | `planned_filling_quantity_units` | Planned Filling Unit Count | Beverage, Dairy | Waste, OR | integer | `units` | batch | Recorded / ERP | Optional | `pre_production` | None | `> 0` |
| 19 | `ambient_temp_forecast_c` | Ambient Facility Temperature | Bakery, Dairy | Waste, Energy | float | `degC` | shift/day | Measured / Sensor | Optional | `pre_production` | Low if measured pre-batch | `-10` to `55` |
| 20 | `waste_quantity_kg` | Realized Avoidable Waste (kg) | Bakery, Dairy, Fruit/Veg | Waste | float | `kg` | batch | Measured / Scale | Target | `post_production` | **TARGET** (never predictor) | `>= 0` |
| 21 | `waste_quantity_l` | Realized Liquid Waste (L) | Beverage, Dairy | Waste | float | `L` | batch | Measured / Flowmeter | Target | `post_production` | **TARGET** (never predictor) | `>= 0` |
| 22 | `rejected_unit_count` | Realized Rejected Units | Beverage, Bakery, Dairy | Waste | integer | `units` | batch | Measured / Checkweigher | Target | `post_production` | **TARGET** (never predictor) | `>= 0` |
| 23 | `actual_accepted_quantity_kg` | Actual Good Output (kg) | All | Business KPI | float | `kg` | batch | Measured / Scale | Post-Prod KPI | `post_production` | **CRITICAL LEAKAGE** | `>= 0` |
| 24 | `actual_yield_pct` | Realized Production Yield (%) | All | Business KPI | float | `%` | batch | Derived | Post-Prod KPI | `post_production` | **CRITICAL LEAKAGE** | `0` to `105` |
| 25 | `brewhouse_efficiency_pct` | Realized Brewhouse Efficiency | Beverage (Brewing) | Business KPI | float | `%` | batch | Measured / Derived | Post-Prod KPI | `post_production` | **CRITICAL LEAKAGE** | `0` to `100` |
| 26 | `quality_score` | Post-Production Quality Score | All | Business KPI | float | `score` | batch | Measured / LIMS | Post-Prod KPI | `post_production` | **CRITICAL LEAKAGE** | `0` to `100` |
| 27 | `total_sales_value` | Realized Batch Revenue | All | Business | float | `currency` | batch | Recorded / ERP | Post-Prod KPI | `post_production` | **CRITICAL LEAKAGE** | `>= 0` |
| 28 | `thermal_load_kw` | Thermal Heating/Cooling Load | Dairy, Bakery, Beverage | Energy | float | `kW` | 1-min / 1-hr | Measured / Meter | Required (M2) | `in_process` | Energy Module Target/Feature | `>= -500` (recovery) |
| 29 | `electric_boiler_kw` | Electric Boiler Power | Dairy, Beverage | Energy | float | `kW` | 1-hr | Measured / Meter | Recommended (M2) | `in_process` | Energy Module Target | `>= 0` |
| 30 | `specific_energy_consumption_wh_l` | Specific Energy Consumption | Dairy, Beverage | Energy | float | `Wh/L` | month/batch | Derived | KPI (M2) | `post_production` | Leakage if used for M1 waste | `>= 0` |
| 31 | `unit_material_cost` | Raw Material Unit Cost | All | Business | float | `currency/kg` | batch | Recorded / ERP | Required (M3) | `pre_production` | None | `>= 0` |
| 32 | `energy_cost_per_kwh` | Electricity / Fuel Tariff | All | Business, Energy | float | `currency/kWh` | month | Configured / Tariff | Required (M3) | `pre_production` | None | `>= 0` |
| 33 | `machine_capacity_units_hr` | Machine Nominal Capacity | All | OR | float | `units/h` | machine | Configured / OEM | Required (M4) | `pre_production` | Constraint parameter | `> 0` |
| 34 | `changeover_setup_min` | Planned Changeover Setup Time | All | OR | float | `min` | transition | Configured / IE | Recommended (M4) | `pre_production` | Decision/Constraint input | `>= 0` |
| 35 | `shift_available_min` | Available Shift Time | All | OR | float | `min` | shift | Configured / HR | Required (M4) | `pre_production` | Constraint parameter | `> 0` |

---

## 3. Detailed Specification Notes (21-Attribute Compliance)

Machine-readable JSON definitions containing all 21 attributes (including aliases, missing-value policy, expected target relationship, evidence source, and remaining semantic uncertainty) are stored in [feature_dictionary.json](file:///d:/TaqaMind/research/food_beverage/feature_dictionary.json).

### Key Semantic Boundaries
- **No `actual_*` predictors in Module 1:** Variables such as `actual_accepted_quantity_kg`, `quality_score`, `brewhouse_efficiency_pct`, and `total_sales_value` are strictly flagged `post_production` and blocked by `SemanticValidator` from entering pre-production waste prediction models.
- **Unit Explicitness:** `kg`, `L`, and `units` are never treated as interchangeable.
- **Planned vs. Actual:** `planned_processing_temp_c` (recipe setpoint) is distinct from `measured_oven_temp_c` (in-process sensor reading).
