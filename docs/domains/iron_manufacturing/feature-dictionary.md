# Feature Dictionary — `iron_manufacturing`

Machine-readable feature dictionaries per process are also generated under:
- `contracts/domains/iron_manufacturing/templates/eaf_meltshop_feature_dictionary.json`
- `contracts/domains/iron_manufacturing/templates/rebar_hot_rolling_feature_dictionary.json`
- `contracts/domains/iron_manufacturing/templates/scrap_yard_reception_feature_dictionary.json`

## 1. EAF Primary Meltshop (`eaf_meltshop`, grain: `one_eaf_heat`)

| Canonical Name | Role | Type | Unit | Required | Availability | Missing Policy | Constraints / Aliases | Leakage Risk & Evidence |
|---|---|---|---|---|---|---|---|---|
| `heat_id` | `identifier` | `string` | — | Yes | `pre_production` | `reject` | Aliases: `heat_no`, `heat_number`, `melt_id`, `رقم الصبة` | Business key; never fed as predictor. |
| `production_order_id` | `metadata` | `string` | — | Yes | `pre_production` | `reject` | Aliases: `order_id`, `work_order_id`, `امر الشغل` | Grouping column for leakage-free splits. |
| `planned_start_ts` | `timestamp` | `datetime` | ISO-8601 | Yes | `pre_production` | `reject` | Aliases: `start_time`, `heat_start_ts`, `تاريخ الصبة` | Temporal split anchor (`t0`). |
| `furnace_id` | `predictor` | `category` | — | Yes | `static` | `reject` | Aliases: `eaf_id`, `furnace_no`, `رقم الفرن` | Static asset ID; safe pre-production. |
| `shift_code` | `predictor` | `category` | — | No | `pre_production` | `impute` | Allowed: `MORNING`, `EVENING`, `NIGHT` | Scheduled shift (`<= t0`). |
| `steel_grade_code` | `predictor` | `category` | — | Yes | `pre_production` | `reject` | Aliases: `steel_grade`, `grade_code`, `رتبة الصلب` | Ordered metallurgy spec (`<= t0`). |
| `planned_scrap_mass_t` | `predictor` | `float` | `metric_ton` | Yes | `pre_production` | `reject` | `[0, 300]` | Planned basket mass (`<= t0`). Must not substitute actual post-melt weight. |
| `planned_dri_mass_t` | `predictor` | `float` | `metric_ton` | Yes | `pre_production` | `reject` | `[0, 300]` | Essential for Egyptian Midrex DRI+EAF route (`<= t0`). |
| `planned_tap_mass_t` | `predictor` | `float` | `metric_ton` | Yes | `pre_production` | `reject` | `(0, 350]` | Ordered tap weight (`<= t0`); NOT actual tapped mass. |
| `planned_scrap_grade_mix_code` | `predictor` | `category` | — | No | `pre_production` | `impute` | Aliases: `scrap_mix_code`, `scrap_class` | Yard mix plan (`<= t0`). |
| `expected_scrap_contaminant_pct` | `predictor` | `float` | `%` | No | `pre_production` | `impute` | `[0, 25]` | Must be pre-charge yard prior (`<= t0`), never post-heat slag calculation. |
| `dri_metallization_pct` | `predictor` | `float` | `%` | No | `pre_production` | `impute` | `[70, 99]` | LIMS lot assay prior to charging (`<= t0`). |
| `dri_gangue_pct` | `predictor` | `float` | `%` | No | `pre_production` | `impute` | `[0, 15]` | LIMS lot assay prior to charging (`<= t0`). |
| `planned_hot_heel_mass_t` | `predictor` | `float` | `metric_ton` | No | `pre_production` | `impute` | `[0, 40]` | Pre-charge hot-heel estimate (`<= t0`). |
| `planned_lime_flux_kg` | `predictor` | `float` | `kg` | No | `pre_production` | `impute` | `[0, 20000]` | Level-2 recipe setpoint (`<= t0`). |
| `planned_oxygen_nm3` | `predictor` | `float` | `Nm3` | No | `pre_production` | `impute` | `[0, 15000]` | Level-2 recipe setpoint (`<= t0`). |
| `target_tap_temp_c` | `predictor` | `float` | `degC` | Yes | `pre_production` | `reject` | `[1520, 1720]` | Ordered aim temperature (`<= t0`); NOT actual measured final temp. |
| `refractory_lining_heat_age` | `predictor` | `integer` | `heats` | No | `pre_production` | `impute` | `[0, 5000]` | Campaign heat count prior to current heat (`<= t0`). |
| `planned_total_metallic_charge_t` | `predictor` (derived) | `float` | `metric_ton` | Yes | `pre_production` | `reject` | `(0, 500]`; `scrap + dri` | Deterministic; also denominator for `metallic_waste_rate_pct`. |
| `planned_dri_share_pct` | `predictor` (derived) | `float` | `%` | No | `pre_production` | `impute` | `[0, 100]` | Deterministic pre-production ratio. |
| `planned_charge_to_tap_ratio` | `predictor` (derived) | `float` | `ratio` | No | `pre_production` | `impute` | `[0.5, 2.0]` | Deterministic pre-production ratio. |
| `planned_flux_kg_per_t_charge` | `predictor` (derived) | `float` | `kg/t` | No | `pre_production` | `impute` | `[0, 500]` | Deterministic pre-production intensity. |
| `planned_oxygen_nm3_per_t_charge` | `predictor` (derived) | `float` | `Nm3/t` | No | `pre_production` | `impute` | `[0, 200]` | Deterministic pre-production intensity. |
| `planned_expected_dirt_mass_t` | `predictor` (derived) | `float` | `metric_ton` | No | `pre_production` | `impute` | `[0, 100]` | Deterministic pre-production expected dirt mass. |
| **`metallic_waste_loss_t`** | **`target`** | `float` | `metric_ton` | Yes | `post_production` | `reject` | `[0, 100]` | Never imputed; `UNCONFIRMED` until Egyptian factory confirmation. |

## 2. Rebar Hot Rolling Mill (`rebar_hot_rolling`, grain: `one_rolling_campaign_lot`)

| Canonical Name | Role | Type | Unit | Required | Availability | Missing Policy | Constraints |
|---|---|---|---|---|---|---|---|
| `rolling_lot_id` | `identifier` | `string` | — | Yes | `pre_production` | `reject` | Business key. |
| `production_order_id` | `metadata` | `string` | — | Yes | `pre_production` | `reject` | Grouping key. |
| `planned_start_ts` | `timestamp` | `datetime` | ISO-8601 | Yes | `pre_production` | `reject` | Temporal split anchor. |
| `mill_line_id` | `predictor` | `category` | — | Yes | `static` | `reject` | Physical rolling line ID. |
| `steel_grade_code` | `predictor` | `category` | — | Yes | `pre_production` | `reject` | Billet grade code. |
| `rebar_diameter_mm` | `predictor` | `float` | `mm` | Yes | `pre_production` | `reject` | `[6, 50]` |
| `planned_input_billet_mass_t` | `predictor` | `float` | `metric_ton` | Yes | `pre_production` | `reject` | `(0, 5000]` |
| `slitting_strands_count` | `predictor` | `integer` | `strands` | No | `pre_production` | `impute` | `[1, 6]` |
| `planned_billet_count` | `predictor` | `integer` | `billets` | No | `pre_production` | `impute` | `[1, 10000]` |
| `charging_mode_hot_or_cold` | `predictor` | `category` | — | No | `pre_production` | `impute` | `HOT_CHARGE`, `COLD_CHARGE` |
| `target_reheat_temp_c` | `predictor` | `float` | `degC` | No | `pre_production` | `impute` | `[950, 1300]` |
| `finishing_roll_groove_tonnage_age_t` | `predictor` | `float` | `metric_ton` | No | `pre_production` | `impute` | `[0, 50000]` |
| `planned_mill_speed_mps` | `predictor` | `float` | `m/s` | No | `pre_production` | `impute` | `[1, 60]` |
| `planned_input_billet_mass_kg` | `predictor` (derived) | `float` | `kg` | Yes | `pre_production` | `reject` | Exact SI `* 1000`; rate denominator. |
| `planned_single_billet_mass_t` | `predictor` (derived) | `float` | `metric_ton` | No | `pre_production` | `impute` | `[0.05, 20.0]` |
| `is_multi_slit_flag` | `predictor` (derived) | `integer` | — | No | `pre_production` | `impute` | `{0, 1}` |
| **`total_rolling_waste_kg`** | **`target`** | `float` | `kg` | Yes | `post_production` | `reject` | `[0, 1000000]`; `UNCONFIRMED` by default. |

## 3. Scrap Yard Reception (`scrap_yard_reception`, grain: `one_scrap_delivery_vehicle`)

| Canonical Name | Role | Type | Unit | Required | Availability | Missing Policy | Constraints |
|---|---|---|---|---|---|---|---|
| `scrap_delivery_id` | `identifier` | `string` | — | Yes | `pre_production` | `reject` | Business key (`ID` in ALCHIMIA). |
| `supplier_group_id` | `metadata` | `string` | — | Yes | `pre_production` | `reject` | Supplier group split key. |
| `entry_ts` | `timestamp` | `datetime` | ISO-8601 | Yes | `pre_production` | `reject` | Entry weighbridge timestamp (`t0`). |
| `supplier_code` | `predictor` | `category` | — | Yes | `pre_production` | `reject` | Supplier code. |
| `scrap_grade_code` | `predictor` | `category` | — | Yes | `pre_production` | `reject` | Commercial scrap grade (`ScrapType`). |
| `gross_delivery_mass_kg` | `predictor` | `float` | `kg` | Yes | `pre_production` | `reject` | `(0, 200000]`; rate denominator. |
| `warehouse_zone` | `predictor` | `category` | — | No | `pre_production` | `impute` | Yard zone (`Warehouse`). |
| `transport_mode` | `predictor` | `category` | — | No | `pre_production` | `impute` | `Truck` / `Rail car`. |
| `delivery_mass_metric_ton` | `predictor` (derived) | `float` | `metric_ton` | Yes | `pre_production` | `reject` | Exact SI `/ 1000`. |
| `entry_hour_of_day` | `predictor` (derived) | `integer` | — | No | `pre_production` | `impute` | `[0, 23]` |
| **`total_contaminant_kg`** | **`target`** | `float` | `kg` | Yes | `post_production` | `reject` | `[0, 200000]`; `UNCONFIRMED` by default. |
