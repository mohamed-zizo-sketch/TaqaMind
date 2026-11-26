# Feature Engineering Specification — `iron_manufacturing`

## 1. Boundary Between Deterministic & Learned Transformations
- **Pre-Split (Row-Level Deterministic Only — Shared Core `DeterministicFeatureEngineer` + `DerivedFeatureSpec`):** Only pure row-wise physical/algebraic transformations over confirmed pre-production (`<= t0`) inputs. Zero dataset-wide aggregations, zero target statistics, zero invented densities.
- **Post-Split (Train-Fold Fitted Only — Shared Core `TabularPreprocessor` inside `ControlledModelTrainer`):** `SimpleImputer(strategy="median")` for numerics, `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` for categoricals, and `StandardScaler()` for linear models (`Ridge`, `ElasticNet`). Fitted strictly on the `TRAIN` partition (`fit_transform(train_df)`) and invoked via `.transform()` only during validation, protected test evaluation, and inference.

## 2. Deterministic Derived Features by Process

### 2.1 EAF Primary Meltshop (`processes/eaf_meltshop.py`)

| Derived Feature | Exact Formula | Inputs | Unit | Granularity | Preconditions / Missing Behavior | Leakage Risk |
|---|---|---|---|---|---|---|
| `planned_total_metallic_charge_t` | `planned_scrap_mass_t + planned_dri_mass_t` | `planned_scrap_mass_t`, `planned_dri_mass_t` | `metric_ton` | `one_eaf_heat` | Sum must be `> 0`; checked by `validate_eaf_physical_rules`. | None (`<= t0`). |
| `planned_dri_share_pct` | `(planned_dri_mass_t / planned_total_metallic_charge_t) * 100` | `planned_dri_mass_t`, `planned_total_metallic_charge_t` | `%` | `one_eaf_heat` | `NaN` if denominator `<= 0`. | None (`<= t0`). |
| `planned_charge_to_tap_ratio` | `planned_total_metallic_charge_t / planned_tap_mass_t` | `planned_total_metallic_charge_t`, `planned_tap_mass_t` | `ratio` | `one_eaf_heat` | `NaN` if `planned_tap_mass_t <= 0`. | None (`<= t0`). |
| `planned_flux_kg_per_t_charge` | `planned_lime_flux_kg / planned_total_metallic_charge_t` | `planned_lime_flux_kg`, `planned_total_metallic_charge_t` | `kg/t` | `one_eaf_heat` | `NaN` if optional flux absent. | None (`<= t0`). |
| `planned_oxygen_nm3_per_t_charge` | `planned_oxygen_nm3 / planned_total_metallic_charge_t` | `planned_oxygen_nm3`, `planned_total_metallic_charge_t` | `Nm3/t` | `one_eaf_heat` | `NaN` if optional oxygen absent. | None (`<= t0`). |
| `planned_expected_dirt_mass_t` | `planned_scrap_mass_t * (expected_scrap_contaminant_pct / 100)` | `planned_scrap_mass_t`, `expected_scrap_contaminant_pct` | `metric_ton` | `one_eaf_heat` | `NaN` if optional contaminant prior absent. | Low (must use prior `<= t0`). |

### 2.2 Rebar Hot Rolling Mill (`processes/rebar_hot_rolling.py`)

| Derived Feature | Exact Formula | Inputs | Unit | Granularity | Preconditions / Missing Behavior | Leakage Risk |
|---|---|---|---|---|---|---|
| `planned_input_billet_mass_kg` | `planned_input_billet_mass_t * 1000.0` | `planned_input_billet_mass_t` | `kg` | `one_rolling_campaign_lot` | Exact SI factor; enables unit-compatible rate denominator (`kg/kg`). | None (`<= t0`). |
| `planned_single_billet_mass_t` | `planned_input_billet_mass_t / planned_billet_count` | `planned_input_billet_mass_t`, `planned_billet_count` | `metric_ton` | `one_rolling_campaign_lot` | `NaN` if optional `planned_billet_count` absent or `<= 0`. | None (`<= t0`). |
| `is_multi_slit_flag` | `1 if slitting_strands_count >= 2 else 0` | `slitting_strands_count` | binary | `one_rolling_campaign_lot` | `NaN` if optional `slitting_strands_count` absent. | None (`<= t0`). |

### 2.3 Scrap Yard Reception (`processes/scrap_yard_reception.py`)

| Derived Feature | Exact Formula | Inputs | Unit | Granularity | Preconditions / Missing Behavior | Leakage Risk |
|---|---|---|---|---|---|---|
| `delivery_mass_metric_ton` | `gross_delivery_mass_kg / 1000.0` | `gross_delivery_mass_kg` | `metric_ton` | `one_scrap_delivery_vehicle` | Exact SI factor. | None (`<= t0`). |
| `entry_hour_of_day` | `hour(entry_ts)` | `entry_ts` | `0–23` | `one_scrap_delivery_vehicle` | `NaN` if timestamp unparseable. | None (`<= t0`). |

## 3. Historical KPIs (Strictly Excluded from Pre-Production Predictors)
- `metallic_yield_pct = 100 * actual_liquid_steel_tapped_t / actual_total_metallic_charge_t`
- `rolling_yield_pct = 100 * actual_good_rebar_output_t / planned_input_billet_mass_t`
- `sec_kwh_per_t_liquid = actual_electrical_energy_kwh / actual_liquid_steel_tapped_t`
These are post-production monitoring KPIs only and are structurally rejected if supplied as predictors.
