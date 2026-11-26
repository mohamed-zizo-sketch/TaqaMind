# TaqaMind V2 — Phase 7: Proposed Energy Contracts & Shared Core Integration

**Status:** Implemented & Verified on `feature/iron_manufacturing/energy-consumption-v2`  
**Domain:** `iron_manufacturing` (`register_iron_manufacturing`)  
**Module:** `PlatformModule.ENERGY_EFFICIENCY` (isolated alongside V1 `PlatformModule.WASTE_MANAGEMENT` tasks)

---

## 1. Multi-Module Task & Dataset Identities

| Process (`process_id`) | V1 Waste Task (Preserved Untouched) | V2 Energy Task (`ENERGY_EFFICIENCY`) | V2 Dataset ID | Granularity |
| :--- | :--- | :--- | :--- | :--- |
| `eaf_meltshop` | `eaf_metallic_waste_loss_prediction` | `eaf_electrical_energy_regression` | `eaf_meltshop_heat_energy` | `one_eaf_heat` |
| `rebar_hot_rolling` | `rebar_rolling_waste_quantity_prediction` | `rebar_electrical_energy_regression` | `rebar_rolling_lot_energy` | `one_rolling_campaign_lot` |
| `scrap_yard_reception` | `scrap_delivery_contaminant_prediction` | `scrap_yard_electrical_energy_regression` | `scrap_yard_lot_energy` | `one_scrap_reception_lot` |

---

## 2. Contract Specifications (`energy_contracts.py`, Version `2.0.0`)

### A. EAF Meltshop (`build_eaf_energy_dataset_contract`)
- **Business Keys:** `("heat_id",)` | **Grouping:** `("production_order_id",)` | **Time Column:** `"planned_start_ts"`
- **Predictors (`PRE_PRODUCTION` / `STATIC`):** `furnace_id`, `steel_grade_code`, `planned_scrap_mass_t` (`metric_ton`), `planned_dri_mass_t` (`metric_ton`), `planned_tap_mass_t` (`metric_ton`), `target_tap_temp_c` (`degC`), `rated_kw` (`kW`), `planned_duration_hr` (`hr`), `planned_utilization_pct` (`%`).
- **Derived (`DERIVED`):** `planned_total_metallic_charge_t`, `planned_dri_share_pct`, `theoretical_nameplate_energy_kwh`.
- **Target:** `total_electrical_energy_kwh` (`kWh`, `minimum >= 0.0`, `UNCONFIRMED` by default).

### B. Rebar Hot Rolling (`build_rebar_energy_dataset_contract`)
- **Business Keys:** `("rolling_lot_id",)` | **Grouping:** `("production_order_id",)` | **Time Column:** `"planned_start_ts"`
- **Predictors (`PRE_PRODUCTION` / `STATIC`):** `mill_line_id`, `planned_billet_mass_t` (`metric_ton`), `target_rebar_diameter_mm` (`mm`), `rated_kw` (`kW`), `planned_duration_hr` (`hr`), `planned_utilization_pct` (`%`).
- **Derived (`DERIVED`):** `theoretical_nameplate_energy_kwh`.
- **Target:** `total_electrical_energy_kwh` (`kWh`, `minimum >= 0.0`, `UNCONFIRMED` by default).

### C. Scrap Yard Reception (`build_scrap_yard_energy_dataset_contract`)
- **Business Keys:** `("reception_lot_id",)` | **Grouping:** `("delivery_batch_id",)` | **Time Column:** `"planned_start_ts"`
- **Predictors (`PRE_PRODUCTION`):** `planned_scrap_mass_t` (`metric_ton`), `rated_kw` (`kW`), `planned_duration_hr` (`hr`), `planned_utilization_pct` (`%`).
- **Derived (`DERIVED`):** `theoretical_nameplate_energy_kwh`.
- **Target:** `total_electrical_energy_kwh` (`kWh`, `minimum >= 0.0`, `UNCONFIRMED` by default).

### D. Shared Target `RateDefinition`
- `name="electrical_energy_to_nameplate_ratio"`, `numerator="total_electrical_energy_kwh"`, `denominator="theoretical_nameplate_energy_kwh"`, `output_unit="ratio"`, `multiplier=1.0`.

---

## 3. Shared Core Integration Assessment

- **Zero Shared Core Edits Required:** `src/taqamind/core/*` required zero modifications.
- **Direct `CorePlatformServices` Wiring (`energy_pipeline.py`):** `IronEnergyPipelineService` uses `CorePlatformServices.create_local(...)` directly for storage, ingestion, profiling, unit-aware mapping, semantic validation, cleaning, deterministic feature engineering (`DeterministicFeatureEngineer`), record membership, readiness checking, `GROUP_TEMPORAL` splitting, leakage validation, `ControlledAutoMLTrainer` (`dummy_regressor`, `historical_rate`, `ridge`, `hist_gradient_boosting`), `ProtectedFinalEvaluator`, `ModelRegistry` activation/promotion, and `PredictionEngine` with zero duplicated adapters.
