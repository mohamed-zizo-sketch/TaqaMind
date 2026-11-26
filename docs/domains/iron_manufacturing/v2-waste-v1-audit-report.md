# TaqaMind V2 — Phase 0: Iron Manufacturing Waste V1 Final Audit Report

**Audit Date:** 2026-11-26  
**Baseline Branch / Commit:** `origin/develop` @ `3bbe83847609bf8edc1278659ff8cb76d9de965e` (V1 Iron Manufacturing integrated from `feature/iron_manufacturing/pipeline-integration`, baseline Core `3e189ddf0b92b4394a516d180dabee05f3e71f80`)  
**Audited Scope:** Independent technical audit of `src/taqamind/domains/iron_manufacturing/` (EAF Meltshop, Rebar Hot Rolling, Scrap Yard Reception Waste V1) and compatibility with Shared Core P1.1–P1.9 (`CorePlatformServices`).

---

## 1. Baseline Verification Results

| Check | Command | Result |
| :--- | :--- | :--- |
| **V1 Iron Manufacturing Unit & E2E Tests** | `.venv/bin/python -m pytest tests/domains/iron_manufacturing -k "not energy_v2" -v` | **20 passed** (`test_contracts_and_validation.py`: 7, `test_lineage_and_isolation.py`: 4, `test_ml_splitting_and_leakage.py`: 5, `test_persistence_prediction_and_retraining.py`: 4) |
| **V1 + V2 Combined Iron Manufacturing Tests** | `.venv/bin/python -m pytest tests/domains/iron_manufacturing -v` | **30 passed** in `17.27s` |
| **Ruff Lint (Domain & Tests)** | `.venv/bin/ruff check src/taqamind/domains/iron_manufacturing tests/domains/iron_manufacturing --no-cache` | **All checks passed** (0 errors) |
| **Shared Core Compatibility** | `AUD-001` unit confirmation (`UnitAwareColumnMapper`), `AUD-004`, `AUD-008` | **Verified compatible** |

---

## 2. Independent Process Audit

### A. EAF Primary Meltshop (`eaf_metallic_waste_loss_prediction`)
- **Dataset & Provenance:** No real Egyptian EAF meltshop CSV exists in supplied data. Supplied `arc_furnace_heats/` (Yandex Practicum Ladle Furnace, 7 CSVs: `14,876` arc rows, `15,907` temperature rows) is quarantined as secondary metallurgy (LF), lacking tonnage, waste target, pre-production predictors, verified units, and commercial license.
- **Target & Boundary:** `metallic_waste_loss_t` (`metric_ton`), `TargetConfirmation.UNCONFIRMED` by default. Secondary KPI `metallic_waste_rate_pct` (`total_metallic_waste / planned_total_metallic_charge_t`).
- **Pre-Production Leakage Guard:** `EAF_FORBIDDEN_PREDICTOR_NAMES` blocks `actual_tap_to_tap_min`, `actual_power_on_min`, `in_process_arc_power_mw`, `actual_final_temp_c`, `actual_liquid_steel_tapped_t`, `slag_mass_t`, and `baghouse_dust_kg`.
- **Feature Engineering:** `build_eaf_derived_feature_specs()` deterministically computes `planned_total_metallic_charge_t`, `planned_dri_share_pct`, `planned_charge_to_tap_ratio`, `planned_flux_kg_per_t_charge`, `planned_oxygen_nm3_per_t_charge`, and `planned_expected_dirt_mass_t` strictly at `t <= t0`.
- **Splitting:** `SplitStrategy.GROUP_TEMPORAL` (`production_order_id` + `planned_start_ts`, `0.70 / 0.15 / 0.15`).

### B. Rebar Hot Rolling Mill (`rebar_rolling_waste_quantity_prediction`)
- **Dataset & Provenance:** No real Egyptian rebar rolling mill CSV exists in supplied data. Contract and deterministic engineering are verified on physics-grounded synthetic fixtures (`common/synthetic.py`).
- **Target & Boundary:** `total_rolling_waste_kg` (`kg`), `TargetConfirmation.UNCONFIRMED` by default. Secondary KPI `rolling_waste_rate_pct` normalized against `planned_input_billet_mass_kg` (`planned_input_billet_mass_t * 1000.0`).
- **Pre-Production Leakage Guard:** `REBAR_FORBIDDEN_PREDICTOR_NAMES` blocks `crop_ends_kg`, `cobble_scrap_kg`, `mill_scale_kg`, `quality_reject_bars_kg`, `actual_good_rebar_output_t`, and `actual_rolling_duration_min`.
- **Feature Engineering:** `build_rebar_derived_feature_specs()` deterministically computes `planned_input_billet_mass_kg`, `planned_single_billet_mass_t`, and `is_multi_slit_flag`.
- **Splitting:** `SplitStrategy.GROUP_TEMPORAL` (`production_order_id` + `planned_start_ts`, `0.70 / 0.15 / 0.15`).

### C. Scrap Yard Reception (`scrap_delivery_contaminant_prediction`)
- **Dataset & Provenance:** Backed for reference/EDA by EU Horizon ALCHIMIA (`scrap_A.csv`: `16,369` rows × `11` cols, `26.32%` zero contaminants, top supplier `45.53%`; `scrap_C.csv`: `9,399` rows × `11` cols, `1.78%` zeros, `130` suppliers) under `CC-BY-4.0`. Non-Egyptian European plants; reference-only.
- **Target & Boundary:** `total_contaminant_kg` (`kg`, representing `soil_kg + water_kg + other_kg`), `TargetConfirmation.UNCONFIRMED` by default until factories confirm entry weighbridge timing and weighing vs visual estimation.
- **Pre-Production Leakage Guard:** `SCRAP_YARD_FORBIDDEN_PREDICTOR_NAMES` excludes `exit_ts`, `exitdate`, `stay_duration_min`, `difference`, `soil_kg`, `water_kg`, `other_kg`, `soil`, `water`, and `other`.
- **Feature Engineering:** `build_scrap_yard_derived_feature_specs()` derives `delivery_mass_metric_ton` and `entry_hour_of_day`.
- **Splitting:** `SplitStrategy.GROUP_TEMPORAL` (`supplier_group_id` + `entry_ts`, `0.70 / 0.15 / 0.15`).

---

## 3. Shared Core Duplication & Engineering Controls

| Area | V1 Status | Finding |
| :--- | :--- | :--- |
| **Core P1.1–P1.5 Wiring** | Wired | `pipeline.py` uses Core `LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`, `LocalDatasetRegistry`, `LocalRecordLineageRegistry`, `CsvDataIngestion`, `PandasDataProfiler`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`, `DeterministicFeatureEngineer`, and `TabularPreprocessor`. |
| **Core P1.6–P1.8 Adapters** | **Duplicated in `adapters/`** | `src/taqamind/domains/iron_manufacturing/adapters/` (`readiness_and_splitting.py`, `training_registry.py`, `data_access.py`, `modeling.py`, `model_registry.py`) duplicates Shared Core P1.6–P1.8 engines now available via `CorePlatformServices`. **Kept untouched on this V2 branch** to isolate V1 maintenance refactoring into a separate dedicated V1 branch. |
| **Baselines & Models** | Verified | Baselines: `dummy_regressor`, `historical_rate`. Challengers: `ridge`, `elastic_net`, `hist_gradient_boosting`, `hurdle_regression` (zero-inflated scrap), plus optional `random_forest` & `xgboost`. |
| **Train-per-Factory & Zero-Retrain** | Verified | `test_lineage_and_isolation.py` and `test_persistence_prediction_and_retraining.py` prove strict `factory_id` isolation and zero `.fit()` / `.fit_transform()` calls during prediction and reload. |

---

## 4. Identified Limitations & V2 Architectural Mandate

1. **Unconfirmed Real Production Targets (By Design):** All 3 V1 targets remain `ConfirmationStatus.UNCONFIRMED` until partner Egyptian facilities supply historical CSVs and confirm measurement boundaries.
2. **V1 Adapter Duplication Left Untouched:** Retire `src/taqamind/domains/iron_manufacturing/adapters/` in a separate V1 maintenance PR—do **not** mix V1 adapter cleanup into `feature/iron_manufacturing/energy-consumption-v2`.
3. **V2 Direct Core Wiring:** `IronEnergyPipelineService` (`energy_pipeline.py`) wires directly to `CorePlatformServices.create_local(...)` from Day 1 with zero duplicated adapters, under `PlatformModule.ENERGY_EFFICIENCY`.

**V1 Readiness Verdict:**  
- **Software Pipeline & Guards:** **READY** (`20/20` V1 tests passing, Ruff clean).  
- **Real Factory Production Claims:** **BLOCKED** pending Egyptian factory CSV onboarding and `TargetConfirmation.CONFIRMED`.
