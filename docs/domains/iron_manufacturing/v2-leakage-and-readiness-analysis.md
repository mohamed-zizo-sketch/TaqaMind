# TaqaMind V2 — Phase 6: Problem Formulation & Leakage Prevention Analysis

**Task Scope:** Pre-Production Electrical Energy Consumption Regression (`PlatformModule.ENERGY_EFFICIENCY`) for `eaf_meltshop`, `rebar_hot_rolling`, and `scrap_yard_reception`.

---

## 1. Exact Problem Formulation

- **What is being predicted:** Cumulative active electrical energy consumption (`total_electrical_energy_kwh`, in `kWh`) for one planned industrial event:
  - `eaf_electrical_energy_regression`: `one_eaf_heat`
  - `rebar_electrical_energy_regression`: `one_rolling_campaign_lot`
  - `scrap_yard_electrical_energy_regression`: `one_scrap_reception_lot`
- **Prediction timing (`t <= t0`):** Prior to physical execution (before basket charging in EAF, billet charging in reheating furnace, or crane/shear/baler operation in the scrap yard), using only static nameplate ratings, planned schedules, and ordered charge/product specifications.
- **Target kind & metrics:** `TaskType.REGRESSION`, `TargetKind.QUANTITY`, evaluated by `ProtectedFinalEvaluator` using primary `MAE` (plus `RMSE`, `WAPE`, `R2`) against `DummyRegressor` and `HistoricalRate` baselines (`minimum_improvement_pct >= 1.0%` default).

---

## 2. Prohibited Future Information & Multi-Layer Leakage Guards

1. **Ingestion-Stage Post-Production Exclusion Guard (`energy_pipeline.py`):**
   - `IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS` (16 telemetry, runtime, final-temperature, output-tonnage, natural-gas, CO₂, and specific-energy columns) is checked case-insensitively on every `ingest_and_prepare()` call and raises `ValueError("CSV contains forbidden post-production/leakage column(s): ...")` if present.
2. **Prediction-Stage Target Exclusion Guard (`energy_pipeline.py`):**
   - When `purpose == DatasetPurpose.PREDICTION`, supplying `total_electrical_energy_kwh` raises `ValueError("Prediction CSV must not contain target column(s): ...")`.
3. **Deterministic Row-Wise Feature Engineering (`DeterministicFeatureEngineer`):**
   - `build_iron_energy_derived_specs()` computes only row-wise pre-production features (`theoretical_nameplate_energy_kwh`, plus EAF `planned_total_metallic_charge_t` and `planned_dri_share_pct`). Zero cross-row aggregations, zero target statistics.
4. **Train-Only Preprocessing (`ControlledAutoMLTrainer`):**
   - `TabularPreprocessor` fits imputers, encoders, and scalers strictly on the `TRAIN` fold; `test_energy_v2_pipeline.py` verifies zero `.fit()` calls during prediction and reload.
5. **Group-Temporal Splitting & Shared Core Leakage Validator:**
   - All 3 processes use `SplitStrategy.GROUP_TEMPORAL` (`0.70 / 0.15 / 0.15`), grouping by `production_order_id` (EAF & Rebar) or `delivery_batch_id` (Scrap Yard) ordered by `planned_start_ts`, validated by `CorePlatformServices.leakage_validator` and `readiness_checker`.
6. **Target Confirmation Gate:**
   - Every production `TargetContract` defaults to `ConfirmationStatus.UNCONFIRMED`. `readiness.require_split_eligibility()` blocks splitting and training unless explicit confirmation (or `build_synthetic_iron_energy_confirmation()` in tests) is provided.
