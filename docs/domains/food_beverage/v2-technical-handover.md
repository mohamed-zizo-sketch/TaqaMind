# TaqaMind V2 — Food & Beverage Electrical Energy Technical Handover

**Domain:** `food_beverage` (`bakery`, `dairy`, `beverage`)  
**Module:** `PlatformModule.ENERGY_EFFICIENCY`  
**Tasks:** `bakery_energy_regression`, `dairy_energy_regression`, `beverage_energy_regression`

---

## 1. Architecture Summary

- **Direct Core Wiring:** `FoodBeverageEnergyPipelineService` (`src/taqamind/domains/food_beverage/energy_pipeline.py`) wires directly to `CorePlatformServices.create_local(root_dir)` with **zero** duplicated P1.6–P1.8 adapters.
- **V1 Preservation:** Existing V1 Waste pipelines (`pipeline.py`, `adapters/*`, `processes/bakery.py`, `dairy.py`, `beverage.py`) remain intact and backwards-compatible.
- **Domain Registration:** `register_food_beverage(...)` registers both V1 Waste (`PlatformModule.WASTE_MANAGEMENT`) and V2 Energy (`PlatformModule.ENERGY_EFFICIENCY`) tasks under `bakery`, `dairy`, and `beverage`.

---

## 2. Key Files Delivered

- **Contracts & Derived Feature Spec:** `src/taqamind/domains/food_beverage/energy_contracts.py`
- **Synthetic Verification Generators:** `src/taqamind/domains/food_beverage/energy_synthetic.py` (strictly for engineering verification, not real factory calibration)
- **Core-Wired Energy Pipeline:** `src/taqamind/domains/food_beverage/energy_pipeline.py`
- **Domain & Process Registration:** `src/taqamind/domains/food_beverage/domain.py`, `processes/{bakery,dairy,beverage}.py`
- **V2 Test Suite:** `tests/domains/food_beverage/test_energy_v2.py`
- **Documentation (Phases 0–7 & 12):** `docs/domains/food_beverage/v2-*.md`, `research/food_beverage/v2_eda_reference_summary.json`

---

## 3. Core Invariants Verified

1. `TargetConfirmation` defaults to `ConfirmationStatus.UNCONFIRMED`; unconfirmed datasets are blocked prior to splitting/training.
2. `RateDefinition` uses `numerator="total_electrical_energy_kwh"`, `denominator="theoretical_nameplate_energy_kwh"`, `output_unit="ratio"`, `multiplier=1.0`.
3. `DeterministicFeatureEngineer` receives `derived_specs` at construction (`DeterministicFeatureEngineer(storage, derived_specs=...)`).
4. Prediction CSVs containing `total_electrical_energy_kwh` are rejected at ingestion.
5. Zero-retraining prediction is enforced across repeated predictions and fresh service reloads, with strict factory-scoped model isolation.
