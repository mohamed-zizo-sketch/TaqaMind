# TaqaMind — Food & Beverage Domain Handover Summary

**Domain ID:** `food_beverage`  
**Branch:** `refactor/food_beverage/retire-v1-adapters`  
**Baseline Core Commit:** `3e189dd` (`feature/core/storage-and-ingestion`)

---

## 1. Delivered Scope

- **Industrial Research & Forensics:** Evidence-based research across Bakery, Dairy, Beverage, and Fruit/Veg; empirical audit of `brewery_data.csv`, Norwegian Dairy Energy (EU ENOUGH), US DOE ITAC (NAICS 311–312), and Babor & Hitzmann Bakery Scheduling (BK15/BK50).
- **Domain Contracts:** Registered `food_beverage` (`1.0.0`) with `bakery`, `dairy`, and `beverage` processes using Shared Core contracts (`DatasetContract`, `FeatureContract`, `TargetContract`), defaulting to `ConfirmationStatus.UNCONFIRMED`.
- **CorePlatformServices Wiring (P1.1–P1.8):** `FoodBeveragePipeline` now obtains **all** P1.1–P1.8 services by calling `CorePlatformServices.create_local(root_dir)` (defined in `taqamind.core.pipeline`). The only domain-specific adapter retained is `UnitAwareColumnMapperAdapter` (AUD-001), which wraps the Core `DeterministicColumnMapper` to apply physical unit conversions (`g↔kg`, `mL↔L`, `h↔min`, etc.).
- **Retired V1 Adapters:** The following duplicated adapter files have been removed via `git rm`:
  - `src/taqamind/domains/food_beverage/adapters/_store.py`
  - `src/taqamind/domains/food_beverage/adapters/data_access_training.py`
  - `src/taqamind/domains/food_beverage/adapters/membership.py`
  - `src/taqamind/domains/food_beverage/adapters/model_registry_prediction.py`
  - `src/taqamind/domains/food_beverage/adapters/readiness_splitting.py`
- **Evaluator:** Created per-contract via `self._services.create_evaluator(contract)` through the `_get_evaluator(contract)` helper on `FoodBeveragePipeline`, not stored as a fixed instance attribute.
- **Verification:** Domain tests passing, `ruff check src tests --no-cache` passing (0 errors).
