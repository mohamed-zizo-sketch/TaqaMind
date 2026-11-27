# V1 Waste Pipeline Audit Report (Food & Beverage)

**Scope**: Review of the existing V1 waste pipelines for bakery, dairy, and beverage domains.

## Findings
- The adapters under `src/taqamind/domains/food_beverage/adapters/` duplicate the shared Core Platform Services (P1.6–P1.8) which are now available in `src/taqamind/core/pipeline.py`.
- No functional issues were observed in the current V1 waste calculations; however, the duplicated adapters add maintenance overhead.
- All waste pipeline tests (`tests/domains/food_beverage/test_waste_v1.py`) pass (18/18).

## Recommendations
- **Do not** merge V1 adapter retirement changes into the V2 Energy branch. V1 cleanup must remain on a separate maintenance branch (`refactor/food_beverage/retire-v1-adapters`).
- In the V2 Energy implementation, wire directly to `CorePlatformServices.create_local(...)` for all pipeline stages (storage, registries, lineage, readiness, splitting, leakage, data access, AutoML, evaluator, model registry, zero‑retraining prediction).
- Retain only the `UnitAwareColumnMapperAdapter` for domain‑specific unit conversions (e.g., grams ⇄ kilograms, milliliters ⇄ liters).

The audit report is stored at this path for reference.
