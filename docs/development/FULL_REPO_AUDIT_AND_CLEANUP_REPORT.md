# TaqaMind — Full Repository Audit, Refactoring & Engineering Report

**Branch:** `refactor/textile/retire-v1-adapters` (branched from `develop` @ `a1fa3be`)  
**Scope:** Repository-wide audit across Shared Core (`src/taqamind/core/`), Domains (`textile`, `iron_manufacturing`, `food_beverage`), Tests (`tests/`), and Documentation (`README.md`, `docs/domains/textile/handover.md`).

---

## Phase 1 & 2 — Prioritized Repository Audit & Refactoring Plan

| Severity | Finding | Affected Files | Action Taken / Plan |
| :--- | :--- | :--- | :--- |
| **High** | **Duplicated P1.6–P1.8 adapters in `textile` V1** (~2,796 lines duplicating `CorePlatformServices` readiness, splitting, leakage, data access, AutoML, evaluation, model registry, and prediction). | `src/taqamind/domains/textile/adapters/*`, `pipeline.py`, `tests/domains/textile/test_training_evaluation.py` | **Fixed in this PR:** Retired 5 duplicated adapter files; rewired `TextilePlatformStores` & `TextilePipelineService` to `CorePlatformServices.create_local(...)`; kept only `UnitAwareColumnMapperAdapter` for textile `yd <-> m`. |
| **High** | **Duplicated P1.6–P1.8 adapters remain in `iron_manufacturing` and `food_beverage` V1.** | `src/taqamind/domains/iron_manufacturing/adapters/*`, `src/taqamind/domains/food_beverage/adapters/*` | **Documented (Out of Textile Ownership):** Left untouched to respect domain boundaries; recommended as separate V1 maintenance PRs by respective domain maintainers. |
| **Medium** | **Target semantics & count-to-length safety (`Rej_and_cut_Piece` / `waste_length_yds`).** | `tests/domains/textile/test_contracts.py`, `weaving/contracts.py` | **Verified & Strengthened:** Weaving/Knitting targets default to `TargetConfirmation.UNCONFIRMED`; added explicit regression test verifying `UnitAwareColumnMapperAdapter` rejects `"pieces"` -> `"yd"` conversion. |
| **Medium** | **Stale documentation (`README.md`, `docs/domains/textile/handover.md`).** | `README.md`, `docs/domains/textile/handover.md` | **Fixed in this PR:** Updated both files to clearly separate implemented & tested V1, engineering-verified V2 extensions, and deferred modules (OR, Business Analytics, API/UI). |
| **Low** | **`sklearn` `ConvergenceWarning` in `ElasticNet` & `LogisticRegression` (Hurdle Stage 1).** | `src/taqamind/core/automl/trainer.py`, `src/taqamind/domains/textile/config.py` | **Documented (No Core Rewrite):** Non-blocking convergence warnings on unscaled synthetic features; all 1,026 tests pass. Future improvement: optional scaling inside linear/logistic estimators. |

---

## Phase 3–5 — Verification & Final Engineering Summary

1. **Files Changed in This PR:**
   - Removed 5 obsolete adapter files (`_store.py`, `data_access_training.py`, `membership.py`, `model_registry_prediction.py`, `readiness_splitting.py`).
   - Updated `src/taqamind/domains/textile/pipeline.py` and `src/taqamind/domains/textile/adapters/__init__.py`.
   - Strengthened `tests/domains/textile/test_contracts.py` (`pieces` -> `yd` rejection) and updated `tests/domains/textile/test_training_evaluation.py` to use `CorePlatformServices.create_evaluator`.
   - Updated `README.md`, `docs/domains/textile/handover.md`, and added this report.
2. **Architectural Improvements:**
   - Single authoritative implementation (`CorePlatformServices`) for both Textile V1 Waste and Textile V2 Energy.
   - `PromotionComparison` now strictly enforced on V1 model promotion transitions.
3. **Verification Results:**
   - Full repository suite (`pytest -q` after retiring adapters): **`1026 passed, 9 warnings in 469.21s`**.
   - Targeted regression check after final test/doc edits (`pytest tests/domains/textile/test_contracts.py -v` + `ruff check src tests --no-cache`) executed before push.
4. **Remaining Issues / Known Limitations:**
   - Real factory target sign-offs (`TargetConfirmation.CONFIRMED`) are still required before production deployment on real mill logs.
   - `iron_manufacturing` and `food_beverage` V1 adapters remain to be retired by their maintainers; `food_beverage` V2 Energy is pending.
