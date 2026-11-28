# Textile Waste V1 — Technical Audit & P0 Compatibility Report

- **Domain:** `textile` (`weaving`, `knitting`)
- **Audit Date:** 2026-11-28
- **Development Baseline (`develop`):** `5455cc8` (includes PR #15 V1 adapter retirement & PR #12 Energy V2)
- **Shared Core Compatibility Reference (`feature/core/v1-integration-audit`):** `849027d` (WP00–WP04)

---

## 1. Component-by-Component Audit

| Component | Existing Implementation | Shared Core Dependency | Duplicated Functionality | Classification | Findings & Status |
|---|---|---|---|---|---|
| **Domain & Process Contracts** | `domain.py`, `weaving/contracts.py`, `knitting/contracts.py` | `taqamind.core.contracts` | None | Ready (with P0 migration) | **HIGH:** `build_weaving_derived_specs()` and `build_knitting_derived_specs()` use legacy Python `compute=` callbacks, blocked under WP01 `UNVERIFIED_CALLBACK_BLOCKED`. |
| **Unit-Aware Column Mapper** | `adapters/unit_mapper.py` (`UnitAwareColumnMapperAdapter`) | `DeterministicColumnMapper`, `LocalDatasetArtifactStorage` | Unit conversion wrapper (`AUD-001`) | **MEDIUM / HIGH** | Uses `pd.read_csv(stream, keep_default_na=True)` instead of `SafeCsvReader` + `CsvResourcePolicy` + `normalize_text_missing` (`TI-001`, `TI-002`, `TI-008`). Also uses `pd.to_numeric(..., errors="coerce")` which silently turns invalid numeric tokens into `NaN` (`TI-009`). |
| **Ingestion & Profiling** | `pipeline.py` via `CorePlatformServices` | `CsvDataIngestion`, `PandasDataProfiler` | None (retired in PR #15) | Clean | Wired directly to `CorePlatformServices.create_local(...)`. |
| **Semantic Validation & Cleaning** | `pipeline.py` via `CorePlatformServices` | `ContractSemanticValidator`, `ContractDataCleaner` | None | Clean | Wired directly to Shared Core. |
| **Feature Engineering** | `pipeline.py` via `DeterministicFeatureEngineer` | `DeterministicFeatureEngineer` | None | **CRITICAL (WP01)** | Blocked at runtime against `849027d` because `DerivedFeatureSpec.compute` callbacks are rejected by WP01 strict leakage-safety policy (`UNVERIFIED_CALLBACK_BLOCKED`). |
| **Record Membership & Lineage** | `pipeline.py` via `services.core.membership_builder` | `RecordMembershipBuilder`, `LocalRecordLineageRegistry` | None (retired in PR #15) | **HIGH (WP02)** | `pipeline.py` (`ingest_and_prepare`) should ensure `DatasetPurpose.PREDICTION` routes through `resolve_prediction_observations` instead of `build_and_register` once WP02 lands in WP10. |
| **Readiness, Splitting & Leakage** | `pipeline.py` via `CorePlatformServices` | `CoreTrainingReadinessChecker`, `CoreDatasetSplitter`, `CoreLeakageValidator` | None (retired in PR #15) | Clean | Wired directly to Shared Core. |
| **AutoML, Evaluation, Registry & Prediction** | `pipeline.py` via `CorePlatformServices` | `ControlledAutoMLTrainer`, `ProtectedFinalEvaluator`, `LocalModelRegistry`, `CorePredictionEngine` | None (retired in PR #15) | Clean | Wired directly to Shared Core; `create_evaluator(contract)` used per contract. |
| **V1 Retired Adapters** | `adapters/__init__.py` | N/A | All 5 duplicated adapter modules removed in PR #15 (`b13be9a`) | Resolved | `_store.py`, `membership.py`, `readiness_splitting.py`, `data_access_training.py`, `model_registry_prediction.py` already retired on `develop`. |

---

## 2. Reproduction of the 9 Known Compatibility Failures (Task 02)

- **Tested Checkpoint:** `feature/core/v1-integration-audit` @ `849027d`
- **Command:** `python -m pytest tests/domains/textile -v --tb=short`
- **Result:** **8 passed, 9 failed** in `7.80s`.
- **Root Cause Classification:** All 9 failures are **intentional WP01 strict safety blocks** (`UNVERIFIED_CALLBACK_BLOCKED`), **not** accidental regressions or weakened assertions.

| # | Test Name | File | Blocked Feature | Exception / Code |
|---|---|---|---|---|
| 1 | `test_end_to_end_synthetic_training_and_prediction[weaving]` | `tests/domains/textile/test_end_to_end.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 2 | `test_end_to_end_synthetic_training_and_prediction[knitting]` | `tests/domains/textile/test_end_to_end.py` | `knitting_tightness_factor` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 3 | `test_lineage_new_exact_corrected_and_factory_isolation` | `tests/domains/textile/test_lineage.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 4 | `test_repeated_predictions_never_retrain_or_fit` | `tests/domains/textile/test_no_retraining.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 5 | `test_prediction_reload_cross_factory_and_missing_model_guards` | `tests/domains/textile/test_prediction.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 6 | `test_unconfirmed_target_blocks_readiness` | `tests/domains/textile/test_splitting_leakage.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 7 | `test_weaving_group_split_disjoint_and_leakage_free` | `tests/domains/textile/test_splitting_leakage.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 8 | `test_knitting_group_temporal_split_passes_temporal_order` | `tests/domains/textile/test_splitting_leakage.py` | `knitting_tightness_factor` | `UNVERIFIED_CALLBACK_BLOCKED` |
| 9 | `test_rejected_challenger_cannot_be_activated` | `tests/domains/textile/test_training_evaluation.py` | `total_thread_density_per_inch` | `UNVERIFIED_CALLBACK_BLOCKED` |

---

## 3. Severity Summary

- **CRITICAL (1):** Legacy `DerivedFeatureSpec.compute` callbacks blocked under WP01 (`total_thread_density_per_inch`, `knitting_tightness_factor`); `knitting_tightness_factor` requires `SQRT` in Core `RowLocalTransform` (`CIR-TEXTILE-001`).
- **HIGH (2):** `UnitAwareColumnMapperAdapter` uses `pd.read_csv(..., keep_default_na=True)` and `pd.to_numeric(..., errors="coerce")`; `pipeline.py` calls `membership_builder.build_and_register` on `PREDICTION` datasets (WP10 migration against WP02).
- **MEDIUM (1):** `energy_pipeline.py` uses unbounded `Path(csv_path).read_bytes()` (`TI-002`).
- **INFORMATIONAL:** All 5 duplicated P1.6–P1.8 V1 adapters are already retired on `develop` (`5455cc8`).
