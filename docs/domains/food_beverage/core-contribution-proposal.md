# TaqaMind — Shared Core Contribution Proposal (from `food_beverage`)

This document catalogs every temporary generic adapter implemented under `src/taqamind/domains/food_beverage/adapters/` on top of Shared Core `3e189dd` (`feature/core/storage-and-ingestion`) that is a candidate for consolidation into `src/taqamind/core/`.

> **Note:** Core P1.1–P1.5 engines (`LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`, `LocalDatasetRegistry`, `CsvDataIngestion`, `PandasDataProfiler`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `LocalRecordLineageRegistry`, `ContractDataCleaner`, `DeterministicFeatureEngineer`, `TabularPreprocessor`) are already wired directly from `taqamind.core` without re-implementation.

---

## 1. Adapter Migration Catalog

| File Path | Class / Helper Name | Core Protocol / Role | Proposed Core Destination | Generic vs. Domain Logic | Required Changes Before Migration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `adapters/_store.py` | `atomic_write_model`, `load_model_json`, `strip_computed_fields` | JSON persistence helper | `src/taqamind/core/registry/` | 100% generic Pydantic v2 serialization stripping `@computed_field`s for strict `extra_forbidden` reload. | Merge into shared registry storage helpers. |
| `adapters/unit_mapper.py` | `UnitAwareColumnMapperAdapter` | Thin wrapper around `DeterministicColumnMapper` (AUD-001) | `src/taqamind/core/mapping/` | Generic unit-conversion normalization using `UNIT_CONVERSION_FACTORS`. | Fold unit normalization into `DeterministicColumnMapper.apply_mapping()`. |
| `adapters/membership.py` | `FoodBeverageRowMembershipBuilder`, `RowLineageSummary` | AUD-008 row-to-record fingerprinting via `LocalRecordLineageRegistry` | `src/taqamind/core/lineage/` | 100% generic business-key & content SHA-256 `DatasetMembershipManifest` builder. | Move directly into `taqamind.core.lineage`. |
| `adapters/readiness_splitting.py` | `FoodBeverageTrainingReadinessChecker`, `FoodBeverageDatasetSplitter`, `FoodBeverageLeakageValidator`, `LocalSplitRegistry`, `LocalRecordUsageRegistry` | `TrainingReadinessChecker`, `DatasetSplitter`, `LeakageValidator`, `SplitRegistry`, `RecordUsageRegistry` (P1.6) | `src/taqamind/core/readiness/`, `splitting/`, `leakage/`, `training_registry/` | 95% generic; minimum row count (15) configurable. | Parameterize minimum row threshold via policy; back registries with DB in production. |
| `adapters/data_access_training.py` | `LocalDevelopmentDataProvider`, `LocalProtectedTestDataProvider`, `LocalTrainingRegistry`, `LocalEvaluationRegistry`, `FoodBeverageControlledAutoMLTrainer`, `FoodBeverageProtectedFinalEvaluator` | `DevelopmentDataProvider`, `ProtectedTestDataProvider`, `TrainingRegistry`, `EvaluationRegistry`, `ModelTrainer`, `FinalModelEvaluator` (P1.7) | `src/taqamind/core/data_access/`, `training/`, `evaluation/` | 95% generic supervised regression using Core `TabularPreprocessor` fitted strictly on `train_df`. | Add Hurdle/two-stage regressor if zero-inflated targets are encountered. |
| `adapters/model_registry_prediction.py` | `LocalModelRegistry`, `LocalActiveModelProvider`, `FoodBeveragePredictionEngine` | `ModelRegistry`, `ActiveModelProvider`, `PredictionEngine` (P1.8) | `src/taqamind/core/registry/`, `prediction/` | 100% generic single-active-model registry and transform-only inference with `DerivedRatePrediction`. | Back registry with transactional DB in production. |
