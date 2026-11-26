# TaqaMind — Shared Core Contribution Proposal (from `food_beverage`)

This document catalogs every temporary generic adapter implemented under `src/taqamind/domains/food_beverage/adapters/` that is a candidate for consolidation into `src/taqamind/core/`.

---

## 1. Adapter Migration Catalog

| File Path | Class Name | Core Protocol / Role | Proposed Core Destination | Generic vs. Domain Logic | Required Changes Before Migration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `adapters/storage.py` | `LocalDatasetArtifactStorage`, `LocalModelArtifactStorage` | `DatasetArtifactStorage`, `ModelArtifactStorage` | `src/taqamind/core/storage/` | 100% generic (SHA-256 verified CSV & joblib storage). | Abstract backend interface for future S3/MinIO object store. |
| `adapters/ingestion.py` | `CsvDataIngestion`, `PandasDataProfiler` | `DataIngestion`, `DataProfiler` | `src/taqamind/core/ingestion/`, `src/taqamind/core/profiling/` | 100% generic CSV header validation & Pandas profiling. | Add Parquet/Excel reader hooks. |
| `adapters/mapping.py` | `DeterministicColumnMapper` | `ColumnMapper` | `src/taqamind/core/mapping/` | Generic exact/alias matching; uses `UNIT_CONVERSION_FACTORS` config. | Move `UNIT_CONVERSION_FACTORS` into Shared Core unit registry. |
| `adapters/validation.py` | `ContractSemanticValidator`, `AuditableDataCleaner` | `SemanticValidator`, `DataCleaner` | `src/taqamind/core/validation/`, `src/taqamind/core/preprocessing/` | 100% generic contract-driven validation & auditable cleaning. | None (already pure contract-driven). |
| `adapters/lineage.py` | `LocalRecordLineageRegistry` | `RecordLineageRegistry` | `src/taqamind/core/lineage/` | 100% generic business-key + content SHA-256 versioning. | Persist index to SQLite/Postgres in production Core. |
| `adapters/readiness_and_splitting.py` | `FoodBeverageReadinessChecker`, `FoodBeverageDatasetSplitter`, `FoodBeverageLeakageValidator` | `TrainingReadinessChecker`, `DatasetSplitter`, `LeakageValidator` | `src/taqamind/core/readiness/`, `splitting/`, `leakage/` | 95% generic; minimum row count (15) configurable. | Parameterize minimum row threshold via policy. |
| `adapters/training_and_evaluation.py` | `ControlledRegressionTrainer`, `ProtectedTestEvaluator` | `ModelTrainer`, `FinalModelEvaluator` | `src/taqamind/core/automl/`, `src/taqamind/core/evaluation/` | 95% generic supervised regression; uses allowlisted scikit-learn/XGBoost. | Add Hurdle/two-stage regressor if zero-inflated targets are encountered. |
| `adapters/model_registry.py` | `LocalModelRegistry` | `ModelRegistry` | `src/taqamind/core/registry/` | 100% generic lifecycle & single-active-model enforcement. | Back with transactional DB in production. |
| `adapters/prediction.py` | `FoodBeveragePredictionEngine` | `PredictionEngine` | `src/taqamind/core/prediction/` | 100% generic transform-only inference with rate calculation. | None. |
