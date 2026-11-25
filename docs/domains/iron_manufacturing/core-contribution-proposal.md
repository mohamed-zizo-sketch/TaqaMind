# Shared Core Contribution & Adapter Migration Proposal — `iron_manufacturing`

As of Core baseline `3e189ddf0b92b4394a516d180dabee05f3e71f80`, all duplicate local P1.1–P1.5 adapters have been **retired** in favor of the real Shared Core engines.

## 1. Retired P1.1–P1.5 Adapters (Migrated to Real Shared Core `3e189dd`)

| Retired Local File / Symbol | Real Shared Core Engine Now Used Directly |
|---|---|
| `adapters/storage.py` (`LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`) | `taqamind.core.registry.LocalDatasetArtifactStorage`, `LocalModelArtifactStorage` |
| `adapters/dataset_registry.py` (`LocalDatasetRegistry`, `LocalRecordLineageRegistry`) | `taqamind.core.registry.LocalDatasetRegistry`, `taqamind.core.lineage.LocalRecordLineageRegistry` |
| `adapters/processing.py` (`LocalCsvDataIngestion`, `LocalDataProfiler`) | `taqamind.core.ingestion.CsvDataIngestion`, `taqamind.core.profiling.PandasDataProfiler` |
| `adapters/processing.py` (`LocalColumnMapper`, `LocalSemanticValidator`) | `taqamind.core.mapping.DeterministicColumnMapper`, `taqamind.core.validation.ContractSemanticValidator` (with thin domain/audit wrappers in `adapters/unit_mapper.py`) |
| `adapters/processing.py` (`LocalDataCleaner`, `LocalFeatureEngineer`) | `taqamind.core.preprocessing.ContractDataCleaner`, `taqamind.core.feature_engineering.DeterministicFeatureEngineer` (`DerivedFeatureSpec`) |
| Custom `ColumnTransformer` in `adapters/modeling.py` | `taqamind.core.preprocessing.TabularPreprocessor` (fit on `train_df` only; `.transform()` only during evaluation and prediction) |

## 2. Remaining Core Audit Mitigation Wrappers (Removable Once Core Audit Items Close)

| Wrapper Class | File Path | Core Audit / Domain Purpose | Removal Condition |
|---|---|---|---|
| `IronUnitAwareColumnMapper` | `src/taqamind/domains/iron_manufacturing/adapters/unit_mapper.py` | Mitigates Core **AUD-001** (verified SI unit conversions during `apply_mapping`) and **AUD-004** (preserve string/category dtypes to prevent leading-zero loss), plus iron ambiguous-header & prediction-mode leakage refusal. | Retire AUD-001/AUD-004 code once `DeterministicColumnMapper` supports verified unit conversion & dtype preservation natively. |
| `IronSemanticValidator` | `src/taqamind/domains/iron_manufacturing/adapters/unit_mapper.py` | Delegates to Core `ContractSemanticValidator` and appends iron physical mass-balance checks (`processes/*.py`). | Keep domain physical rules injected once Core exposes a custom rule hook. |
| `ProcessFeatureEngineerRouter` | `src/taqamind/domains/iron_manufacturing/adapters/unit_mapper.py` | Dispatches each process (`eaf_meltshop`, `rebar_hot_rolling`, `scrap_yard_reception`) to its own real Core `DeterministicFeatureEngineer` configured with `DerivedFeatureSpec`. | Thin permanent multi-process facade or Core process-scoped registry. |
| `register_dataset_lineage` | `src/taqamind/domains/iron_manufacturing/services/lineage_service.py` | Focused row-to-record fingerprinting & membership builder (**AUD-008** mitigation) calling Core `LocalRecordLineageRegistry`. | Promote generic row fingerprinting helper to `taqamind.core.lineage`. |
| `strip_computed_fields` | `src/taqamind/domains/iron_manufacturing/adapters/_store.py` | Recursively strips `ValidationReport` `@computed_field` keys (`status`, `summary`, `has_blocking_issues`) before JSON persistence so strict `model_validate_json()` succeeds. | Promote into `taqamind.core.registry._json_store` / `taqamind.core.interfaces.serialization`. |

## 3. Remaining True Core Gap Adapters (P1.6–P1.8)

| Core Phase | Adapter Class | File Path | Core Protocol Implemented | Proposed Shared Core Destination | Migration Notes |
|---|---|---|---|---|---|
| **P1.6** | `LocalTrainingReadinessChecker`, `LocalDatasetSplitter`, `LocalLeakageValidator` | `src/taqamind/domains/iron_manufacturing/adapters/readiness_and_splitting.py` | `TrainingReadinessChecker`, `DatasetSplitter`, `LeakageValidator` | `taqamind.core.{readiness,splitting}` | 100% generic contract-driven readiness, `GROUP_TEMPORAL`/`TEMPORAL`/`GROUP`/`RANDOM` splitting, and all 8 `LeakageCheck` categories. |
| **P1.6** | `LocalSplitRegistry`, `LocalRecordUsageRegistry` | `src/taqamind/domains/iron_manufacturing/adapters/training_registry.py` | `SplitRegistry`, `RecordUsageRegistry` | `taqamind.core.registry.ml_governance` | Immutable split/authorization persistence and historical exposure tracking. |
| **P1.7** | `LocalTrainingRegistry`, `LocalEvaluationRegistry` | `src/taqamind/domains/iron_manufacturing/adapters/training_registry.py` | `TrainingRegistry`, `EvaluationRegistry` | `taqamind.core.registry.ml_governance` | `TrainingReuseKey` deduplication and atomic single-use protected test reservations. |
| **P1.7** | `LocalDevelopmentDataProvider`, `LocalProtectedTestDataProvider` | `src/taqamind/domains/iron_manufacturing/adapters/data_access.py` | `DevelopmentDataProvider`, `ProtectedTestDataProvider` | `taqamind.core.data_access.local` | Physical partition CSV materialization and exposure-before-release enforcement. |
| **P1.7** | `ControlledModelTrainer`, `ProtectedFinalModelEvaluator` | `src/taqamind/domains/iron_manufacturing/adapters/modeling.py` | `ModelTrainer`, `FinalModelEvaluator` | `taqamind.core.{automl,evaluation}` | Uses Core `TabularPreprocessor` train-only; allowlisted families (`DummyRegressor`, `HistoricalRateRegressor`, `Ridge`, `ElasticNet`, `RandomForest`, `HistGradientBoosting`, `XGBoost`, `TwoStageHurdleRegressor`). |
| **P1.8** | `LocalModelRegistry` | `src/taqamind/domains/iron_manufacturing/adapters/model_registry.py` | `ModelRegistry` | `taqamind.core.registry.model` | Lifecycle transitions (`CANDIDATE -> VALIDATED/REJECTED -> ACTIVE -> ARCHIVED`), max 1 active per scope, atomic audit trail. |
| **P1.8** | `LocalActiveModelProvider`, `ZeroRetrainingPredictionEngine` | `src/taqamind/domains/iron_manufacturing/adapters/data_access.py`, `adapters/modeling.py` | `ActiveModelProvider`, `PredictionEngine` | `taqamind.core.{data_access,prediction}` | SHA-256-verified bundle loading and strict `.transform()` + `.predict()` zero-fit inference. |

## 4. Replacement Strategy
1. Promote P1.6–P1.8 generic adapters into `taqamind.core.*` behind the exact existing protocols.
2. Swap bindings inside `IronManufacturingPipeline.__init__` (zero changes required to `domain.py`, `processes/`, or `contracts/domains/iron_manufacturing/`).
3. Re-run `pytest tests/domains/iron_manufacturing -v` to confirm 100% behavioral parity.
