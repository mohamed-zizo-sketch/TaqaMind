# Shared Core Adapter Retirement Summary — `iron_manufacturing`

All duplicated local P1.1–P1.8 adapters under `src/taqamind/domains/iron_manufacturing/adapters/` have been **completely retired** in favor of `CorePlatformServices.create_local(...)` from `taqamind.core.pipeline`.

## 1. Retired Duplicate Adapters (100% Migrated to Shared Core)

| Phase | Retired Local File | Replaced By Real Shared Core Engine (`CorePlatformServices`) |
| :--- | :--- | :--- |
| **P1.1** | `adapters/storage.py` | `taqamind.core.registry.LocalDatasetArtifactStorage`, `LocalModelArtifactStorage` |
| **P1.2–P1.3, P1.5** | `adapters/processing.py` | `CsvDataIngestion`, `PandasDataProfiler`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`, `DeterministicFeatureEngineer` |
| **P1.4** | `adapters/dataset_registry.py` | `LocalDatasetRegistry`, `LocalRecordLineageRegistry` |
| **P1.6** | `adapters/readiness_and_splitting.py` | `CoreTrainingReadinessChecker`, `CoreDatasetSplitter`, `CoreLeakageValidator` |
| **P1.6–P1.7** | `adapters/training_registry.py` | `LocalSplitRegistry`, `LocalRecordUsageRegistry`, `LocalTrainingRegistry`, `LocalEvaluationRegistry` |
| **P1.7–P1.8** | `adapters/data_access.py`, `adapters/modeling.py` | `LocalDevelopmentDataProvider`, `LocalProtectedTestDataProvider`, `ControlledAutoMLTrainer`, `ProtectedFinalEvaluator`, `LocalActiveModelProvider`, `CorePredictionEngine` |
| **P1.8** | `adapters/model_registry.py`, `adapters/_store.py` | `LocalModelRegistry` & Core `_json_store` |

## 2. Retained Domain-Specific Wrappers (`adapters/unit_mapper.py`)

Only thin iron-manufacturing domain wrappers remain under `src/taqamind/domains/iron_manufacturing/adapters/`:

| Symbol | Purpose |
| :--- | :--- |
| `IronUnitAwareColumnMapper` | Wraps Core `DeterministicColumnMapper` for verified SI unit conversions (**AUD-001**), string/category dtype preservation (**AUD-004**), ambiguous raw header refusal, and prediction-mode target/post-production leakage blocking. |
| `IronSemanticValidator` | Delegates to Core `ContractSemanticValidator` and appends iron physical mass-balance rules (`processes/*.py`). |
| `ProcessFeatureEngineerRouter` | Dispatches each iron process (`eaf_meltshop`, `rebar_hot_rolling`, `scrap_yard_reception`) to its dedicated Core `DeterministicFeatureEngineer` configured with `DerivedFeatureSpec`. |
