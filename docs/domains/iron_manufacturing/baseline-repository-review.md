# Baseline Repository Review — `iron_manufacturing`

## 1. Baseline Git & Environment State
- **Repository:** `https://github.com/mohamed-zizo-sketch/TaqaMind.git`
- **Core Baseline Commit SHA:** `3e189ddf0b92b4394a516d180dabee05f3e71f80` (`feat(core): implement local dataset and record lineage registries`, implementing Shared Core P1.1–P1.5 natively; branched initially from `develop` at `d9cdb911d094f49c887d223144746b2acad462c5` and synced cleanly onto `3e189dd`).
- **Working Branch:** `feature/iron_manufacturing/pipeline-integration`
- **Python Version:** `3.13.1` (satisfies `requires-python = ">=3.13,<3.14"`)
- **Operating System:** Windows (PowerShell)

## 2. Shared Core P1.1–P1.5 Direct Wiring vs Remaining P1.6–P1.8 Gap Adapters

| Shared Core Phase / Component | Status at Core `3e189dd` | Action Taken for `iron_manufacturing` |
|---|---|---|
| Core Contracts & `DomainRegistry` | **Implemented** | Reused directly via `register_iron_manufacturing(registry)`. |
| **P1.1 Storage:** `LocalDatasetArtifactStorage`, `LocalModelArtifactStorage` (`taqamind.core.registry`) | **Implemented in Core** | **Retired duplicate `adapters/storage.py`**; wired Core engines directly. |
| **P1.2 Ingestion & Profiling:** `CsvDataIngestion` (`taqamind.core.ingestion`), `PandasDataProfiler` (`taqamind.core.profiling`) | **Implemented in Core** | **Retired duplicate ingestion/profiling code**; wired Core engines directly. |
| **P1.3 Mapping & Validation:** `DeterministicColumnMapper` (`taqamind.core.mapping`), `ContractSemanticValidator` (`taqamind.core.validation`) | **Implemented in Core** | Wired Core engines directly with thin wrappers (`IronUnitAwareColumnMapper` for Core AUD-001/AUD-004 unit conversion & string dtype preservation + ambiguous/post-production guards; `IronSemanticValidator` for iron physical balance checks). |
| **P1.4 Dataset & Lineage Registries:** `LocalDatasetRegistry` (`taqamind.core.registry`), `LocalRecordLineageRegistry` (`taqamind.core.lineage`) | **Implemented in Core** | **Retired duplicate `adapters/dataset_registry.py`**; wired Core registries directly + focused row-to-record membership/fingerprinting helper (`services/lineage_service.py`, AUD-008 mitigation). |
| **P1.5 Cleaning, Feature Engineering & Preprocessing:** `ContractDataCleaner` (`taqamind.core.preprocessing`), `DeterministicFeatureEngineer` + `DerivedFeatureSpec` (`taqamind.core.feature_engineering`), `TabularPreprocessor` (`taqamind.core.preprocessing`) | **Implemented in Core** | **Retired duplicate `adapters/processing.py` and custom `ColumnTransformer`**; wired Core `ContractDataCleaner`, per-process `DeterministicFeatureEngineer` (via `ProcessFeatureEngineerRouter`), and train-only `TabularPreprocessor` inside `ControlledModelTrainer` / `ZeroRetrainingPredictionEngine`. |
| **P1.6 Readiness, Splitting & Leakage:** `TrainingReadinessChecker`, `DatasetSplitter`, `LeakageValidator`, `SplitRegistry`, `RecordUsageRegistry` | **Protocols only (Core gap)** | Retained protocol-compatible adapters in `adapters/readiness_and_splitting.py` and `adapters/training_registry.py`. |
| **P1.7 Controlled AutoML & Protected Evaluation:** `DevelopmentDataProvider`, `ProtectedTestDataProvider`, `TrainingRegistry`, `EvaluationRegistry`, `ModelTrainer`, `FinalModelEvaluator` | **Protocols only (Core gap)** | Retained protocol-compatible adapters in `adapters/data_access.py`, `adapters/training_registry.py`, and `adapters/modeling.py`. |
| **P1.8 Model Registry & Zero-Retraining Prediction:** `ModelRegistry`, `ActiveModelProvider`, `PredictionEngine` | **Protocols only (Core gap)** | Retained protocol-compatible adapters in `adapters/model_registry.py`, `adapters/data_access.py`, and `adapters/modeling.py`. |

## 3. Dependency, Ownership & Pre-Existing Core Notes
- Shared Core (`src/taqamind/core/`), `tests/core/`, `tests/integration/`, root `pyproject.toml`, workflows, and other domains were **never modified**.
- All changes remain strictly within `iron_manufacturing`-owned paths.
- **Pre-existing Windows `MAX_PATH` finding in Shared Core `3e189dd`:** On default Windows `AppData\Local\Temp` pytest paths, `tests/core/test_artifact_storage.py::test_model_put_exists_and_open_verified` raises `FileNotFoundError: [WinError 3]` at `src/taqamind/core/registry/artifact_storage.py:128` (`os.replace` on a 277-char path without `\\?\` prefix). Running with a short temp directory (`pytest tests/core/test_artifact_storage.py -q --basetemp=D:\tm_pt`) passes `9/9`. Left untouched in Core per read-only Core policy; `tests/domains/iron_manufacturing/conftest.py` uses a short `tempfile.TemporaryDirectory(prefix="tm_")` fixture.

## 4. Implementation & Core-Sync Checklist
- [x] Phase 0: Baseline inspection & Core `3e189dd` sync
- [x] Phase 1–2: Supplied research & raw dataset forensic audit + reproducible EDA script/notebook
- [x] Phase 3–4: Domain/Process/Task/Dataset contracts, `register_iron_manufacturing(registry)`, and customer CSV templates
- [x] Phase 5–8: Real Shared Core P1.1–P1.5 engines wired directly (duplicate `storage.py`, `dataset_registry.py`, `processing.py` retired)
- [x] Phase 9–14: Protocol-compatible P1.6–P1.8 gap adapters, train-only Core `TabularPreprocessor`, protected evaluation, and zero-retraining inference
- [x] Phase 15–18: CLI, offline automated tests (including zero-retraining spy on `TabularPreprocessor.fit`/`fit_transform`), full documentation & handover manifest
