# TaqaMind Shared Core P1.6–P1.9 Completion Report & Domain Migration Guide

**Branch:** `feature/core/ml-engines` (branched from `develop` @ `e574ae8`)
**Scope:** Shared Core ML Engines (`src/taqamind/core/`), Core Unit & E2E Tests (`tests/core/`), and Core Documentation (`docs/development/`).
**Zero Domain Edits:** `src/taqamind/domains/*`, `tests/domains/*`, and `pyproject.toml` were intentionally left untouched.

---

## 1. Delivered Shared Core Capabilities

### Phase P1.6 — Training Readiness, Splitting, Leakage Prevention & Split Registries
- **`CoreTrainingReadinessChecker`** ([readiness_checker.py](file:///E:/TaqaMind/src/taqamind/core/readiness/readiness_checker.py))
  - Verifies confirmed target (`ConfirmationStatus.CONFIRMED`), `DatasetState.ENGINEERED`, registered `DatasetMembershipManifest`, non-empty usable target/predictors, and sufficient rows.
- **`CoreDatasetSplitter`** ([dataset_splitter.py](file:///E:/TaqaMind/src/taqamind/core/splitting/dataset_splitter.py))
  - Supports `GROUP`, `TEMPORAL`, `GROUP_TEMPORAL`, and `RANDOM` split strategies using `SafeCsvReader` and `RecordLineageRegistry` without mutating raw/engineered artifacts.
- **`CoreLeakageValidator`** ([leakage_validator.py](file:///E:/TaqaMind/src/taqamind/core/splitting/leakage_validator.py))
  - Evaluates all 8 `LeakageCheck` gates (partition overlap, group overlap, temporal ordering, target confirmation, predictor availability, engineered target exclusion, historical protected test exposure, and duplicate record identities).
- **`LocalSplitRegistry` & `LocalRecordUsageRegistry`** ([split_registry.py](file:///E:/TaqaMind/src/taqamind/core/splitting/split_registry.py))
  - Persists immutable `SplitManifest`, `LeakageValidationResult`, `TrainingAuthorization`, and `RecordUsage` history using `locked_path` and `dump_json_model`.

### Phase P1.7 — Restricted Data Access, Training/Evaluation Registries, AutoML & Protected Evaluator
- **`LocalDevelopmentDataProvider` & `LocalProtectedTestDataProvider`** ([data_access.py](file:///E:/TaqaMind/src/taqamind/core/automl/data_access.py))
  - Enforces strict column allowlisting (`predictor_names | {target}`) and partition isolation; records `RecordUsage` (`TRAIN`, `VALIDATION`, `PROTECTED_TEST`) before yielding bytes.
- **`LocalTrainingRegistry` & `LocalEvaluationRegistry`** ([training_registry.py](file:///E:/TaqaMind/src/taqamind/core/automl/training_registry.py))
  - Tracks `TrainingIntent`, content-based `TrainingReuseKey`, `TrainingRunResult`, single-use `EvaluationReservation`, and `FinalEvaluationResult`.
- **`ControlledAutoMLTrainer` & `TabularPreprocessor`** ([trainer.py](file:///E:/TaqaMind/src/taqamind/core/automl/trainer.py))
  - Fits `TabularPreprocessor` strictly on the training partition (`fit` on train only; `transform` on validation/test/prediction).
  - Supports all 8 allowlisted `ModelFamily` baselines and challengers (`DUMMY_REGRESSOR`, `HISTORICAL_RATE`, `RIDGE`, `ELASTIC_NET`, `RANDOM_FOREST`, `HIST_GRADIENT_BOOSTING`, `HURDLE_REGRESSION`, and `XGBOOST` with clean skip if optional dependency is absent).
- **`ProtectedFinalEvaluator`** ([evaluator.py](file:///E:/TaqaMind/src/taqamind/core/evaluation/evaluator.py))
  - Reserves protected test access once, exposes protected test records atomically (single-use), scores baseline + frozen challenger, and enforces `AcceptancePolicy`.

### Phase P1.8 — Model Registry, Zero-Retraining Prediction Engine & Unified Core Orchestrator
- **`LocalModelRegistry`** ([model_registry.py](file:///E:/TaqaMind/src/taqamind/core/registry/model_registry.py))
  - Factory- and task-scoped `ModelRegistry` with SHA-256 verification on registration/activation, single-active-model invariant, `FIRST_ACTIVATION` / `PROMOTION` / `ROLLBACK` lifecycle transitions, and append-only `transitions.jsonl` audit log.
- **`LocalActiveModelProvider` & `CorePredictionEngine`** ([prediction_engine.py](file:///E:/TaqaMind/src/taqamind/core/prediction/prediction_engine.py))
  - Enforces `ModelStatus.ACTIVE`, verifies SHA-256 integrity and predictor feature order, transforms input data without ever calling `.fit()`, and computes optional `DerivedRatePrediction`.
- **`CorePlatformServices`** ([pipeline.py](file:///E:/TaqaMind/src/taqamind/core/pipeline.py))
  - Injectable local platform wiring P1.1–P1.8 under a single `root_dir`.

### Windows & Pydantic Hardening
- **`_concurrency.py`:** `locked_path` uses `Path(os.path.abspath(lock_path))` and retries `(FileExistsError, PermissionError)` to avoid Windows NTFS delete-pending / `\$Extend\$Deleted` race conditions.
- **`_json_store.py`:** `dump_json_model` recursively strips `@computed_field` keys before JSON persistence so `model_validate_json` reloads cleanly with `extra="forbid"`.

---

## 2. Domain Migration Guide (for `iron_manufacturing` & `textile`)

Domain teams can progressively replace domain-local adapters with Shared Core engines by constructing `CorePlatformServices.create_local(root_dir)` and passing their domain `DatasetContract`:

1. **Readiness, Splitting & Leakage:** Replace domain-specific splitters/leakage validators with `services.readiness_checker`, `services.splitter`, `services.leakage_validator`, `services.split_registry`, and `services.usage_registry`.
2. **Data Access & Training:** Replace domain-local `DevelopmentDataProvider` / `ModelTrainer` adapters with `services.development_data_provider`, `services.training_registry`, and `services.trainer`.
3. **Protected Evaluation & Registry:** Replace domain-local evaluators and model registries with `services.create_evaluator(contract)`, `services.evaluation_registry`, and `services.model_registry`.
4. **Prediction:** Replace domain-local prediction engines with `services.active_model_provider` and `services.prediction_engine`.
