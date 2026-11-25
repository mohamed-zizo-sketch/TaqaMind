# Model Registry, Persistence & Zero-Retraining Prediction — `iron_manufacturing`

## 1. Local Model Bundle & Registry Structure
- **Bundle Format (`taqamind_iron_manufacturing_bundle_v1`):** Joblib-compressed dictionary storing exact `feature_names` (ordered), `target_name`, `target_unit`, `candidate_family`, runtime `dependency_versions`, and the fitted `_TabularModelWrapper` pairing Shared Core `TabularPreprocessor` (fit train-only; `.transform()` only during evaluation and prediction) with the regression estimator.
- **Storage Integrity (`LocalModelArtifactStorage`):** Stored immutably under `factories/{factory_id}/training/{execution_id}/candidates/{candidate_id}/{artifact_id}.bundle` with full SHA-256 and byte-size verification before every read. Corrupted bytes raise `ArtifactIntegrityError`. Only internally registered trusted bundles are ever deserialized.
- **Registry Lifecycle (`LocalModelRegistry`):**
  - States: `CANDIDATE` → `VALIDATED` or `REJECTED` → `ACTIVE` → `ARCHIVED`.
  - Enforces **at most one `ACTIVE` model** per `(factory_id, domain, process, task_id)`.
  - `REJECTED` models can never transition to `ACTIVE`.
  - Atomic JSON persistence with full transition audit trail (`first_activation`, `promotion`, `rollback`).

## 2. Mandatory Train-Once / Predict-Many Architecture
- `IronPredictionService` (`services/prediction_service.py`) has **zero imports** from `training_service.py` or `ControlledModelTrainer`.
- Prediction execution steps:
  1. Resolve `(factory_id, domain, process, task_id)` via `ActiveModelProvider.get_active(...)`. If no `ACTIVE` model exists, return `status="blocked"` immediately — **never triggers training**.
  2. Verify persisted `contract_sha256` matches current `DatasetContract`.
  3. Run `DataIngestion -> ColumnMapper -> SemanticValidator -> DataCleaner -> FeatureEngineer`.
  4. Match rows via `RecordLineageRegistry` using trusted business keys to distinguish `previously_known_record_count` vs `new_record_count` without retraining.
  5. Open SHA-256-verified bundle via `ActiveModelProvider.open_bundle(...)`, verify exact saved feature ordering, and call **only** `.predict(X_pred)`.
  6. Validate contract target bounds, compute `DerivedRatePrediction` only when denominator is finite and strictly positive (otherwise populate `undefined_reason`), and return `PredictionBatchResult` (no fabricated confidence percentages).

## 3. Explicit Retraining, Reuse, Promotion & Rollback
- Retraining requires an explicit `TrainingIntent(mode=TrainingMode.RETRAINING, previous_model_version_id=...)`.
- `TrainingReuseKey` hashes factory/task identity, engineered dataset content SHA-256, contract SHA-256, exact split+record-version fingerprint, effective training configuration+policy fingerprint, and implementation/dependency fingerprint. Equivalent completed requests reuse verified results instead of redundant retraining.
- Retraining **never** replaces the active model automatically: a rejected challenger leaves the current `ACTIVE` model untouched, and a `VALIDATED` challenger requires explicit `promote_model(...)` with a passing `PromotionComparison` on the same independent test records.
- `rollback_model(...)` restores any previously validated `ARCHIVED` version with compatible contract hash.
