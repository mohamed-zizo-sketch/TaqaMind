# Model Registry & Prediction Lifecycle

## Registry Invariants (`LocalModelRegistry`)

- Scoped by `(factory_id, domain, process, task_id)`.
- Enforces at most **one** `ACTIVE` model per scope.
- Transitions (`FIRST_ACTIVATION`, `PROMOTION`, `ROLLBACK`) are validated via `validate_model_transition` and recorded in an audit log.
- Rejected models (`EvaluationDecision.REJECTED`) cannot transition to `ACTIVE`.

## Train-Once / Predict-Many (`TextilePredictionEngine`)

1. Resolves active `ModelRegistryEntry` via `LocalActiveModelProvider`.
2. Verifies bundle SHA-256 checksum via `LocalModelArtifactStorage.open_verified`.
3. Validates `bundle["feature_names"] == active.version.feature_specification.feature_names`.
4. Executes `preprocessor.transform(...)` and `estimator.predict(...)` **only** — zero calls to `.fit`, `.fit_transform`, or `.partial_fit`.
5. Computes `DerivedRatePrediction` with explicit `undefined_reason` when the planned quantity denominator is missing or `<= 0`.
