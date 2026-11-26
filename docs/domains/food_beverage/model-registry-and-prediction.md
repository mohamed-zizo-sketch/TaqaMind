# TaqaMind — Model Registry and No-Retraining Prediction

---

## 1. Factory-Scoped Model Registry (`LocalModelRegistry`)

- **Isolation Scope:** `(factory_id, domain, process, task_id)`.
- **Lifecycle States:** `CANDIDATE` -> `VALIDATED` -> `ACTIVE` -> `ARCHIVED` (or `REJECTED`).
- **Rules:**
  - Maximum one `ACTIVE` model per factory/process/task scope.
  - `REJECTED` models are permanently blocked from activation.
  - Retraining preserves the existing `ACTIVE` model until explicit promotion approval.
  - Every model bundle stores `preprocessor`, `estimator`, `feature_names`, `target_name`, `target_unit`, and is verified by SHA-256 checksum prior to `joblib.load`.

## 2. Transform-Only Prediction (`FoodBeveragePredictionEngine`)

- **No-Retraining Guarantee:** Prediction never calls `fit`, `fit_transform`, `partial_fit`, or `ModelTrainer.train`.
- **Execution Steps:** Resolve `ACTIVE` model -> validate scope & contract version -> verify bundle SHA-256 -> align exact feature column order -> `preprocessor.transform(X)` -> `estimator.predict(X_t)` -> validate finite & non-negative predictions -> compute `DerivedRatePrediction` only when denominator is positive and finite.
