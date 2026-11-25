# Training, Splitting, Leakage Prevention & Protected Evaluation — `iron_manufacturing`

## 1. Train-Per-Factory Policy
Every initial model is trained exclusively on the labeled historical dataset of one factory (`factory_id`), one process, and one industrial task. No foreign pretrained model is deployed across Egyptian plants.

## 2. Training Readiness Gate (`LocalTrainingReadinessChecker`)
Training is **blocked** before splitting whenever any of the following hold:
- `contract.target.is_confirmed` is `False` (default for all production contracts until factory sign-off).
- Target column is missing, contains nulls/non-finite values, has zero variance, or has fewer than 5 positive observations.
- Required pre-production predictors are missing.
- Record membership manifest is missing, contains repeated `record_id`s, or contains unconfirmed/ambiguous identities.
- Dataset has fewer than 30 rows or fewer than 3 distinct groups for group-disjoint splitting.

## 3. Leakage-Safe Splitting & Verification
- **Preferred Strategy:** `SplitStrategy.GROUP_TEMPORAL` (chronologically orders groups such as `production_order_id` or `supplier_group_id` into 70/15/15 Train/Validation/Protected-Test partitions).
- **Leakage Checks Enforced (`LocalLeakageValidator`):**
  1. `RECORD_OVERLAP` — zero overlapping `record_id`s across partitions.
  2. `RECORD_COVERAGE` — every row in `DatasetMembershipManifest` accounted for.
  3. `RECORD_IDENTITY` — every split record has `RecordIdentityStatus.CONFIRMED`.
  4. `DUPLICATE_CONTENT` — zero shared `record_version_id` across partitions.
  5. `GROUP_OVERLAP` — zero shared `production_order_id` / `supplier_group_id` across partitions.
  6. `TEMPORAL_ORDER` — strict monotonic non-lookahead ordering.
  7. `HISTORICAL_USAGE` — conservative V1 policy: zero previously exposed `record_id`s (including corrected versions of prior records!) may enter a new protected test partition.
  8. `FEATURE_AVAILABILITY` — every predictor is available at `PRE_PRODUCTION`.

## 4. Controlled Candidate Training (`ControlledModelTrainer`)
- Receives **only** `DevelopmentDataProvider` (Train + Validation CSVs materialized separately; protected test bytes are never accessible to the trainer).
- **Train-Only Learned Preprocessing:** `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore")`, and `StandardScaler` are fitted inside scikit-learn `Pipeline`s strictly on the `TRAIN` partition and applied via `.transform()` on `VALIDATION` and `TEST`.
- **Mandatory Baselines:** `DummyRegressor(strategy="median")` + `HistoricalRateRegressor` (only when a valid pre-production unit-compatible denominator is defined).
- **Allowlisted Challengers:** `Ridge`, `ElasticNet`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `XGBRegressor`, plus `TwoStageHurdleRegressor` **only when zero-target fraction >= 15%**.
- **Metrics:** Primary `MAE`; secondary `RMSE`, `WAPE`, `R2` (with explicit `undefined_reason` when variance or denominator sum is zero; never calls `R2` classification accuracy).

## 5. Protected Final Evaluation (`ProtectedFinalModelEvaluator`)
- Predefined `AcceptancePolicy` must be created **before** `PipelineContext.created_at`.
- `FrozenModelSelection.from_training_run(...)` freezes best baseline and best challenger **before** reserving protected test data.
- `EvaluationRegistry.reserve_test_access(...)` verifies the original registered `SplitManifest` and prevents competing overlapping test reservations.
- `ProtectedTestDataProvider.open_test(...)` persists `RecordUsageRole.PROTECTED_TEST` **before** releasing bytes so exposure survives crashes.
- Exact frozen baseline and challenger bundles are loaded from `ModelArtifactStorage` and evaluated **without refitting**. Models failing the acceptance policy are honestly marked `REJECTED` and blocked from activation.
