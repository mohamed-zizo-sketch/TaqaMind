# TaqaMind — Food & Beverage Pipeline Architecture

**Package:** `src/taqamind/domains/food_beverage/`  
**Orchestrator:** `FoodBeveragePipeline` (`pipeline.py`)

---

## 1. End-to-End Stage Flow

1. **Ingestion (`CsvDataIngestion`):** Validates CSV readability, header uniqueness, computes SHA-256, stores immutable `RAW` and `INGESTED` artifacts.
2. **Profiling (`PandasDataProfiler`):** Computes column-level missing counts, unique counts, min/max bounds, and exact duplicate row counts (`ProfilingResult`).
3. **Column Mapping (`DeterministicColumnMapper`):** Matches exact canonical names and contract aliases; normalizes compatible units via `UNIT_CONVERSION_FACTORS`; rejects incompatible units or target leakage in prediction mode.
4. **Semantic Validation (`ContractSemanticValidator`):** Validates required columns, data types, finite values, physical constraints, missing policies, and prediction-time availability.
5. **Cleaning (`AuditableDataCleaner`):** Deterministic row-level filtering with audit issues; never imputes missing targets and preserves zero-waste rows.
6. **Feature Engineering (`DomainFeatureEngineer`):** Computes leak-free pre-production features (`planned_month`, `planned_dayofweek`) and strips post-production metadata.
7. **Lineage (`LocalRecordLineageRegistry`):** Assigns deterministic business-key hashes, `RecordIdentity`, `RecordVersion`, `DatasetMembershipManifest`, and tracks `RecordUsage`.
8. **Readiness & Splitting (`FoodBeverageReadinessChecker`, `FoodBeverageDatasetSplitter`, `FoodBeverageLeakageValidator`):** Enforces confirmed target semantics, creates immutable `SplitManifest` (excluding previously exposed test records), and issues `TrainingAuthorization`.
9. **Controlled Training & Protected Evaluation (`ControlledRegressionTrainer`, `ProtectedTestEvaluator`):** Fits `ColumnTransformer` strictly on `TRAIN`, evaluates candidates on `VALIDATION`, freezes `FrozenModelSelection`, and evaluates once on `TEST`.
10. **Model Registry & Inference (`LocalModelRegistry`, `FoodBeveragePredictionEngine`):** Persists versioned model bundles with SHA-256 verification and runs transform-only prediction.
