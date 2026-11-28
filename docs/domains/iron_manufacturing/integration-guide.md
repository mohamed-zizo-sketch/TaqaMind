# Integration Guide — `iron_manufacturing`

## 1. Domain Registration
```python
from taqamind.core.registry import DomainRegistry
from taqamind.domains.iron_manufacturing import register_iron_manufacturing

registry = DomainRegistry()
domain_contract = register_iron_manufacturing(registry)
```

## 2. Public Python Entrypoints
All public symbols are exported from `taqamind.domains.iron_manufacturing`:
- `register_iron_manufacturing(registry, *, confirmations=None) -> DomainContract`
- `build_iron_manufacturing_domain(*, confirmations=None) -> DomainContract`
- `get_iron_task(process_id, task_id=None, *, confirmation=None) -> IndustrialTaskContract`
- `get_iron_dataset_contract(process_id, task_id=None, *, confirmation=None) -> DatasetContract`
- `generate_process_templates(process_id, output_dir, *, sample_rows=3)`
- `generate_all_contracts_and_templates(contracts_root=...)`
- `IronManufacturingPipeline(storage_root=..., domain_registry=..., column_overrides=..., unit_overrides=...)`:
  - `.inspect_dataset(factory_id=..., process_id=..., csv_path=...)`
  - `.validate_dataset(factory_id=..., process_id=..., csv_path=..., purpose=..., confirmation=...)`
  - `.train_initial(factory_id=..., process_id=..., csv_path=..., confirmation=..., ...)`
  - `.retrain(factory_id=..., process_id=..., csv_path=..., confirmation=..., ...)`
  - `.predict(factory_id=..., process_id=..., csv_path=..., confirmation=..., output_csv_path=...)`
  - `.get_model_info(factory_id=..., process_id=...)`
  - `.promote_model(factory_id=..., process_id=..., target_model_version_id=..., comparison=..., ...)`
  - `.rollback_model(factory_id=..., process_id=..., target_archived_version_id=..., ...)`

## 3. CLI Usage & Reproducibility Commands
```powershell
python -m taqamind.domains.iron_manufacturing.cli --help

# Generate customer templates
python -m taqamind.domains.iron_manufacturing.cli template --process eaf_meltshop

# Inspect a dataset
python -m taqamind.domains.iron_manufacturing.cli inspect `
  --factory-id 11111111-1111-4111-8111-111111111111 `
  --process eaf_meltshop --csv path/to/historical.csv

# Validate against DatasetContract
python -m taqamind.domains.iron_manufacturing.cli validate `
  --factory-id 11111111-1111-4111-8111-111111111111 `
  --process eaf_meltshop --csv path/to/historical.csv --purpose training

# Train (explicitly synthetic confirmation flag for offline demos only)
python -m taqamind.domains.iron_manufacturing.cli train `
  --factory-id 11111111-1111-4111-8111-111111111111 `
  --process eaf_meltshop --csv path/to/synthetic_train.csv --synthetic-confirmation

# Predict using saved ACTIVE model (zero retraining)
python -m taqamind.domains.iron_manufacturing.cli predict `
  --factory-id 11111111-1111-4111-8111-111111111111 `
  --process eaf_meltshop --csv path/to/synthetic_pred.csv `
  --output-csv path/to/predictions.csv --synthetic-confirmation

# Inspect registered model versions
python -m taqamind.domains.iron_manufacturing.cli model-info `
  --factory-id 11111111-1111-4111-8111-111111111111 --process eaf_meltshop
```

## 4. Local Storage & Registry Layout
Under `--storage-root` (default `artifacts/iron_manufacturing/store/`, gitignored):
- `factories/{factory_id}/datasets/{artifact_id}.csv` — immutable raw & derived CSV revisions.
- `factories/{factory_id}/training/{execution_id}/candidates/{candidate_id}/{artifact_id}.bundle` — SHA-256-verified joblib bundles.
- `partitions/{factory_id}/{split_id}/{train|validation|test}_{id}.csv` — physically separated split partitions.
- `registries/{datasets,lineage,splits,usage,training,evaluations,models}/` — atomic JSON registries.

## 5. Direct Shared Core `CorePlatformServices` Wiring (P1.1–P1.8)
Both V1 Waste (`IronManufacturingPipeline`) and V2 Electrical Energy (`IronEnergyPipelineService`) wire directly to `CorePlatformServices.create_local(storage_root)` from `taqamind.core.pipeline` for all P1.1–P1.8 engines (`LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`, `LocalDatasetRegistry`, `LocalRecordLineageRegistry`, `LocalSplitRegistry`, `LocalRecordUsageRegistry`, `LocalTrainingRegistry`, `LocalEvaluationRegistry`, `LocalModelRegistry`, `CsvDataIngestion`, `PandasDataProfiler`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`, `DeterministicFeatureEngineer`, `RecordMembershipBuilder`, `CoreTrainingReadinessChecker`, `CoreDatasetSplitter`, `CoreLeakageValidator`, `LocalDevelopmentDataProvider`, `LocalProtectedTestDataProvider`, `ControlledAutoMLTrainer`, `ProtectedFinalEvaluator`, `LocalActiveModelProvider`, and `CorePredictionEngine`).
All duplicated local P1.1–P1.8 adapters have been retired; only thin iron-specific domain wrappers remain in `adapters/unit_mapper.py`.

## 6. Error Conventions & Dependencies
- Contract/lifecycle violations raise `ValueError` or Shared Core exceptions (`ArtifactIntegrityError`, `ArtifactConflictError`, `ArtifactScopeError`, `RegistryNotFoundError`, `RegistryConflictError`, `RegistryScopeError`, `DataAccessDeniedError`).
- Operational gating issues (missing active model, unconfirmed target, validation failure) return structured `PredictionOutcome` / `TrainingWorkflowOutcome` objects with `status="blocked"` and actionable `ValidationReport` issues.
- Uses existing repository dependencies only (`numpy`, `pandas`, `pydantic`, `PyYAML`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `seaborn`, `pytest`, `ruff`); zero changes to `pyproject.toml`.
