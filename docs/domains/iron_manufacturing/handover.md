# Owner Handover Summary — `iron_manufacturing`

- **Domain ID:** `iron_manufacturing` (`v1.0.0`)
- **Git Branch:** `feature/iron_manufacturing/pipeline-integration`
- **Baseline Core Commit:** `3e189ddf0b92b4394a516d180dabee05f3e71f80` (Shared Core P1.1–P1.5)
- **Shared Core / Other Domains Modified:** **None** (100% strictly within owned paths).

## Delivered Package Overview
1. **Contracts & Domain Registration (`src/taqamind/domains/iron_manufacturing/domain.py`, `processes/`, `contracts/domains/iron_manufacturing/`):**
   - Authoritative Shared Core contracts for `eaf_meltshop`, `rebar_hot_rolling`, and `scrap_yard_reception`.
   - `register_iron_manufacturing(registry)` entrypoint tested against `DomainRegistry`.
   - Customer historical training CSV templates, prediction-only CSV templates, feature dictionaries, mapping examples, and `domain_contract_v1.yaml`.
2. **Real Shared Core P1.1–P1.5 Integration + P1.6–P1.8 Gap Adapters (`src/taqamind/domains/iron_manufacturing/pipeline.py`, `adapters/`):**
   - Duplicate P1.1–P1.5 adapters (`storage.py`, `dataset_registry.py`, `processing.py`) **retired**.
   - Directly wires Shared Core `LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`, `LocalDatasetRegistry`, `LocalRecordLineageRegistry`, `CsvDataIngestion`, `PandasDataProfiler`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`, `DeterministicFeatureEngineer` (`DerivedFeatureSpec`), and train-only `TabularPreprocessor`.
   - Keeps local adapters strictly for Core P1.6–P1.8 gaps and focused Core audit mitigations (`AUD-001`, `AUD-004`, `AUD-008`, and recursive `ValidationReport` `@computed_field` stripping).
   - Full migration blueprint in [core-contribution-proposal.md](./core-contribution-proposal.md).
3. **Services, Pipeline Facade & CLI (`services/`, `pipeline.py`, `cli.py`):**
   - Strict architectural isolation between training (`IronTrainingService`) and zero-retraining inference (`IronPredictionService`).
   - CLI supporting `inspect`, `template`, `validate`, `train`, `predict`, `retrain`, `model-info`, and `rollback`.
4. **Research & EDA (`research/iron_manufacturing/audit_raw_datasets.py`, `notebooks/iron_manufacturing/01_iron_dataset_audit_and_eda.ipynb`):**
   - Reproducible SHA-256 forensic audit & diagnostic plotting for all 5 supplied raw datasets.
5. **Automated Tests (`tests/domains/iron_manufacturing/`):**
   - Deterministic offline contract, Core P1.1–P1.5 wiring, protocol conformance, mapping, unit conversion, lineage, isolation, readiness, leakage, train-only `TabularPreprocessor` fitting, protected evaluation, corruption detection, mandatory zero-retraining spy test (covering `TabularPreprocessor.fit`/`fit_transform`), explicit retraining/promotion/rollback, and CLI end-to-end tests.
6. **Machine-Readable Manifest:** See [handover-manifest.json](./handover-manifest.json) for full metadata, entrypoints, retired vs remaining adapters, actual test/lint/coverage metrics, and open blockers.

## Recommended Next Integration Steps for Repository Owner
1. Review `git diff --stat 3e189dd..HEAD` to confirm zero Shared Core or other-domain modifications.
2. Run verification commands listed in [handover-manifest.json](./handover-manifest.json).
3. Promote remaining P1.6–P1.8 adapters from [core-contribution-proposal.md](./core-contribution-proposal.md) into `taqamind.core.*`.
4. Obtain signed-off Egyptian partner factory historical CSVs and target confirmations before enabling production training.
