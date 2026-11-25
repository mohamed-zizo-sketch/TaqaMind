# Owner Handover Summary — `iron_manufacturing`

- **Domain ID:** `iron_manufacturing` (`v1.0.0`)
- **Git Branch:** `feature/iron_manufacturing/pipeline-integration`
- **Baseline Core Commit:** `d9cdb911d094f49c887d223144746b2acad462c5`
- **Shared Core / Other Domains Modified:** **None** (100% strictly within owned paths).

## Delivered Package Overview
1. **Contracts & Domain Registration (`src/taqamind/domains/iron_manufacturing/domain.py`, `processes/`, `contracts/domains/iron_manufacturing/`):**
   - Authoritative Shared Core contracts for `eaf_meltshop`, `rebar_hot_rolling`, and `scrap_yard_reception`.
   - `register_iron_manufacturing(registry)` entrypoint tested against `DomainRegistry`.
   - Customer historical training CSV templates, prediction-only CSV templates, feature dictionaries, mapping examples, and `domain_contract_v1.yaml`.
2. **Protocol-Compatible Local Adapters (`src/taqamind/domains/iron_manufacturing/adapters/`):**
   - Full local implementations of all Shared Core storage, registry, processing, lineage, readiness, splitting, leakage, data access, training, protected evaluation, and zero-retraining prediction protocols.
   - Windows long-path safe (`\\?\`) for nested artifact keys.
   - Full migration blueprint in [core-contribution-proposal.md](./core-contribution-proposal.md).
3. **Services, Pipeline Facade & CLI (`services/`, `pipeline.py`, `cli.py`):**
   - Strict architectural isolation between training (`IronTrainingService`) and zero-retraining inference (`IronPredictionService`).
   - CLI supporting `inspect`, `template`, `validate`, `train`, `predict`, `retrain`, `model-info`, and `rollback`.
4. **Research & EDA (`research/iron_manufacturing/audit_raw_datasets.py`, `notebooks/iron_manufacturing/01_iron_dataset_audit_and_eda.ipynb`):**
   - Reproducible SHA-256 forensic audit & diagnostic plotting for all 5 supplied raw datasets.
5. **Automated Tests (`tests/domains/iron_manufacturing/`):**
   - Deterministic offline contract, protocol conformance, mapping, unit, lineage, isolation, readiness, leakage, train-only preprocessing, protected evaluation, corruption detection, mandatory zero-retraining spy test, explicit retraining/promotion/rollback, and CLI end-to-end tests.
6. **Machine-Readable Manifest:** See [handover-manifest.json](./handover-manifest.json) for full metadata, entrypoints, adapters, actual test/lint/coverage metrics, and open blockers.

## Recommended Next Integration Steps for Repository Owner
1. Review `git diff --stat develop..HEAD` to confirm zero shared Core or other-domain modifications.
2. Run verification commands listed in `handover-manifest.json`.
3. Select reusable adapters from `core-contribution-proposal.md` for promotion into `taqamind.core.*`.
4. Obtain signed-off Egyptian partner factory historical CSVs and target confirmations before enabling production training.
