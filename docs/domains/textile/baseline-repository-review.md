# Textile Domain — Baseline Repository Review

## Baseline Git & Environment Verification

- **Base Commit:** `3e189dd` (`feat(core): implement local dataset and record lineage registries`)
- **Working Branch:** `feature/textile/pipeline-integration`
- **Python Version:** `3.13.17` (`.venv` at `E:\TaqaMind\.venv`)
- **Dependency Health:** `python -m pip check` passed with zero conflicts.
- **Baseline Test Suite:** `815 passed` (`pytest -q` before textile implementation).
- **Baseline Linter:** `ruff check src tests --no-cache` — all checks passed.

## Shared Core Baseline Capabilities (P1.1–P1.5)

- **P1.1 Local Storage:** `LocalDatasetArtifactStorage`, `LocalModelArtifactStorage` with SHA-256 verification and atomic writes.
- **P1.2 Ingestion & Profiling:** `CsvDataIngestion`, `PandasDataProfiler`.
- **P1.3 Mapping & Semantic Validation:** `DeterministicColumnMapper`, `ContractSemanticValidator`.
- **P1.4 Dataset & Lineage Registries:** `LocalDatasetRegistry`, `LocalRecordLineageRegistry`.
- **P1.5 Cleaning, Engineering & Preprocessing:** `ContractDataCleaner`, `DeterministicFeatureEngineer`, `TabularPreprocessor` (train-only fitting).

## Known Core Audit Findings Worked Around in Textile Adapters

- **AUD-001:** `DeterministicColumnMapper.apply_mapping` does not convert units; mitigated via `UnitAwareColumnMapperAdapter`.
- **AUD-008:** Core lineage lacks a built-in row-to-record membership builder; mitigated via `TextileRowMembershipBuilder`.
- **Missing Core P1.6–P1.8 Engines:** Readiness, splitting, leakage validation, AutoML training, protected evaluation, model registry, and prediction engine are implemented as replaceable adapters under `src/taqamind/domains/textile/adapters/`.
