# Textile Domain — Development Handover

## Status Summary

- **Branch:** `feature/textile/pipeline-integration` (branched from `3e189dd`; NOT merged).
- **Processes Implemented:** `weaving` (`weaving_waste_regression`) and `knitting` (`knitting_waste_regression`).
- **Scope:** V1 pre-production waste quantity regression (`Train Per Factory`), with deterministic derived rate calculation.
- **Shared Core Integrity:** Zero modifications to `src/taqamind/core/`, `tests/core/`, or `tests/integration/`.

## Verification Results

- `python -m pytest tests/domains/textile -v`: **17 passed** (including `test_no_retraining.py` proving zero `.fit` / `.fit_transform` calls across 3+ prediction requests & process reload, and `test_end_to_end.py` training real scikit-learn/XGBoost estimators).
- `python -m pytest tests/domains/textile --cov=taqamind.domains.textile --cov-report=term-missing`: **81% coverage** across `taqamind.domains.textile`.
- `python -m ruff check src tests --no-cache`: **All checks passed**.
- `python -m taqamind.domains.textile.cli --help` + CLI E2E smoke test (`train` -> `model-info` -> `predict`): **Passed** (`hist_gradient_boosting` activated with 8.67% MAE improvement on synthetic weaving).

## Critical Guardrails for Production Onboarding

1. **Confirm Target Semantics First:** Production contracts ship with `TargetConfirmation.UNCONFIRMED`. Do not activate production factory models until industrial owners formally sign off on target definitions and unit provenance.
2. **Real Dataset Exploratory Only:** Raw reference datasets (`full_weaving_dataset.csv`, `knitting_dataset.csv`, `raw_data_production_reports_20222.xlsx`) contain unresolved anomalies documented in [dataset-audit.md](file:///E:/TaqaMind/docs/domains/textile/dataset-audit.md) and [limitations-and-research-gaps.md](file:///E:/TaqaMind/docs/domains/textile/limitations-and-research-gaps.md).
3. **Adapter Retirement:** Replace `src/taqamind/domains/textile/adapters/` with Shared Core P1.6–P1.8 implementations as they land (see [core-contribution-proposal.md](file:///E:/TaqaMind/docs/domains/textile/core-contribution-proposal.md)).
