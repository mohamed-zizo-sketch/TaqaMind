# Textile Domain — Development Handover

## Status Summary

- **Processes Implemented:** `weaving` and `knitting`.
- **V1 Scope (Implemented & Tested):** Pre-production waste quantity regression (`weaving_waste_regression`, `knitting_waste_regression` under `PlatformModule.WASTE_MANAGEMENT`) with `Train-per-Factory`, leakage guards, and deterministic derived rate calculation.
- **V2 Scope (Implemented & Verified on Synthetic Engineering Fixtures):** Pre-production electrical energy regression (`weaving_energy_regression`, `knitting_energy_regression` under `PlatformModule.ENERGY_EFFICIENCY`), documented in [v2-technical-handover.md](file:///E:/TaqaMind/docs/domains/textile/v2-technical-handover.md).
- **Shared Core Integration:** Both V1 ([pipeline.py](file:///E:/TaqaMind/src/taqamind/domains/textile/pipeline.py)) and V2 ([energy_pipeline.py](file:///E:/TaqaMind/src/taqamind/domains/textile/energy_pipeline.py)) wire directly to `CorePlatformServices` (`src/taqamind/core/pipeline.py`). Only [unit_mapper.py](file:///E:/TaqaMind/src/taqamind/domains/textile/adapters/unit_mapper.py) (`UnitAwareColumnMapperAdapter`) remains in `adapters/` for textile-specific imperial length conversion (`m <-> yd`).
- **Shared Core Integrity:** Zero modifications to `src/taqamind/core/`.

## Verification Results

- `pytest tests/domains/textile -v`: **22 passed** (17 V1 tests + 5 V2 Energy tests), including zero-retraining verification across repeated predictions & process reload, unit conversion & count-to-length blocking, leakage checks, and factory isolation.
- `ruff check src tests --no-cache`: **All checks passed** (0 errors).

## Critical Guardrails for Production Onboarding

1. **Confirm Target Semantics First:** Production contracts ship with `TargetConfirmation.UNCONFIRMED`. Do not activate production factory models until industrial owners formally sign off on target definitions (e.g., whether `Rej_and_cut_Piece` is measured in `yd` vs `pieces`) and unit provenance.
2. **Real Dataset Exploratory Only:** Raw reference datasets (`full_weaving_dataset.csv`, `knitting_dataset.csv`, `raw_data_production_reports_20222.xlsx`) contain unresolved anomalies documented in [dataset-audit.md](file:///E:/TaqaMind/docs/domains/textile/dataset-audit.md) and [limitations-and-research-gaps.md](file:///E:/TaqaMind/docs/domains/textile/limitations-and-research-gaps.md).
