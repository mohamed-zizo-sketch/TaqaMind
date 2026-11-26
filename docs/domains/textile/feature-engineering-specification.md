# Feature Engineering Specification

All derived features are implemented in `src/taqamind/domains/textile/common/formulas.py` and registered via `DerivedFeatureSpec` in `DeterministicFeatureEngineer` (Core P1.5).

## Derived Predictors

1. **Weaving — `total_thread_density_per_inch`**
   - Formula: `EPI + PPI` (`warp_ends_per_inch + weft_picks_per_inch`)
   - Preconditions: `EPI > 0` and `PPI > 0`; invalid/missing yields `NaN` (no fabrication).
   - Leakage safety: Uses only pre-production loom specification inputs.

2. **Knitting — `knitting_tightness_factor`**
   - Formula: `sqrt(yarn_count_tex_1) / (stitch_length_mm / 10.0)`
   - Preconditions: `yarn_count_tex_1 > 0` and `stitch_length_mm > 0`; invalid/missing yields `NaN`.
   - Leakage safety: Uses only pre-production yarn linear density and planned stitch length.

## Diagnostic & Unit Utilities

- `compute_knitted_fabric_gsm(...)`: Supports both metric (`density_unit="cm"`, divisor `100.0`) and imperial (`density_unit="inch"`, divisor `645.16`) density inputs.
- `classify_setup_scrap_regime(...)`: Tags `SETUP_CHANGEOVER` vs `RUNNING_STEADY` without deleting legitimate scrap events.
- `recover_excel_yarn_count(...)`: Recovers known Jan-2022 Excel date serials (`44573`–`44591`) with explicit `requires_confirmation=True`.
