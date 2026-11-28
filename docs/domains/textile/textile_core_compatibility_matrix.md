# Textile — Shared Core Compatibility Matrix (Task 04)

- **Domain:** `textile` (`weaving`, `knitting`)
- **Development Branch:** `develop` @ `5455cc8`
- **Shared Core Reference:** `feature/core/v1-integration-audit` @ `849027d` (WP00–WP04)

---

## 1. Component Classification (A / B / C / D)

| Component | Classification | Rationale & WP10 Action |
|---|---|---|
| `UnitAwareColumnMapperAdapter` (`adapters/unit_mapper.py`) | **A** (Genuine Textile-specific) + **C** (Internal CSV read) | Keep AUD-001 unit conversion (`m <-> yds`, `g <-> kg`, etc.) until Core natively supports unit-aware mapping (`AUD-001`). In WP10, replace internal `pd.read_csv(..., keep_default_na=True)` with Core `SafeCsvReader` + `read_verified_csv` + `normalize_text_missing`. |
| `common/units.py` | **A** (Genuine Textile-specific) | Textile physical unit conversion table (`yds`, `m`, `kg`, `g`, `tex`, `dtex`, `den`, etc.). |
| `common/formulas.py` | **A** (Genuine Textile-specific) | Domain industrial formulas (`compute_weaving_thread_density_sum`, `compute_knitting_tightness_factor`, `compute_knitted_fabric_gsm`, `classify_setup_scrap_regime`). |
| `TextilePlatformStores` (`pipeline.py`) | **B** (Shared Core reused) | Already wired to `CorePlatformServices.create_local(...)` via PR #15. |
| `TextileEnergyPipelineService` (`energy_pipeline.py`) | **B** (Shared Core reused) | Already wired to `CorePlatformServices.create_local(...)` via PR #12; replace `Path.read_bytes()` with bounded `SafeCsvReader` / `CsvResourcePolicy` in WP10. |
| `UnaryOperator.SQRT` in `RowLocalExpr` | **D** (Missing from Shared Core) | Required for `knitting_tightness_factor`; requested via `CIR-TEXTILE-001`. |

---

## 2. Team Integration Register (`TI-001` – `TI-020`) Readiness

| ID | Requirement | Source | Textile Current State | WP10 Action |
|---|---|---|---|---|
| **TI-001** | Adopt `SafeCsvReader` | WP01 | `unit_mapper.py:127` uses `pd.read_csv` | Replace with `SafeCsvReader` + `read_verified_csv` |
| **TI-002** | Enforce CSV Resource Policies | WP01 | `energy_pipeline.py:172` uses `read_bytes()`; `unit_mapper.py` unbounded | Apply `resolve_internal_csv_policy` |
| **TI-003** | Use verified artifact reads | WP01 | `unit_mapper.py` uses `open_verified`, `energy_pipeline.py` reads raw path | Use `read_verified_csv` consistently |
| **TI-004** | Preserve lexical identifiers | WP01 | `unit_mapper.py` uses `_string_dtype_hints` | Align with `SafeCsvReader` string preservation |
| **TI-005** | Preserve `Int64` precision | WP01 | Handled by Core once `SafeCsvReader` is active | Verified in WP10 |
| **TI-006** | Preserve datetime nanoseconds | WP01 | Handled by Core `ContractTypeConverter` | Verified in WP10 |
| **TI-007** | Use shared type conversion | WP01 | Handled by Core `convert_dataframe_by_contract` | Verified in WP10 |
| **TI-008** | Use shared missing-value normalization | WP01 | `unit_mapper.py:127` uses `keep_default_na=True` | Switch to `normalize_text_missing` |
| **TI-009** | Reject invalid values before processing | WP01 | `unit_mapper.py:137` used `errors="coerce"` | Hardened to `errors="raise"` in P0 |
| **TI-010** | Migrate legacy feature callbacks | WP01 | 4 `DerivedFeatureSpec.compute` callbacks | Migrate `total_thread_density_per_inch` & nameplate kWh; `SQRT` via `CIR-TEXTILE-001` |
| **TI-011** | Validate derived feature outputs | WP01 | Handled by Core `DeterministicFeatureEngineer` | Verified in WP10 |
| **TI-012** | Prevent unverified precomputed features | WP01 | Handled by Core `DeterministicFeatureEngineer` | Verified in WP10 |
| **TI-013** | Enforce numerical safety | WP01 | Hardened `unit_mapper.py` finite/numeric check | Verified in WP10 |
| **TI-014** | Prevent pre-split aggregate leakage | WP01 | All Textile derived features are row-local | Confirmed row-local |
| **TI-015** | Unified business key canonicalization | WP02 | Handled by Core `RecordMembershipBuilder` | Synced in WP10 |
| **TI-016** | Fingerprint version compatibility | WP02 | Handled by Core `LocalRecordLineageRegistry` | Synced in WP10 |
| **TI-017** | Historical content fingerprint schema | WP02 | Handled by Core `v2.0.0` | Synced in WP10 |
| **TI-018** | Separate prediction observations | WP02 | `pipeline.py:350` & `energy_pipeline.py:296` call `build_and_register` on `PREDICTION` | Route `PREDICTION` to `resolve_prediction_observations` in WP10 |
| **TI-019** | Complete record scope verification | WP02 | Handled by Core `RecordMembershipBuilder` | Synced in WP10 |
| **TI-020** | Preserve legacy record identities | WP02 | Handled by Core `1.0.0` compatibility | Synced in WP10 |
