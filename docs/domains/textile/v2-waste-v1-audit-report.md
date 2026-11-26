# TaqaMind V2 — Phase 0: Textile Waste Prediction V1 Final Audit Report

**Audit Date:** 2026-11-26  
**Audited Branch / Commit:** `origin/develop` @ `4cdcb58e3c7c6dd8b80a3734c999b311bf85e9bc` (inspected from `feature/textile/energy-consumption-v2`)  
**Scope:** Independent technical audit of `src/taqamind/domains/textile/` (Weaving & Knitting Waste V1) and compatibility with Shared Core P1.1–P1.9 (`CorePlatformServices`).

---

## 1. Baseline Verification Results

| Check | Command | Result |
| :--- | :--- | :--- |
| **Textile Unit & E2E Tests** | `pytest tests/domains/textile -v` | **17 passed** in `84.27s` |
| **Ruff Lint (Textile)** | `ruff check src/taqamind/domains/textile tests/domains/textile --no-cache` | **All checks passed** (0 errors) |
| **Shared Core Compatibility** | `AUD-001` unit confirmation (`PR #9`) | **Verified compatible** |

---

## 2. Independent Process Audit

### A. Weaving (`weaving_waste_regression`)
- **Dataset & Provenance:** Reference CSV `full_weaving_dataset.csv` (`121,148` rows × `18` columns) documented in [dataset-audit.md](file:///E:/TaqaMind/docs/domains/textile/dataset-audit.md). Exact duplicate rows: `74,448` (`46,700` unique rows; max repeat `156`). `1,009` unique `ID` values (`693` show varying targets across rows).
- **Target & Anomalies:** `Rej_and_cut_Piece` -> canonical `waste_length_yds` (`yd`), `105,721` zeros and `15,427` positive rows (`max = 11,353.0 yds`, skewness `10.86`). `140` rows have `Rej_and_cut_Piece > Req_beam_length(yds)` (up to `129.77x`).
- **Pre-Production Leakage Guard:** `WEAVING_POST_PRODUCTION_EXCLUDED_COLUMNS` explicitly excludes `act_shrink%`, `Total_pdn_m/c`, `Total_pdn_per_order`, and `Previous_pdn`.
- **Feature Engineering:** `compute_weaving_thread_density_sum` (`warp_density + weft_density`), derived strictly from pre-production predictors.
- **Splitting:** `SplitStrategy.GROUP` (`order_id`, `0.70 / 0.15 / 0.15`) — correctly chosen because real weaving logs only contain month names (`January`–`September`) without calendar dates/years.

### B. Knitting (`knitting_waste_regression`)
- **Dataset & Provenance:** Reference `knitting_dataset.csv` (`12,569` rows × `38` columns) and `raw_data_production_reports_20222.xlsx` (`12,569` data rows × `21` headers). `6` exact duplicates.
- **Target & Anomalies:** `Reject Qty` -> canonical `waste_qty_kg` (`kg`), `4,449` invalid/null (`4,448` NaN + `1` non-numeric string), `7,822` zeros, `298` positive rows (`max = 41.0 kg`). `10` rows exceed `100%` of `Shift Total`; `959` rows have `Shift Total <= 5 kg`. Excel `Yarn Count` contains `5,150` date-corrupted tokens (recovered deterministically via `common/formulas.py` / `test_excel_yarn_count_recovery`).
- **Pre-Production Leakage Guard:** `KNITTING_POST_PRODUCTION_EXCLUDED_COLUMNS` excludes `Shift Total`, `Machine Total`, `SHIFT-A`, `SHIFT-B`, `SHIFT-C`, `Without Shift(Outbound)`.
- **Feature Engineering:** `compute_knitting_tightness_factor` (pre-production yarn count + stitch/gauge parameters).
- **Splitting:** `SplitStrategy.GROUP_TEMPORAL` (`order_lot_id` + `production_date`, `0.70 / 0.15 / 0.15`) — appropriate because knitting records include `production_date` and `order_lot_id`.

---

## 3. Shared Core Duplication & Engineering Controls

| Area | V1 Status | Finding |
| :--- | :--- | :--- |
| **Core P1.1–P1.5 Wiring** | Wired | Uses Core `LocalDatasetArtifactStorage`, `LocalDatasetRegistry`, `LocalRecordLineageRegistry`, `CsvDataIngestion`, `DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`, `DeterministicFeatureEngineer`. |
| **Core P1.6–P1.8 Adapters** | **Duplicated in `adapters/`** | `src/taqamind/domains/textile/adapters/` (`readiness_splitting.py`, `data_access_training.py`, `model_registry_prediction.py`) duplicates Shared Core P1.6–P1.8 engines that now exist in `CorePlatformServices` (`PR #10`). |
| **Baselines & Models** | Verified | Baselines: `DummyRegressor` (median), `HistoricalRate`. Challengers: `Ridge`, `ElasticNet`, `HistGradientBoosting`, `HurdleRegression`, plus optional `RandomForest` & `XGBoost`. Primary metric: `MAE`; secondary: `RMSE`, `WAPE`, `R2`. |
| **Train-per-Factory & Zero-Retrain** | Verified | `test_lineage.py`, `test_prediction.py`, and `test_no_retraining.py` confirm factory isolation and zero `.fit()` / `.fit_transform()` calls during repeated prediction & process reload. |
| **Unit Mapping (`AUD-001`)** | Verified | `UnitAwareColumnMapperAdapter` + `pipeline.py` properly confirm canonical units and convert `m -> yd` / `g -> kg`. |

---

## 4. Identified Defects, Limitations & Readiness Verdict

1. **Unconfirmed Real Production Targets (By Design):** Both `weaving_waste_regression` and `knitting_waste_regression` default to `TargetConfirmation.UNCONFIRMED`. Real factory deployment is blocked until domain owners sign off on target definitions and anomaly rules (`140` weaving >100% rows, `10` knitting >100% rows, missing `run_id` in real knitting).
2. **Domain Adapter Duplication:** `src/taqamind/domains/textile/adapters/` duplicates `CorePlatformServices` (merged in `PR #10`). **Recommendation:** Retire `textile/adapters/` in favor of `CorePlatformServices` on a dedicated V1 maintenance branch — **do not** mix this refactor into `feature/textile/energy-consumption-v2`.
3. **V2 Separation Requirement:** For TaqaMind V2 Energy Consumption, use `CorePlatformServices` directly from Day 1 and **never** duplicate P1.6–P1.8 engines in `domains/textile/adapters/`.

**V1 Readiness Verdict:**  
- **Engineering & Contract Pipeline:** **READY** (17/17 tests passing, Ruff clean, zero-retraining and leakage controls verified on synthetic fixtures).  
- **Real Factory Production Claims:** **BLOCKED** pending industrial target sign-off (`TargetConfirmation.CONFIRMED`) and data-governance resolution of historical log anomalies.
