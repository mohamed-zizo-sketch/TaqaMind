# Textile Energy V2 — Implementation Audit (Task 06)

- **Domain:** `textile` (`weaving`, `knitting`)
- **Branch:** `feature/textile/energy-v2-validation` (from `develop` @ `5455cc8`)
- **Overall State:** **Implemented and tested** (against `develop` @ `5455cc8`), with **WP10 Shared Core compatibility items** identified against `849027d`.

---

## 1. Component Status

| Component | File | Current State | Findings & WP10 Actions |
|---|---|---|---|
| **Weaving Energy Contracts** | `weaving/energy_contracts.py` | **Implemented and tested** | Defines `weaving_energy_regression` (`total_electrical_energy_kwh`, `beam_batch` grain, `UNCONFIRMED` by default). Uses legacy `DerivedFeatureSpec.compute` callbacks (`total_thread_density_per_inch`, `theoretical_nameplate_energy_kwh`) -> migrate to `RowLocalTransform` in WP10. |
| **Knitting Energy Contracts** | `knitting/energy_contracts.py` | **Implemented and tested** | Defines `knitting_energy_regression` (`total_electrical_energy_kwh`, `shift_run` grain, `UNCONFIRMED` by default). `knitting_tightness_factor` blocked by missing `SQRT` in `RowLocalExpr` (`CIR-TEXTILE-001`). |
| **Energy Pipeline Service** | `energy_pipeline.py` | **Implemented and tested** | Wired directly to `CorePlatformServices.create_local(...)`. Two WP10 items: (1) `Path(csv_path).read_bytes()` at line 172 is unbounded (`TI-002`); (2) `membership_builder.build_and_register(...)` at line 296 runs even when `purpose == PREDICTION` (`TI-018`). |
| **Synthetic Verification Data** | `weaving/energy_synthetic.py`, `knitting/energy_synthetic.py` | **Implemented and tested** | Explicitly scoped to engineering verification only; never used as proof of real factory accuracy. |
| **Energy V2 Test Suite** | `tests/domains/textile/test_energy_v2.py` | **Implemented and tested** (`5/5` passed) | Covers registration, unconfirmed-target blocking, E2E zero-retraining prediction (`weaving`, `knitting`), and retraining + leakage guard. |
