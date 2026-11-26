# TaqaMind V2 — Textile Energy Consumption Final Technical Handover

**Branch:** `feature/textile/energy-consumption-v2` (branched from `origin/develop` @ `4cdcb58`)  
**Module:** `PlatformModule.ENERGY_EFFICIENCY` (`weaving_energy_regression` & `knitting_energy_regression`)  
**Zero Core / Cross-Domain Edits:** `src/taqamind/core/*`, `iron_manufacturing`, `food_beverage`, and `pyproject.toml` were untouched.

---

## 1. Delivered Phases & Artifacts

- **Phase 0 — Waste V1 Audit:** [v2-waste-v1-audit-report.md](file:///E:/TaqaMind/docs/domains/textile/v2-waste-v1-audit-report.md)
- **Phase 1 — Industrial Energy Research:**
  - [v2-weaving-energy-research.md](file:///E:/TaqaMind/docs/domains/textile/v2-weaving-energy-research.md)
  - [v2-knitting-energy-research.md](file:///E:/TaqaMind/docs/domains/textile/v2-knitting-energy-research.md)
- **Phase 2 — Target & Measurement Boundary:** [v2-target-and-measurement-boundary.md](file:///E:/TaqaMind/docs/domains/textile/v2-target-and-measurement-boundary.md)
- **Phase 3 — Dataset Discovery:** [v2-dataset-discovery-report.md](file:///E:/TaqaMind/docs/domains/textile/v2-dataset-discovery-report.md)
- **Phase 4 — Dataset Inspection & EDA:** [v2-dataset-inspection-eda.md](file:///E:/TaqaMind/docs/domains/textile/v2-dataset-inspection-eda.md) & [v2_eda_reference_summary.json](file:///E:/TaqaMind/research/textile/v2_eda_reference_summary.json)
- **Phase 5 — Feature Registry:** [v2-universal-energy-feature-registry.md](file:///E:/TaqaMind/docs/domains/textile/v2-universal-energy-feature-registry.md)
- **Phase 6 — Problem Formulation & Leakage:** [v2-leakage-and-readiness-analysis.md](file:///E:/TaqaMind/docs/domains/textile/v2-leakage-and-readiness-analysis.md)
- **Phase 7 — Proposed Contracts & Core Check:** [v2-proposed-energy-contracts.md](file:///E:/TaqaMind/docs/domains/textile/v2-proposed-energy-contracts.md)
- **Phase 8–10 — Domain Implementation (Wired Directly to `CorePlatformServices`):**
  - `compute_theoretical_nameplate_energy_kwh` in [formulas.py](file:///E:/TaqaMind/src/taqamind/domains/textile/common/formulas.py)
  - Weaving Energy contract & physics-calibrated synthetic verifier: [energy_contracts.py](file:///E:/TaqaMind/src/taqamind/domains/textile/weaving/energy_contracts.py), [energy_synthetic.py](file:///E:/TaqaMind/src/taqamind/domains/textile/weaving/energy_synthetic.py)
  - Knitting Energy contract & physics-calibrated synthetic verifier: [energy_contracts.py](file:///E:/TaqaMind/src/taqamind/domains/textile/knitting/energy_contracts.py), [energy_synthetic.py](file:///E:/TaqaMind/src/taqamind/domains/textile/knitting/energy_synthetic.py)
  - `register_textile` multi-module registration: [domain.py](file:///E:/TaqaMind/src/taqamind/domains/textile/domain.py)
  - `TextileEnergyPipelineService` (`CorePlatformServices`-backed): [energy_pipeline.py](file:///E:/TaqaMind/src/taqamind/domains/textile/energy_pipeline.py)
- **Phase 11 — Regression & E2E Tests:** [test_energy_v2.py](file:///E:/TaqaMind/tests/domains/textile/test_energy_v2.py)

---

## 2. Reproducible Verification Commands

```bash
pytest tests/domains/textile -v
ruff check src tests --no-cache
```

## 3. Production Onboarding Disclaimer

- Both `weaving_energy_regression` and `knitting_energy_regression` ship with `TargetConfirmation.UNCONFIRMED` by default.
- Synthetic datasets in `energy_synthetic.py` are strictly for **engineering verification** and **must not** be presented as real-world factory accuracy. Real factories must upload sub-metered `kWh` logs and confirm target/meter boundaries via `TargetConfirmation.CONFIRMED`.
