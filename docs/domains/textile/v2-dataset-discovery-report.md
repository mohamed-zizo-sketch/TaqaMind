# TaqaMind V2 — Phase 3: Textile Energy Dataset Discovery Report

**Search Scope:** Kaggle, UCI Machine Learning Repository, Zenodo, Mendeley Data, Figshare, Hugging Face Datasets, GitHub, IEEE DataPort, and Open-Access Academic Supplementary Repositories (MDPI, ScienceDirect, Springer).

---

## 1. Search Summary & Classification

| Dataset / Source | Platform / Origin | Coverage & Granularity | Energy Target (`kWh`)? | Classification | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`full_weaving_dataset.csv`** (V1 Reference) | Kaggle / TaqaMind V1 | 121,148 rows × 18 cols; Weaving beam batches | **No** (waste `Rej_and_cut_Piece` only) | **Unsuitable** for energy training; **Reference** for weaving feature schema | Lacks electrical `kWh` / `kW` meter readings and loom technology/speed columns. |
| **`knitting_dataset.csv` & `raw_data_production_reports_20222.xlsx`** (V1 Reference) | TaqaMind V1 | 12,569 rows × 38 cols; Circular knitting shift runs | **No** (waste `Reject Qty` only) | **Unsuitable** for energy training; **Reference** for knitting feature schema | Lacks electrical `kWh` / `kW` meter readings, RPM, and run duration. |
| **IndustryHeat-EU / Mopo** | Zenodo | Aggregated European sector energy carriers | Macro sector GJ/TWh only | **Unsuitable** | Macro-economic sector level; not machine/batch-level textile production. |
| **UCI Steel Industry Energy Consumption** | UCI / GitHub | 15-min plant electrical load (`kWh`, kVarh) | Yes (`kWh`) | **Unsuitable** | Steel industry arc-furnace/plant load; does not represent textile weaving/knitting physics. |
| **Smart Manufacturing Resource Efficiency** | Kaggle | Generic synthetic/multi-industry sensor logs | Generic energy | **Unsuitable** | Generic manufacturing; lacks loom/knitting-specific predictors and boundary provenance. |
| **MDPI / ScienceDirect / Springer Textile Energy Papers** | Open-Access Journals | Air-jet air consumption, workshop digital twin, knitting LCA | Literature benchmarks only (no raw public CSV/XLSX) | **Suitable for Reference Benchmarks Only** | Provides verified `kWh/kg`, `kW`, and `kWh/Nm³` ranges; raw SCADA/PLC tables are proprietary. |

---

## 2. Applicability to Egyptian Factories & Engineering Decision

1. **Public Data Gap:** Granular machine-level or batch-level `.csv`/`.xlsx` datasets combining textile production parameters with sub-metered `kWh` are proprietary SCADA/PLC assets and are not publicly available open-access.
2. **Approved Hybrid Strategy (`Physics-Informed + Train-per-Factory`):**
   - **Real Factory Onboarding:** Contracts and `CorePlatformServices` pipelines accept any factory's sub-metered CSV via `UnitAwareColumnMapper` and train exclusively per factory (`Training-per-Factory`).
   - **Engineering Verification:** Verified physical equations and reference ranges from Phase 1 (LBNL-3970E, UNIDO Egypt, IFTH/EDIPTEX) are used to build a deterministic **Physics-Informed Engineering Baseline** and **Synthetic Verification Datasets** (`Engineering Verification Only`, explicitly tagged non-production).
