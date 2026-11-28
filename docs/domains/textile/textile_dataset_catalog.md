# Textile Dataset Catalog (Task 11)

- **Search Scope:** Kaggle, UCI ML Repository, Zenodo, Mendeley Data, Figshare, Hugging Face, IEEE DataPort, MDPI/ScienceDirect/Springer open-access supplements, and existing TaqaMind `research/textile/` artifacts.
- **Verification Rule:** Only datasets directly present in repo research or verified in `v2-dataset-discovery-report.md` are listed; no unverified dimensions are fabricated.

| Dataset / Source | Access & Origin | Rows × Cols | Granularity | Target | Classification | Notes |
|---|---|---|---|---|---|---|
| `full_weaving_dataset.csv` | TaqaMind V1 / Kaggle | `121,148 × 18` | Weaving beam batches | Waste (`Rej_and_cut_Piece`) — **No `kWh`** | **C** (V1 Waste / V2 Schema Reference) | `74,448` exact duplicate feature/row entries; lacks electrical `kWh` & loom `kW`/RPM. |
| `knitting_dataset.csv` | TaqaMind V1 | `12,569 × 38` | Circular knitting shift runs | Waste (`Reject Qty`) — **No `kWh`** | **C** (V1 Waste / V2 Schema Reference) | `6` exact duplicates; lacks electrical `kWh`, `kW`, RPM, and run duration. |
| UCI Steel Industry Energy | UCI / GitHub | Plant 15-min load | Steel plant 15-min | `kWh` (Steel) | **E** (Not suitable) | Steel arc-furnace physics; invalid for textile weaving/knitting. |
| IndustryHeat-EU / Mopo | Zenodo | Macro EU sector | Sector-level | Macro GJ/TWh | **E** (Not suitable) | Macro-economic aggregate, not machine/batch level. |
| MDPI / ScienceDirect Textile Energy Literature | Open-Access Papers | Aggregate tables only | Literature benchmarks | `kWh/kg` ranges | **C / D** (Benchmarks only) | No raw row-level SCADA/PLC CSVs published. |
