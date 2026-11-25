# Dataset Inventory — `iron_manufacturing`

All supplied raw datasets are stored locally under `data/iron_manufacturing/raw/` (gitignored; raw bytes are never committed or mutated). SHA-256 checksums and row/column dimensions were verified directly by `research/iron_manufacturing/audit_raw_datasets.py`.

## Supplied Datasets Summary

| # | Dataset Group | Source & Provenance | License | Country / Plant | Granularity | Suitability Classification |
|---|---|---|---|---|---|---|
| 1 | **EU Horizon ALCHIMIA Scrap** (`scrap_A.csv`, `scrap_C.csv`) | Zenodo DOIs `10.5281/zenodo.17630721` & `10.5281/zenodo.17633305` (Scuola Superiore Sant'Anna) | `CC-BY-4.0` | Anonymised European electric steel plants (**NOT Egyptian**) | 1 scrap delivery vehicle (`ID`) | **CONDITIONALLY SUITABLE (Reference & EDA only)** — real non-metallic contaminant target (`Soil+Water+Other`), requires factory confirmation on measurement method and cannot validate Egyptian plants. |
| 2 | **Yandex Practicum Ladle Furnace** (`data_arc_en.csv`, `data_temp_en.csv`, `data_bulk_en.csv`, `data_bulk_time_en.csv`, `data_wire_en.csv`, `data_wire_time_en.csv`, `data_gas_en.csv`) | GitHub mirror `shradha289/steel-temperature-prediction` | **UNKNOWN / UNSTATED** | Undisclosed (**NOT Egyptian**) | Multi-table Ladle Furnace treatment (`key`) | **UNSUITABLE** — secondary metallurgy LF temperature control (NOT EAF melting), zero waste target, zero tonnage, 100% in-process features, unverified units, unstated license. |
| 3 | **UCI Steel Industry Energy Consumption (ID 851)** (`Steel_industry_data.csv`) | UCI ML Repository ID 851 (Daewoo Steel, Gwangyang, Korea, 2018) | `CC-BY-4.0` | South Korea (**NOT Egyptian**) | 15-minute plant electricity meter | **UNSUITABLE FOR WASTE PREDICTION** — energy-only meter series with zero production tonnage, zero heat/lot IDs, zero scrap/waste target, and 59.9% zero CO2 rows. |
| 4 | **Low-Carbon EAF Scrap Blending** (`materials.csv`, `product_quality_scenarios.csv`) | Zenodo DOI `10.5281/zenodo.22162000` | `CC-BY-4.0` | Literature / US border anchors | 7 charge material parameter rows | **REFERENCE PARAMETERS ONLY** — useful as metallurgical/OR prior bounds, not factory training observations. |
| 5 | **EAF-LF-CC Flexibility & Demand Reshaping** (`steel_vpp_reproducibility`) | Zenodo DOI `10.5281/zenodo.21447633` | `CC-BY-4.0` / `MIT` | Author-declared synthetic | 8 synthetic scheduling heats | **UNSUITABLE FOR ML** — synthetic OR scheduling reference only. |

## Missing Essential Inputs (Blocked Real Production Training)
- **Egyptian EAF Meltshop historical CSV:** No real labeled Egyptian heat-level CSV with pre-production planned charge/DRI/recipe inputs and confirmed `metallic_waste_loss_t` is currently available.
- **Egyptian Rebar Hot Rolling historical CSV:** No real labeled Egyptian lot-level CSV with pre-production billet/pass-design/groove-age inputs and confirmed `total_rolling_waste_kg` is currently available.
- **Egyptian Scrap Yard Reception historical CSV:** No Egyptian weighbridge delivery CSV with confirmed `total_contaminant_kg` measurement semantics is currently available.

## Non-Fabrication Policy
- Earlier prototype calibrated files (e.g., synthetic fusion of LF telemetry with simulated `cal_*` columns) are **excluded** from this repository and never used as industrial evidence.
- Automated tests and CLI demonstrations use **explicitly synthetic fixtures** (`common/synthetic.py` + `make_synthetic_confirmation()`) that never claim industrial production readiness.
