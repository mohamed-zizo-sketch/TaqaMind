# Dataset Collection Log — Food & Beverage Domain

**Log Date:** 2026-11-25  
**Storage Directory:** `data/food_beverage/raw/`  

## 1. Pre-Downloaded Dataset Verification

All local raw datasets were verified for file integrity, readability, and SHA-256 checksums without modifying any source bytes:

1. `brewery_data.csv` — Verified UTF-8 CSV, 583 data rows, 20 columns. SHA-256: `a1c0bf3c67e754e81ef9a16b4efe7b7484df6f2c7efdb3eb98163f53819854ef`.
2. `00_README.txt`, `01_DATA_Figure_2_specific_energy_consumption.csv`, `02_DATA_thermal_loads_in_kW.csv`, `03_DATA_electric_boiler_consumption_in_kW.csv` — Verified EU ENOUGH Norwegian Dairy dataset. SHA-256 hashes recorded in `dataset_catalog.csv`.
3. `ITAC_Database_20261124.xlsx` — Verified Excel workbook with `Terms`, `ASSESS`, `RECC` sheets. Filtered programmatically during audit to NAICS `311*` and `312*` (Food & Beverage). SHA-256: `01e993603e185643fcbc26a397e944d7b29350ba2348b6f387fc153545b6db06`.
4. `Babor M-Hitzmann B-2022-Small and medium-sized bakery production data for scheduling-BK15.xlsx` & `BK50.xlsx` — Verified Excel workbooks with `Description` and scheduling sheets. SHA-256 hashes recorded.

## 2. External Reference Notes

- **FAO FLW Database:** Accessed online reference documentation for Processing and Storage stages in Egypt/NENA. Macro-level percentages (e.g., 10%–20% traditional storage loss) are documented as contextual baselines in `industrial-domain-research.md`, not mixed into batch-level ML training tables.
- **Exclusion Rule:** Household, restaurant, and retail food waste datasets were explicitly excluded from industrial manufacturing ML training per Section 9 of the Master Instructions.
