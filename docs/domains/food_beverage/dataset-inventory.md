# TaqaMind — Food & Beverage Dataset Inventory

This inventory summarizes all collected and verified datasets for the `food_beverage` domain. Detailed forensic metrics are stored in [forensics_summary.json](file:///d:/TaqaMind/research/food_beverage/forensics_summary.json) and [dataset_catalog.csv](file:///d:/TaqaMind/research/food_beverage/dataset_catalog.csv).

## 1. Summary Table

| ID | Dataset | Raw Files | Rows | Columns | Module | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fb_brewery_01` | Brewery Operations & Market Analysis | `brewery_data.csv` | 583 | 20 | Module 1 (Waste Benchmark) | Audited — Conditionally Suitable (Benchmark) |
| `fb_dairy_energy_01` | Norwegian Dairy Thermal & Boiler Data | `00_README.txt`, `01_DATA_...`, `02_DATA_...`, `03_DATA_...` | 12 / 525,564 / 8,784 | 25 / 8 / 2 | Module 2 (Energy) | Audited — Suitable for M2 |
| `fb_itac_01` | US DOE ITAC (NAICS 311–312 F&B) | `ITAC_Database_20261124.xlsx` | 1,619 assess / 12,692 recc | Multi-sheet | Modules 2 & 3 (Energy & Cost) | Audited — Suitable for M2 & M3 |
| `fb_bakery_sched_01` | Babor & Hitzmann Bakery Scheduling | `...BK15.xlsx`, `...BK50.xlsx` | 159 (BK15) / 448 (BK50) | 15 | Module 4 (OR) | Audited — Suitable for M4 |
| `fb_fao_flw_ref` | FAO FLW Database (Processing/Storage) | External Reference | Macro | — | Module 1 (Egypt Context) | Referenced — Macro Baseline Only |
