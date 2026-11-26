# Forensic Dataset Audit & EDA Report — `iron_manufacturing`

Reproducible command:
```powershell
python research/iron_manufacturing/audit_raw_datasets.py
```
Notebook: [01_iron_dataset_audit_and_eda.ipynb](../../../notebooks/iron_manufacturing/01_iron_dataset_audit_and_eda.ipynb)  
Generated diagnostics & plots (gitignored): `reports/iron_manufacturing/dataset_audit_report.json` and `reports/iron_manufacturing/figures/*.png`.

## 1. EU Horizon ALCHIMIA Scrap Yard (`scrap_A.csv`, `scrap_C.csv`)
- **Classification:** `CONDITIONALLY_SUITABLE_REFERENCE_ONLY`
- **SHA-256:**
  - `scrap_A.csv`: `b83385d1859e549d40fd6c2bc9516b26ef1460746ecf8bd7dd49efe76077fc4b` (`16,369` rows × `11` cols, `0` missing, `0` exact duplicates, unique `ID`).
  - `scrap_C.csv`: `e71ef2dba0a5449edb4350f5a1c0260f9b6bdbc17a7210320bd352bf14773692` (`9,399` rows × `11` cols, `0` missing, `0` exact duplicates, unique `ID`).
- **Temporal Coverage:**
  - Plant A: `2024-07-01 06:03:18` to `2024-09-30 20:02:33` (`0` rows with `ExitDate < EntryDate`).
  - Plant C: `2024-01-02 07:58:43` to `2024-06-29 13:36:57` (`0` rows with `ExitDate < EntryDate`).
- **Target Analysis (`total_contaminant_kg = Soil + Water + Other`):**
  - Plant A: **26.32% exact zeros** (strongly zero-inflated — justifies evaluating a two-stage hurdle candidate), median `400 kg`, p95 `1,900 kg`, weighted contaminant share `2.926%` of gross weight (`158` distinct values, indicating quantized/visual deductions).
  - Plant C: **1.78% zeros**, median `300 kg`, p95 `1,150 kg`, weighted contaminant share `1.607%` (`164` distinct values).
- **Group Dependence & Concentration Risk:**
  - Plant A has `37` suppliers, with the **top 1 supplier accounting for `45.53%`** of deliveries (`scrap_C` has `130` suppliers, top supplier `5.84%`). Group-temporal splitting is mandatory to avoid supplier memorization.
- **Leakage Risks Identified:**
  - `ExitDate` and `Difference` (stay duration) are post-unloading variables and are strictly blocked from pre-inspection predictors.
  - `Soil`, `Water`, and `Other` are target components and strictly blocked from predictors.
- **Unresolved Confirmation Required:** Whether `Weight` is gross entry weighbridge weight or post-unloading net weight, and whether contaminant deductions are physically weighed vs visually estimated.

## 2. Yandex Practicum Ladle Furnace (`data/iron_manufacturing/raw/arc_furnace_heats/`)
- **Classification:** `UNSUITABLE_FOR_WASTE_PREDICTION` (Quarantined)
- **Key Findings:**
  - Multi-table secondary refining (Ladle Furnace) records (`14,876` arc pulses in `data_arc_en.csv`, `15,907` temperature samples in `data_temp_en.csv`, `3,129` rows in `data_bulk_en.csv`, `3,081` in `data_wire_en.csv`, `3,239` in `data_gas_en.csv`).
  - **Missingness:** `18.24%` missing `Temperature` (`739` keys masked as competition holdout); `10.3%` to `99.97%` missing across `Bulk 1..15`.
  - **Physical Anomalies:** Negative reactive power sensor fault (`-715.505` on `key=2116`); `7` temperature samples `< 1400°C`.
  - **Why Excluded:** No scrap or metallic waste target exists, no steel tonnage exists, units are undocumented, all arc/bulk/wire/gas signals occur **in-process**, and the dataset lacks a cleared commercial license.

## 3. UCI Steel Industry Energy Consumption (`Steel_industry_data.csv`, ID 851)
- **Classification:** `UNSUITABLE_FOR_WASTE_PREDICTION`
- **SHA-256:** `9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc` (`35,040` rows × `11` cols, 15-minute intervals across 2018).
- **Key Findings:** Zero missing values and zero duplicates, but **zero production tonnage, zero heat/lot IDs, zero process stage, zero waste/scrap target**, and **`59.9%` exact zero `CO2(tCO2)`** readings. Energy consumption datasets cannot be repurposed into waste prediction targets.

## 4. Zenodo Parameter & Synthetic References (`22162000` & `21447633`)
- **Classification:** `REFERENCE_ONLY` (not factory ML training datasets). Provide metallurgical yield/tramp-element bounds (`7` charge materials) and 3-mode EAF demand benchmarks (`75/85/95 MW`) documented for future Energy and Operations Research modules.
