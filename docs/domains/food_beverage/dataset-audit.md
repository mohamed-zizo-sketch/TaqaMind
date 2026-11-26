# TaqaMind — Food & Beverage Dataset Forensics & Suitability Audit

**Audit Script:** [eda_forensics.py](file:///d:/TaqaMind/research/food_beverage/eda_forensics.py)  
**Machine-Readable Output:** [forensics_summary.json](file:///d:/TaqaMind/research/food_beverage/forensics_summary.json)  
**Generated Figures:** `reports/food_beverage/figures/`

---

## 1. Module 1 Audit: `brewery_data.csv` (`fb_brewery_01`)

### Structural & Quality Forensics
- **Shape & Integrity:** 583 rows, 20 columns, 0 missing values, 0 exact duplicate rows, 583 unique `Batch_ID` values (`2020-01-01` to `2020-12-30`).
- **Categorical Cardinality:** `Beer_Style` (8 values: Pilsner 94, Ale 80, Lager 72, Sour 70, Porter 70, Stout 69, Wheat Beer 66, IPA 62), `SKU` (4 values: Pints, Bottles, Cans, Kegs), `Location` (10 Bangalore neighborhoods), `Ingredient_Ratio` (370 unique strings).
- **Target Analysis (`Loss_During_Brewing`, `Loss_During_Fermentation`, `Loss_During_Bottling_Kegging`):**
  - Zero-inflation: **0 zero-loss rows** across all 3 loss columns (minimum `Total_Loss` = 5.3, mean = 9.16, std = 1.58, max = 14.8).
  - **Unit Ambiguity:** The raw CSV does not specify whether loss values are in percentages (%), hectoliters (hL), or liters (L).
- **Correlation & Predictive Signal:**
  - Every pre-production candidate feature exhibits near-zero linear correlation with `Total_Loss`: `Fermentation_Time` ($r = 0.0542$), `Temperature` ($r = 0.0113$), `pH_Level` ($r = 0.0215$), `Gravity` ($r = -0.0349$), `Volume_Produced` ($r = 0.0496$).
- **Leakage Risks Identified:**
  - `Volume_Produced`, `Total_Sales`, `Quality_Score`, and `Brewhouse_Efficiency` are **post-production** realizations. Using them as predictors constitutes severe target/temporal leakage.
- **Suitability Classification:** **Conditionally Suitable / Benchmark-Only (Requires Industrial Confirmation)**. It is not suitable for claiming real industrial predictive accuracy in Egyptian factories, because loss units are unconfirmed and predictors show synthetic-like independence from the target.

![Brewery Loss Distributions](file:///d:/TaqaMind/reports/food_beverage/figures/brewery_loss_distributions.png)
![Brewery Correlation Matrix](file:///d:/TaqaMind/reports/food_beverage/figures/brewery_correlation_matrix.png)
![Brewery Group and Temporal](file:///d:/TaqaMind/reports/food_beverage/figures/brewery_group_and_temporal.png)

---

## 2. Module 2 Audit: Norwegian Dairy Energy (`fb_dairy_energy_01`)

- **Real Industrial Origin:** EU Horizon 2020 ENOUGH project (Köster et al., 2025), 2024 full-year SCADA/meter data.
- **Resolution & Completeness:**
  - `01_DATA` (Monthly SEC): 12 rows, 0 missing, 20,546 m³ processed milk, 26,378 m³ water.
  - `02_DATA` (1-min Thermal Loads): 525,564 rows. Minor sensor gaps (<0.15% missing). Contains negative kW values (e.g., -38.69 kW in Glycol Process Cooling), reflecting heat-recovery/sensor-zero behavior rather than invalid rows.
  - `03_DATA` (1-hr Electric Boiler): 8,784 rows, 0 missing, mean 101.70 kW, max 484.00 kW.
- **Suitability Classification:** **Suitable for Module 2 (Energy Efficiency)**; **Unsuitable for Module 1 (Waste Prediction)** (no waste quantities recorded).

![Dairy Energy Profiles](file:///d:/TaqaMind/reports/food_beverage/figures/dairy_energy_profiles.png)

---

## 3. Modules 2 & 3 Audit: US DOE ITAC (`fb_itac_01`)

- **Filtering:** Strictly filtered `ITAC_Database_20261124.xlsx` to NAICS `311*` (Food Manufacturing) and `312*` (Beverage & Tobacco).
- **Yield:** **1,619 F&B plant assessments** and **12,692 energy/cost recommendations** (FY 2002–2026), including **5,461 implemented recommendations** (`IMPSTATUS == 'I'`). Top subsectors: `3121` Beverage (302), `3119` Other Food (262), `3118` Bakeries (211), `3114` Fruit/Veg Preserving (200), `3115` Dairy (185).
- **Suitability Classification:** **Suitable for Modules 2 & 3 (Energy & Cost Benchmarking)**; **Unsuitable for Module 1 (Batch Waste Prediction)**.

---

## 4. Module 4 Audit: Babor & Hitzmann Bakery Scheduling (`fb_bakery_sched_01`)

- **Scope:** Real SME bakery flow-shop scheduling data (`BK15`: 159 stage rows, 3 products, 10 machines, 164 batches; `BK50`: 448 stage rows, 6 products, 24 machines, 677 batches).
- **Data Hygiene Findings:** `BK50` `stage_name` contains trailing whitespace and casing variants (`"Baking"` vs `"Baking "`, `"Dividing"` vs `"DIviding"`), requiring deterministic string normalization in OR ingestion.
- **Suitability Classification:** **Suitable for Module 4 (Operations Research)**; **Unsuitable for Module 1 (Waste Prediction)**.

![Bakery BK50 Stage Durations](file:///d:/TaqaMind/reports/food_beverage/figures/bakery_bk50_stage_durations.png)

---

## 5. Overall Feasibility Conclusion for Module 1 (Pre-Production Waste Prediction)

No public dataset currently in `data/food_beverage/raw/` provides factory-confirmed, real-world pre-production batch inputs paired with unit-verified avoidable waste targets. Therefore, per Section 10 of the Master Instructions:
1. We document this data gap transparently.
2. We implement the complete, production-grade software & ML pipeline for `bakery`, `dairy`, and `beverage` processes.
3. All automated training, evaluation, and no-retraining tests use **explicitly labeled synthetic industrial fixtures** with realistic physical relationships, while production factory training remains gated on `TargetConfirmation(status=CONFIRMED)`.
