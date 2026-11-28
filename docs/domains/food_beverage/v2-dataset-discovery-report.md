# v2-dataset-discovery-report.md

## Real Open‑Access Electrical Energy Datasets for Food & Beverage

| ID | Title | Source | URL | License | Country / Context | Process | Data Type | Status |
|----|-------|--------|-----|---------|-------------------|---------|-----------|--------|
| `fb_brewery_01` | Brewery Operations and Market Analysis Dataset | Kaggle | https://www.kaggle.com/datasets/aqmarh11/brewery-operations-and-market-analysis-dataset | Educational/Public | India (simulated style) | beverage | CSV | **Benchmark** – inspected, but loss units are unconfirmed; suitable for pipeline benchmark only. |
| `fb_dairy_energy_01` | Norwegian Dairy Integrated CO₂ Refrigeration & Thermal Energy | Zenodo (EU ENOUGH) | https://zenodo.org/records/22877098 | CC BY‑4.0 | Norway | dairy | CSV+TXT | **Validation** – real industrial plant data; contains thermal and electric boiler loads but **no batch‑level waste target**; can be used for energy model validation after mapping. |
| `fb_itac_01` | US DOE ITAC Database (NAICS 311‑312 Food & Beverage) | US DOE ITAC | https://itac.university/download | Public Domain (US Govt) | USA | multi‑process | XLSX | **Development** – plant‑level annual audits; provides kWh and cost data, but not batch‑level pre‑production targets. |
| `fb_bakery_sched_01` | Small & Medium‑Sized Bakery Production Data for Scheduling (BK15 & BK50) | Mendeley Data | https://data.mendeley.com/datasets/dhgbssb8ns/1 | CC BY‑4.0 | Germany/Europe | bakery | XLSX | **Unsuitable for Energy** – contains scheduling times and makespan, no waste or electricity measurements. |
| `fb_fao_flw_ref` | FAO Food Loss and Waste Database (Processing & Storage Filter) | FAO | https://www.fao.org/platform-food-loss-waste/flw-data/en/ | CC BY‑NC‑SA | Egypt / Global | macro reference | CSV (referenced) | **Reference Only** – macro‑aggregated loss percentages; not usable for batch‑level ML. |

### Classification Summary
- **Development**: `fb_itac_01` – real plant‑level data, useful for feature engineering and cost modeling.
- **Benchmark**: `fb_brewery_01` – provides a complete set of columns; can be used to benchmark the pipeline (target columns are loss metrics, not energy).
- **Validation**: `fb_dairy_energy_01` – high‑resolution electrical/thermal data, ideal for validating energy models after adaptation.
- **Unsuitable**: `fb_bakery_sched_01` (no energy metrics) and `fb_fao_flw_ref` (macro level).

The above datasets will be referenced in subsequent contract definitions and synthetic data generators.
