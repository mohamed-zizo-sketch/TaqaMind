# TaqaMind V2 — Phase 3: Iron & Steel Energy Dataset Discovery Report

**Search Scope:** Existing repository raw audit (`research/iron_manufacturing/audit_raw_datasets.py`), UCI Machine Learning Repository, Kaggle, Zenodo, Mendeley Data, Figshare, GitHub, and Open-Access Academic Supplementary Material.

---

## 1. Search Summary & Honest Dataset Classification

| Dataset / Source | Platform / Origin | Granularity & Fields | Pre-Production & Energy `kWh`? | Classification | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UCI Steel Industry Energy Consumption (`Steel_industry_data.csv`, ID 851)** | UCI ML Repository (Daewoo Steel, Gwangyang, South Korea, 2018; `CC-BY-4.0`) | `35,040` rows × `11` cols; **15-minute plant meter series** (`Usage_kWh`, `Lagging/Leading_kVarh`, `CO2(tCO2)`, `PF`, `Load_Type`) | Has `Usage_kWh` (`15-min` series), **NO pre-production heat/lot features** | **Unsuitable for V2 Pre-Production Training** *(Benchmarking / Telemetry Reference Only)* | **Critical limitations:** (1) 15-minute time-series granularity cannot map to `one_eaf_heat`, `one_rolling_campaign_lot`, or `one_scrap_reception_lot`; (2) **zero** steel tonnage, charge mix, billet diameter, or equipment ratings (`rated_kw`); (3) **zero** process-stage split (`EAF` vs `Rolling` vs `Yard`); (4) `59.9%` exact zero `CO2(tCO2)` rows; (5) Korean plant load profile does **not** represent Egyptian DRI+EAF factories. |
| **Yandex Practicum Ladle Furnace (`arc_furnace_heats/`)** | GitHub mirror (`shradha289/steel-temperature-prediction`; **License: UNKNOWN**) | Multi-table secondary refining (`14,876` arc pulses, `15,907` temp rows, `key` grain) | **No `kWh` target**; in-process `Active/Reactive power` only | **Unsuitable (Quarantined)** | Secondary metallurgy LF temperature prediction; zero `kWh` meter total, zero tonnage, 100% in-process signals, sensor faults (`-715.505` reactive power), undocumented units, and unstated commercial license. |
| **EU Horizon ALCHIMIA Scrap (`scrap_A.csv`, `scrap_C.csv`)** | Zenodo DOIs `10.5281/zenodo.17630721` & `17633305` (`CC-BY-4.0`) | `16,369` & `9,399` delivery rows (`1` vehicle grain) | **No electrical `kWh` or equipment rating** | **Unsuitable for Energy Training** *(Scrap Schema Reference Only)* | European scrap delivery contaminant logs; lacks crane/shear/baler electrical `kWh` meter readings and equipment ratings. |
| **Low-Carbon EAF Scrap Blending (`materials.csv`)** | Zenodo DOI `10.5281/zenodo.22162000` (`CC-BY-4.0`) | `7` charge material parameter rows | Parameter table only | **Reference Parameters Only** | Provides metallurgical yield and tramp-element bounds, not time-stamped factory heat observations. |
| **EAF-LF-CC Flexibility & Demand Reshaping** | Zenodo DOI `10.5281/zenodo.21447633` (`CC-BY-4.0` / `MIT`) | `8` author-declared synthetic OR scheduling heats (`75/85/95 MW` modes) | Synthetic OR scheduling only | **Unsuitable for ML** *(Power Scale Reference Only)* | Confirms realistic `75–95 MW` EAF power-mode scale, but only `8` synthetic scheduling cases—unusable for supervised ML. |
| **Kaggle / Mendeley / Academic EAF & Rolling Papers** | Open-Access Literature & Repositories | Aggregate plant audits or proprietary SCADA studies | Literature `kWh/t` ranges only | **Suitable for Reference Benchmarks Only** | No open-access public CSV pairs pre-production MES/ERP charge/rolling plans (`<= t0`) with heat- or lot-level sub-metered `kWh`. |

---

## 2. Applicability to Egyptian Factories & Engineering Strategy

1. **Verified Public Data Gap:** Zero public open-access datasets simultaneously satisfy **(a)** Egyptian or comparable DRI+EAF / rebar rolling context, **(b)** heat/lot/reception-batch granularity, **(c)** pre-production (`t <= t0`) predictors, and **(d)** sub-metered active electrical energy (`kWh`).
2. **Why UCI ID 851 Must Not Be Forced into V2:** Forcing 15-minute `Steel_industry_data.csv` into TaqaMind V2 would violate row granularity, require fabricating tonnage/recipes, and train models on reactive power/CO2 telemetry (`t > t0` leakage) rather than pre-production schedules.
3. **Approved Architecture (`CorePlatformServices` + Per-Factory Onboarding + Physics-Consistent Verification):**
   - **Production Path:** Factories onboard their own Level-2 MES/ERP + sub-metered `kWh` CSVs via `IronEnergyPipelineService` (`Train-per-Factory`, `factory_id`-isolated, gated on `TargetConfirmation.CONFIRMED`).
   - **Verification Path:** Physics-consistent synthetic data (`common/energy_synthetic.py`) calibrated to Phase 1 benchmarks (`350–700+ kWh/t` EAF, `50–120 kWh/t` rolling, crane/shear `rated_kw` envelopes) validates pipeline mechanics **without** claiming real-plant predictive accuracy.
