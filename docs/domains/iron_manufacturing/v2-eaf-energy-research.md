# TaqaMind V2 — Phase 1: EAF Primary Meltshop Industrial Energy Research

**Scope:** Electrical energy consumption in Electric Arc Furnace (EAF) primary steelmaking, prioritizing Egyptian steelmaking conditions (DRI + Scrap EAF route) alongside traceable international engineering benchmarks.

---

## 1. Industrial Workflow & Major Electrical Loads

1. **Basket Charging & Roof Swing:**
   - Overhead charging cranes, roof-lift hydraulics, and furnace tilting drives (minor auxiliary electrical load relative to melting).
2. **Melting & Refining (Primary Electrical Load — EAF Main Transformer):**
   - Three-phase AC (or DC) graphite electrodes strike electric arcs onto cold scrap and Direct Reduced Iron (DRI/HBI).
   - **EAF Furnace Transformer** dominates total heat electricity (~**85%–92%** of EAF shop electricity), rated typically in **40–150 MVA** (~**32–120 MW** active power depending on vessel capacity and tap size).
   - Chemical energy inputs (natural gas oxy-fuel burners, oxygen lancing, carbon injection for foamy slag) supplement electrical arc energy—these are **non-electrical** thermal/chemical inputs that alter electrical `kWh/t` without counting as metered `kWh`.
3. **Fume Extraction, Cooling & Dedicated Auxiliaries:**
   - High-capacity off-gas baghouse / dedusting ID fans, primary/secondary water-cooling pumps, and hydraulic systems represent significant continuous motor-driven auxiliary load during tap-to-tap duration.

---

## 2. Reference Electrical Energy Benchmarks (Traceable Sources)

| Operating Route / Configuration | Typical Specific Electricity Consumption (`kWh/t` liquid steel) | Typical Active Power Scale | Primary Energy Variance Drivers |
| :--- | :--- | :--- | :--- |
| **100% Scrap-Fed EAF (International Best Practice / Typical)** | **350 – 500 kWh/t** (older/unoptimized configurations up to ~**600 kWh/t**) | **40 – 120 MW** (`40,000 – 120,000 kW` furnace transformer) | Tap-to-tap duration, scrap bulk density/contaminants, oxy-fuel/O₂ assistance, power-off delays |
| **DRI + Scrap Mixed EAF (Cold DRI Charge)** | **500 – 700+ kWh/t** (UNIDO IEE benchmark ~**595 kWh/t** for mixed DRI/scrap EAF) | **50 – 120 MW** | DRI gangue (`SiO2`/`Al2O3`) requiring extra slag melting + flux, unreduced `FeO` endothermic reduction, lower metallization |
| **Hot-Charged DRI EAF** | Substantially lower than cold-DRI baseline | **50 – 120 MW** | Retains sensible heat (~500–650 °C) directly from shaft furnace, reducing electrical arc demand |

> **IEA / Worldsteel Boundary Distinction:** IEA reports ~**2.1 GJ/t** final energy for scrap-EAF vs ~**17.1 GJ/t** for full DRI-EAF (which includes upstream natural-gas shaft reduction). TaqaMind V2 isolates **electrical `kWh` metered at the EAF transformer and dedicated dedusting feeder per heat**—excluding natural gas in DRI shaft furnaces and oxy-fuel burners.

---

## 3. Pre-Production Operating Parameters Driving EAF `kWh`

1. **Nameplate Rating, Duration & Planned Utilization (`rated_kw`, `planned_duration_hr`, `planned_utilization_pct`):**
   - Establishes the physics-informed theoretical nameplate energy envelope: `rated_kw * planned_duration_hr * (planned_utilization_pct / 100)`.
2. **Metallic Charge Mix (`planned_scrap_mass_t`, `planned_dri_mass_t`, `planned_dri_share_pct`):**
   - Higher DRI share increases specific electrical demand (`kWh/t`) compared to clean heavy scrap because DRI gangue must be fluxed and unreduced iron oxides reduced endothermically.
3. **Target Tap Weight & Tapping Temperature (`planned_tap_mass_t`, `target_tap_temp_c`):**
   - Larger batch mass scales total `kWh` near-linearly; higher aim superheat (`target_tap_temp_c`, typically `1610–1665 °C`) increases power-on holding energy and radiative/wall losses.
4. **Steel Grade & Furnace Vessel (`steel_grade_code`, `furnace_id`):**
   - Grade metallurgy dictates refining duration and aim temperature; distinct EAF vessels differ in transformer impedance, shell geometry, and electrode regulation.

---

## 4. Egyptian Steel Industry Context (UNIDO / World Steel / Regulatory)

- **DRI + EAF Dominance:** Unlike blast-furnace (BF-BOF) steelmakers, Egypt's steel sector is predominantly **natural-gas DRI + EAF** combined with scrap-based mini-mills (producing reinforcing bar and flat products).
- **UNIDO Egypt IEE & Electric Motor Assessments:**
  - UNIDO & GEF Industrial Energy Efficiency (IEE) benchmarking in Egypt highlights higher EAF electricity intensity (~**595 kWh/t** range for DRI/scrap routes) alongside significant savings opportunities from foamy slag practices (up to ~20% arc efficiency improvement via wall shielding) and VSD upgrades on fume-extraction ID fans and cooling pumps.
  - Across Egyptian manufacturing, electric motor-driven systems account for **>70%** of industrial electricity consumption; Egypt Ministerial Decree **#463/2020** established MEPS for industrial electric motors.
- **EgyptERA Power-Factor Compliance:** Industrial HV/MV tariffs enforce reactive power / power-factor penalties (minimum `0.92` PF threshold noted in [domain-specification.md](./domain-specification.md)), making pre-production `kWh` forecasting valuable for demand planning and TOU scheduling—while actual reactive power (`kvar`) and actual PF remain strictly post-production outcomes excluded from predictors.

---

## 5. Traceable Sources

1. **UNIDO & GEF (Egypt IEE Project):** *Benchmarking Report for the Iron and Steel Sector in Egypt* & *Electric Motor-Driven Systems Assessments in Egypt* (Ministerial Decree #463/2020).
2. **International Energy Agency (IEA):** *Iron and Steel Technology Roadmap* (Scrap-EAF vs. DRI-EAF final energy intensity and route boundaries).
3. **World Steel Association (worldsteel):** *Energy Use in the Steel Industry* (Scrap-EAF and DRI-EAF primary energy boundaries).
4. **AIST / Advanced Energy:** EAF electrical power profile benchmarks (`350–500 kWh/t` scrap-EAF; `500–700+ kWh/t` high-DRI EAF).
