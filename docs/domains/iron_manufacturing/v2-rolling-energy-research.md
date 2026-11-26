# TaqaMind V2 — Phase 1: Rebar Hot Rolling Industrial Energy Research

**Scope:** Electrical energy consumption in billet-to-rebar hot rolling mills (`rebar_hot_rolling`), prioritizing Egyptian long-product mills and traceable LBNL / EU BAT / worldsteel benchmarks.

---

## 1. Industrial Workflow & Energy Split (Electrical vs. Thermal)

A rebar hot rolling mill converts cast steel billets (typically `130×130 mm` or `150×150 mm`) into deformed reinforcing bars (`10–32 mm` diameter) through two fundamentally different energy carriers:

1. **Billet Reheating Furnace (Primarily Thermal — Natural Gas):**
   - Walking-beam or pusher reheating furnaces heat cold or hot-charged billets to ~**1,100–1,250 °C**.
   - Natural gas combustion typically accounts for **60%–70% of total rolling mill energy** (**1.5–2.5 GJ/t** typical; **1.1–1.4 GJ/t** EU BAT with regenerative burners / waste-heat recovery).
   - **Boundary Rule:** Natural gas thermal energy (`GJ` / `Nm³`) is **excluded** from the V2 electrical energy target (`total_electrical_energy_kwh`), while combustion air blowers, furnace skid-cooling pumps, and pusher drives *are* electrical auxiliaries if sub-metered on the mill/furnace MCC.
2. **Roughing, Intermediate & Finishing Mill Stands (Primary Electrical Load):**
   - Multi-stand AC/DC variable-speed main drives deform the billet into target rebar cross-sections, including multi-strand **slitting** on smaller diameters (`10–12 mm`, 2–3 strands).
3. **Auxiliary Systems (Pumps, Cooling Bed & Finishing):**
   - High-pressure water descaling pumps, Quenching & Self-Tempering (QST / Thermex) water pumps, flying shears, cooling-bed drives, and bundling presses.

---

## 2. Reference Electrical Energy Benchmarks (LBNL / EU FMP BREF)

| Process Scope | Typical Electricity Intensity (`kWh/t` rolled rebar) | Best Available Technique (BAT / LBNL) | Typical Mill Drive Power Scale |
| :--- | :--- | :--- | :--- |
| **Rebar Hot Rolling Mill (Drives + Auxiliaries)** | **80 – 120 kWh/t** (broader LBNL/LCA route figures ~**60–80 kWh/t** on modern mills) | **50 – 70 kWh/t** (high hot-charge ratio, VSDs on pumps/fans, optimized pass design) | **10 – 25 MW** (`10,000 – 25,000 kW` combined stand drives + auxiliary MCC) |
| **Reheating Furnace Natural Gas (Thermal — Separate)** | **1.5 – 2.5 GJ/t** (older pusher furnaces up to 3.0+ GJ/t) | **1.1 – 1.4 GJ/t** | Excluded from electrical `kWh` target |

---

## 3. Pre-Production Operating Parameters Driving Rolling `kWh`

1. **Nameplate Rating, Duration & Planned Utilization (`rated_kw`, `planned_duration_hr`, `planned_utilization_pct`):**
   - Establishes the pre-production electrical envelope (`theoretical_nameplate_energy_kwh`). Idle delays (mill cobbles, roll changes) leave water-cooling pumps and furnace blowers running while output drops, inflating `kWh/t`.
2. **Planned Billet Charge Mass (`planned_billet_mass_t`):**
   - Scales total mechanical deformation work and rolling lot duration.
3. **Target Rebar Diameter (`target_rebar_diameter_mm`):**
   - **Inverse relationship with electrical SEC (`kWh/t`):** Smaller rebar diameters (`10 mm`, `12 mm`) undergo higher total elongation/reduction ratios across more finishing stands and higher rolling speeds than larger bars (`25 mm`, `32 mm`), increasing specific electricity consumption per ton.
4. **Mill Line Identity (`mill_line_id`):**
   - Captures structural equipment differences between separate rolling lines (drive efficiency, motor topology, pump VSD retrofits).

---

## 4. Egyptian Rebar Rolling Mill Context

- **Product Mix Concentration:** Rebar for domestic construction dominates Egyptian long-steel output (`10–32 mm`, B500DWR / B500B grades), with frequent multi-strand slitting on `10–12 mm` campaigns.
- **Motor & Pump Efficiency Opportunity:** Consistent with UNIDO Egypt electric motor-driven system findings (>70% of manufacturing electricity in motors; ~45% of motor electricity in pumps/fans/compressors), descaling and QST water-cooling pumps are high-leverage electrical loads on Egyptian rolling lines where VSDs are not yet deployed.
- **Hot vs. Cold Charging:** Integrated meltshop-to-rolling facilities that hot-charge continuous-cast billets reduce reheating furnace fuel and auxiliary fan runtimes compared to standalone re-rolling mills charging ambient billets.

---

## 5. Traceable Sources

1. **European Commission Joint Research Centre (JRC):** *Best Available Techniques (BAT) Reference Document for the Ferrous Metals Processing Industry (FMP BREF)* — hot rolling electricity (`kWh/t`) and reheating thermal benchmarks (`GJ/t`).
2. **Worrell, E., Blinde, P., Neelis, M., Blomen, E., & Masanet, E. (LBNL):** *Energy Efficiency Improvement and Cost Saving Opportunities for the U.S. Iron and Steel Industry* — hot rolling electricity intensity (`60–80 kWh/t` route benchmark) and VSD/pump retrofits.
3. **UNIDO & Global Efficiency Intelligence (Egypt):** Industrial motor, pump, and fan efficiency assessments in Egypt.
