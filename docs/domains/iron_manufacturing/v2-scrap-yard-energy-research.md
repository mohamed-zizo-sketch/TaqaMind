# TaqaMind V2 — Phase 1: Scrap Yard Reception Industrial Energy Research

**Scope:** Electrical energy consumption in ferrous scrap yard receiving, handling, shearing, and baling operations (`scrap_yard_reception`) prior to EAF charging.

---

## 1. Industrial Workflow & Energy Carrier Separation

Scrap yard operations prepare raw incoming ferrous scrap (heavy melting scrap, shredded scrap, turnings, light sheet) into furnace-ready charge mixes:

1. **Entry Weighing & Unloading (Mixed Electrical / Diesel):**
   - Truck/rail entry weighbridges consume negligible electrical power (load cells + barrier/IT).
   - Overhead **electromagnetic bridge/gantry cranes** consume grid electricity for lifting magnets and hoist/trolley drives; conversely, **mobile wheeled/tracked material handlers (excavators/loaders)** typically burn **diesel fuel** and must be **excluded** from the electrical `kWh` boundary.
2. **Size Reduction & Compaction (Primary Electrical Load):**
   - **Hydraulic Scrap Shears** (typically **300–1,200+ kW** electric hydraulic power packs) cut oversized structural steel and demolition scrap to EAF basket dimensions.
   - **Scrap Balers / Compactors** (typically **150–600 kW** hydraulic drives) compress light sheet/turnings into high-density bundles.
   - **Shredder Lines** (when electrically driven on large yards, **1,000–4,000+ kW** main induction/synchronous drives plus magnetic drum separators and dedusting fans).
3. **Yard Lighting & Conveyor / Dust Collection Auxiliaries:**
   - Dedicated conveyor motors and dust-extraction fans on shear/shredder lines.

---

## 2. Reference Equipment Ratings & Boundary Caveat

| Equipment Class | Typical Electrical Rating (`kW`) | Primary Energy Driver | Literature Benchmark Caveat |
| :--- | :--- | :--- | :--- |
| **Electromagnetic Overhead / Gantry Cranes** | **75 – 350 kW** (hoist + bridge + DC magnet rectifier) | Tonnage lifted, lift cycles, magnet duty cycle | No standardized universal `kWh/t` benchmark across yards due to site layout differences |
| **Hydraulic Scrap Shears & Balers** | **150 – 1,200+ kW** | Scrap thickness/grade, hydraulic pressure cycles, idling time | Duty-cycle dependent; **no fabricated `kWh/t` figure is claimed here** |
| **Electric Scrap Shredder + Separators** | **1,000 – 4,000+ kW** | Feed scrap density/gauge, throughput (`t/h`) | Only applicable if the facility operates an electric shredder fed from the yard substation |

> **Honesty & Non-Fabrication Note:** Unlike EAF melting (`350–700+ kWh/t`) and rebar hot rolling (`50–120 kWh/t`), public IEA/LBNL/worldsteel steel sector reports rarely publish a standardized standalone `kWh/t` SEC benchmark solely for scrap-yard crane/shear/baler feeders, because yards vary heavily in mobile diesel vs. electric crane share and whether shearing/shredding occurs on-site. Accordingly, TaqaMind V2 relies on **site-specific nameplate power (`rated_kw`), `planned_duration_hr`, `planned_utilization_pct`, and `planned_scrap_mass_t`** trained strictly **per factory** (`train_and_evaluate` per `factory_id`).

---

## 3. Pre-Production Predictors & Boundary Discipline

1. **Nameplate Rating, Planned Duration & Utilization (`rated_kw`, `planned_duration_hr`, `planned_utilization_pct`):**
   - Captures the scheduled electrical power envelope of the assigned crane/shear/baler line prior to batch processing.
2. **Planned Scrap Mass (`planned_scrap_mass_t`):**
   - Scheduled ferrous scrap tonnage for the reception/processing batch (`delivery_batch_id` grouping).
3. **Strict Carrier Isolation:**
   - Diesel mobile loaders/forklifts and upstream meltshop EAF transformers are explicitly listed in `excluded_components` inside `build_scrap_yard_energy_dataset_contract()`.

---

## 4. Egyptian Steel Mill Context

- **Substation Sub-Metering Reality:** In Egyptian steel plants, scrap yards are frequently metered at a secondary yard substation feeder serving overhead magnet cranes and stationary hydraulic shears/balers, while mobile grapple excavators run on diesel. Confirming whether crane + shear feeders are cleanly separated from mobile diesel operations and main EAF transformers is a mandatory `TargetConfirmation` gate before production training.
- **Idling Power Waste:** Consistent with UNIDO Egypt electric motor studies, running hydraulic power packs and magnet rectifiers unloaded during truck queuing or crane delays creates avoidable base-load `kWh`.

---

## 5. Traceable Sources

1. **UNIDO & Global Efficiency Intelligence (Egypt):** *Industrial Energy Efficiency & Electric Motor-Driven Systems Assessments in Egypt* (hydraulic, fan, and crane motor duty-cycle efficiency).
2. **EU Horizon ALCHIMIA Reference Context (Scrap Yard Logistics):** Physical reception, unloading, and scrap classification boundary prior to EAF charging (`docs/domains/iron_manufacturing/dataset-audit.md`).
3. **EU FMP / Iron & Steel BREF:** Qualitative separation of raw material handling / scrap preparation electrical auxiliaries vs. primary EAF melting.
