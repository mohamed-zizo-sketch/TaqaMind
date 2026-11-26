# TaqaMind V2 — Phase 2: Energy Measurement & Target Definition Report

**Status:** Approved for V2 design baseline  
**Task Scope:** Pre-Production Electrical Energy Consumption Prediction (`Weaving` & `Knitting`)

---

## 1. Energy Quantities & Unit Standardization

| Concept | Canonical Quantity | Canonical Unit | Notes |
| :--- | :--- | :--- | :--- |
| **Electrical Power (Instantaneous)** | Active electrical power | `kW` (or `W`, `MW`) | Instantaneous demand rate; **never** equated to total consumed energy (`kWh`). |
| **Electrical Energy (Consumed)** | Cumulative electrical work | **`kWh`** (or `Wh`, `MWh`) | $\text{Energy (kWh)} = \int P(t)\,dt$; primary prediction target for a production event. |
| **Specific Energy Intensity (KPI)** | Energy per unit output | **`kWh/kg`** and **`kWh/m`** | Derived post-prediction KPI using confirmed planned output mass or length. |
| **Thermal Energy (Out of V2 Primary Scope)** | Steam / fuel / heat | `MJ` or `GJ` | Kept strictly separate from electrical `kWh` (e.g., sizing steam or dyeing boilers). |
| **Compressed Air Demand** | Pneumatic flow / volume | `Nm³/h` or `Nm³` | Key explanatory / auxiliary factor for Air-Jet weaving; converted to `kWh` only with verified compressor `kWh/Nm³`. |
| **Energy Cost & Equipment Efficiency** | Monetary cost / ratio | `EGP/kWh`, `%` | Derived business KPI (`EGP`) and machine/motor efficiency (`%`). |

> **Core Guardrail:** Machine nameplate **rated power (`kW`)** is a static machine specification, **not** actual consumed electrical energy (`kWh`).

---

## 2. Measurement Boundaries

TaqaMind V2 strictly separates measurement boundaries and prohibits mixing machine-level and factory-level energy figures in the same contract:

1. **Primary V2 Prediction Boundary — `Machine-Run / Shift-Run / Batch` Level:**
   - **Weaving:** One planned beam batch or loom shift-run (`total_electrical_energy_kwh` = direct loom drive electricity + allocated compressed air electricity where sub-metered).
   - **Knitting:** One planned circular/flat knitting shift-run or lot batch (`total_electrical_energy_kwh` = machine drive + positive yarn feeders + control panel).
2. **Excluded from Machine-Level Target (unless explicitly sub-metered & contractually separated):**
   - Whole-factory HVAC, central humidification, general lighting, office loads, and wet-processing thermal boilers (`Line / Factory Level`).

---

## 3. Approved Target & Secondary Derived KPIs

- **Primary Target:**
  - `total_electrical_energy_kwh` (`kWh`, `DataType.FLOAT`, `minimum >= 0.0`, `FeatureAvailability.POST_PRODUCTION`, `TargetKind.QUANTITY`, `TaskType.REGRESSION`).
- **Secondary Derived KPIs (`RateDefinition` / Post-Prediction):**
  - **Mass-based Energy Intensity:** `kWh/kg` = `total_electrical_energy_kwh / planned_qty_kg` (primary KPI for Knitting; applicable to Weaving when planned fabric mass is present).
  - **Length-based Energy Intensity:** `kWh/m` (or `kWh/yd`) = `total_electrical_energy_kwh / planned_length_m` (primary KPI for Weaving).

All targets remain `TargetConfirmation.UNCONFIRMED` by default until an industrial owner confirms the factory's meter boundary and unit provenance.
