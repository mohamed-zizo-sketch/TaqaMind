# v2-target-and-measurement-boundary.md

## Measurement Boundary
- **Target variable**: `total_electrical_energy_kwh` – the net electrical energy consumed (kWh) by a batch or shift run.
- **Boundary**: per *factory* → *production line* → *batch* (or *shift*) granularity. No aggregation across factories.
- **Exclusions**:
  - Thermal energy (MJ/GJ) from gas, steam, or other fuels.
  - Indirect electricity (facility lighting, office HVAC) not tied to the production line.
- **Source metric**: Direct meter readings from machine‑level electricity meters (or aggregated line meters) calibrated to kWh.

## Naming Conventions
- Columns ending with `_kwh` represent **electrical** consumption.
- Columns ending with `_mj` or `_gj` represent **thermal** energy and are **not** part of the target.

## Unit Safety
- All electrical energy columns must use unit `kWh`.
- Any conversion from power (kW) × time (h) must be performed **before** target calculation.

## Validation Rules
- `total_electrical_energy_kwh` must be non‑negative.
- Missing values are **rejected** (dataset ingestion will fail).
- Target confirmation defaults to `UNCONFIRMED` for real factories; synthetic fixtures may set `CONFIRMED`.
