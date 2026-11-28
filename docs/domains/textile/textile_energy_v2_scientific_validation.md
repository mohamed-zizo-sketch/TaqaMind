# Textile Energy V2 — Scientific Validation & Target Architecture (Task 07)

- **Primary Target:** `total_electrical_energy_kwh` (`kWh`, `TargetKind.QUANTITY`, `>= 0`).
- **Secondary KPI (Deferred to WP12 / Business Analytics):** Specific Energy Consumption (`SEC = kWh / kg`).

## 1. Physical Distinctions Enforced
- **Power (`kW`) vs Energy (`kWh`):** `machine_rated_power_kw` is an instantaneous nameplate capacity; `total_electrical_energy_kwh` is time-integrated electrical work ($E = \int P(t)\,dt$).
- **Electrical vs Non-Electrical Carriers:** Electricity (`kWh`) is kept strictly separate from thermal energy, natural gas, steam, and compressed air (`Nm^3`) — incompatible carriers are never summed into one target.
- **No Waste V1 Rate Coercion:** `kWh / kg` is a cross-unit intensity ratio, **not** a dimensionless same-unit percentage rate (`RateDefinition` in Shared Core). Therefore, `kWh/kg` is not forced into Waste V1's `RateDefinition` and is documented in `CIR-TEXTILE-002` for WP12.

## 2. Granularity Alignment
- **Weaving:** `beam_batch` (`batch_record_id`, grouped by `order_id`, ordered by `scheduled_date`).
- **Knitting:** `shift_run` (`run_id`, grouped by `order_lot_id`, ordered by `production_date`).
- **Reference SEC Ranges (Literature Only — Non-Egyptian Benchmarks):**
  - Weaving: `1.58 – 5.8 kWh/kg` (UNIDO / LBNL-3970E / IFTH).
  - Circular Knitting: `0.5 – 1.5 kWh/kg`; Flat Panel: `0.85 – 1.2 kWh/kg`; Flat Fashioned: `4.59 – 5.42 kWh/kg`.
