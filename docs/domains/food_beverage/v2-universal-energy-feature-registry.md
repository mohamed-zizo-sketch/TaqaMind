# v2-universal-energy-feature-registry.md

## Re‑used V1 Features
| Feature | Description | Role |
|---------|-------------|------|
| `factory_id` | Unique factory identifier (tenant isolation). | Identifier |
| `batch_id` | Unique production batch identifier. | Identifier |
| `production_line_id` | Line/oven identifier within the factory. | Identifier |
| `product_category` | Product family (e.g., bread, cheese, soda). | Predictor |
| `planned_batch_quantity_kg` | Planned output mass for the batch. | Predictor |
| `planned_raw_material_kg` | Planned ingredient mass. | Predictor (optional) |
| `planned_processing_temp_c` | Planned oven or processing temperature. | Predictor |
| `planned_processing_duration_min` | Planned duration of the process. | Predictor |
| `planned_shift` | Scheduled production shift. | Predictor |

## New Energy‑Specific Features (V2)
| Feature | Description | Unit | Role |
|---------|-------------|------|------|
| `machine_id` | Unique machine/loom/boiler identifier. | – | Identifier |
| `machine_rated_power_kw` | Nameplate electrical motor rating (kW). | kW | Predictor |
| `planned_operating_duration_hr` | Planned run time for the machine (hours). | h | Predictor |
| `planned_utilization_pct` | Expected utilization as a percentage (0‑100%). | % | Predictor (optional) |
| `ambient_temp_c` | Ambient temperature in the factory area. | °C | Predictor (optional) |
| `theoretical_nameplate_energy_kwh` | Deterministic calculation: `machine_rated_power_kw * planned_operating_duration_hr * (planned_utilization_pct/100)`. | kWh | Predictor (derived) |
| `total_electrical_energy_kwh` | **Target** – actual electricity consumed for the batch. | kWh | Target |

## Derived Feature Specification (used by the pipeline)
The pipeline will automatically generate `theoretical_nameplate_energy_kwh` via a `DerivedFeatureSpec` that applies the formula above. All other features are ingested directly.
