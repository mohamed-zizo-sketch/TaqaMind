# Customer Data Requirements & Templates — `iron_manufacturing`

## 1. Template Artifacts
All templates and examples live under `contracts/domains/iron_manufacturing/templates/`:
- **EAF Meltshop:**
  - Historical training CSV (includes target): `eaf_meltshop_historical_training_template.csv`
  - Prediction-only CSV (no target / no post-production columns): `eaf_meltshop_prediction_only_template.csv`
  - Feature dictionary & mapping example: `eaf_meltshop_feature_dictionary.json`, `eaf_meltshop_mapping_example.yaml`
- **Rebar Hot Rolling:**
  - Historical training CSV: `rebar_rolling_historical_training_template.csv` (`rebar_hot_rolling_historical_training_template.csv`)
  - Prediction-only CSV: `rebar_hot_rolling_prediction_only_template.csv`
  - Feature dictionary & mapping example: `rebar_hot_rolling_feature_dictionary.json`, `rebar_hot_rolling_mapping_example.yaml`
- **Scrap Yard Reception:**
  - Historical training CSV: `scrap_yard_reception_historical_training_template.csv`
  - Prediction-only CSV: `scrap_yard_reception_prediction_only_template.csv`
  - Feature dictionary & mapping example: `scrap_yard_reception_feature_dictionary.json`, `scrap_yard_reception_mapping_example.yaml`

Regenerate any template via CLI:
```powershell
python -m taqamind.domains.iron_manufacturing.cli template --process eaf_meltshop
```

## 2. What the Factory Must Know Before Production (`t <= t0`)
- **Stable unique business key** (`heat_id`, `rolling_lot_id`, or `scrap_delivery_id`) — never reused across different production events.
- **Grouping key** (`production_order_id` or `supplier_group_id`) — links dependent batches for leakage-free splitting.
- **Planned timestamp** (`planned_start_ts` or `entry_ts`, ISO-8601) — establishes strict chronological ordering.
- **Mandatory planned scale & specification inputs** in explicit documented units (`metric_ton`, `kg`, `mm`, `degC`). Never submit unlabeled `Weight` or `Waste` headers without mapping confirmation.

## 3. What Must Be Collected After Production (Monitoring & Future Explicit Retraining)
- **Measured target quantity** (`metallic_waste_loss_t`, `total_rolling_waste_kg`, or `total_contaminant_kg`) using signed-off scale reconciliation.
- **Zero-waste records** must be logged honestly as `0.0` (never left blank; missing targets are rejected, never imputed as zero).
- **Corrections** to historical records must retain the original business key and include a confirmed correction reason; they never replace the original immutable version.
- Post-production measurements are used **only** for offline monitoring and **explicitly authorized retraining** — never passed into pre-production inference.
