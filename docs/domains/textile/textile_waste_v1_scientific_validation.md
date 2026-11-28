# Textile Waste V1 — Scientific Validation (Task 05)

- **Domain:** `textile` (`weaving`, `knitting`)
- **Modeling Strategy:** Training-per-Factory (no cross-factory pooling assumed)

---

## 1. Target Definitions & Industrial Waste Boundaries

1. **Weaving (`weaving_waste_regression`):**
   - **Primary Target:** `rejected_fabric_yds` (`yds`, `FLOAT`, `>= 0`), representing rejected/defective woven fabric length per **beam batch** (`granularity="beam_batch"`).
   - **Secondary Rate (Optional/Confirmed):** `waste_rate_pct` when a valid positive denominator (`planned_beam_length_yds > 0`) is confirmed.
   - **Distinction:** Separates loom-beam batch fabric rejection from downstream garment cutting losses and rework.

2. **Knitting (`knitting_waste_regression`):**
   - **Primary Target:** `knitting_waste_kg` (`kg`, `FLOAT`, `>= 0`), representing circular knitting material waste mass per **shift run** (`granularity="shift_run"`).
   - **Distinction:** Captures yarn/fabric waste generated during the knitting shift run; incompatible with weaving length-based rejection without factory-specific conversion.

---

## 2. Feature Availability & Leakage Guardrails

| Process | Excluded Post-Production Columns (Never Pre-Production Predictors) | Business Keys | Grouping Columns | Time Column |
|---|---|---|---|---|
| **Weaving** | `act_shrink%`, `Total_pdn_m/c`, `Total_pdn_per_order`, `Previous_pdn` | `beam_batch_id` (or declared beam key) | `order_id` / `beam_group` | `run_timestamp` |
| **Knitting** | `Shift Total`, `Machine Total`, `SHIFT-A`, `SHIFT-B`, `SHIFT-C`, `Without Shift(Outbound)` | `shift_run_id` | `machine_order_group` | `run_timestamp` |

- **Post-Production Leakage Rule:** Post-production measurements (`FeatureAvailability.POST_PRODUCTION`) are retained in historical dataset contracts where needed for target/audit, but strictly rejected as pre-production predictors by `DatasetContract.validate_prediction_features()` and `CoreLeakageValidator`.
- **Granularity Consistency:** Weaving targets and predictors operate at `beam_batch` grain; Knitting targets and predictors operate at `shift_run` grain. Cross-grain denominators are prohibited.
