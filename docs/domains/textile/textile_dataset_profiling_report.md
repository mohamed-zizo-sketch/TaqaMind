# Textile Dataset Profiling Report (Task 12)

- **Source Evidence:** `research/textile/v2_eda_reference_summary.json` and `docs/domains/textile/dataset-audit.md`.
- **Limitation Disclosed:** Real Egyptian factory-level electrical sub-metering CSVs (`kWh` per beam batch / shift run) are **not publicly available**; synthetic datasets (`weaving/energy_synthetic.py`, `knitting/energy_synthetic.py`) are used strictly for contract, numerical safety, and pipeline engineering verification — **not** as evidence of real-world predictive accuracy.

## 1. Existing V1 Reference Datasets
- **Weaving (`full_weaving_dataset.csv`):**
  - Dimensions: `121,148` rows × `18` columns.
  - Exact duplicates: `74,448` rows (must not be blindly dropped without distinguishing repeated machine settings vs accidental duplicate exports).
  - Reusable pre-production predictors: `planned_beam_length_yds`, `planned_finish_fabric_yds`, `planned_grey_fabric_yds`, `fabric_allowance_pct`, `shrink_allowance_pct`, `warp_count_ne`, `weft_count_ne`, `warp_ends_per_inch`, `weft_picks_per_inch`, `fabric_construction`, `total_thread_density_per_inch`.
- **Knitting (`knitting_dataset.csv`):**
  - Dimensions: `12,569` rows × `38` columns.
  - Exact duplicates: `6` rows.
  - Reusable pre-production predictors: `planned_qty_kg`, `machine_brand`, `machine_diameter_in`, `machine_gauge_e`, `yarn_count_tex_1`, `stitch_length_mm`, `target_gsm`, `number_of_feeders`, `knitting_tightness_factor`.
