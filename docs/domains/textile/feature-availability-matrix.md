# Feature Availability Matrix

Only **PRE_PRODUCTION** features are allowed as predictors in V1. Post-production measurements and targets are strictly excluded at prediction time (`DatasetContract.validate_prediction_features()`).

| Feature / Column | Process | Pre-Production Planning | Machine Setup | Mid-Run | Post-Production | V1 Predictor? |
|---|---|---|---|---|---|---|
| `planned_beam_length_yds`, `planned_grey_fabric_yds`, `planned_finish_fabric_yds` | Weaving | Yes | Yes | Yes | Yes | **Yes** |
| `warp_count_ne`, `weft_count_ne`, `warp_ends_per_inch`, `weft_picks_per_inch` | Weaving | Yes | Yes | Yes | Yes | **Yes** |
| `total_thread_density_per_inch` (derived) | Weaving | Yes | Yes | Yes | Yes | **Yes** |
| `act_shrink%`, `Total_pdn_m/c`, `Total_pdn_per_order`, `Previous_pdn` | Weaving | No | No | Partial | Yes | **NO (Excluded)** |
| `waste_length_yds` (`Rej_and_cut_Piece`) | Weaving | No | No | No | Yes | **Target Only** |
| `planned_qty_kg`, `machine_diameter_in`, `machine_gauge_e`, `yarn_count_tex_1`, `stitch_length_mm` | Knitting | Yes | Yes | Yes | Yes | **Yes** |
| `knitting_tightness_factor` (derived) | Knitting | Yes | Yes | Yes | Yes | **Yes** |
| `Shift Total`, `Machine Total`, `SHIFT-A/B/C`, `Without Shift(Outbound)` | Knitting | No | No | Partial | Yes | **NO (Excluded)** |
| `waste_qty_kg` (`Reject Qty`) | Knitting | No | No | No | Yes | **Target Only** |
