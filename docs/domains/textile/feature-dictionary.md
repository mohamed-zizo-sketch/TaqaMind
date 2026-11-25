# Textile Feature Dictionary

## Weaving (`weaving_beam_batches`)

| Canonical Name | Role | Type | Unit | Availability | Missing Policy | Aliases |
|---|---|---|---|---|---|---|
| `batch_record_id` | IDENTIFIER | STRING | - | PRE_PRODUCTION | REJECT | `Record_ID` |
| `order_id` | IDENTIFIER | STRING | - | PRE_PRODUCTION | REJECT | `ID` |
| `planned_beam_length_yds` | PREDICTOR | FLOAT | `yd` | PRE_PRODUCTION | REJECT | `Req_beam_length(yds)`, `Rec_Beam_length(yds)` |
| `planned_finish_fabric_yds` | PREDICTOR | FLOAT | `yd` | PRE_PRODUCTION | REJECT | `Req_Finish_Fabrics` |
| `planned_grey_fabric_yds` | PREDICTOR | FLOAT | `yd` | PRE_PRODUCTION | REJECT | `Req_grey_fabric` |
| `fabric_allowance_pct` | PREDICTOR | FLOAT | `%` | PRE_PRODUCTION | IMPUTE | `Fabric_Allowance` |
| `shrink_allowance_pct` | PREDICTOR | FLOAT | `%` | PRE_PRODUCTION | IMPUTE | `Shrink_allow` |
| `warp_count_ne` | PREDICTOR | FLOAT | `Ne` | PRE_PRODUCTION | REJECT | `warp_count` |
| `weft_count_ne` | PREDICTOR | FLOAT | `Ne` | PRE_PRODUCTION | REJECT | `weft_count` |
| `warp_ends_per_inch` | PREDICTOR | FLOAT | `ends/in` | PRE_PRODUCTION | REJECT | `epi` |
| `weft_picks_per_inch` | PREDICTOR | FLOAT | `picks/in` | PRE_PRODUCTION | REJECT | `ppi` |
| `fabric_construction` | PREDICTOR | CATEGORY | - | PRE_PRODUCTION | IMPUTE | `Construction` |
| `total_thread_density_per_inch` | PREDICTOR (DERIVED) | FLOAT | `threads/in` | PRE_PRODUCTION | IMPUTE | - |
| `waste_length_yds` | TARGET | FLOAT | `yd` | POST_PRODUCTION | REJECT | `Rej_and_cut_Piece` |

## Knitting (`knitting_shift_runs`)

| Canonical Name | Role | Type | Unit | Availability | Missing Policy | Aliases |
|---|---|---|---|---|---|---|
| `run_id` | IDENTIFIER | STRING | - | PRE_PRODUCTION | REJECT | - |
| `order_lot_id` | IDENTIFIER | STRING | - | PRE_PRODUCTION | REJECT | `Lot` |
| `production_date` | TIMESTAMP | DATE | - | PRE_PRODUCTION | REJECT | `Production Date` |
| `planned_qty_kg` | PREDICTOR | FLOAT | `kg` | PRE_PRODUCTION | REJECT | - |
| `machine_brand` | PREDICTOR | CATEGORY | - | PRE_PRODUCTION | IMPUTE | `M/C Brand` |
| `machine_diameter_in` | PREDICTOR | FLOAT | `in` | PRE_PRODUCTION | REJECT | `m/c_dia` |
| `machine_gauge_e` | PREDICTOR | FLOAT | `E` | PRE_PRODUCTION | REJECT | `m/c_gauge` |
| `yarn_count_tex_1` | PREDICTOR | FLOAT | `tex` | PRE_PRODUCTION | REJECT | - |
| `stitch_length_mm` | PREDICTOR | FLOAT | `mm` | PRE_PRODUCTION | REJECT | `stitch_1`, `stitch` |
| `target_gsm` | PREDICTOR | FLOAT | `g/m2` | PRE_PRODUCTION | IMPUTE | `GSM` |
| `number_of_feeders` | PREDICTOR | INTEGER | - | PRE_PRODUCTION | IMPUTE | - |
| `knitting_tightness_factor` | PREDICTOR (DERIVED) | FLOAT | `tex^0.5/cm` | PRE_PRODUCTION | IMPUTE | - |
| `waste_qty_kg` | TARGET | FLOAT | `kg` | POST_PRODUCTION | REJECT | `Reject Qty` |
