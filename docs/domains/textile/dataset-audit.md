# Textile Dataset Audit Summary

Reproduced independently via `research/textile/audit_datasets.py` and recorded in `reports/textile/eda_summary.json`.

## Weaving (`full_weaving_dataset.csv`)

- **Rows / Columns:** 121,148 rows × 18 columns.
- **Exact Duplicate Rows:** 74,448 exact duplicate rows (46,700 unique rows; max repeat 156).
- **Order Grouping:** 1,009 unique `ID` values; 693 IDs exhibit varying target values across rows (top ID repeats 2,806 times).
- **Target (`Rej_and_cut_Piece`):** 105,721 zeros, 15,427 positive rows, 0 nulls; positive median = 28.0 yds, p95 = 329.0 yds, max = 11,353.0 yds, skewness = 10.86.
- **Physical Anomalies:** 140 rows have `Rej_and_cut_Piece > Req_beam_length(yds)` (max ratio 129.77x).
- **Missingness & Literals:** `act_shrink%` has 15,218 literal `"na"` strings; `Previous_pdn` has 13,637 nulls; `Total_pdn_m/c` has 13,630 nulls.
- **Time Granularity:** Only month names (`January`–`September`, no year/day) → temporal split is impossible; `SplitStrategy.GROUP` by `order_id` is mandatory.

## Knitting (`knitting_dataset.csv` & `raw_data_production_reports_20222.xlsx`)

- **Rows / Columns:** 12,569 rows × 38 columns (CSV), 12,569 data rows × 21 headers (Excel sheet `368_1661742005 (2)`).
- **Exact Duplicate Rows:** 6.
- **Target (`Reject Qty`):** 4,449 invalid/null values (4,448 NaN + 1 non-numeric text message), 7,822 zeros, 298 positive rows; positive median = 4.9 kg, p95 = 15.105 kg, max = 41.0 kg, skewness = 2.76.
- **Ratio Anomalies:** 10 rows with `Reject Qty / Shift Total > 100%` (max 840x); 959 rows with `Shift Total <= 5 kg` (setup/changeover regime).
- **Excel Yarn Count Corruption:** 5,150 cells in `Yarn Count` are stored as Jan-2022 datetime values (e.g., `2022-01-30` × 1,919, `2022-01-24` × 1,381) corresponding to `30/1`, `24/1`, etc.
- **Schema Notes:** No NBSP characters detected in CSV headers; secondary yarn columns are named `yarn_count_tex_2` and `yarn_count_tex_3`.
