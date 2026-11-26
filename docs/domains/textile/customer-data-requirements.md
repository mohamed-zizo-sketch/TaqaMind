# Customer Data Requirements & Templates

Customers onboard a new factory by exporting CSV files matching the generated templates under `contracts/domains/textile/templates/`.

## Template Files

- **Weaving:**
  - `contracts/domains/textile/templates/weaving/weaving_historical_training_template.csv`
  - `contracts/domains/textile/templates/weaving/weaving_prediction_only_template.csv`
  - `contracts/domains/textile/templates/weaving/weaving_data_dictionary.json`
- **Knitting:**
  - `contracts/domains/textile/templates/knitting/knitting_historical_training_template.csv`
  - `contracts/domains/textile/templates/knitting/knitting_prediction_only_template.csv`
  - `contracts/domains/textile/templates/knitting/knitting_data_dictionary.json`

## Key Requirements

1. **Preserve String Identifiers:** Do not strip leading zeros from `batch_record_id`, `order_id`, `run_id`, or `order_lot_id`.
2. **Pre-Production Only at Prediction Time:** Never supply post-production columns (`act_shrink%`, `Total_pdn_m/c`, `Shift Total`, `Machine Total`, or target columns) in prediction files.
3. **Explicit Units:** Declare source units if different from canonical units (`yd` for weaving lengths, `kg` for knitting mass, `tex` for knitting linear density).
4. **Yarn Count Formatting:** Export knitting yarn counts as text (`30/1`, `24/1`) rather than Excel date serials.
