# TaqaMind — Food & Beverage Customer Data Collection Requirements

**Domain ID:** `food_beverage`  
**Supported Processes:** `bakery`, `dairy`, `beverage`  
**Templates Location:** `contracts/domains/food_beverage/templates/` (also generatable via CLI `template` command)

---

## 1. Timeline of Data Collection (Pre-Production vs. Post-Production)

To prevent data leakage, factories must separate variables recorded **before** batch start from those measured **after** batch completion:

| Timing | Required / Recommended Fields | Purpose |
| :--- | :--- | :--- |
| **Before Production** (`pre_production`) | `factory_id`, `batch_id`, `production_timestamp`, `production_line_id`, `product_category`, `planned_batch_quantity_kg` (or `planned_batch_volume_l`), `planned_processing_temp_c`, `planned_processing_duration_min` | Used in **both** historical training and live pre-production prediction. |
| **During Production** (`in_production`) | Actual sensor temperatures, line stoppages, boiler kW | Reserved for Module 2 (Energy) & monitoring; **excluded** from pre-production waste prediction. |
| **After Production** (`post_production`) | `waste_quantity_kg` (or `waste_quantity_l`) | Required **only** in historical training CSVs (and optional post-hoc monitoring). **Never** included in prediction CSVs. |

---

## 2. Target Measurement Rules by Process

- **Bakery (`waste_quantity_kg`):** Weigh spoiled dough, burnt/collapsed loaves, slicing scrap, and QC rejects. **Exclude** baking moisture evaporation, reused rework dough, and animal feed by-products.
- **Dairy (`waste_quantity_kg`):** Measure CIP milk-water push-out loss, filling spillage, and rejected packs. **Exclude** recoverable cheese whey and CIP rinse water.
- **Beverage (`waste_quantity_l`):** Measure line purge, overfill/foaming spillage, and rejected filled containers. **Exclude** spent grain/yeast and empty bottle/can scrap.

---

## 3. Formatting & Unit Standards

- **File Format:** UTF-8 CSV with comma delimiter and single header row.
- **Units:** Mass in `kg`, liquid volume in `L`, temperature in `degC`, duration in `min`. Approved aliases and convertible units (e.g., `g`, `ton`, `hL`, `h`) are normalized automatically only after explicit unit confirmation.
- **Missing Values:** Leave truly unknown values blank; **never** enter `0` for an unmeasured waste quantity.
