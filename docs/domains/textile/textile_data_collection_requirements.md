# Textile Data Collection Requirements for Egyptian Factory Onboarding (Task 12)

To train factory-specific Waste V1 and Energy V2 models (`Training-per-Factory`), each Egyptian textile mill must provide:

1. **Unique Business Keys & Grouping Identifiers:**
   - Weaving: `batch_record_id` (unique per beam batch), `order_id` (for group-safe splitting), `machine_id`, `scheduled_date` (ISO 8601).
   - Knitting: `run_id` (unique per shift run), `order_lot_id`, `machine_id`, `production_date` (ISO 8601).
2. **Pre-Production Machine & Production Settings:**
   - Nameplate `machine_rated_power_kw` (`kW`), `loom_type` / `knitting_machine_type`, planned speed (`rpm`), planned operating duration (`h`), planned length (`m`/`yd`) or mass (`kg`), yarn counts (`Ne` or `tex`), fabric density / gauge / stitch length.
3. **Sub-Metered Target Measurements (Explicit Factory Sign-Off Required):**
   - Waste V1: Confirmed `waste_yds` (weaving) or `waste_kg` (knitting) with clear boundary (setup scrap vs defective rejection).
   - Energy V2: Sub-metered electrical `total_electrical_energy_kwh` (`kWh`) aligned to the exact same `beam_batch` or `shift_run` grain.
4. **Unit & Provenance Declaration:**
   - Explicit source units for every column (`m` vs `yd`, `g` vs `kg`, `tex` vs `Ne`); never mix unconfirmed post-production actuals into pre-production predictors.
