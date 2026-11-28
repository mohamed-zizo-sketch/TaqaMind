# Textile Energy V2 — Core Integration Requests

## `CIR-TEXTILE-001` — Unary `SQRT` Operator in `RowLocalExpr`
- **Request ID:** `CIR-TEXTILE-001`
- **Affected Module:** `taqamind.core.feature_engineering.row_local`
- **Affected Process:** `knitting` (`knitting_tightness_factor` in Waste V1 & Energy V2)
- **Limitation:** `RowLocalExpr` lacks `UnaryOperator.SQRT`, blocking exact representation of $K = \sqrt{\text{tex}} / (\ell / 10)$.
- **Priority / Blocking:** **HIGH / Blocking** for WP10 Knitting migration.

## `CIR-TEXTILE-002` — Cross-Unit Intensity KPI Support (`kWh/kg`) in WP12
- **Request ID:** `CIR-TEXTILE-002`
- **Affected Module:** `taqamind.core.contracts.target` (`RateDefinition`) & `CorePredictionEngine` (WP12 Shared Energy Capabilities)
- **Affected Process:** `weaving`, `knitting` (Energy V2 secondary KPI)
- **Existing Shared Core Interface:** `RateDefinition` assumes same-unit percentage/rate (`output_unit = "%"`, `multiplier = 100.0`).
- **Limitation:** Specific Energy Consumption (`SEC = total_electrical_energy_kwh / planned_qty_kg`) divides electrical energy (`kWh`) by production mass (`kg`), producing `kWh/kg` rather than a dimensionless percentage.
- **Proposed Shared Capability:** Add `IntensityKPIDefinition` (or generalize `RateDefinition` for WP12) supporting cross-unit numerator/denominator with strict `denominator > 0` and `PRE_PRODUCTION` planned denominator guard.
- **Priority / Blocking:** **MEDIUM / Deferred to WP12** (does not block primary `kWh` regression).
