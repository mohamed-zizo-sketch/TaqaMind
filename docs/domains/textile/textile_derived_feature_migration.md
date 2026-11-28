# Textile Derived Feature Migration & Core Integration Request (Task 03)

- **Domain:** `textile` (`weaving`, `knitting`)
- **Shared Core Reference:** `feature/core/v1-integration-audit` @ `849027d` (`src/taqamind/core/feature_engineering/row_local.py`)
- **Policy:** Never enable `allow_unverified_callbacks`; never invent mathematically different approximations for `knitting_tightness_factor`.

---

## 1. Inventory of Textile `DerivedFeatureSpec` Definitions

| Feature Name | Process / Module | Current Callback | Mathematical Formula | RowLocalTransform Compatibility |
|---|---|---|---|---|
| `total_thread_density_per_inch` | `weaving` (`weaving/contracts.py`, `weaving/energy_contracts.py`) | `compute_weaving_thread_density_sum(epi, ppi)` | $\text{EPI} + \text{PPI}$ | **100% Supported** (`BinaryExpr(ADD)`) |
| `knitting_tightness_factor` | `knitting` (`knitting/contracts.py`, `knitting/energy_contracts.py`) | `compute_knitting_tightness_factor(tex, \ell)` | $K = \frac{\sqrt{\text{yarn\_count\_tex\_1}}}{\text{stitch\_length\_mm} / 10}$ | **Blocked — Missing `SQRT`** (`CIR-TEXTILE-001`) |
| `theoretical_nameplate_energy_kwh` | `weaving` & `knitting` (`*/energy_contracts.py`) | `compute_theoretical_nameplate_energy_kwh(kw, h)` | $P_{\text{rated}} \times t_{\text{planned}}$ | **100% Supported** (`BinaryExpr(MUL)`) |

---

## 2. Detailed Feature Specifications

### 2.1 Weaving — `total_thread_density_per_inch`
- **Industrial Meaning:** Total woven thread count per linear inch (warp ends per inch + weft picks per inch), governing fabric cover factor, warp/weft tension, and loom stop/waste propensity.
- **Formula:** `warp_ends_per_inch + weft_picks_per_inch`
- **Required Inputs & Units:** `warp_ends_per_inch` (`1/inch`, `FLOAT`, `> 0`), `weft_picks_per_inch` (`1/inch`, `FLOAT`, `> 0`).
- **Output Unit:** `1/inch`
- **Valid Range:** `(0, +inf)` (contract `minimum=0.0`, `minimum_inclusive=False`).
- **Missing / Invalid Input Behavior:** Propagates `NA` if nullable or fails contract validation before engineering; never fabricates values.
- **Zero-Denominator Behavior:** Not applicable (addition).
- **Availability & Leakage:** `PRE_PRODUCTION`, `FeatureOrigin.DERIVED`; depends strictly on pre-production loom/fabric spec inputs.
- **WP10 `RowLocalTransform` Equivalent:**
```python
BinaryExpr(
    op=BinaryOperator.ADD,
    left=ColumnRef(name="warp_ends_per_inch"),
    right=ColumnRef(name="weft_picks_per_inch"),
)
```

### 2.2 Knitting — `knitting_tightness_factor`
- **Industrial Meaning:** Munden's tightness factor $K$, quantifying knitted loop compactness from yarn linear density (Tex) and loop/stitch length (mm).
- **Formula:** $K = \frac{\sqrt{\text{yarn\_count\_tex\_1}}}{\text{stitch\_length\_mm} / 10}$
- **Required Inputs & Units:** `yarn_count_tex_1` (`tex`, `FLOAT`, `> 0`), `stitch_length_mm` (`mm`, `FLOAT`, `> 0`).
- **Output Unit:** `tex^0.5/cm`
- **Valid Range:** `(0, +inf)` (strictly positive).
- **Missing / Zero-Denominator Behavior:** Must reject/propagate `NA` on missing inputs and reject `stitch_length_mm <= 0` or `yarn_count_tex_1 <= 0`.
- **Availability & Leakage:** `PRE_PRODUCTION`, `FeatureOrigin.DERIVED`.
- **Current Blocker:** `taqamind.core.feature_engineering.row_local.RowLocalExpr` supports `ColumnRef`, `ConstantValue`, `BinaryExpr` (`ADD`, `SUB`, `MUL`, `DIV`, `SAFE_DIV`, `SAFE_DIV_POSITIVE`, comparisons), `UnitScaleExpr`, `WhereExpr`, and `DatetimePartExpr` — it has **no unary `SQRT` operator**. Per Task 03 rules, no polynomial/linear approximation may be substituted.

---

## 3. Core Integration Request (`CIR-TEXTILE-001`)

- **Request ID:** `CIR-TEXTILE-001`
- **Affected Module:** Shared Core Feature Engineering (`taqamind.core.feature_engineering.row_local`)
- **Affected Textile Process:** `knitting` (Waste V1 `knitting_tightness_factor`, Energy V2 `knitting_tightness_factor`)
- **Existing Shared Core Interface:** `RowLocalExpr` (`BinaryExpr`, `UnitScaleExpr`, `WhereExpr`, `DatetimePartExpr`)
- **Description of Limitation:** No unary square-root (`SQRT`) expression node exists in `RowLocalTransform`, blocking exact representation of Munden's tightness factor $K = \sqrt{\text{tex}} / (\ell / 10)$.
- **Scientific Justification:** Munden's tightness factor is a standard physical dimensionless/semi-dimensionless knitting parameter; altering the exponent changes physical meaning and breaks comparability across yarn counts.
- **Proposed Shared Capability:** Add `UnaryOperator.SQRT` / `UnaryExpr(op=UnaryOperator.SQRT, operand=RowLocalExpr)` with strict domain guard:
  - Reject negative inputs (`operand < 0` raises `ValueError` — never silently coerces to `NaN`!).
  - Preserve `NA` for missing values.
  - Reject boolean inputs.
  - Verify finite `Float64` output (`_ensure_finite_non_null`).
- **Example Input / Output:** `yarn_count_tex_1 = 25.0`, `stitch_length_mm = 2.5` -> `sqrt(25.0) / (2.5 / 10.0) = 5.0 / 0.25 = 20.0`.
- **Priority / Blocking Status:** **HIGH / Blocking** for WP10 Textile Knitting migration.
