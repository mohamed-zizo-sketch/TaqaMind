# Textile Domain Specification

## Overview

The `textile` domain (`TEXTILE_DOMAIN_ID = "textile"`, version `1.0.0`) models Egyptian textile manufacturing processes under the TaqaMind **Train Per Factory** architecture.

## Registered Processes & V1 Scope

| Process ID | Task ID | Dataset ID | Module | Task Type | Target | Unit |
|---|---|---|---|---|---|---|
| `weaving` | `weaving_waste_regression` | `weaving_beam_batches` | `waste_management` | `regression` | `waste_length_yds` | `yd` |
| `knitting` | `knitting_waste_regression` | `knitting_shift_runs` | `waste_management` | `regression` | `waste_qty_kg` | `kg` |

## Architectural Principles

1. **Train Per Factory:** Models, splits, lineage, and registries are strictly scoped by `factory_id`.
2. **Pre-Production Prediction Only:** Only features available before machine execution are permitted as predictors.
3. **Unconfirmed Target Default:** `DatasetContract.target.confirmation` defaults to `UNCONFIRMED`, blocking production training until an explicit industrial sign-off is provided.
4. **Derived Rate:** Quantity predictions are primary; waste rate (%) is computed deterministically from `predicted_quantity / planned_quantity * 100` with explicit undefined guards when denominator <= 0.
