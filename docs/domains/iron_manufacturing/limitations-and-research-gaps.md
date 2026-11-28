# Limitations, Platform Constraints & Research Gaps — `iron_manufacturing`

## 1. Software vs Industrial Production Status (V1 Waste & V2 Energy)
- **Software Pipeline Status (V1 Waste + V2 Electrical Energy):** **Implemented & Verified** directly on Shared Core `CorePlatformServices.create_local(...)` (P1.1–P1.8) across all 3 processes (`eaf_meltshop`, `rebar_hot_rolling`, `scrap_yard_reception`) with zero duplicated P1.6–P1.8 adapters.
- **Industrial Production Readiness Status:** **`BLOCKED` (NOT production-ready yet)** — all V1 waste targets and V2 electrical energy targets ship with `ConfirmationStatus.UNCONFIRMED`. Offline synthetic tests verify software architecture, leakage prevention, unit safety, factory isolation, and zero-retraining inference — **not** predictive accuracy on real Egyptian steel plants.

## 2. Unresolved Industrial Research Gaps & Factory Confirmations Required
1. **No Real Egyptian Labeled Factory Data Yet (V1 & V2):** Public datasets audited (`ALCHIMIA` EU, `Yandex LF`, `UCI 851` Korea 15-min plant meter, `Zenodo 22162000/21447633`) are foreign, wrong-granularity, or synthetic/parameter tables. Real 6–12 month Egyptian Level-2 MES + ERP + sub-metered `kWh` CSVs are required per facility.
2. **V1 EAF Target & Hot-Heel Accounting (`eaf_meltshop`):** Partner factories must confirm `actual_liquid_steel_tapped_t` measurement source, retained hot-heel estimation, and ladle/tundish skull crediting.
3. **V1 Rolling Mill Scale & Commercial Short Bars (`rebar_hot_rolling`):** Confirm whether `mill_scale_kg` is weighed vs estimated and whether commercial short bars are excluded from waste.
4. **V1 Scrap Yard Weighbridge Timing (`scrap_yard_reception`):** Confirm whether `gross_delivery_mass_kg` is captured at entry weighbridge prior to unloading and whether contaminant deductions are weighed vs visually estimated.
5. **V2 Sub-Metering Boundary Confirmation (All 3 Processes):** Confirm sub-metered active `kWh` isolation from natural gas thermal fuels (oxy-fuel burners, DRI shaft gas, reheating furnace gas), diesel mobile loaders, and whole-plant HVAC/office feeders before confirming V2 targets.

## 3. Local Runtime Constraints
- **Local Filesystem Persistence (`CorePlatformServices.create_local`):** Suitable for single-node local validation and pilot workflows; concurrent multi-worker production deployment requires transactional database/object storage backends behind Shared Core protocols.
- **Windows `MAX_PATH` Note:** Deep Windows temp paths (>260 chars) combined with nested `CorePlatformServices` lineage/model keys can trigger `WinError 3` in Core `_json_store.py` unless a short storage root is used (or Windows long paths are enabled). Domain tests use a short `storage_root` fixture in `tests/domains/iron_manufacturing/conftest.py`.
- **Future Modules Still Out of Scope:** Business Analytics / Cost Optimization (`EGP/t`) and Operations Research (`OR-1` TOU scheduling, `OR-2` charge-mix blending) remain documented scope only.
