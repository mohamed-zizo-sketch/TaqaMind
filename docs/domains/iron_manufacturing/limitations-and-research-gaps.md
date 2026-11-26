# Limitations, Local Adapter Constraints & Research Gaps — `iron_manufacturing`

## 1. Production Readiness Status
- **Software Pipeline Status:** **Implemented & Verified** on deterministic synthetic fixtures across all 3 processes (`eaf_meltshop`, `rebar_hot_rolling`, `scrap_yard_reception`).
- **Industrial Production Readiness Status:** **`BLOCKED` (NOT production-ready yet)** — synthetic tests verify software correctness, leakage guards, and zero-retraining architecture, **not** predictive accuracy on real Egyptian steel plants.

## 2. Unresolved Industrial Research Gaps & Factory Confirmations Required
1. **No Real Egyptian Labeled Factory Data Yet:** All 5 supplied raw datasets are foreign or synthetic/parameter tables (`ALCHIMIA` EU, `Yandex LF`, `UCI 851` Korea, `Zenodo 22162000/21447633`). Real 6–12 month Egyptian Level-2 MES + ERP historical CSVs are required per facility.
2. **EAF Target & Hot-Heel Accounting Confirmation (`eaf_meltshop`):** Partner Egyptian factories must confirm whether `actual_liquid_steel_tapped_t` comes from ladle crane load cells or billet counts, how retained hot heel is estimated, and whether recovered ladle/tundish skulls are credited back.
3. **Rolling Mill Scale Measurement (`rebar_hot_rolling`):** Factories must confirm whether `mill_scale_kg` is directly weighed or estimated, and whether commercial short bars are excluded from waste.
4. **Scrap Yard Weighbridge Timing (`scrap_yard_reception`):** Confirm whether `gross_delivery_mass_kg` is recorded at entry weighbridge before unloading and whether `Soil`/`Water`/`Other` deductions are physically weighed or visually estimated.
5. **DRI Lab Assay Timing:** Confirm `dri_metallization_pct` and `dri_gangue_pct` from DRP LIMS are available prior to `t0`.

## 3. Technical Limitations of Temporary Local Adapters
- **Local Filesystem Concurrency:** Atomic `os.replace` writes prevent torn JSON files in single-process workflows, but multi-worker concurrent production deployments require transactional database locks (PostgreSQL + RLS).
- **Resource Budget Enforcement:** `TrainingBudget.max_trials` and `max_parallel_jobs=1` are strictly enforced; hard wall-clock/memory kill limits (`max_training_seconds`, `max_memory_mb`) are metadata-documented in the local adapter and require OS worker-process isolation when migrated to Shared Core.
- **No Production Auth / DB / API:** Explicitly deferred out of scope per TaqaMind architecture phase.
- **Future Modules Deferred:** Energy Efficiency (`SEC`, EgyptERA `0.92` PF), Business/Cost (`EGP/t` leakage), and Operations Research (`OR-1` TOU scheduling, `OR-2` charge-mix optimization) are documented in `domain-specification.md` and intentionally not executed in this V1 waste-regression package.
