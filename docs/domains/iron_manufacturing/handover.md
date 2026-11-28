# Owner Handover Summary — `iron_manufacturing`

- **Domain ID:** `iron_manufacturing` (`v1.0.0`)
- **Cleanup Branch:** `refactor/iron_manufacturing/retire-v1-adapters` (branched from `origin/develop` @ `5b5e4b6`)
- **Shared Core Integration:** Direct `CorePlatformServices.create_local(...)` wiring for **P1.1–P1.8** across **both** V1 Waste (`IronManufacturingPipeline`) and V2 Electrical Energy (`IronEnergyPipelineService`).
- **Shared Core / Other Domains Modified:** **None** (100% strictly within `iron_manufacturing`-owned paths).

## 1. V1 Waste vs V2 Electrical Energy Status

| Platform Module | Processes & Tasks | Core Platform Wiring | Software Verification | Real Egyptian Factory Production Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 `WASTE_MANAGEMENT`** | `eaf_meltshop` (`eaf_metallic_waste_loss_prediction`), `rebar_hot_rolling` (`rebar_rolling_waste_quantity_prediction`), `scrap_yard_reception` (`scrap_delivery_contaminant_prediction`) | Direct `CorePlatformServices.create_local(...)` for P1.1–P1.8 + thin domain wrappers in `adapters/unit_mapper.py` (AUD-001/004 & iron physical rules) | **VERIFIED** (`20/20` V1 tests passing on synthetic fixtures) | **BLOCKED** — all targets default to `ConfirmationStatus.UNCONFIRMED` pending Egyptian factory CSV onboarding and target sign-off |
| **V2 `ENERGY_EFFICIENCY`** | `eaf_meltshop` (`eaf_electrical_energy_regression`), `rebar_hot_rolling` (`rebar_electrical_energy_regression`), `scrap_yard_reception` (`scrap_yard_electrical_energy_regression`) | Direct `CorePlatformServices.create_local(...)` for P1.1–P1.8 (zero local ML adapters) | **VERIFIED** (`10/10` V2 tests passing across Windows & Linux on synthetic fixtures) | **BLOCKED** — all targets default to `ConfirmationStatus.UNCONFIRMED` pending factory sub-meter boundary confirmation |

## 2. Retired Duplicated Adapters (P1.1–P1.8 Cleanup Complete)
All duplicated local P1.1–P1.8 engines under `src/taqamind/domains/iron_manufacturing/adapters/` have been retired:
- **Retired in P1.1–P1.5 sync:** `storage.py`, `dataset_registry.py`, `processing.py`.
- **Retired in `refactor/iron_manufacturing/retire-v1-adapters` (P1.6–P1.8):** `_store.py`, `data_access.py`, `model_registry.py`, `modeling.py`, `readiness_and_splitting.py`, `training_registry.py`.
- **Retained domain-only wrapper:** `adapters/unit_mapper.py` (`IronUnitAwareColumnMapper` for verified SI unit conversions / AUD-001, string dtype preservation / AUD-004, ambiguous header & post-production rejection; `IronSemanticValidator` for iron physical balance checks; `ProcessFeatureEngineerRouter` dispatching to Core `DeterministicFeatureEngineer`).

## 3. Delivered Package Overview
1. **Contracts & Multi-Module Domain Registration (`domain.py`, `processes/`, `energy_contracts.py`):**
   - `register_iron_manufacturing(registry)` registers all 3 processes with strictly separated V1 `WASTE_MANAGEMENT` and V2 `ENERGY_EFFICIENCY` tasks (`UNCONFIRMED` by default).
2. **Services, Pipelines & CLI (`pipeline.py`, `energy_pipeline.py`, `services/`, `cli.py`):**
   - Both V1 (`IronManufacturingPipeline`) and V2 (`IronEnergyPipelineService`) delegate P1.1–P1.8 execution directly to `CorePlatformServices.create_local(...)`.
   - CLI (`python -m taqamind.domains.iron_manufacturing.cli`) supports `--module waste|energy` across `inspect`, `template`, `validate`, `train`, `predict`, `retrain`, `model-info`, and `rollback`.
3. **Automated Tests (`tests/domains/iron_manufacturing/`):**
   - Full V1 + V2 offline suite (`30` tests): contracts, unit-safety (unknown/count-to-mass/mass-to-length rejection), Core P1.1–P1.8 wiring conformance, lineage, factory isolation, readiness, leakage, train-only preprocessing, zero-retraining spy tests, explicit retraining/promotion/rollback, and CLI parity.
4. **Machine-Readable Manifest & Proposal:** See [handover-manifest.json](./handover-manifest.json), [core-contribution-proposal.md](./core-contribution-proposal.md), and [v2-technical-handover.md](./v2-technical-handover.md).
