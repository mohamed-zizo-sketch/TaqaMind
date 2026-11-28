"""Domain-specific thin wrappers around Shared Core processing engines.

All Shared Core P1.1-P1.8 engines (storage, registries, lineage, ingestion,
profiling, mapping, validation, cleaning, feature engineering, readiness,
splitting, leakage validation, AutoML training, protected evaluation, model
registry, and zero-retraining prediction) are provided directly by
`CorePlatformServices.create_local(...)` from `taqamind.core.pipeline`.

Only iron_manufacturing domain wrappers remain here:
    - `IronUnitAwareColumnMapper`: AUD-001 SI unit conversions, AUD-004 string/category
      dtype preservation, ambiguous header refusal, and post-production leakage blocking.
    - `IronSemanticValidator`: composes Core `ContractSemanticValidator` with iron
      physical mass-balance rules.
    - `ProcessFeatureEngineerRouter`: dispatches each iron process to its dedicated
      Core `DeterministicFeatureEngineer` configured with `DerivedFeatureSpec`.
"""

from taqamind.domains.iron_manufacturing.adapters.unit_mapper import (
    IronSemanticValidator,
    IronUnitAwareColumnMapper,
    ProcessFeatureEngineerRouter,
)

__all__ = [
    "IronSemanticValidator",
    "IronUnitAwareColumnMapper",
    "ProcessFeatureEngineerRouter",
]
