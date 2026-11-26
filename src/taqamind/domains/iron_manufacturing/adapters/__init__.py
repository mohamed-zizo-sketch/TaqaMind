"""Protocol-compatible local adapters for Shared Core P1.6-P1.8 gaps and audit mitigations.

Shared Core P1.1-P1.5 engines (`LocalDatasetArtifactStorage`, `LocalModelArtifactStorage`,
`LocalDatasetRegistry`, `LocalRecordLineageRegistry`, `CsvDataIngestion`, `PandasDataProfiler`,
`DeterministicColumnMapper`, `ContractSemanticValidator`, `ContractDataCleaner`,
`DeterministicFeatureEngineer`, and `TabularPreprocessor`) are used directly from
`taqamind.core.*`.

Remaining adapters in this package cover ONLY:
    - P1.6: Readiness, splitting, leakage validation, split & usage registries.
    - P1.7: Development/protected-test data providers, training/evaluation registries,
      controlled AutoML trainer, protected final evaluator.
    - P1.8: Model registry, active model provider, zero-retraining prediction engine.
    - Thin Core audit mitigation wrappers (`IronUnitAwareColumnMapper` for AUD-001/AUD-004,
      `IronSemanticValidator` for iron physical mass-balance checks, and
      `ProcessFeatureEngineerRouter` dispatching to Core `DeterministicFeatureEngineer`).
"""

from taqamind.domains.iron_manufacturing.adapters._store import (
    atomic_write_json,
    df_to_csv_bytes,
    read_artifact_df,
    read_json,
    strip_computed_fields,
)
from taqamind.domains.iron_manufacturing.adapters.data_access import (
    LocalActiveModelProvider,
    LocalDevelopmentDataProvider,
    LocalProtectedTestDataProvider,
)
from taqamind.domains.iron_manufacturing.adapters.model_registry import (
    LocalModelRegistry,
)
from taqamind.domains.iron_manufacturing.adapters.modeling import (
    ControlledModelTrainer,
    HistoricalRateRegressor,
    ProtectedFinalModelEvaluator,
    TwoStageHurdleRegressor,
    ZeroRetrainingPredictionEngine,
    compute_regression_metrics,
    deserialize_trusted_bundle,
    serialize_trusted_bundle,
)
from taqamind.domains.iron_manufacturing.adapters.readiness_and_splitting import (
    LocalDatasetSplitter,
    LocalLeakageValidator,
    LocalTrainingReadinessChecker,
    build_record_scope,
)
from taqamind.domains.iron_manufacturing.adapters.training_registry import (
    LocalEvaluationRegistry,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    LocalTrainingRegistry,
)
from taqamind.domains.iron_manufacturing.adapters.unit_mapper import (
    IronSemanticValidator,
    IronUnitAwareColumnMapper,
    ProcessFeatureEngineerRouter,
)

__all__ = [
    "ControlledModelTrainer",
    "HistoricalRateRegressor",
    "IronSemanticValidator",
    "IronUnitAwareColumnMapper",
    "LocalActiveModelProvider",
    "LocalDatasetSplitter",
    "LocalDevelopmentDataProvider",
    "LocalEvaluationRegistry",
    "LocalLeakageValidator",
    "LocalModelRegistry",
    "LocalProtectedTestDataProvider",
    "LocalRecordUsageRegistry",
    "LocalSplitRegistry",
    "LocalTrainingReadinessChecker",
    "LocalTrainingRegistry",
    "ProcessFeatureEngineerRouter",
    "ProtectedFinalModelEvaluator",
    "TwoStageHurdleRegressor",
    "ZeroRetrainingPredictionEngine",
    "atomic_write_json",
    "build_record_scope",
    "compute_regression_metrics",
    "deserialize_trusted_bundle",
    "df_to_csv_bytes",
    "read_artifact_df",
    "read_json",
    "serialize_trusted_bundle",
    "strip_computed_fields",
]
