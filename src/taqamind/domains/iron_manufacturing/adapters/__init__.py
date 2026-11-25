"""Temporary protocol-compatible local adapters for iron_manufacturing.

Every class exported here implements an authoritative Shared Core protocol and
can be swapped out via dependency injection once Shared Core concrete engines
are implemented.
"""

from taqamind.domains.iron_manufacturing.adapters.data_access import (
    LocalActiveModelProvider,
    LocalDevelopmentDataProvider,
    LocalProtectedTestDataProvider,
)
from taqamind.domains.iron_manufacturing.adapters.dataset_registry import (
    LocalDatasetRegistry,
    LocalRecordLineageRegistry,
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
from taqamind.domains.iron_manufacturing.adapters.processing import (
    LocalColumnMapper,
    LocalCsvDataIngestion,
    LocalDataCleaner,
    LocalDataProfiler,
    LocalFeatureEngineer,
    LocalSemanticValidator,
)
from taqamind.domains.iron_manufacturing.adapters.readiness_and_splitting import (
    LocalDatasetSplitter,
    LocalLeakageValidator,
    LocalTrainingReadinessChecker,
    build_record_scope,
)
from taqamind.domains.iron_manufacturing.adapters.storage import (
    LocalDatasetArtifactStorage,
    LocalModelArtifactStorage,
)
from taqamind.domains.iron_manufacturing.adapters.training_registry import (
    LocalEvaluationRegistry,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    LocalTrainingRegistry,
)

__all__ = [
    "ControlledModelTrainer",
    "HistoricalRateRegressor",
    "LocalActiveModelProvider",
    "LocalColumnMapper",
    "LocalCsvDataIngestion",
    "LocalDataCleaner",
    "LocalDataProfiler",
    "LocalDatasetArtifactStorage",
    "LocalDatasetRegistry",
    "LocalDatasetSplitter",
    "LocalDevelopmentDataProvider",
    "LocalEvaluationRegistry",
    "LocalFeatureEngineer",
    "LocalLeakageValidator",
    "LocalModelArtifactStorage",
    "LocalModelRegistry",
    "LocalProtectedTestDataProvider",
    "LocalRecordLineageRegistry",
    "LocalRecordUsageRegistry",
    "LocalSemanticValidator",
    "LocalSplitRegistry",
    "LocalTrainingReadinessChecker",
    "LocalTrainingRegistry",
    "ProtectedFinalModelEvaluator",
    "TwoStageHurdleRegressor",
    "ZeroRetrainingPredictionEngine",
    "build_record_scope",
    "compute_regression_metrics",
    "deserialize_trusted_bundle",
    "serialize_trusted_bundle",
]
