"""Temporary protocol-compatible adapters pending Shared Core P1.6-P1.8 integration."""

from taqamind.domains.textile.adapters.data_access_training import (
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
    TextileControlledAutoMLTrainer,
    TextileProtectedFinalEvaluator,
    compute_regression_metrics,
)
from taqamind.domains.textile.adapters.membership import (
    RowLineageSummary,
    TextileRowMembershipBuilder,
)
from taqamind.domains.textile.adapters.model_registry_prediction import (
    LocalActiveModelProvider,
    LocalModelRegistry,
    TextilePredictionEngine,
)
from taqamind.domains.textile.adapters.readiness_splitting import (
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    TextileDatasetSplitter,
    TextileLeakageValidator,
    TextileTrainingReadinessChecker,
    hash_contract,
)
from taqamind.domains.textile.adapters.unit_mapper import UnitAwareColumnMapperAdapter

__all__ = [
    "LocalActiveModelProvider",
    "LocalDevelopmentDataProvider",
    "LocalEvaluationRegistry",
    "LocalModelRegistry",
    "LocalProtectedTestDataProvider",
    "LocalRecordUsageRegistry",
    "LocalSplitRegistry",
    "LocalTrainingRegistry",
    "RowLineageSummary",
    "TextileControlledAutoMLTrainer",
    "TextileDatasetSplitter",
    "TextileLeakageValidator",
    "TextilePredictionEngine",
    "TextileProtectedFinalEvaluator",
    "TextileRowMembershipBuilder",
    "TextileTrainingReadinessChecker",
    "UnitAwareColumnMapperAdapter",
    "compute_regression_metrics",
    "hash_contract",
]
