"""Protocol-compliant local adapters ONLY for Core P1.6-P1.8 / AUD-001 / AUD-008 gaps."""

from taqamind.domains.food_beverage.adapters.data_access_training import (
    FoodBeverageControlledAutoMLTrainer,
    FoodBeverageProtectedFinalEvaluator,
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
)
from taqamind.domains.food_beverage.adapters.membership import (
    FoodBeverageRowMembershipBuilder,
    RowLineageSummary,
)
from taqamind.domains.food_beverage.adapters.model_registry_prediction import (
    FoodBeveragePredictionEngine,
    LocalActiveModelProvider,
    LocalModelRegistry,
)
from taqamind.domains.food_beverage.adapters.readiness_splitting import (
    FoodBeverageDatasetSplitter,
    FoodBeverageLeakageValidator,
    FoodBeverageTrainingReadinessChecker,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    hash_contract,
)
from taqamind.domains.food_beverage.adapters.unit_mapper import UnitAwareColumnMapperAdapter

__all__ = [
    "FoodBeverageControlledAutoMLTrainer",
    "FoodBeverageDatasetSplitter",
    "FoodBeverageLeakageValidator",
    "FoodBeveragePredictionEngine",
    "FoodBeverageProtectedFinalEvaluator",
    "FoodBeverageRowMembershipBuilder",
    "FoodBeverageTrainingReadinessChecker",
    "LocalActiveModelProvider",
    "LocalDevelopmentDataProvider",
    "LocalEvaluationRegistry",
    "LocalModelRegistry",
    "LocalProtectedTestDataProvider",
    "LocalRecordUsageRegistry",
    "LocalSplitRegistry",
    "LocalTrainingRegistry",
    "RowLineageSummary",
    "UnitAwareColumnMapperAdapter",
    "hash_contract",
]
