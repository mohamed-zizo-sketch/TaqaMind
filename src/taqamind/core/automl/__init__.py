"""Core AutoML data access, registries, and controlled training."""

from taqamind.core.automl.data_access import (
    LocalDevelopmentDataProvider,
    LocalProtectedTestDataProvider,
)
from taqamind.core.automl.trainer import (
    ControlledAutoMLTrainer,
    HistoricalRateEstimator,
    HurdleRegressorEstimator,
    TabularPreprocessor,
)
from taqamind.core.automl.training_registry import (
    LocalEvaluationRegistry,
    LocalTrainingRegistry,
)

__all__ = [
    "ControlledAutoMLTrainer",
    "HistoricalRateEstimator",
    "HurdleRegressorEstimator",
    "LocalDevelopmentDataProvider",
    "LocalEvaluationRegistry",
    "LocalProtectedTestDataProvider",
    "LocalTrainingRegistry",
    "TabularPreprocessor",
]
