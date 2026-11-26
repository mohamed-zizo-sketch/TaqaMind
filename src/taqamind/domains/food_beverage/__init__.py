"""Food & Beverage industrial domain package for TaqaMind."""

from taqamind.domains.food_beverage.config import (
    DOMAIN_ID,
    DOMAIN_VERSION,
    SUPPORTED_PROCESSES,
    FoodBeverageStorageConfig,
)
from taqamind.domains.food_beverage.domain import (
    build_food_beverage_domain,
    make_synthetic_target_confirmation,
    register_food_beverage,
)
from taqamind.domains.food_beverage.pipeline import (
    FoodBeveragePipeline,
    TrainingWorkflowResult,
)

__all__ = [
    "DOMAIN_ID",
    "DOMAIN_VERSION",
    "FoodBeveragePipeline",
    "FoodBeverageStorageConfig",
    "SUPPORTED_PROCESSES",
    "TrainingWorkflowResult",
    "build_food_beverage_domain",
    "make_synthetic_target_confirmation",
    "register_food_beverage",
]
