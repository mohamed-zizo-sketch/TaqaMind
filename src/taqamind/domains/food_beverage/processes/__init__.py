"""Process registry and contract builders for Food & Beverage."""

from taqamind.domains.food_beverage.processes.bakery import (
    BAKERY_ALIAS_MAP,
    build_bakery_dataset_contract,
    build_bakery_process_contract,
)
from taqamind.domains.food_beverage.processes.beverage import (
    BEVERAGE_ALIAS_MAP,
    build_beverage_dataset_contract,
    build_beverage_process_contract,
)
from taqamind.domains.food_beverage.processes.dairy import (
    DAIRY_ALIAS_MAP,
    build_dairy_dataset_contract,
    build_dairy_process_contract,
)

__all__ = [
    "BAKERY_ALIAS_MAP",
    "BEVERAGE_ALIAS_MAP",
    "DAIRY_ALIAS_MAP",
    "build_bakery_dataset_contract",
    "build_bakery_process_contract",
    "build_beverage_dataset_contract",
    "build_beverage_process_contract",
    "build_dairy_dataset_contract",
    "build_dairy_process_contract",
]
