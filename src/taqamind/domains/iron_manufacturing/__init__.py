"""TaqaMind Iron Manufacturing industrial domain package."""

from taqamind.domains.iron_manufacturing.config import (
    DOMAIN_ID,
    DOMAIN_VERSION,
    IMPLEMENTATION_VERSION,
    IronDomainPaths,
)
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    build_iron_manufacturing_domain,
    get_iron_dataset_contract,
    get_iron_task,
    make_synthetic_confirmation,
    register_iron_manufacturing,
    with_target_confirmation,
)
from taqamind.domains.iron_manufacturing.pipeline import IronManufacturingPipeline
from taqamind.domains.iron_manufacturing.services import (
    PredictionOutcome,
    TrainingWorkflowOutcome,
    generate_all_contracts_and_templates,
    generate_process_templates,
)

__all__ = [
    "DOMAIN_ID",
    "DOMAIN_VERSION",
    "IMPLEMENTATION_VERSION",
    "IronDomainPaths",
    "IronManufacturingPipeline",
    "PROCESS_DEFAULT_TASKS",
    "PredictionOutcome",
    "TrainingWorkflowOutcome",
    "build_iron_manufacturing_domain",
    "generate_all_contracts_and_templates",
    "generate_process_templates",
    "get_iron_dataset_contract",
    "get_iron_task",
    "make_synthetic_confirmation",
    "register_iron_manufacturing",
    "with_target_confirmation",
]
