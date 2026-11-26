"""TaqaMind Iron Manufacturing industrial domain package."""

from taqamind.domains.iron_manufacturing.common.energy_synthetic import (
    generate_synthetic_iron_energy_dataset,
)
from taqamind.domains.iron_manufacturing.config import (
    DOMAIN_ID,
    DOMAIN_VERSION,
    IMPLEMENTATION_VERSION,
    IronDomainPaths,
)
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    PROCESS_ENERGY_TASKS,
    build_iron_manufacturing_domain,
    get_iron_dataset_contract,
    get_iron_task,
    make_synthetic_confirmation,
    register_iron_manufacturing,
    with_target_confirmation,
)
from taqamind.domains.iron_manufacturing.energy_contracts import (
    EAF_ENERGY_DATASET_ID,
    EAF_ENERGY_TASK_ID,
    IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS,
    REBAR_ENERGY_DATASET_ID,
    REBAR_ENERGY_TASK_ID,
    SCRAP_YARD_ENERGY_DATASET_ID,
    SCRAP_YARD_ENERGY_TASK_ID,
    build_eaf_energy_dataset_contract,
    build_iron_energy_derived_specs,
    build_iron_energy_task_contract,
    build_rebar_energy_dataset_contract,
    build_scrap_yard_energy_dataset_contract,
    build_synthetic_iron_energy_confirmation,
    get_iron_energy_dataset_contract,
)
from taqamind.domains.iron_manufacturing.energy_pipeline import (
    IronEnergyPipelineService,
    IronEnergyTrainingOutcome,
    PreparedIronEnergyDataset,
)
from taqamind.domains.iron_manufacturing.pipeline import IronManufacturingPipeline
from taqamind.domains.iron_manufacturing.services import (
    PredictionOutcome,
    TrainingWorkflowOutcome,
    generate_all_contracts_and_templates,
    generate_energy_process_templates,
    generate_process_templates,
)

__all__ = [
    "DOMAIN_ID",
    "DOMAIN_VERSION",
    "EAF_ENERGY_DATASET_ID",
    "EAF_ENERGY_TASK_ID",
    "IMPLEMENTATION_VERSION",
    "IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS",
    "IronDomainPaths",
    "IronEnergyPipelineService",
    "IronEnergyTrainingOutcome",
    "IronManufacturingPipeline",
    "PROCESS_DEFAULT_TASKS",
    "PROCESS_ENERGY_TASKS",
    "PredictionOutcome",
    "PreparedIronEnergyDataset",
    "REBAR_ENERGY_DATASET_ID",
    "REBAR_ENERGY_TASK_ID",
    "SCRAP_YARD_ENERGY_DATASET_ID",
    "SCRAP_YARD_ENERGY_TASK_ID",
    "TrainingWorkflowOutcome",
    "build_eaf_energy_dataset_contract",
    "build_iron_energy_derived_specs",
    "build_iron_energy_task_contract",
    "build_iron_manufacturing_domain",
    "build_rebar_energy_dataset_contract",
    "build_scrap_yard_energy_dataset_contract",
    "build_synthetic_iron_energy_confirmation",
    "generate_all_contracts_and_templates",
    "generate_energy_process_templates",
    "generate_process_templates",
    "generate_synthetic_iron_energy_dataset",
    "get_iron_dataset_contract",
    "get_iron_energy_dataset_contract",
    "get_iron_task",
    "make_synthetic_confirmation",
    "register_iron_manufacturing",
    "with_target_confirmation",
]
