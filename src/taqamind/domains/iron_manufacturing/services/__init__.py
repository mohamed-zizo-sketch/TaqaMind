"""Domain application services for iron_manufacturing."""

from taqamind.domains.iron_manufacturing.services.lineage_service import (
    LineageResolutionSummary,
    register_dataset_lineage,
)
from taqamind.domains.iron_manufacturing.services.prediction_service import (
    IronPredictionService,
    PredictionOutcome,
)
from taqamind.domains.iron_manufacturing.services.templates import (
    build_feature_dictionary_records,
    generate_all_contracts_and_templates,
    generate_process_templates,
)
from taqamind.domains.iron_manufacturing.services.training_service import (
    IronTrainingService,
    TrainingWorkflowOutcome,
    build_default_split_configuration,
    build_default_training_configuration,
)

__all__ = [
    "IronPredictionService",
    "IronTrainingService",
    "LineageResolutionSummary",
    "PredictionOutcome",
    "TrainingWorkflowOutcome",
    "build_default_split_configuration",
    "build_default_training_configuration",
    "build_feature_dictionary_records",
    "generate_all_contracts_and_templates",
    "generate_process_templates",
    "register_dataset_lineage",
]
