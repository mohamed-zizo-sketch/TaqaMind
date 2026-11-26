"""Process-specific contracts, validation rules, and feature engineering specs."""

from taqamind.domains.iron_manufacturing.processes.eaf_meltshop import (
    EAF_PROCESS_ID,
    EAF_TASK_ID,
    build_eaf_meltshop_process,
    engineer_eaf_features,
    validate_eaf_physical_rules,
)
from taqamind.domains.iron_manufacturing.processes.rebar_hot_rolling import (
    REBAR_PROCESS_ID,
    REBAR_TASK_ID,
    build_rebar_hot_rolling_process,
    engineer_rebar_features,
    validate_rebar_physical_rules,
)
from taqamind.domains.iron_manufacturing.processes.scrap_yard_reception import (
    SCRAP_YARD_PROCESS_ID,
    SCRAP_YARD_TASK_ID,
    build_scrap_yard_process,
    engineer_scrap_yard_features,
    validate_scrap_yard_physical_rules,
)

__all__ = [
    "EAF_PROCESS_ID",
    "EAF_TASK_ID",
    "REBAR_PROCESS_ID",
    "REBAR_TASK_ID",
    "SCRAP_YARD_PROCESS_ID",
    "SCRAP_YARD_TASK_ID",
    "build_eaf_meltshop_process",
    "build_rebar_hot_rolling_process",
    "build_scrap_yard_process",
    "engineer_eaf_features",
    "engineer_rebar_features",
    "engineer_scrap_yard_features",
    "validate_eaf_physical_rules",
    "validate_rebar_physical_rules",
    "validate_scrap_yard_physical_rules",
]
