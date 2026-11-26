"""Iron Manufacturing domain contract assembly and Shared Core registration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DomainContract,
    IndustrialTaskContract,
    ProcessContract,
    TargetConfirmation,
    ValidationIssue,
)
from taqamind.core.feature_engineering import DerivedFeatureSpec
from taqamind.core.registry import DomainRegistry
from taqamind.domains.iron_manufacturing.config import DOMAIN_ID, DOMAIN_VERSION
from taqamind.domains.iron_manufacturing.processes.eaf_meltshop import (
    EAF_FORBIDDEN_PREDICTOR_NAMES,
    EAF_PROCESS_ID,
    EAF_TASK_ID,
    build_eaf_derived_feature_specs,
    build_eaf_meltshop_process,
    engineer_eaf_features,
    validate_eaf_physical_rules,
)
from taqamind.domains.iron_manufacturing.processes.rebar_hot_rolling import (
    REBAR_FORBIDDEN_PREDICTOR_NAMES,
    REBAR_PROCESS_ID,
    REBAR_TASK_ID,
    build_rebar_derived_feature_specs,
    build_rebar_hot_rolling_process,
    engineer_rebar_features,
    validate_rebar_physical_rules,
)
from taqamind.domains.iron_manufacturing.processes.scrap_yard_reception import (
    SCRAP_YARD_FORBIDDEN_PREDICTOR_NAMES,
    SCRAP_YARD_PROCESS_ID,
    SCRAP_YARD_TASK_ID,
    build_scrap_yard_derived_feature_specs,
    build_scrap_yard_process,
    engineer_scrap_yard_features,
    validate_scrap_yard_physical_rules,
)

PROCESS_DEFAULT_TASKS: dict[str, str] = {
    EAF_PROCESS_ID: EAF_TASK_ID,
    REBAR_PROCESS_ID: REBAR_TASK_ID,
    SCRAP_YARD_PROCESS_ID: SCRAP_YARD_TASK_ID,
}

PROCESS_PHYSICAL_VALIDATORS: dict[str, Callable[[pd.DataFrame], list[ValidationIssue]]] = {
    EAF_PROCESS_ID: validate_eaf_physical_rules,
    REBAR_PROCESS_ID: validate_rebar_physical_rules,
    SCRAP_YARD_PROCESS_ID: validate_scrap_yard_physical_rules,
}

PROCESS_FEATURE_ENGINEERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    EAF_PROCESS_ID: engineer_eaf_features,
    REBAR_PROCESS_ID: engineer_rebar_features,
    SCRAP_YARD_PROCESS_ID: engineer_scrap_yard_features,
}

PROCESS_DERIVED_FEATURE_SPECS: dict[str, tuple[DerivedFeatureSpec, ...]] = {
    EAF_PROCESS_ID: build_eaf_derived_feature_specs(),
    REBAR_PROCESS_ID: build_rebar_derived_feature_specs(),
    SCRAP_YARD_PROCESS_ID: build_scrap_yard_derived_feature_specs(),
}

PROCESS_FORBIDDEN_PREDICTORS: dict[str, frozenset[str]] = {
    EAF_PROCESS_ID: EAF_FORBIDDEN_PREDICTOR_NAMES,
    REBAR_PROCESS_ID: REBAR_FORBIDDEN_PREDICTOR_NAMES,
    SCRAP_YARD_PROCESS_ID: SCRAP_YARD_FORBIDDEN_PREDICTOR_NAMES,
}


def build_iron_manufacturing_domain(
    *,
    confirmations: dict[str, TargetConfirmation] | None = None,
) -> DomainContract:
    """Build the versioned Iron Manufacturing DomainContract.

    All production targets default to ConfirmationStatus.UNCONFIRMED until
    explicit factory confirmation is supplied.
    """
    conf_map = confirmations or {}
    processes: tuple[ProcessContract, ...] = (
        build_eaf_meltshop_process(confirmation=conf_map.get(EAF_PROCESS_ID)),
        build_rebar_hot_rolling_process(confirmation=conf_map.get(REBAR_PROCESS_ID)),
        build_scrap_yard_process(confirmation=conf_map.get(SCRAP_YARD_PROCESS_ID)),
    )
    return DomainContract(
        domain_id=DOMAIN_ID,
        version=DOMAIN_VERSION,
        display_name="Iron & Steel Manufacturing",
        description=(
            "Configurable industrial ML domain for Egyptian iron and steel facilities "
            "(EAF primary meltshop, rebar hot rolling mill, and scrap yard reception). "
            "Current executable ML scope is pre-production waste/scrap quantity prediction."
        ),
        processes=processes,
    )


def register_iron_manufacturing(
    registry: DomainRegistry,
    *,
    confirmations: dict[str, TargetConfirmation] | None = None,
) -> DomainContract:
    """Register the Iron Manufacturing domain into the Shared Core DomainRegistry."""
    domain = build_iron_manufacturing_domain(confirmations=confirmations)
    return registry.register(domain)


def with_target_confirmation(
    task: IndustrialTaskContract,
    confirmation: TargetConfirmation,
) -> IndustrialTaskContract:
    """Return a new IndustrialTaskContract with an updated TargetConfirmation."""
    dataset = task.dataset_contract
    updated_target = dataset.target.model_copy(update={"confirmation": confirmation})
    updated_dataset = dataset.model_copy(update={"target": updated_target})
    return task.model_copy(update={"dataset_contract": updated_dataset})


SYNTHETIC_CONFIRMATION_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def make_synthetic_confirmation(
    *,
    confirmed_by: str = "synthetic_fixture_owner",
    notes: str = "Explicit synthetic test fixture confirmation — NOT production-confirmed.",
    confirmed_at: datetime | None = None,
) -> TargetConfirmation:
    """Create an explicitly synthetic TargetConfirmation for offline tests and demos."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by=confirmed_by,
        confirmed_at=confirmed_at or SYNTHETIC_CONFIRMATION_TIMESTAMP,
        notes=notes,
    )


def get_iron_task(
    process_id: str,
    task_id: str | None = None,
    *,
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Resolve an IndustrialTaskContract for a supported Iron Manufacturing process."""
    if process_id not in PROCESS_DEFAULT_TASKS:
        raise KeyError(
            f"Unsupported iron_manufacturing process '{process_id}'. "
            f"Supported: {sorted(PROCESS_DEFAULT_TASKS)}."
        )
    resolved_task_id = task_id or PROCESS_DEFAULT_TASKS[process_id]
    confirmations = {process_id: confirmation} if confirmation is not None else None
    domain = build_iron_manufacturing_domain(confirmations=confirmations)
    return domain.get_task(process_id=process_id, task_id=resolved_task_id)


def get_iron_dataset_contract(
    process_id: str,
    task_id: str | None = None,
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Resolve the DatasetContract for a supported Iron Manufacturing process."""
    return get_iron_task(
        process_id,
        task_id,
        confirmation=confirmation,
    ).dataset_contract
