"""Food & Beverage domain contract definition and registration."""

from __future__ import annotations

from datetime import UTC, datetime

from taqamind.core.contracts import (
    ConfirmationStatus,
    DomainContract,
    TargetConfirmation,
)
from taqamind.core.registry import DomainRegistry, domain_registry
from taqamind.domains.food_beverage.config import DOMAIN_ID, DOMAIN_VERSION
from taqamind.domains.food_beverage.processes.bakery import (
    build_bakery_process_contract,
)
from taqamind.domains.food_beverage.processes.beverage import (
    build_beverage_process_contract,
)
from taqamind.domains.food_beverage.processes.dairy import (
    build_dairy_process_contract,
)


def make_synthetic_target_confirmation(
    confirmed_by: str = "food_beverage_synthetic_fixture",
    notes: str = "Explicit synthetic test confirmation; not valid for production factory claims.",
) -> TargetConfirmation:
    """Create an explicit synthetic TargetConfirmation for automated testing."""

    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by=confirmed_by,
        confirmed_at=datetime.now(UTC),
        notes=notes,
    )


def build_food_beverage_domain(
    confirmations: dict[str, TargetConfirmation] | None = None,
) -> DomainContract:
    """Construct the Food & Beverage DomainContract."""

    conf = confirmations or {}
    return DomainContract(
        domain_id=DOMAIN_ID,
        version=DOMAIN_VERSION,
        display_name="Food & Beverage Manufacturing",
        description=(
            "Configurable industrial ML domain for Bakery, Dairy, and Beverage "
            "manufacturing facilities."
        ),
        processes=(
            build_bakery_process_contract(confirmation=conf.get("bakery")),
            build_dairy_process_contract(confirmation=conf.get("dairy")),
            build_beverage_process_contract(confirmation=conf.get("beverage")),
        ),
    )


def register_food_beverage(
    registry: DomainRegistry | None = None,
    confirmations: dict[str, TargetConfirmation] | None = None,
) -> DomainContract:
    """Register the Food & Beverage domain into a DomainRegistry instance."""

    target_registry = registry if registry is not None else domain_registry
    domain = build_food_beverage_domain(confirmations=confirmations)
    return target_registry.register(domain)
