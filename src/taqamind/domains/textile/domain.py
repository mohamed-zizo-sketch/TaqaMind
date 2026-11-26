"""Textile DomainContract definition and DomainRegistry registration."""

from __future__ import annotations

from taqamind.core.contracts import DomainContract, TargetConfirmation
from taqamind.core.registry import DomainRegistry
from taqamind.domains.textile.knitting import (
    build_knitting_process_contract,
    build_synthetic_knitting_confirmation,
)
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_process_contract,
)

TEXTILE_DOMAIN_ID = "textile"
TEXTILE_DOMAIN_VERSION = "1.0.0"


def build_textile_domain_contract(
    *,
    weaving_confirmation: TargetConfirmation | None = None,
    knitting_confirmation: TargetConfirmation | None = None,
    use_synthetic_confirmations: bool = False,
) -> DomainContract:
    """Build the Textile DomainContract covering Weaving and Knitting."""
    w_conf = weaving_confirmation or (
        build_synthetic_weaving_confirmation() if use_synthetic_confirmations else None
    )
    k_conf = knitting_confirmation or (
        build_synthetic_knitting_confirmation() if use_synthetic_confirmations else None
    )
    return DomainContract(
        domain_id=TEXTILE_DOMAIN_ID,
        version=TEXTILE_DOMAIN_VERSION,
        display_name="Textile Manufacturing",
        description=(
            "Egyptian textile manufacturing domain covering Weaving and Knitting "
            "with pre-production waste quantity prediction."
        ),
        processes=(
            build_weaving_process_contract(confirmation=w_conf),
            build_knitting_process_contract(confirmation=k_conf),
        ),
    )


def register_textile(
    registry: DomainRegistry,
    *,
    weaving_confirmation: TargetConfirmation | None = None,
    knitting_confirmation: TargetConfirmation | None = None,
    use_synthetic_confirmations: bool = False,
) -> DomainContract:
    """Register the textile domain version explicitly in DomainRegistry."""
    domain = build_textile_domain_contract(
        weaving_confirmation=weaving_confirmation,
        knitting_confirmation=knitting_confirmation,
        use_synthetic_confirmations=use_synthetic_confirmations,
    )
    return registry.register(domain)
