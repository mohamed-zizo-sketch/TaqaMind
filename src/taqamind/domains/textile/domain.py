"""Textile DomainContract definition and DomainRegistry registration."""

from __future__ import annotations

from taqamind.core.contracts import DomainContract, TargetConfirmation
from taqamind.core.registry import DomainRegistry
from taqamind.domains.textile.knitting import (
    build_knitting_process_contract,
    build_synthetic_knitting_confirmation,
)
from taqamind.domains.textile.knitting.energy_contracts import (
    build_synthetic_knitting_energy_confirmation,
)
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_process_contract,
)
from taqamind.domains.textile.weaving.energy_contracts import (
    build_synthetic_weaving_energy_confirmation,
)

TEXTILE_DOMAIN_ID = "textile"
TEXTILE_DOMAIN_VERSION = "1.0.0"


def build_textile_domain_contract(
    *,
    weaving_confirmation: TargetConfirmation | None = None,
    knitting_confirmation: TargetConfirmation | None = None,
    weaving_energy_confirmation: TargetConfirmation | None = None,
    knitting_energy_confirmation: TargetConfirmation | None = None,
    use_synthetic_confirmations: bool = False,
) -> DomainContract:
    """Build the Textile DomainContract covering Weaving and Knitting (Waste V1 & Energy V2)."""
    w_conf = weaving_confirmation or (
        build_synthetic_weaving_confirmation() if use_synthetic_confirmations else None
    )
    k_conf = knitting_confirmation or (
        build_synthetic_knitting_confirmation() if use_synthetic_confirmations else None
    )
    we_conf = weaving_energy_confirmation or (
        build_synthetic_weaving_energy_confirmation() if use_synthetic_confirmations else None
    )
    ke_conf = knitting_energy_confirmation or (
        build_synthetic_knitting_energy_confirmation() if use_synthetic_confirmations else None
    )
    return DomainContract(
        domain_id=TEXTILE_DOMAIN_ID,
        version=TEXTILE_DOMAIN_VERSION,
        display_name="Textile Manufacturing",
        description=(
            "Egyptian textile manufacturing domain covering Weaving and Knitting "
            "with pre-production waste quantity and electrical energy consumption prediction."
        ),
        processes=(
            build_weaving_process_contract(confirmation=w_conf, energy_confirmation=we_conf),
            build_knitting_process_contract(confirmation=k_conf, energy_confirmation=ke_conf),
        ),
    )


def register_textile(
    registry: DomainRegistry,
    *,
    weaving_confirmation: TargetConfirmation | None = None,
    knitting_confirmation: TargetConfirmation | None = None,
    weaving_energy_confirmation: TargetConfirmation | None = None,
    knitting_energy_confirmation: TargetConfirmation | None = None,
    use_synthetic_confirmations: bool = False,
) -> DomainContract:
    """Register the textile domain version explicitly in DomainRegistry."""
    domain = build_textile_domain_contract(
        weaving_confirmation=weaving_confirmation,
        knitting_confirmation=knitting_confirmation,
        weaving_energy_confirmation=weaving_energy_confirmation,
        knitting_energy_confirmation=knitting_energy_confirmation,
        use_synthetic_confirmations=use_synthetic_confirmations,
    )
    return registry.register(domain)
