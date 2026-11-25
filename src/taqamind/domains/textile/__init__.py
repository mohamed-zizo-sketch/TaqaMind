"""TaqaMind Textile Manufacturing Domain (Weaving & Knitting)."""

from taqamind.domains.textile.config import (
    build_default_acceptance_policy,
    build_default_split_configuration,
    build_default_training_configuration,
)
from taqamind.domains.textile.domain import (
    TEXTILE_DOMAIN_ID,
    TEXTILE_DOMAIN_VERSION,
    build_textile_domain_contract,
    register_textile,
)
from taqamind.domains.textile.pipeline import TextilePipelineService, TextilePlatformStores

__all__ = [
    "TEXTILE_DOMAIN_ID",
    "TEXTILE_DOMAIN_VERSION",
    "TextilePipelineService",
    "TextilePlatformStores",
    "build_default_acceptance_policy",
    "build_default_split_configuration",
    "build_default_training_configuration",
    "build_textile_domain_contract",
    "register_textile",
]
