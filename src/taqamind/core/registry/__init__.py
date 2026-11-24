"""Shared registry implementations."""

from taqamind.core.registry.domain_registry import (
    DomainAlreadyRegisteredError,
    DomainNotFoundError,
    DomainRegistry,
    DomainRegistryError,
)

__all__ = [
    "DomainAlreadyRegisteredError",
    "DomainNotFoundError",
    "DomainRegistry",
    "DomainRegistryError",
]