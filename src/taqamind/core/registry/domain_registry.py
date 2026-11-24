"""Registration and discovery of industrial domain contracts."""

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.domain import (
    DomainContract,
    IndustrialTaskContract,
    ProcessContract,
)
from taqamind.core.contracts.enums import PlatformModule
from taqamind.core.contracts.target import ContractVersion


class DomainRegistryError(Exception):
    """Base exception for domain registration errors."""


class DomainAlreadyRegisteredError(DomainRegistryError):
    """A conflicting domain version is already registered."""


class DomainNotFoundError(DomainRegistryError):
    """The requested domain version does not exist."""


class DomainRegistry:
    """In-memory registry for industrial domain contracts.

    Registration is explicit and performed during initialization.
    No domain-specific business rules are embedded in the registry.
    """

    def __init__(self) -> None:
        self._domains: dict[
            tuple[str, str],
            DomainContract,
        ] = {}

    def register(
        self,
        domain: DomainContract,
    ) -> DomainContract:
        """Register a domain or reuse an identical registration."""

        key = (
            domain.domain_id,
            domain.version,
        )

        existing = self._domains.get(key)

        if existing is not None:
            if existing != domain:
                raise DomainAlreadyRegisteredError(
                    f"Conflicting domain registration: {key}"
                )

            return existing

        self._domains[key] = domain

        return domain

    def get_domain(
        self,
        domain_id: str,
        version: ContractVersion,
    ) -> DomainContract:
        """Retrieve an explicitly selected domain version."""

        key = (
            domain_id,
            version,
        )

        if key not in self._domains:
            raise DomainNotFoundError(
                f"Domain version not registered: {key}"
            )

        return self._domains[key]

    def has_domain(
        self,
        domain_id: str,
        version: ContractVersion,
    ) -> bool:
        return (domain_id, version) in self._domains

    def list_domains(self) -> tuple[DomainContract, ...]:
        """Return registered domains in deterministic order."""

        return tuple(
            self._domains[key]
            for key in sorted(self._domains)
        )

    @property
    def domain_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({
                domain_id
                for domain_id, _ in self._domains
            })
        )

    @property
    def registration_count(self) -> int:
        return len(self._domains)

    def list_versions(
        self,
        domain_id: str,
    ) -> tuple[str, ...]:
        """List available versions in numeric version order."""

        versions = [
            version
            for registered_id, version in self._domains
            if registered_id == domain_id
        ]

        return tuple(
            sorted(
                versions,
                key=lambda version: tuple(
                    int(part)
                    for part in version.split(".")
                ),
            )
        )

    def get_process(
        self,
        *,
        domain_id: str,
        domain_version: ContractVersion,
        process_id: str,
    ) -> ProcessContract:
        domain = self.get_domain(
            domain_id,
            domain_version,
        )

        return domain.get_process(process_id)

    def get_task(
        self,
        *,
        domain_id: str,
        domain_version: ContractVersion,
        process_id: str,
        task_id: str,
    ) -> IndustrialTaskContract:
        process = self.get_process(
            domain_id=domain_id,
            domain_version=domain_version,
            process_id=process_id,
        )

        return process.get_task(task_id)

    def get_dataset_contract(
        self,
        *,
        domain_id: str,
        domain_version: ContractVersion,
        process_id: str,
        task_id: str,
    ) -> DatasetContract:
        task = self.get_task(
            domain_id=domain_id,
            domain_version=domain_version,
            process_id=process_id,
            task_id=task_id,
        )

        return task.dataset_contract

    def find_tasks_by_module(
        self,
        module: PlatformModule,
    ) -> tuple[
        tuple[str, str, str, IndustrialTaskContract],
        ...,
    ]:
        """Discover tasks across all registered domain versions."""

        results = []

        for domain in self.list_domains():
            for process_id, task in domain.get_tasks_by_module(
                module
            ):
                results.append(
                    (
                        domain.domain_id,
                        domain.version,
                        process_id,
                        task,
                    )
                )

        return tuple(results)