"""Hierarchical contracts for industrial domains and tasks."""

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    PlatformModule,
    TaskType,
)
from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import ContractVersion


class IndustrialTaskContract(BaseModel):
    """Definition of a supported industrial ML task."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    task_id: FeatureName

    version: ContractVersion = "1.0.0"

    display_name: NonEmptyString

    description: NonEmptyString

    module: PlatformModule

    task_type: TaskType

    dataset_contract: DatasetContract

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if self.task_type != self.dataset_contract.target.task_type:
            raise ValueError(
                "Industrial task type must match "
                "the dataset target task type."
            )

        return self


class ProcessContract(BaseModel):
    """Definition of an industrial manufacturing process."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    process_id: FeatureName

    version: ContractVersion = "1.0.0"

    display_name: NonEmptyString

    description: NonEmptyString

    tasks: tuple[IndustrialTaskContract, ...] = Field(
        min_length=1,
    )

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return registered task identifiers."""

        return tuple(
            task.task_id
            for task in self.tasks
        )

    def get_task(self, task_id: str) -> IndustrialTaskContract:
        """Retrieve an industrial task by its identifier."""

        for task in self.tasks:
            if task.task_id == task_id:
                return task

        raise KeyError(
            f"Unknown task '{task_id}' "
            f"in process '{self.process_id}'."
        )

    def get_tasks_by_module(
        self,
        module: PlatformModule,
    ) -> tuple[IndustrialTaskContract, ...]:
        """Return tasks belonging to a platform module."""

        return tuple(
            task
            for task in self.tasks
            if task.module == module
        )

    @model_validator(mode="after")
    def validate_process(self) -> Self:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError(
                "Task identifiers must be unique within a process."
            )

        for task in self.tasks:
            if task.dataset_contract.process != self.process_id:
                raise ValueError(
                    f"Task '{task.task_id}' has an incompatible "
                    "dataset process."
                )

        return self


class DomainContract(BaseModel):
    """Complete definition of an industrial domain."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    domain_id: FeatureName

    version: ContractVersion = "1.0.0"

    display_name: NonEmptyString

    description: NonEmptyString

    processes: tuple[ProcessContract, ...] = Field(
        min_length=1,
    )

    @property
    def process_ids(self) -> tuple[str, ...]:
        """Return registered process identifiers."""

        return tuple(
            process.process_id
            for process in self.processes
        )

    @property
    def supported_modules(self) -> tuple[PlatformModule, ...]:
        """Return platform modules declared in this domain."""

        modules = {
            task.module
            for process in self.processes
            for task in process.tasks
        }

        return tuple(
            sorted(modules, key=lambda module: module.value)
        )

    @property
    def total_tasks(self) -> int:
        """Return the number of declared industrial tasks."""

        return sum(
            len(process.tasks)
            for process in self.processes
        )

    def get_process(self, process_id: str) -> ProcessContract:
        """Retrieve an industrial process."""

        for process in self.processes:
            if process.process_id == process_id:
                return process

        raise KeyError(
            f"Unknown process '{process_id}' "
            f"in domain '{self.domain_id}'."
        )

    def get_task(
        self,
        process_id: str,
        task_id: str,
    ) -> IndustrialTaskContract:
        """Retrieve a task within a specific process."""

        process = self.get_process(process_id)

        return process.get_task(task_id)

    def get_dataset_contract(
        self,
        process_id: str,
        task_id: str,
    ) -> DatasetContract:
        """Retrieve the dataset contract for an industrial task."""

        task = self.get_task(
            process_id=process_id,
            task_id=task_id,
        )

        return task.dataset_contract

    def get_tasks_by_module(
        self,
        module: PlatformModule,
    ) -> tuple[tuple[str, IndustrialTaskContract], ...]:
        """Return matching tasks together with their process IDs."""

        return tuple(
            (process.process_id, task)
            for process in self.processes
            for task in process.tasks
            if task.module == module
        )

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if len(self.process_ids) != len(set(self.process_ids)):
            raise ValueError(
                "Process identifiers must be unique within a domain."
            )

        dataset_contracts: dict[
            tuple[str, str],
            DatasetContract,
        ] = {}

        for process in self.processes:
            for task in process.tasks:
                dataset = task.dataset_contract

                if dataset.domain != self.domain_id:
                    raise ValueError(
                        f"Task '{task.task_id}' has an "
                        "incompatible dataset domain."
                    )

                identity = (
                    dataset.dataset_id,
                    dataset.version,
                )

                existing = dataset_contracts.get(identity)

                if existing is not None and existing != dataset:
                    raise ValueError(
                        f"Conflicting dataset definitions "
                        f"for '{dataset.dataset_id}' "
                        f"version '{dataset.version}'."
                    )

                dataset_contracts[identity] = dataset

        return self