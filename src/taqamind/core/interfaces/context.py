"""Shared execution context for industrial pipeline operations."""

from datetime import UTC, datetime
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.domain import IndustrialTaskContract
from taqamind.core.contracts.enums import DatasetPurpose
from taqamind.core.contracts.feature import FeatureName
from taqamind.core.contracts.target import ContractVersion


class PipelineContext(BaseModel):
    """Immutable identity and scope of a pipeline execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    execution_id: UUID = Field(
        default_factory=uuid4
    )

    factory_id: UUID

    dataset_revision_id: UUID

    domain: FeatureName

    process: FeatureName

    task_id: FeatureName

    dataset_id: FeatureName

    contract_version: ContractVersion

    task_version: ContractVersion

    target_version: ContractVersion

    purpose: DatasetPurpose

    model_version_id: UUID | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @property
    def task_key(self) -> tuple[str, str, str]:
        """Return the hierarchical industrial task identity."""

        return (
            self.domain,
            self.process,
            self.task_id,
        )

    @property
    def factory_task_key(
        self,
    ) -> tuple[UUID, str, str, str]:
        """Return the factory-scoped task identity."""

        return (
            self.factory_id,
            self.domain,
            self.process,
            self.task_id,
        )

    def validate_dataset_contract(
        self,
        contract: DatasetContract,
    ) -> None:
        """Verify that a dataset contract matches this execution."""

        expected = {
            "domain": self.domain,
            "process": self.process,
            "dataset_id": self.dataset_id,
            "version": self.contract_version,
        }

        for field, expected_value in expected.items():
            actual_value = getattr(contract, field)

            if actual_value != expected_value:
                raise ValueError(
                    f"Dataset contract mismatch for '{field}'."
                )

        if contract.target.version != self.target_version:
            raise ValueError(
                "Dataset target version mismatch."
            )

    def validate_task_contract(
        self,
        task: IndustrialTaskContract,
    ) -> None:
        """Verify that a task matches this execution."""

        if task.task_id != self.task_id:
            raise ValueError(
                "Industrial task identifier mismatch."
            )

        if task.version != self.task_version:
            raise ValueError(
                "Industrial task version mismatch."
            )

        self.validate_dataset_contract(
            task.dataset_contract
        )

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        """Validate operation-specific context requirements."""

        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Execution timestamp must be timezone-aware."
            )

        if (
            self.purpose == DatasetPurpose.PREDICTION
            and self.model_version_id is None
        ):
            raise ValueError(
                "Prediction requires a registered model version."
            )

        return self

    @classmethod
    def from_task(
        cls,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
        task: IndustrialTaskContract,
        purpose: DatasetPurpose,
        model_version_id: UUID | None = None,
    ) -> Self:
        """Create an execution context from an industrial task."""

        dataset = task.dataset_contract

        return cls(
            factory_id=factory_id,
            dataset_revision_id=dataset_revision_id,
            domain=dataset.domain,
            process=dataset.process,
            task_id=task.task_id,
            dataset_id=dataset.dataset_id,
            contract_version=dataset.version,
            task_version=task.version,
            target_version=dataset.target.version,
            purpose=purpose,
            model_version_id=model_version_id,
        )