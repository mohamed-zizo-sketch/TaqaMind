"""Restricted data access contracts for industrial ML workflows."""

from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.feature import FeatureName
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.evaluation import FrozenModelSelection
from taqamind.core.interfaces.leakage import TrainingAuthorization
from taqamind.core.interfaces.model_registry import (
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
)
from taqamind.core.interfaces.splitting import (
    SplitPartition,
    SplitRole,
)
from taqamind.core.interfaces.training_registry import (
    EvaluationReservation,
)


class DataAccessError(Exception):
    """Base exception for restricted data access failures."""


class DataAccessDeniedError(DataAccessError):
    """Requested data access is not permitted."""


class PartitionDataReference(BaseModel):
    """Metadata for an independently materialized data partition.

    The partition contains only its registered records.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition_artifact_id: UUID

    factory_id: UUID

    dataset_revision_id: UUID

    split_id: UUID

    role: SplitRole

    record_ids: tuple[UUID, ...] = Field(min_length=1)

    columns: tuple[FeatureName, ...] = Field(min_length=1)

    content_sha256: SHA256Digest

    size_bytes: int = Field(gt=0)

    @property
    def row_count(self) -> int:
        return len(self.record_ids)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError(
                "Partition record identifiers must be unique."
            )

        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "Partition column names must be unique."
            )

        return self

    def validate_partition(
        self,
        *,
        context: PipelineContext,
        split_id: UUID,
        dataset_revision_id: UUID,
        partition: SplitPartition,
    ) -> None:
        if self.factory_id != context.factory_id:
            raise DataAccessDeniedError(
                "Partition factory mismatch."
            )

        if self.split_id != split_id:
            raise DataAccessDeniedError(
                "Partition split mismatch."
            )

        if self.dataset_revision_id != dataset_revision_id:
            raise DataAccessDeniedError(
                "Partition dataset revision mismatch."
            )

        if self.role != partition.role:
            raise DataAccessDeniedError(
                "Partition role mismatch."
            )

        if set(self.record_ids) != set(partition.record_ids):
            raise DataAccessDeniedError(
                "Partition record coverage mismatch."
            )

    def validate_columns(
        self,
        contract: DatasetContract,
    ) -> None:
        """Validate the supervised learning column selection."""

        predictors = set(contract.predictor_names)

        target = contract.target.feature.name

        allowed = predictors | {target}

        observed = set(self.columns)

        if not observed <= allowed:
            raise DataAccessDeniedError(
                "Partition contains unapproved learning columns."
            )

        required = {
            feature.name
            for feature in contract.features
            if feature.name in predictors and feature.required
        }

        required.add(target)

        if not required <= observed:
            raise DataAccessDeniedError(
                "Partition is missing required learning columns."
            )

        if not observed & predictors:
            raise DataAccessDeniedError(
                "Partition must contain at least one predictor."
            )

        for name in observed & predictors:
            feature = contract.get_feature(name)

            if not contract.is_available_at_prediction(feature):
                raise DataAccessDeniedError(
                    f"Predictor '{name}' is unavailable at prediction time."
                )


class DevelopmentDataAccess(BaseModel):
    """Restricted access metadata for training and validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization: TrainingAuthorization

    train: PartitionDataReference

    validation: PartitionDataReference

    @model_validator(mode="after")
    def validate_access(self) -> Self:
        development = self.authorization.development_split

        for reference, partition in (
            (self.train, development.train),
            (self.validation, development.validation),
        ):
            reference.validate_partition(
                context=development.context,
                split_id=development.split_id,
                dataset_revision_id=(
                    development.dataset_revision_id
                ),
                partition=partition,
            )

        if (
            self.train.partition_artifact_id
            == self.validation.partition_artifact_id
        ):
            raise DataAccessDeniedError(
                "Development partitions require separate artifacts."
            )

        if self.train.columns != self.validation.columns:
            raise DataAccessDeniedError(
                "Development partition schemas must match."
            )

        return self

    def validate_contract(
        self,
        contract: DatasetContract,
    ) -> None:
        context = (
            self.authorization.development_split.context
        )

        context.validate_dataset_contract(contract)

        self.train.validate_columns(contract)
        self.validation.validate_columns(contract)


class ProtectedTestDataAccess(BaseModel):
    """Restricted test access bound to a frozen evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation: EvaluationReservation

    selection: FrozenModelSelection

    test: PartitionDataReference

    @model_validator(mode="after")
    def validate_access(self) -> Self:
        development = (
            self.selection.training_run.development_split
        )

        context = development.context

        self.reservation.scope.validate_context(context)

        if (
            self.reservation.selection_id
            != self.selection.selection_id
        ):
            raise DataAccessDeniedError(
                "Evaluation selection mismatch."
            )

        if (
            self.reservation.execution_id
            != context.execution_id
        ):
            raise DataAccessDeniedError(
                "Evaluation execution mismatch."
            )

        if self.reservation.split_id != development.split_id:
            raise DataAccessDeniedError(
                "Evaluation split mismatch."
            )

        if self.reservation.reserved_at < self.selection.selected_at:
            raise DataAccessDeniedError(
                "Test reservation precedes model selection."
            )

        expected_partition = SplitPartition(
            role=SplitRole.TEST,
            record_ids=self.reservation.test_record_ids,
        )

        self.test.validate_partition(
            context=context,
            split_id=development.split_id,
            dataset_revision_id=(
                development.dataset_revision_id
            ),
            partition=expected_partition,
        )

        development_ids = (
            set(development.train.record_ids)
            | set(development.validation.record_ids)
        )

        if set(self.test.record_ids) & development_ids:
            raise DataAccessDeniedError(
                "Protected test overlaps development records."
            )

        return self

    def validate_contract(
        self,
        contract: DatasetContract,
    ) -> None:
        context = (
            self.selection.training_run.development_split.context
        )

        context.validate_dataset_contract(contract)

        self.test.validate_columns(contract)


def validate_active_model_access(
    *,
    context: PipelineContext,
    entry: ModelRegistryEntry,
    contract: DatasetContract,
) -> None:
    """Validate access to an active registered prediction model."""

    if entry.status != ModelStatus.ACTIVE:
        raise DataAccessDeniedError(
            "Prediction requires an active registered model."
        )

    entry.version.identity.validate_context(context)

    context.validate_dataset_contract(contract)

    entry.version.feature_specification.validate_contract(
        contract
    )


@runtime_checkable
class DevelopmentDataProvider(Protocol):
    """Provides only approved training and validation partitions."""

    def prepare(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
    ) -> DevelopmentDataAccess:
        """Verify authorization against trusted registries.

        Implementations must materialize only the approved
        development partitions and verify record membership.
        """
        ...

    def open_train(
        self,
        *,
        access: DevelopmentDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        """Record training exposure before releasing verified data."""
        ...

    def open_validation(
        self,
        *,
        access: DevelopmentDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        """Record validation exposure before releasing verified data."""
        ...


@runtime_checkable
class ProtectedTestDataProvider(Protocol):
    """Provides test data only for registered final evaluations."""

    def prepare(
        self,
        *,
        reservation: EvaluationReservation,
        selection: FrozenModelSelection,
        contract: DatasetContract,
    ) -> ProtectedTestDataAccess:
        """Verify registered reservation and frozen model selection."""
        ...

    def open_test(
        self,
        *,
        access: ProtectedTestDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        """Persist test exposure before releasing verified data.

        A failed evaluation must not erase the exposure history.
        """
        ...


@runtime_checkable
class ActiveModelProvider(Protocol):
    """Read-only model access for production prediction."""

    def get_active(
        self,
        *,
        scope: ModelScope,
    ) -> ModelRegistryEntry | None:
        """Resolve the active model without triggering training."""
        ...

    def open_bundle(
        self,
        *,
        context: PipelineContext,
        entry: ModelRegistryEntry,
        contract: DatasetContract,
    ) -> AbstractContextManager[BinaryIO]:
        """Return a verified trusted model bundle.

        Implementations must resolve the registered entry,
        enforce factory scope and verify artifact integrity.
        """
        ...