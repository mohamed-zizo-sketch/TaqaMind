"""Dataset splitting and immutable split manifest contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from math import isclose
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.enums import DatasetPurpose
from taqamind.core.contracts.feature import NonEmptyString
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.readiness import TrainingReadinessResult


class SplitStrategy(StrEnum):
    """Supported industrial data splitting strategies."""

    TEMPORAL = "temporal"
    GROUP = "group"
    GROUP_TEMPORAL = "group_temporal"
    RANDOM = "random"


class SplitRole(StrEnum):
    """Purpose of each dataset partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitConfiguration(BaseModel):
    """Requested dataset splitting configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    strategy: SplitStrategy

    train_ratio: float = Field(gt=0, lt=1)

    validation_ratio: float = Field(gt=0, lt=1)

    test_ratio: float = Field(gt=0, lt=1)

    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
    )

    justification: NonEmptyString

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        total = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if not isclose(
            total,
            1.0,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Split ratios must sum to 1."
            )

        return self


class SplitPartition(BaseModel):
    """Registered record identities belonging to one partition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    role: SplitRole

    record_ids: tuple[UUID, ...] = Field(
        min_length=1,
    )

    @property
    def row_count(self) -> int:
        return len(self.record_ids)

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError(
                "Partition record identifiers must be unique."
            )

        return self


class SplitManifest(BaseModel):
    """Immutable assignment of registered records to ML partitions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    split_id: UUID = Field(
        default_factory=uuid4,
    )

    context: PipelineContext

    dataset_revision_id: UUID

    configuration: SplitConfiguration

    train: SplitPartition

    validation: SplitPartition

    test: SplitPartition

    excluded_record_ids: tuple[UUID, ...] = ()

    exclusion_reason: NonEmptyString | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def total_rows(self) -> int:
        """Return the number of records assigned to model partitions."""

        return (
            self.train.row_count
            + self.validation.row_count
            + self.test.row_count
        )

    @property
    def accounted_rows(self) -> int:
        """Include explicitly excluded records."""

        return self.total_rows + len(self.excluded_record_ids)

    @property
    def actual_ratios(self) -> dict[str, float]:
        """Return actual partition proportions."""

        return {
            "train": self.train.row_count / self.total_rows,
            "validation": (
                self.validation.row_count / self.total_rows
            ),
            "test": self.test.row_count / self.total_rows,
        }

    def get_partition(
        self,
        role: SplitRole,
    ) -> SplitPartition:
        """Retrieve a partition by its declared role."""

        return {
            SplitRole.TRAIN: self.train,
            SplitRole.VALIDATION: self.validation,
            SplitRole.TEST: self.test,
        }[role]

    def validate_record_coverage(
        self,
        expected_record_ids: tuple[UUID, ...],
    ) -> None:
        """Verify complete coverage of registered source records."""

        expected = set(expected_record_ids)

        if len(expected) != len(expected_record_ids):
            raise ValueError(
                "Expected source record IDs must be unique."
            )

        actual = (
            set(self.train.record_ids)
            | set(self.validation.record_ids)
            | set(self.test.record_ids)
            | set(self.excluded_record_ids)
        )

        if actual != expected:
            raise ValueError(
                "Split manifest does not cover the expected records."
            )

    def validate_readiness(
        self,
        readiness: TrainingReadinessResult,
    ) -> None:
        """Check compatibility with the assessed dataset."""

        readiness.require_split_eligibility()

        result = readiness.result

        if self.context != result.context:
            raise ValueError(
                "Split and readiness execution contexts must match."
            )

        metadata = result.input_artifact.metadata

        if self.dataset_revision_id != metadata.revision_id:
            raise ValueError(
                "Split dataset revision mismatch."
            )

        if self.accounted_rows != metadata.row_count:
            raise ValueError(
                "Split record count does not match dataset row count."
            )

        contract = readiness.contract

        if self.configuration.strategy in {
            SplitStrategy.TEMPORAL,
            SplitStrategy.GROUP_TEMPORAL,
        }:
            if contract.time_column is None:
                raise ValueError(
                    "Temporal splitting requires a time column."
                )

        if self.configuration.strategy in {
            SplitStrategy.GROUP,
            SplitStrategy.GROUP_TEMPORAL,
        }:
            if not contract.grouping_columns:
                raise ValueError(
                    "Group splitting requires grouping columns."
                )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.context.purpose != DatasetPurpose.TRAINING:
            raise ValueError(
                "Split manifest requires a training context."
            )

        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Split timestamp must be timezone-aware."
            )

        if self.created_at < self.context.created_at:
            raise ValueError(
                "Split cannot precede execution creation."
            )

        expected_roles = (
            (self.train, SplitRole.TRAIN),
            (self.validation, SplitRole.VALIDATION),
            (self.test, SplitRole.TEST),
        )

        for partition, role in expected_roles:
            if partition.role != role:
                raise ValueError(
                    f"Invalid partition role: expected '{role.value}'."
                )

        train_ids = set(self.train.record_ids)
        validation_ids = set(self.validation.record_ids)
        test_ids = set(self.test.record_ids)
        excluded_ids = set(self.excluded_record_ids)

        if train_ids & validation_ids:
            raise ValueError(
                "Record leakage between train and validation."
            )

        if train_ids & test_ids:
            raise ValueError(
                "Record leakage between train and test."
            )

        if validation_ids & test_ids:
            raise ValueError(
                "Record leakage between validation and test."
            )

        if excluded_ids & (
            train_ids | validation_ids | test_ids
        ):
            raise ValueError(
                "Excluded records cannot belong to a partition."
            )

        if len(excluded_ids) != len(self.excluded_record_ids):
            raise ValueError(
                "Excluded record identifiers must be unique."
            )

        if self.excluded_record_ids and self.exclusion_reason is None:
            raise ValueError(
                "Excluded records require an exclusion reason."
            )

        return self