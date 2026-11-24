"""Structured outputs for industrial data processing."""

from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.enums import (
    PipelineStage,
)
from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
)


class MappingMethod(StrEnum):
    """Method used to identify a canonical column."""

    EXACT = "exact"
    ALIAS = "alias"
    RULE_BASED = "rule_based"
    LLM_SUGGESTED = "llm_suggested"
    MANUAL = "manual"


class ColumnProfile(BaseModel):
    """Observed statistics for a single dataset column."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    name: NonEmptyString

    physical_dtype: NonEmptyString

    missing_count: int = Field(ge=0)

    unique_count: int = Field(ge=0)

    numeric_minimum: float | None = None

    numeric_maximum: float | None = None

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        if (
            self.numeric_minimum is not None
            and self.numeric_maximum is not None
            and self.numeric_minimum > self.numeric_maximum
        ):
            raise ValueError(
                "Column minimum cannot exceed maximum."
            )

        return self


class DatasetProfile(BaseModel):
    """Observed statistics for one dataset revision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_revision_id: UUID

    factory_id: UUID

    row_count: int = Field(ge=0)

    columns: tuple[ColumnProfile, ...] = Field(
        min_length=1,
    )

    exact_duplicate_row_count: int = Field(
        default=0,
        ge=0,
    )

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        names = [
            column.name
            for column in self.columns
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Profile column names must be unique."
            )

        if self.exact_duplicate_row_count > max(
            self.row_count - 1,
            0,
        ):
            raise ValueError(
                "Duplicate row count exceeds possible duplicates."
            )

        for column in self.columns:
            if column.missing_count > self.row_count:
                raise ValueError(
                    "Missing count cannot exceed row count."
                )

            non_missing = (
                self.row_count - column.missing_count
            )

            if column.unique_count > non_missing:
                raise ValueError(
                    "Unique count cannot exceed non-missing rows."
                )

        return self


class ProfilingResult(BaseModel):
    """Stage result together with observed dataset statistics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    result: StageResult

    profile: DatasetProfile | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.result.stage != PipelineStage.PROFILING:
            raise ValueError(
                "Profiling result requires the profiling stage."
            )

        if self.result.status == ExecutionStatus.SUCCEEDED:
            if self.profile is None:
                raise ValueError(
                    "Successful profiling requires a dataset profile."
                )

        elif self.profile is not None:
            raise ValueError(
                "Unsuccessful profiling cannot expose a profile."
            )

        if self.profile is None:
            return self

        metadata = self.result.input_artifact.metadata

        if self.profile.factory_id != metadata.factory_id:
            raise ValueError(
                "Profile factory mismatch."
            )

        if (
            self.profile.dataset_revision_id
            != metadata.revision_id
        ):
            raise ValueError(
                "Profile dataset revision mismatch."
            )

        if self.profile.row_count != metadata.row_count:
            raise ValueError(
                "Profile row count mismatch."
            )

        observed_schema = tuple(
            (column.name, column.physical_dtype)
            for column in self.profile.columns
        )

        expected_schema = tuple(
            (column.name, column.physical_dtype)
            for column in metadata.columns
        )

        if observed_schema != expected_schema:
            raise ValueError(
                "Profile schema does not match input artifact."
            )

        return self


class ColumnMappingEntry(BaseModel):
    """Proposed relationship between source and canonical columns."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_column: NonEmptyString

    canonical_column: FeatureName

    method: MappingMethod

    source_unit: NonEmptyString | None = None

    canonical_unit: NonEmptyString | None = None

    requires_confirmation: bool = True

    reason: NonEmptyString | None = None


class MappingProposal(BaseModel):
    """Candidate mapping that has not yet been applied."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    proposal_id: UUID = Field(default_factory=uuid4)

    factory_id: UUID

    execution_id: UUID

    dataset_revision_id: UUID

    entries: tuple[ColumnMappingEntry, ...] = ()

    unmapped_columns: tuple[NonEmptyString, ...] = ()

    @property
    def requires_review(self) -> bool:
        return any(
            entry.requires_confirmation
            for entry in self.entries
        )

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        sources = [
            entry.source_column
            for entry in self.entries
        ]

        targets = [
            entry.canonical_column
            for entry in self.entries
        ]

        if len(sources) != len(set(sources)):
            raise ValueError(
                "Source columns must be unique."
            )

        if len(targets) != len(set(targets)):
            raise ValueError(
                "Canonical columns must be unique."
            )

        if len(self.unmapped_columns) != len(
            set(self.unmapped_columns)
        ):
            raise ValueError(
                "Unmapped columns must be unique."
            )

        if set(sources) & set(self.unmapped_columns):
            raise ValueError(
                "A source column cannot be mapped and unmapped."
            )

        return self


class ApprovedMapping(BaseModel):
    """Mapping approved for application to a specific revision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    proposal: MappingProposal

    approved_by: NonEmptyString

    confirmed_source_columns: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_confirmation(self) -> Self:
        confirmed = set(self.confirmed_source_columns)

        if len(confirmed) != len(self.confirmed_source_columns):
            raise ValueError(
                "Confirmed columns must be unique."
            )

        known = {
            entry.source_column
            for entry in self.proposal.entries
        }

        required = {
            entry.source_column
            for entry in self.proposal.entries
            if entry.requires_confirmation
        }

        if not confirmed <= known:
            raise ValueError(
                "Confirmation references unknown source columns."
            )

        if not required <= confirmed:
            raise ValueError(
                "Required mapping confirmations are missing."
            )

        return self