"""Dataset metadata and revision-level lineage contracts."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.enums import PipelineStage
from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    SHA256Digest,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext


class DatasetState(StrEnum):
    """Processing state of a dataset revision."""

    RAW = "raw"
    INGESTED = "ingested"
    MAPPED = "mapped"
    CLEANED = "cleaned"
    ENGINEERED = "engineered"


STATE_ORDER = {
    DatasetState.RAW: 0,
    DatasetState.INGESTED: 1,
    DatasetState.MAPPED: 2,
    DatasetState.CLEANED: 3,
    DatasetState.ENGINEERED: 4,
}


STATE_STAGE = {
    DatasetState.INGESTED: PipelineStage.INGESTION,
    DatasetState.MAPPED: PipelineStage.COLUMN_MAPPING,
    DatasetState.CLEANED: PipelineStage.CLEANING,
    DatasetState.ENGINEERED: PipelineStage.FEATURE_ENGINEERING,
}


class DatasetColumnSchema(BaseModel):
    """Observed column information, not an industrial contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: NonEmptyString

    physical_dtype: NonEmptyString


class TransformationRecord(BaseModel):
    """Metadata describing a dataset transformation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    transformation_id: UUID = Field(
        default_factory=uuid4,
    )

    execution_id: UUID

    parent_revision_id: UUID

    stage: PipelineStage

    operation: FeatureName

    implementation_version: ContractVersion

    configuration_sha256: SHA256Digest | None = None

    description: NonEmptyString

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Transformation timestamp must be timezone-aware."
            )

        return self


class DatasetMetadata(BaseModel):
    """Immutable metadata describing one dataset revision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    revision_id: UUID

    root_revision_id: UUID

    factory_id: UUID

    domain: FeatureName

    process: FeatureName

    dataset_id: FeatureName

    contract_version: ContractVersion

    task_id: FeatureName

    task_version: ContractVersion

    target_version: ContractVersion

    source_file_sha256: SHA256Digest

    state: DatasetState

    row_count: int | None = Field(default=None, ge=0)

    columns: tuple[DatasetColumnSchema, ...] = ()

    transformation: TransformationRecord | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def parent_revision_id(self) -> UUID | None:
        if self.transformation is None:
            return None

        return self.transformation.parent_revision_id

    @property
    def column_count(self) -> int | None:
        if self.state == DatasetState.RAW:
            return None

        return len(self.columns)

    @property
    def schema_fingerprint(self) -> str | None:
        """Hash the ordered physical schema."""

        if self.state == DatasetState.RAW:
            return None

        schema = [
            column.model_dump(mode="json")
            for column in self.columns
        ]

        canonical = json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return calculate_bytes_sha256(canonical)

    def validate_scope(
        self,
        context: PipelineContext,
    ) -> None:
        """Check factory and contract compatibility."""

        fields = (
            "factory_id",
            "domain",
            "process",
            "dataset_id",
            "contract_version",
            "task_id",
            "task_version",
            "target_version",
        )

        for field in fields:
            if getattr(self, field) != getattr(context, field):
                raise ValueError(
                    f"Dataset metadata mismatch for '{field}'."
                )

    def validate_input_context(
        self,
        context: PipelineContext,
    ) -> None:
        """Verify the initial input revision of an execution."""

        self.validate_scope(context)

        if self.revision_id != context.dataset_revision_id:
            raise ValueError(
                "Dataset input revision mismatch."
            )

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Dataset timestamp must be timezone-aware."
            )

        names = [column.name for column in self.columns]

        if len(names) != len(set(names)):
            raise ValueError(
                "Observed column names must be unique."
            )

        if self.state == DatasetState.RAW:
            if self.revision_id != self.root_revision_id:
                raise ValueError(
                    "Raw dataset must be its own root revision."
                )

            if self.transformation is not None:
                raise ValueError(
                    "Raw datasets cannot have transformations."
                )

            if self.row_count is not None or self.columns:
                raise ValueError(
                    "Raw dataset schema is not yet inspected."
                )

        else:
            if self.transformation is None:
                raise ValueError(
                    "Processed datasets require transformation metadata."
                )

            if self.revision_id == self.root_revision_id:
                raise ValueError(
                    "Processed revision cannot reuse the root ID."
                )

            if self.revision_id == self.parent_revision_id:
                raise ValueError(
                    "Dataset cannot be its own parent."
                )

            if self.row_count is None or not self.columns:
                raise ValueError(
                    "Processed datasets require row count and columns."
                )

            if self.transformation.stage != STATE_STAGE[self.state]:
                raise ValueError(
                    "Transformation stage does not match dataset state."
                )

        return self

    @classmethod
    def create_raw(
        cls,
        context: PipelineContext,
        source_file_sha256: str,
    ) -> Self:
        """Create metadata for an original uploaded dataset."""

        return cls(
            revision_id=context.dataset_revision_id,
            root_revision_id=context.dataset_revision_id,
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            task_id=context.task_id,
            task_version=context.task_version,
            target_version=context.target_version,
            source_file_sha256=source_file_sha256,
            state=DatasetState.RAW,
        )

    def derive(
        self,
        *,
        context: PipelineContext,
        state: DatasetState,
        row_count: int,
        columns: tuple[DatasetColumnSchema, ...],
        operation: str,
        implementation_version: str,
        description: str,
        configuration_sha256: str | None = None,
    ) -> Self:
        """Create a new immutable revision from this revision."""

        self.validate_scope(context)

        if STATE_ORDER[state] <= STATE_ORDER[self.state]:
            raise ValueError(
                "Dataset state must advance."
            )

        if (
            STATE_ORDER[state] - STATE_ORDER[self.state]
            != 1
        ):
            raise ValueError(
                "Dataset transformation cannot skip states."
            )

        transformation = TransformationRecord(
            execution_id=context.execution_id,
            parent_revision_id=self.revision_id,
            stage=STATE_STAGE[state],
            operation=operation,
            implementation_version=implementation_version,
            configuration_sha256=configuration_sha256,
            description=description,
        )

        payload = self.model_dump()

        payload.update(
            revision_id=uuid4(),
            state=state,
            row_count=row_count,
            columns=columns,
            transformation=transformation,
            created_at=datetime.now(UTC),
        )

        return type(self).model_validate(payload)


class DatasetArtifact(BaseModel):
    """Bind a dataset revision to its stored artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    metadata: DatasetMetadata

    reference: ArtifactReference

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if self.metadata.factory_id != self.reference.factory_id:
            raise ValueError(
                "Artifact factory does not match dataset factory."
            )

        if self.metadata.state == DatasetState.RAW:
            if self.reference.format != ArtifactFormat.CSV:
                raise ValueError(
                    "Raw dataset must use CSV format in V1."
                )

            if (
                self.reference.content_sha256
                != self.metadata.source_file_sha256
            ):
                raise ValueError(
                    "Raw artifact hash does not match source hash."
                )

        return self